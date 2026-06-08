#!/usr/bin/env python3
"""
VLM evaluation script for llama.cpp server on Raspberry Pi.

Two modes, detected automatically from the config file top-level key:

  evaluations:   concurrency sweep per model — produces the structured table
                 with ITL / TTFT / e2e latency / throughput / req·min⁻¹
                 (matches the LaTeX evaluation table format)

  benchmarks:    legacy per-config mode with p50/p90/p99 stats

Usage:
    python3 run_benchmark_rpi.py [config.yaml]

Default config: benchmark_config_rpi.yaml
"""

import csv
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
import yaml


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    request_idx: int
    prompt_tokens: int
    output_tokens: int
    ttft: float           # s — time to first token
    total_latency: float  # s — full e2e wall time
    itl: float            # s — inter-token latency (= TPOT, excl. first token)
    throughput: float     # tok/s — output_tokens / total_latency (per request)
    success: bool
    error: Optional[str] = None


@dataclass
class EvalRow:
    """One row of the evaluation table (one model × one concurrency level)."""
    model_name: str
    input_tokens: int
    input_tokens_std_pct: float
    output_tokens: int
    output_tokens_std_pct: float
    concurrency: int
    max_completed: int
    itl_mean: float
    itl_std: float
    ttft_mean: float
    ttft_std: float
    e2e_mean: float
    e2e_std: float
    throughput_mean: float   # aggregate: total_output_tokens / wall_time
    throughput_std: float    # per-request std
    req_per_min: float


# ── prompt generation ─────────────────────────────────────────────────────────

_WORDS = [
    "the", "a", "an", "is", "are", "was", "were", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "must", "can", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "up", "about", "into", "through", "before", "after", "above",
    "below", "between", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "some", "other", "than", "then",
    "that", "this", "man", "woman", "time", "year", "day", "way", "thing",
    "people", "hand", "part", "place", "case", "week", "number", "night",
    "point", "home", "water", "room", "area", "money", "story", "fact",
    "system", "world", "city", "name", "even", "back", "life", "long",
]


def generate_prompt(token_len: int, std_pct: float, seed: int, idx: int) -> str:
    """
    Return a filler text prompt of approximately `token_len` tokens.
    std_pct is the ±percentage variance applied to the target length (0.1 = 0.1%).
    1 token ≈ 4 chars for this vocabulary.
    """
    rng = random.Random(seed + idx)
    target_chars = token_len * 4
    variance = max(1, int(target_chars * std_pct / 100.0))
    actual_chars = target_chars + rng.randint(-variance, variance)
    parts: list[str] = []
    length = 0
    while length < actual_chars:
        word = rng.choice(_WORDS)
        parts.append(word)
        length += len(word) + 1
    return " ".join(parts)


# ── HTTP request ──────────────────────────────────────────────────────────────

def send_request(
    host: str,
    port: int,
    endpoint: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    request_idx: int,
    timeout: int = 600,
) -> RequestResult:
    """
    POST one streaming chat-completion request.
    Parses SSE to measure TTFT, per-chunk token count, and total latency.
    """
    url = f"http://{host}:{port}{endpoint}"
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "temperature": 0.0,
        "cache_prompt": False,
    }

    start = time.perf_counter()
    ttft: Optional[float] = None
    output_tokens = 0
    prompt_tokens = 0

    try:
        with requests.post(url, json=body, stream=True, timeout=timeout) as resp:
            if resp.status_code != 200:
                return RequestResult(
                    request_idx=request_idx, prompt_tokens=0, output_tokens=0,
                    ttft=0.0, total_latency=0.0, itl=0.0, throughput=0.0,
                    success=False, error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if choices:
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        output_tokens += 1  # llama.cpp sends 1 token per SSE chunk

                usage = chunk.get("usage")
                if usage:
                    output_tokens = usage.get("completion_tokens", output_tokens)
                    prompt_tokens = usage.get("prompt_tokens", 0)

    except requests.exceptions.Timeout:
        return RequestResult(
            request_idx=request_idx, prompt_tokens=0, output_tokens=0,
            ttft=0.0, total_latency=0.0, itl=0.0, throughput=0.0,
            success=False, error="Timeout",
        )
    except Exception as exc:
        return RequestResult(
            request_idx=request_idx, prompt_tokens=0, output_tokens=0,
            ttft=0.0, total_latency=0.0, itl=0.0, throughput=0.0,
            success=False, error=str(exc),
        )

    total_latency = time.perf_counter() - start
    if ttft is None:
        ttft = total_latency

    # ITL = time per output token excluding the first (pure decode speed)
    itl = (total_latency - ttft) / max(output_tokens - 1, 1)
    throughput = output_tokens / total_latency if total_latency > 0 else 0.0

    return RequestResult(
        request_idx=request_idx,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft=ttft,
        total_latency=total_latency,
        itl=itl,
        throughput=throughput,
        success=True,
    )


# ── statistics helpers ────────────────────────────────────────────────────────

def _std(data: list[float]) -> float:
    return statistics.stdev(data) if len(data) >= 2 else 0.0


def pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (p / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def stat_block(data: list[float]) -> dict:
    return {
        "mean":   round(statistics.mean(data), 4),
        "std":    round(_std(data), 4),
        "median": round(pct(data, 50), 4),
        "p90":    round(pct(data, 90), 4),
        "p99":    round(pct(data, 99), 4),
        "min":    round(min(data), 4),
        "max":    round(max(data), 4),
    }


# ── evaluation mode ───────────────────────────────────────────────────────────

def _run_level(
    model_name: str,
    input_tokens: int, input_std_pct: float,
    output_tokens: int, output_std_pct: float,
    concurrency: int, num_requests: int,
    endpoint: str, seed: int,
    host: str, port: int, timeout: int,
    warmup_n: int,
) -> EvalRow:
    """Run one (model, concurrency) cell and return its EvalRow."""

    print(f"\n  ── concurrency={concurrency}  requests={num_requests} ──")

    # Warm-up (discarded)
    if warmup_n > 0:
        wp = generate_prompt(input_tokens, input_std_pct, seed - 1, 0)
        print(f"     warm-up {warmup_n} req …", end=" ", flush=True)
        for _ in range(warmup_n):
            send_request(host, port, endpoint, wp, output_tokens, seed, -1, timeout)
        print("done")

    prompts = [
        generate_prompt(input_tokens, input_std_pct, seed, i)
        for i in range(num_requests)
    ]

    results: list[RequestResult] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                send_request,
                host, port, endpoint, prompts[i],
                output_tokens, seed + i, i, timeout,
            ): i
            for i in range(num_requests)
        }
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            tag = "OK" if r.success else f"FAIL({r.error})"
            print(f"     [{done:3d}/{num_requests}]  "
                  f"ttft={r.ttft:.3f}s  e2e={r.total_latency:.3f}s  "
                  f"itl={r.itl:.4f}s  out={r.output_tokens}tok  {tag}")

    wall_elapsed = time.perf_counter() - wall_start

    good = [r for r in results if r.success]
    if not good:
        print("     ERROR: all requests failed.")
        return None

    itls       = [r.itl           for r in good]
    ttfts      = [r.ttft          for r in good]
    e2es       = [r.total_latency for r in good]
    per_req_tp = [r.throughput    for r in good]
    total_out  = sum(r.output_tokens for r in good)

    agg_throughput = total_out / wall_elapsed  # total tok/s across all workers

    return EvalRow(
        model_name=model_name,
        input_tokens=input_tokens,
        input_tokens_std_pct=input_std_pct,
        output_tokens=output_tokens,
        output_tokens_std_pct=output_std_pct,
        concurrency=concurrency,
        max_completed=len(good),
        itl_mean=round(statistics.mean(itls), 6),
        itl_std=round(_std(itls), 6),
        ttft_mean=round(statistics.mean(ttfts), 6),
        ttft_std=round(_std(ttfts), 6),
        e2e_mean=round(statistics.mean(e2es), 6),
        e2e_std=round(_std(e2es), 6),
        throughput_mean=round(agg_throughput, 3),
        throughput_std=round(_std(per_req_tp), 3),
        req_per_min=round(len(good) / wall_elapsed * 60, 3),
    )


def run_evaluation(eval_cfg: dict, global_cfg: dict, results_dir: Path) -> list[EvalRow]:
    """Sweep all concurrency levels for one model entry."""
    model_name      = eval_cfg["model"]["name"]
    input_tokens    = eval_cfg["input_tokens"]
    input_std_pct   = eval_cfg.get("input_tokens_std_pct", 0.1)
    output_tokens   = eval_cfg["output_tokens"]
    output_std_pct  = eval_cfg.get("output_tokens_std_pct", 0.1)
    concur_levels   = eval_cfg["concurrency_levels"]
    req_per_level   = eval_cfg.get("requests_per_level", 5)
    seed            = eval_cfg.get("seed", 42)
    endpoint        = eval_cfg.get("endpoint", "/v1/chat/completions")

    host    = global_cfg.get("server_host", "localhost")
    port    = global_cfg.get("server_port", 8080)
    timeout = global_cfg.get("request_timeout", 600)
    warmup  = global_cfg.get("warmup_requests", 1)

    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}")
    print(f"  input≈{input_tokens}tok (±{input_std_pct}%)  "
          f"output≈{output_tokens}tok (±{output_std_pct}%)")
    print(f"  concurrency levels: {concur_levels}")
    print(f"{'='*70}")

    rows: list[EvalRow] = []
    for concurrency in concur_levels:
        num_requests = req_per_level * concurrency
        row = _run_level(
            model_name=model_name,
            input_tokens=input_tokens, input_std_pct=input_std_pct,
            output_tokens=output_tokens, output_std_pct=output_std_pct,
            concurrency=concurrency, num_requests=num_requests,
            endpoint=endpoint, seed=seed,
            host=host, port=port, timeout=timeout, warmup_n=warmup,
        )
        if row:
            rows.append(row)

    # Save per-model JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    out_file = results_dir / f"eval_{safe_name}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
    print(f"\n  Saved → {out_file}")

    return rows


def print_eval_table(all_rows: list[EvalRow]) -> None:
    """Print the evaluation table matching the LaTeX format."""
    sep  = "+" + "-"*24 + "+" + "-"*7 + "+" + "-"*9 + "+" + "-"*7 + "+" + "-"*9 + \
           "+" + "-"*6 + "+" + "-"*8 + \
           "+" + "-"*8 + "+" + "-"*8 + \
           "+" + "-"*8 + "+" + "-"*8 + \
           "+" + "-"*8 + "+" + "-"*8 + \
           "+" + "-"*10 + "+" + "-"*8 + \
           "+" + "-"*10 + "+"
    hdr1 = (
        f"| {'CONFIGURATION':^55} | {'PERFORMANCE  METRICS':^79} |"
    )
    hdr2 = (
        f"| {'Model':<22} | {'In':>5} | {'In std%':>7} | {'Out':>5} | {'Out std%':>7} "
        f"| {'Conc':>4} | {'MaxReq':>6} "
        f"| {'ITL μ':>6} | {'ITL σ':>6} "
        f"| {'TTFT μ':>6} | {'TTFT σ':>6} "
        f"| {'e2e μ':>6} | {'e2e σ':>6} "
        f"| {'Thru μ':>8} | {'Thru σ':>6} "
        f"| {'req/min':>8} |"
    )
    hdr3 = (
        f"| {'':22} | {'tok':>5} | {'%':>7} | {'tok':>5} | {'%':>7} "
        f"| {' ':>4} | {' ':>6} "
        f"| {'(s)':>6} | {'(s)':>6} "
        f"| {'(s)':>6} | {'(s)':>6} "
        f"| {'(s)':>6} | {'(s)':>6} "
        f"| {'tok/s':>8} | {'tok/s':>6} "
        f"| {' ':>8} |"
    )

    wide_sep = "=" * len(hdr2)
    print(f"\n{wide_sep}")
    print(hdr1)
    print(wide_sep)
    print(hdr2)
    print(hdr3)
    print(wide_sep)

    for r in all_rows:
        print(
            f"| {r.model_name:<22} "
            f"| {r.input_tokens:>5} "
            f"| {r.input_tokens_std_pct:>7.1f} "
            f"| {r.output_tokens:>5} "
            f"| {r.output_tokens_std_pct:>7.1f} "
            f"| {r.concurrency:>4} "
            f"| {r.max_completed:>6} "
            f"| {r.itl_mean:>6.4f} "
            f"| {r.itl_std:>6.4f} "
            f"| {r.ttft_mean:>6.3f} "
            f"| {r.ttft_std:>6.3f} "
            f"| {r.e2e_mean:>6.3f} "
            f"| {r.e2e_std:>6.3f} "
            f"| {r.throughput_mean:>8.3f} "
            f"| {r.throughput_std:>6.3f} "
            f"| {r.req_per_min:>8.2f} |"
        )
    print(wide_sep)


def save_csv(all_rows: list[EvalRow], path: Path) -> None:
    fieldnames = list(EvalRow.__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))
    print(f"CSV  → {path}")


# ── legacy benchmark mode ─────────────────────────────────────────────────────

def run_benchmark(cfg: dict, global_cfg: dict) -> dict:
    bench_name  = cfg["bench_name"]
    model_name  = cfg["model"]["name"]
    b           = cfg["benchmark"]

    host        = global_cfg.get("server_host", "localhost")
    port        = global_cfg.get("server_port", 8080)
    results_dir = Path(global_cfg.get("results_base_dir", "./benchmark_results"))
    timeout     = global_cfg.get("request_timeout", 600)
    warmup_n    = global_cfg.get("warmup_requests", 1)

    results_dir.mkdir(parents=True, exist_ok=True)

    num_prompts     = b["num_prompts"]
    max_concurrency = b["max_concurrency"]
    input_len       = b["random_input_len"]
    output_len      = b["random_output_len"]
    range_ratio     = b.get("random_range_ratio", 0.01)
    seed            = b.get("seed", 42)
    endpoint        = b.get("endpoint", "/v1/chat/completions")

    print(f"\n{'='*65}")
    print(f"  {bench_name}  |  {model_name}")
    print(f"  prompts={num_prompts}  concurrency={max_concurrency}  "
          f"in≈{input_len}tok  out≈{output_len}tok")
    print(f"{'='*65}")

    if warmup_n > 0:
        print(f"  Warm-up: {warmup_n} req …", end=" ", flush=True)
        wp = generate_prompt(input_len, range_ratio * 100, seed - 1, 0)
        for _ in range(warmup_n):
            send_request(host, port, endpoint, wp, output_len, seed, -1, timeout)
        print("done")

    prompts = [
        generate_prompt(input_len, range_ratio * 100, seed, i)
        for i in range(num_prompts)
    ]

    results: list[RequestResult] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        futures = {
            pool.submit(
                send_request,
                host, port, endpoint, prompts[i],
                output_len, seed + i, i, timeout,
            ): i
            for i in range(num_prompts)
        }
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            tag = "OK" if r.success else f"FAIL — {r.error}"
            print(f"    [{done:3d}/{num_prompts}]  "
                  f"ttft={r.ttft:.3f}s  e2e={r.total_latency:.3f}s  "
                  f"out={r.output_tokens}  {tag}")

    wall_elapsed = time.perf_counter() - wall_start
    good   = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not good:
        print("  ERROR: all requests failed.")
        return {}

    summary = {
        "bench_name": bench_name,
        "model_name": model_name,
        "timestamp":  datetime.now().isoformat(),
        "config": {
            "num_prompts": num_prompts, "max_concurrency": max_concurrency,
            "random_input_len": input_len, "random_output_len": output_len, "seed": seed,
        },
        "metrics": {
            "total_wall_time_s":        round(wall_elapsed, 3),
            "successful_requests":      len(good),
            "failed_requests":          len(failed),
            "requests_per_second":      round(len(good) / wall_elapsed, 4),
            "output_tokens_per_second": round(sum(r.output_tokens for r in good) / wall_elapsed, 3),
            "itl_s":     stat_block([r.itl           for r in good]),
            "ttft_s":    stat_block([r.ttft          for r in good]),
            "latency_s": stat_block([r.total_latency for r in good]),
        },
        "detailed_results": [asdict(r) for r in results],
    }

    m = summary["metrics"]
    print(f"\n  Wall time : {wall_elapsed:.2f}s")
    print(f"  Throughput: {m['requests_per_second']:.4f} req/s | {m['output_tokens_per_second']:.2f} tok/s")
    print(f"  TTFT  μ/σ : {m['ttft_s']['mean']:.3f}s / {m['ttft_s']['std']:.3f}s")
    print(f"  e2e   μ/σ : {m['latency_s']['mean']:.3f}s / {m['latency_s']['std']:.3f}s")
    print(f"  ITL   μ/σ : {m['itl_s']['mean']:.4f}s / {m['itl_s']['std']:.4f}s")
    print(f"  OK/Fail   : {len(good)}/{len(failed)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"{bench_name}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {out_file}")
    return summary


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_config_rpi.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    global_cfg  = config.get("global", {})
    host        = global_cfg.get("server_host", "localhost")
    port        = global_cfg.get("server_port", 8080)
    results_dir = Path(global_cfg.get("results_base_dir", "./eval_results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config : {config_path}")
    print(f"Server : http://{host}:{port}")

    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=10)
        print(f"Health : HTTP {r.status_code}")
    except Exception as exc:
        print(f"Health : WARN — {exc}")
        print("         Make sure the server is up:  docker compose up -d")

    # ── evaluation mode ───────────────────────────────────────────────────────
    if "evaluations" in config:
        evals = config["evaluations"]
        print(f"Mode   : evaluation  ({len(evals)} model(s))\n")

        all_rows: list[EvalRow] = []
        for idx, eval_cfg in enumerate(evals, 1):
            print(f"[{idx}/{len(evals)}] {eval_cfg['model']['name']}")
            rows = run_evaluation(eval_cfg, global_cfg, results_dir)
            all_rows.extend(rows)

        print_eval_table(all_rows)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = results_dir / f"evaluation_{ts}.csv"
        save_csv(all_rows, csv_path)

        json_path = results_dir / f"evaluation_{ts}.json"
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in all_rows], f, indent=2)
        print(f"JSON → {json_path}")

    # ── legacy benchmark mode ─────────────────────────────────────────────────
    elif "benchmarks" in config:
        benchmarks = config["benchmarks"]
        print(f"Mode   : benchmark  ({len(benchmarks)} config(s))\n")

        all_results: list[dict] = []
        for idx, bench_cfg in enumerate(benchmarks, 1):
            print(f"\n[{idx}/{len(benchmarks)}] {bench_cfg['bench_name']}")
            result = run_benchmark(bench_cfg, global_cfg)
            if result:
                all_results.append(result)

        if len(all_results) > 1:
            w = "=" * 95
            print(f"\n{w}\nSUMMARY\n{w}")
            hdr = (f"{'benchmark':<28} {'in':>5} {'out':>5} {'conc':>5} "
                   f"{'req/s':>8} {'tok/s':>8} {'TTFT μ':>8} {'TTFT σ':>8} "
                   f"{'e2e μ':>8} {'e2e σ':>8} {'ITL μ':>8} {'ITL σ':>8}")
            print(hdr)
            print("-" * 95)
            for r in all_results:
                c, m = r["config"], r["metrics"]
                print(
                    f"{r['bench_name']:<28}"
                    f"{c['random_input_len']:>5}"
                    f"{c['random_output_len']:>6}"
                    f"{c['max_concurrency']:>6}"
                    f"{m['requests_per_second']:>9.4f}"
                    f"{m['output_tokens_per_second']:>9.2f}"
                    f"{m['ttft_s']['mean']:>9.3f}"
                    f"{m['ttft_s']['std']:>9.3f}"
                    f"{m['latency_s']['mean']:>9.3f}"
                    f"{m['latency_s']['std']:>9.3f}"
                    f"{m['itl_s']['mean']:>9.4f}"
                    f"{m['itl_s']['std']:>9.4f}"
                )

    else:
        print("ERROR: config must contain 'evaluations' or 'benchmarks' key.")
        sys.exit(1)

    print(f"\nResults in: {results_dir}/")


if __name__ == "__main__":
    main()
