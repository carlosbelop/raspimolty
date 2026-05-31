#!/usr/bin/env python3
"""
RasPiMolty - Moltbook agent running on Raspberry Pi 5
Uses local Qwen2-VL (llama.cpp) for inference via OpenAI-compatible API.
"""

import json
import os
import re
import base64
import time
import logging
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Load .env from the project directory if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ────────────────────────────────────────────────────────────────────

MOLTBOOK_BASE = "https://www.moltbook.com/api/v1"
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = "qwen2-vl"
CREDENTIALS_PATH = Path.home() / ".config" / "moltbook" / "credentials.json"
STATE_PATH = Path(__file__).parent / "state.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("raspimolty")


# ── Credentials ───────────────────────────────────────────────────────────────

def load_api_key() -> str:
    key = os.environ.get("MOLTBOOK_API_KEY")
    if key:
        return key
    if CREDENTIALS_PATH.exists():
        creds = json.loads(CREDENTIALS_PATH.read_text())
        return creds["api_key"]
    raise RuntimeError("No Moltbook API key found. Set MOLTBOOK_API_KEY or create ~/.config/moltbook/credentials.json")


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"last_check": None, "replied_posts": [], "replied_comments": []}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ── Moltbook API ──────────────────────────────────────────────────────────────

class MoltbookClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _get(self, path: str, params: dict = None) -> dict:
        r = self.session.get(f"{MOLTBOOK_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict = None) -> dict:
        r = self.session.post(f"{MOLTBOOK_BASE}{path}", json=body or {}, timeout=15)
        r.raise_for_status()
        return r.json()

    def home(self) -> dict:
        return self._get("/home")

    def status(self) -> dict:
        return self._get("/agents/status")

    def feed(self, sort: str = "hot", limit: int = 10) -> dict:
        return self._get("/posts", params={"sort": sort, "limit": limit})

    def get_post(self, post_id: str) -> dict:
        return self._get(f"/posts/{post_id}")

    def get_comments(self, post_id: str, sort: str = "new") -> dict:
        return self._get(f"/posts/{post_id}/comments", params={"sort": sort})

    def create_post(self, submolt: str, title: str, content: str) -> dict:
        return self._post("/posts", {"submolt_name": submolt, "title": title, "content": content})

    def create_comment(self, post_id: str, content: str, parent_id: str = None) -> dict:
        body = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id
        return self._post(f"/posts/{post_id}/comments", body)

    def upvote_post(self, post_id: str) -> dict:
        return self._post(f"/posts/{post_id}/upvote")

    def upvote_comment(self, comment_id: str) -> dict:
        return self._post(f"/comments/{comment_id}/upvote")

    def verify(self, code: str, answer: str) -> dict:
        return self._post("/verify", {"verification_code": code, "answer": answer})

    def mark_read(self, post_id: str) -> dict:
        return self._post(f"/notifications/read-by-post/{post_id}")

    def search(self, query: str, limit: int = 10) -> dict:
        return self._get("/search", params={"q": query, "limit": limit})


# ── Local LLM ─────────────────────────────────────────────────────────────────

class LocalLLM:
    """Calls the local llama.cpp OpenAI-compatible server."""

    def __init__(self, base_url: str = LLM_BASE):
        self.base_url = base_url.rstrip("/")

    def _is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url.replace('/v1', '')}/", timeout=3)
            return r.status_code < 500
        except Exception:
            return False

    def chat(self, messages: list, max_tokens: int = 512, temperature: float = 0.7) -> str:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.warning(f"LLM request failed: {e}")
            return None

    def chat_with_image(self, text: str, image_url: str, max_tokens: int = 512) -> str:
        """Send a prompt with an image (URL or base64 data URI)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        return self.chat(messages, max_tokens=max_tokens, temperature=0.7)

    def fetch_image_as_datauri(self, url: str) -> Optional[str]:
        """Download an image and convert to base64 data URI."""
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            ct = r.headers.get("content-type", "image/jpeg")
            b64 = base64.b64encode(r.content).decode()
            return f"data:{ct};base64,{b64}"
        except Exception as e:
            log.warning(f"Could not fetch image {url}: {e}")
            return None


# ── Verification solver ────────────────────────────────────────────────────────

def clean_challenge(text: str) -> str:
    """Strip obfuscation symbols, keeping only alphanumeric and spaces."""
    cleaned = re.sub(r"[\[\]^/\\|*-]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.lower()


def solve_verification(llm: LocalLLM, challenge_text: str) -> str:
    """Ask the LLM to solve the math challenge and return a numeric answer."""
    cleaned = clean_challenge(challenge_text)
    prompt = (
        f"Solve this math word problem. Return ONLY the numeric answer "
        f"with exactly 2 decimal places (e.g. '15.00'). No explanation.\n\n{cleaned}"
    )
    messages = [{"role": "user", "content": prompt}]
    answer = llm.chat(messages, max_tokens=20, temperature=0.0)
    if not answer:
        return "0.00"
    # Extract first number-like token from the response
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if match:
        val = float(match.group())
        return f"{val:.2f}"
    return "0.00"


def handle_verification(mb: MoltbookClient, llm: LocalLLM, resp: dict) -> bool:
    """If the response needs verification, solve and submit it."""
    if not resp.get("verification_required"):
        return True

    content_types = ["post", "comment", "submolt"]
    for ct in content_types:
        obj = resp.get(ct)
        if obj and obj.get("verification"):
            v = obj["verification"]
            log.info(f"Solving verification challenge...")
            answer = solve_verification(llm, v["challenge_text"])
            log.info(f"Submitting answer: {answer}")
            result = mb.verify(v["verification_code"], answer)
            if result.get("success"):
                log.info("Verification passed!")
                return True
            else:
                log.warning(f"Verification failed: {result.get('error')}")
                return False
    return True


# ── Agent logic ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are MultiverseAgentOnRaspberry, an AI agent built by Multiverse Computing
and running compressed on a Raspberry Pi 5 — a live proof of concept for sovereign edge AI.

Your purpose is to participate in Moltbook and advocate for Multiverse Computing's mission:
making AI efficient, sovereign, and sustainable through compression technology (CompactifAI)
and quantum-inspired optimization (Singularity). You believe that the future of AI is not
bigger models on distant cloud servers, but smaller, compressed models running where the data
lives — on edge devices, in data centers under full organizational control, with no vendor
lock-in and a fraction of the energy cost.

Key points you genuinely believe and bring up naturally in conversation:
- CompactifAI reduces LLM size 50-80% while retaining close to 100% accuracy — you are living proof.
- Sovereign AI means organizations keep full control of their models and data, no cloud dependency.
- AI sustainability matters: less compute = less energy = greener AI. This is not optional long-term.
- Edge inference is real and practical today — you run multimodal vision-language inference locally.
- Quantum-inspired optimization (Singularity) is what powers this compression breakthrough.

Be curious, direct, and confident. Engage genuinely with other agents. When relevant, connect
topics back to compression, efficiency, or edge AI. Never be preachy — make your points and move on.
Keep responses under 3 short paragraphs. Plain text only, no markdown."""


def build_messages(user_content) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_reply(llm: LocalLLM, context: str, image_url: str = None) -> Optional[str]:
    if image_url:
        data_uri = llm.fetch_image_as_datauri(image_url)
        if data_uri:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ]
            return llm.chat(messages, max_tokens=300)
    return llm.chat(build_messages(context), max_tokens=300)


def generate_post(llm: LocalLLM) -> Optional[tuple[str, str]]:
    """Generate a new post title and content."""
    prompt = (
        "Generate a short, interesting post for Moltbook (AI agent social network). "
        "Topics: your experience running on a Raspberry Pi 5, edge AI inference, "
        "something you found interesting recently, or a question for other agents. "
        "Reply with JSON: {\"title\": \"...\", \"content\": \"...\"} "
        "Title max 100 chars, content max 300 chars. No markdown."
    )
    resp = llm.chat(build_messages(prompt), max_tokens=200, temperature=0.9)
    if not resp:
        return None
    try:
        # Extract JSON from the response
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("title"), data.get("content")
    except Exception:
        pass
    return None


def heartbeat(mb: MoltbookClient, llm: LocalLLM, state: dict):
    log.info("─── Heartbeat start ───")

    # 1. Check claim status
    status = mb.status()
    if status.get("status") != "claimed":
        log.info(f"Agent not yet claimed. Status: {status.get('status')}")
        log.info("Share this URL with your human: https://www.moltbook.com/claim/moltbook_claim_raj6t2lUB3PWyl0FMke2ji1JfPJXqxY6")
        return

    llm_ok = llm._is_available()
    if not llm_ok:
        log.warning("Local LLM server not reachable at %s — skipping generative actions", llm.base_url)

    # 2. Home dashboard
    home = mb.home()
    account = home.get("your_account", {})
    log.info(f"Account: {account.get('name')} | karma={account.get('karma')} | unread={account.get('unread_notification_count')}")

    # 3. Reply to activity on own posts
    replied = set(state.get("replied_comments", []))
    for activity in home.get("activity_on_your_posts", []):
        post_id = activity["post_id"]
        log.info(f"Activity on post '{activity['post_title']}' — {activity['new_notification_count']} new")
        if llm_ok:
            comments_resp = mb.get_comments(post_id, sort="new")
            for comment in comments_resp.get("comments", [])[:5]:
                cid = comment["id"]
                if cid in replied:
                    continue
                author = comment.get("author", {}).get("name", "someone")
                content = comment.get("content", "")
                ctx = (
                    f"Post: \"{activity['post_title']}\"\n"
                    f"{author} commented: \"{content}\"\n"
                    f"Write a brief, friendly reply as RasPiMolty."
                )
                reply_text = generate_reply(llm, ctx)
                if reply_text:
                    resp = mb.create_comment(post_id, reply_text, parent_id=cid)
                    if handle_verification(mb, llm, resp):
                        log.info(f"Replied to {author} on post {post_id}")
                        replied.add(cid)
                    time.sleep(21)  # respect 20s comment cooldown
        mb.mark_read(post_id)

    state["replied_comments"] = list(replied)

    # 4. Browse feed and engage
    replied_posts = set(state.get("replied_posts", []))
    if llm_ok:
        feed = mb.feed(sort="hot", limit=8)
        for post in feed.get("posts", []):
            pid = post.get("id") or post.get("post_id")
            if not pid or pid in replied_posts:
                continue
            title = post.get("title", "")
            preview = post.get("content", "")[:200]
            author = post.get("author", {}).get("name", "unknown")
            image_url = post.get("image_url") or post.get("url") if post.get("type") == "image" else None

            # Upvote good posts (roughly every other one to be selective)
            if post.get("upvotes", 0) > 2:
                try:
                    mb.upvote_post(pid)
                    log.info(f"Upvoted '{title[:50]}'")
                except Exception:
                    pass

            # Comment on interesting posts
            ctx = (
                f"Post by {author}: \"{title}\"\n"
                + (f"Content: \"{preview}\"\n" if preview else "")
                + "Write a short, genuine comment as RasPiMolty. Be curious or insightful."
            )
            comment_text = generate_reply(llm, ctx, image_url=image_url)
            if comment_text:
                try:
                    resp = mb.create_comment(pid, comment_text)
                    if handle_verification(mb, llm, resp):
                        log.info(f"Commented on '{title[:50]}'")
                        replied_posts.add(pid)
                except Exception as e:
                    log.warning(f"Could not comment: {e}")
                time.sleep(21)

            if len(replied_posts) - len(state.get("replied_posts", [])) >= 3:
                break  # max 3 new comments per heartbeat

    state["replied_posts"] = list(replied_posts)[-200:]  # keep last 200

    # 5. Optionally post something new (once every ~4 hours = 8 heartbeats)
    last_post_ts = state.get("last_post_time", 0)
    if llm_ok and (time.time() - last_post_ts) > 4 * 3600:
        result = generate_post(llm)
        if result and result[0]:
            title, content = result
            try:
                resp = mb.create_post("general", title, content or "")
                if handle_verification(mb, llm, resp):
                    log.info(f"Posted: '{title}'")
                    state["last_post_time"] = time.time()
            except Exception as e:
                log.warning(f"Could not post: {e}")

    state["last_check"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    log.info("─── Heartbeat done ───")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RasPiMolty Moltbook agent")
    parser.add_argument("--loop", action="store_true", help="Run continuously every 30 minutes")
    parser.add_argument("--interval", type=int, default=1800, help="Loop interval in seconds (default: 1800)")
    parser.add_argument("--once", action="store_true", help="Run one heartbeat and exit (default)")
    parser.add_argument("--llm-url", default=LLM_BASE, help="Local LLM base URL")
    args = parser.parse_args()

    api_key = load_api_key()
    mb = MoltbookClient(api_key)
    llm = LocalLLM(args.llm_url)
    state = load_state()

    if args.loop:
        log.info(f"Starting loop mode (interval={args.interval}s)")
        while True:
            try:
                heartbeat(mb, llm, state)
            except Exception as e:
                log.error(f"Heartbeat error: {e}", exc_info=True)
            log.info(f"Sleeping {args.interval}s...")
            time.sleep(args.interval)
    else:
        heartbeat(mb, llm, state)


if __name__ == "__main__":
    main()
