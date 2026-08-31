"""
ML Gateway client for Delphi judging pipeline.
Wraps Claude (Anthropic /v1/messages) and OpenAI-compatible (/v1/chat/completions) models.
All models route through the same gateway base URL.
"""
import os, time, json, base64, logging
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv(Path(__file__).parent.parent / ".env")

BASE_URL = os.getenv("LLM_API_BASE_URL", "").rstrip("/")
API_KEY  = os.getenv("LLM_API_KEY", "")

JUDGES = {
    "claude": {
        "model_id": "anthropic/claude-opus-4-6",
        "endpoint": "/v1/messages",
        "api_type": "anthropic",
        "max_tokens": 4096,
    },
    "gpt": {
        "model_id": "openai/gpt-4.1",
        "endpoint": "/v1/chat/completions",
        "api_type": "openai",
        "max_tokens": 4096,
    },
    "gemini": {
        "model_id": "google/gemini-2.5-pro",
        "endpoint": "/v1/chat/completions",
        "api_type": "openai",
        "max_tokens": 3000,  # thinking model; full 4096 slows it without adding output quality
    },
}

logger = logging.getLogger(__name__)


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _call_anthropic(model_id: str, system: str, user_text: str,
                    image_b64: str, max_tokens: int = 1024,
                    retries: int = 3) -> dict:
    content = []
    if image_b64:
        content.append({"type": "image", "source": {"type": "base64",
            "media_type": "image/png", "data": image_b64}})
    content.append({"type": "text", "text": user_text})

    payload = {"model": model_id, "system": system,
               "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens}
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE_URL}/v1/messages",
                headers={"Authorization": API_KEY, "Content-Type": "application/json"},
                json=payload, timeout=120)
            if r.status_code == 200:
                return {"ok": True, "text": r.json()["content"][0]["text"]}
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10)
            else:
                return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "max retries exceeded"}


def _call_openai_compat(model_id: str, system: str, user_text: str,
                         image_b64: str, max_tokens: int = 1024,
                         retries: int = 3) -> dict:
    user_content = []
    if image_b64:
        user_content.append({"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    user_content.append({"type": "text", "text": user_text})

    payload = {"model": model_id, "max_completion_tokens": max_tokens,
               "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user_content}]}
    for attempt in range(retries):
        try:
            r = requests.post(f"{BASE_URL}/v1/chat/completions",
                headers={"Authorization": API_KEY, "Content-Type": "application/json"},
                json=payload, timeout=120)
            if r.status_code == 200:
                body = r.json()
                choices = body.get("choices") or []
                if not choices:
                    return {"ok": False, "error": f"Malformed response (no choices): {str(body)[:300]}"}

                first = choices[0]

                # Truncated response — finish_reason=length with no content (common in
                # thinking models that exhaust max_tokens on reasoning before writing output)
                if first.get("finish_reason") == "length" and not first.get("message"):
                    reasoning = (body.get("usage") or {}).get(
                        "completion_tokens_details", {}).get("reasoning_tokens", "?")
                    return {"ok": False, "error": (
                        f"Response truncated (finish_reason=length): model used {reasoning} "
                        f"reasoning tokens and produced no output — increase max_tokens")}

                # Most OpenAI-compatible gateways return message.content, but some variants
                # return text directly or content as a list of typed chunks.
                if isinstance(first.get("text"), str):
                    return {"ok": True, "text": first["text"]}

                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        if not content.strip():
                            logger.warning("Empty content from model %s — full body: %s",
                                           model_id, str(body)[:1000])
                            return {"ok": False, "error": f"Empty response content from {model_id}"}
                        return {"ok": True, "text": content}
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text_parts.append(part["text"])
                        if text_parts:
                            return {"ok": True, "text": "\n".join(text_parts)}

                return {"ok": False, "error": f"Malformed response schema: {str(body)[:600]}"}
            if r.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(10)
            else:
                return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "max retries exceeded"}


def call_judge(judge_name: str, system: str, user_text: str,
               image_path: str = None, max_tokens: int = None) -> dict:
    """Call a judge and return {'ok': bool, 'text': str, 'error': str}."""
    cfg = JUDGES[judge_name]
    if max_tokens is None:
        max_tokens = cfg["max_tokens"]
    image_b64 = _encode_image(image_path) if image_path else None
    if cfg["api_type"] == "anthropic":
        return _call_anthropic(cfg["model_id"], system, user_text, image_b64, max_tokens)
    else:
        return _call_openai_compat(cfg["model_id"], system, user_text, image_b64, max_tokens)


def summarize_dissent(criterion: str, scores: list, reasonings: list) -> str:
    """Use Claude to produce anonymized dissent summary between rounds."""
    system = "You are aggregating expert disagreement for a Delphi process. Be factual and neutral."
    text = (
        f'Below are reasoning texts from three anonymous judges who rated a 3D asset on '
        f'"{criterion}". Their scores were {scores}. '
        f'Reasoning texts:\n\n'
        + "\n\n---\n\n".join(f"Judge {i+1}: {r}" for i, r in enumerate(reasonings))
        + "\n\nProduce a 3–4 sentence summary capturing the main points of disagreement, "
          "without revealing which judge said what."
    )
    result = _call_anthropic(JUDGES["claude"]["model_id"], system, text, None, 512)
    if result["ok"]:
        return result["text"]
    logger.warning("Dissent summary failed: %s", result.get("error"))
    return "Judges differed on this criterion; see raw round data."


def test_all_judges() -> bool:
    """Smoke-test all three judges. Returns True if all respond."""
    ok = True
    for name in JUDGES:
        result = call_judge(name,
            system="You are a helpful assistant.",
            user_text="Reply with exactly: OK",
            max_tokens=200)  # thinking models (Gemini 2.5 Pro) need headroom beyond reasoning tokens
        status = "✓" if result["ok"] else f"✗ ({result.get('error','')})"
        print(f"  {name} ({JUDGES[name]['model_id']}): {status}")
        if not result["ok"]:
            ok = False
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing ML gateway connectivity...")
    all_ok = test_all_judges()
    print("All judges OK:", all_ok)
