"""Jev (TypeSafe System One) via TypeSafe, OpenRouter, or Vercel AI Gateway.

Not a chat model. Do not send it to /chat/completions.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .paths import ROOT
from .traces import maybe_traceable

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
GATEWAY_URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
TYPESAFE_MODEL = "jev-1.13.0"
OPENROUTER_MODEL = "typesafe/jev-1.13"
GATEWAY_MODEL = "typesafe-ai/jev"


@dataclass(frozen=True)
class JevAnswers:
    answers: dict[str, dict[str, Any]]
    model: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, Any]

    def choice(self, qid: str) -> str:
        answer = self.answers[qid]
        picked = answer.get("choice")
        if picked:
            return str(picked)
        probs = answer.get("probabilities") or {}
        if not probs:
            raise KeyError(f"no choice in {qid!r}")
        return max(probs, key=probs.get)

    def probability(self, qid: str) -> float:
        answer = self.answers[qid]
        if "noul" in answer:
            return float(answer["noul"])
        if "probability" in answer:
            return float(answer["probability"])
        probs = answer.get("probabilities") or {}
        if not probs:
            raise KeyError(f"no probability in {qid!r}")
        return float(max(probs.values()))


def load_env() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")


def _maybe_refresh_oidc() -> None:
    if os.getenv("VERCEL_OIDC_TOKEN") or os.getenv("AI_GATEWAY_API_KEY"):
        return
    if not (ROOT / ".vercel" / "project.json").exists():
        return
    handle, tmp_name = tempfile.mkstemp(prefix="xword-oidc-", suffix=".env")
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        subprocess.run(
            ["vercel", "env", "pull", str(tmp), "--yes"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=90,
        )
        load_dotenv(tmp, override=False)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    finally:
        tmp.unlink(missing_ok=True)


def resolve_jev() -> dict[str, str]:
    load_env()
    if os.getenv("TYPESAFE_API_KEY"):
        return {"kind": "typesafe", "url": TYPESAFE_URL, "model": TYPESAFE_MODEL}
    if os.getenv("OPENROUTER_API_KEY"):
        return {"kind": "openrouter", "url": OPENROUTER_URL, "model": OPENROUTER_MODEL}
    _maybe_refresh_oidc()
    if os.getenv("AI_GATEWAY_API_KEY") or os.getenv("VERCEL_OIDC_TOKEN"):
        return {"kind": "gateway", "url": GATEWAY_URL, "model": GATEWAY_MODEL}
    raise RuntimeError(
        "No Jev credentials. Set TYPESAFE_API_KEY, OPENROUTER_API_KEY, "
        "AI_GATEWAY_API_KEY, or VERCEL_OIDC_TOKEN."
    )


def _translate_questions(questions: dict[str, Any], *, kind: str) -> dict[str, Any]:
    if kind != "gateway":
        return questions
    translated = {}
    for qid, question in questions.items():
        item = dict(question)
        if item.get("type") == "noul":
            item["type"] = "boolean"
        translated[qid] = item
    return translated


def _headers(kind: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if kind == "typesafe":
        headers["Authorization"] = f"Bearer {os.environ['TYPESAFE_API_KEY']}"
    elif kind == "openrouter":
        headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"
    else:
        token = os.getenv("AI_GATEWAY_API_KEY") or os.environ["VERCEL_OIDC_TOKEN"]
        headers["Authorization"] = f"Bearer {token}"
        headers["ai-gateway-protocol-version"] = "0.0.1"
        headers["ai-gateway-auth-method"] = (
            "api-key" if os.getenv("AI_GATEWAY_API_KEY") else "oidc"
        )
        headers["ai-evaluation-model-specification-version"] = "4"
        headers["ai-model-id"] = GATEWAY_MODEL
    return headers


def _payload(kind: str, model: str, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "state": state,
        "questions": _translate_questions(questions, kind=kind),
    }
    if kind != "gateway":
        body["model"] = model
    return body


@maybe_traceable("jev.system_one")
def system_one(
    state: Any,
    questions: dict[str, Any],
    *,
    timeout: float = 30.0,
    retries: int = 4,
) -> JevAnswers:
    route = resolve_jev()
    body = json.dumps(_payload(route["kind"], route["model"], state, questions)).encode()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            route["url"], data=body, headers=_headers(route["kind"]), method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            last_error = RuntimeError(f"Jev {route['kind']} HTTP {exc.code}: {detail}")
            if exc.code not in {408, 429, 503, 529} or attempt == retries:
                raise last_error from exc
            time.sleep(1.5 * (attempt + 1))
        except TimeoutError as exc:
            last_error = exc
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    else:
        raise last_error or RuntimeError("Jev request failed")

    answers = raw.get("answers") or {}
    usage = raw.get("usage") or {}
    return JevAnswers(
        answers=answers,
        model=str(raw.get("model") or route["model"]),
        endpoint=route["kind"],
        input_tokens=int(usage.get("input_tokens") or usage.get("inputTokens") or 0),
        output_tokens=int(usage.get("output_tokens") or usage.get("outputTokens") or 0),
        raw=raw,
    )
