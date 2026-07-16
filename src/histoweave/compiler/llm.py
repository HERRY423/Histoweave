"""LLM providers, including a deterministic offline compiler used by CI."""

from __future__ import annotations

import json
from typing import Any

from .templates import template_for_question


class CompilerProviderError(RuntimeError):
    """The configured model provider could not return a usable response."""


def _reject_nonfinite_json(token: str) -> None:
    raise ValueError(f"LLM JSON contains non-finite constant {token!r}")


def mock_response(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compile common questions without network access or model credentials."""
    q = question.casefold()
    context = context or {}
    assay = str(context.get("assay") or ("xenium" if "xenium" in q else "unknown"))
    response = template_for_question(question)
    response["assay_assumed"] = assay

    if "margin" in q or "boundary" in q:
        response["gaps"].append(
            {
                "concept": "explicit invasive-margin or boundary ROI",
                "reason": "the registry has no region-operations category",
                "degraded_to": "BANKSY domains plus spatial-neighbour graph approximation",
            }
        )
    return response


def request_plan(
    *,
    model: str,
    messages: list[dict[str, str]],
    question: str,
    context: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Return one decoded strict-JSON plan from mock or LiteLLM."""
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise ValueError("model must be a non-empty string of at most 256 characters")
    if timeout is not None and not 1.0 <= float(timeout) <= 600.0:
        raise ValueError("timeout must be between 1 and 600 seconds")
    if model == "mock" or model.startswith("mock/"):
        return mock_response(question, context)
    try:
        from litellm import completion
    except ImportError as exc:
        raise CompilerProviderError(
            "LiteLLM is required for non-mock compilation; install "
            "'histoweave-spatial[compiler]' or use --model mock"
        ) from exc
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 4096,
    }
    if timeout is not None:
        request["timeout"] = float(timeout)
    try:
        response = completion(**request)
    except Exception as exc:
        raise CompilerProviderError(
            f"model provider request failed ({type(exc).__name__})"
        ) from exc
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise CompilerProviderError("model provider returned an invalid response envelope") from exc
    if not isinstance(content, str) or not content.strip():
        raise CompilerProviderError("LLM returned an empty or non-text response")
    if len(content.encode("utf-8")) > 1_000_000:
        raise ValueError("LLM response exceeds the 1000000-byte compiler limit")
    decoded = json.loads(content, parse_constant=_reject_nonfinite_json)
    if not isinstance(decoded, dict):
        raise ValueError("LLM response must decode to one JSON object")
    return decoded
