"""LLM providers, including a deterministic offline compiler used by CI."""

from __future__ import annotations

import json
from typing import Any

from .templates import template_for_question


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
) -> dict[str, Any]:
    """Return a decoded JSON plan from mock or LiteLLM."""
    if model == "mock" or model.startswith("mock/"):
        return mock_response(question, context)
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError(
            "LiteLLM is required for non-mock compilation; install "
            "'histoweave-spatial[compiler]' or use --model mock"
        ) from exc
    response = completion(model=model, messages=messages, response_format={"type": "json_object"})
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("LLM returned an empty or non-text response")
    decoded = json.loads(content)
    if not isinstance(decoded, dict):
        raise RuntimeError("LLM response must decode to a JSON object")
    return decoded
