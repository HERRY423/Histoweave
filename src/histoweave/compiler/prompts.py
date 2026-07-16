"""Prompt construction for the spatial pipeline compiler."""

from __future__ import annotations

import json
from typing import Any

from .templates import MOCK_TEMPLATES

SYSTEM_PROMPT = """You compile natural-language spatial transcriptomics questions into
HistoWeave pipelines. Return exactly one JSON object and no markdown. Use only methods
from CATALOG. Never emit Python, R, shell, URLs, or invented methods. The required shape
is {rationale, steps:[{category,method,params,purpose}], gaps:[{concept,reason,degraded_to}],
assay_assumed}. If the catalog lacks a requested capability, schedule the closest safe
approximation and record it in gaps. Prefer qc before normalization and analysis."""


def _few_shot_messages(catalog: list[dict[str, Any]]) -> list[dict[str, str]]:
    available = {(row["category"], row["name"]) for row in catalog}
    messages: list[dict[str, str]] = []
    for template in MOCK_TEMPLATES:
        plan = template["plan"]
        required = {(step["category"], step["method"]) for step in plan["steps"]}
        if not required <= available:
            continue
        messages.extend(
            [
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": template["question"], "data_context": {}},
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        plan,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
            ]
        )
    return messages


def build_messages(
    question: str,
    catalog: list[dict[str, Any]],
    *,
    context: dict[str, Any] | None = None,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "data_context": context or {},
        "catalog": catalog,
    }
    if validation_error:
        payload["previous_validation_error"] = validation_error
        payload["instruction"] = "Regenerate a corrected full plan."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_few_shot_messages(catalog),
        {
            "role": "user",
            "content": json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ),
        },
    ]
