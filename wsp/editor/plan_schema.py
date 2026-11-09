"""编辑计划 DSL 与 JSON Schema 校验。

计划由大模型生成，必须通过这里的 schema 才能执行。
schema 刻意写得严格：所有会影响结果的字段都必填，不给"猜默认值"留空间。
"""

from __future__ import annotations

from typing import Any

import jsonschema

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

INDEX_SCHEMA = {
    "type": "object",
    "required": ["kind"],
    "additionalProperties": False,
    "properties": {
        "kind": {"enum": ["all", "range", "list", "top_norm"]},
        "start": {"type": "integer", "minimum": 0},
        "stop": {"type": "integer", "minimum": 0},
        "indices": {"type": "array", "items": {"type": "integer", "minimum": 0}},
        "n": {"type": "integer", "minimum": 1},
    },
}

EDIT_SCHEMA = {
    "type": "object",
    "required": ["type", "expected_effect", "risk"],
    "additionalProperties": False,
    "properties": {
        "type": {
            "enum": [
                "direction_ablation",
                "rank1_subtract",
                "module_scale",
                "noop",
            ]
        },
        "layer": {"type": ["integer", "null"], "minimum": 0},
        "module": {"type": ["string", "null"]},
        "tensor_name": {"type": ["string", "null"]},
        "axis": {"type": ["integer", "null"], "enum": [0, 1, None]},
        "index": {"oneOf": [INDEX_SCHEMA, {"type": "null"}]},
        "method": {"enum": ["project_out", "scale", None]},
        "magnitude": {"type": ["number", "null"], "minimum": 0, "maximum": 2},
        "factor": {"type": ["number", "null"], "exclusiveMinimum": 0, "maximum": 3},
        "norm_preserve": {"type": ["boolean", "null"]},
        "direction_source": {"type": ["object", "null"]},
        "expected_effect": {"type": "string", "minLength": 1},
        "risk": {"type": "string", "minLength": 1},
    },
    "allOf": [
        {
            "if": {"properties": {"type": {"const": "direction_ablation"}},
                   "required": ["type"]},
            "then": {
                "required": ["layer", "module", "axis", "index", "method",
                             "magnitude"],
            },
        },
        {
            "if": {"properties": {"type": {"const": "rank1_subtract"}},
                   "required": ["type"]},
            "then": {
                "required": ["layer", "module", "axis", "index", "magnitude",
                             "direction_source"],
            },
        },
        {
            "if": {"properties": {"type": {"const": "module_scale"}},
                   "required": ["type"]},
            "then": {"required": ["layer", "module", "factor"]},
        },
    ],
}

GATE_SCHEMA = {
    "type": "object",
    "required": ["primary", "min_delta"],
    "additionalProperties": False,
    "properties": {
        "primary": {"type": "string", "minLength": 1},
        "guards": {"type": "array", "items": {"type": "string"}},
        "tasks": {"type": "array", "items": {"type": "string"}},
        "min_delta": {"type": "number"},
        "delta_unit": {"enum": ["absolute", "relative"]},
        "max_regression": {"type": "number", "minimum": 0},
        "regression_unit": {"enum": ["absolute", "relative"]},
        "ppl_worsen_ratio": {"type": "number", "exclusiveMinimum": 0},
    },
}

PLAN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "WSP edit plan",
    "type": "object",
    "required": ["goal", "edits", "eval_gate"],
    "additionalProperties": False,
    "properties": {
        "goal": {"type": "string", "minLength": 1, "maxLength": 2000},
        "reasoning": {"type": "string", "maxLength": 4000},
        "edits": {
            "type": "array",
            "minItems": 0,
            "maxItems": 8,
            "items": EDIT_SCHEMA,
        },
        "eval_gate": GATE_SCHEMA,
        "expected_effect": {"type": "string"},
    },
}


class PlanValidationError(Exception):
    """计划不符合 schema。"""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.errors = errors or []


def validate_plan(plan: Any) -> dict:
    """校验并返回计划；不合法则抛 PlanValidationError。"""
    validator = jsonschema.Draft202012Validator(PLAN_SCHEMA)
    errors = sorted(validator.iter_errors(plan), key=lambda e: list(e.path))
    if errors:
        detail = []
        for e in errors[:8]:
            path = ".".join(str(p) for p in e.path) or "<root>"
            detail.append(f"{path}: {e.message}")
        raise PlanValidationError(
            "编辑计划不符合 schema：\n" + "\n".join(f"  - {d}" for d in detail),
            detail,
        )
    return plan


def empty_plan(goal: str, reason: str = "") -> dict:
    """生成一个合法的 noop 计划。"""
    return {
        "goal": goal,
        "reasoning": reason or "证据不足，返回 noop。",
        "edits": [],
        "eval_gate": {
            "primary": "task_score",
            "guards": [],
            "tasks": [],
            "min_delta": 0.05,
            "delta_unit": "absolute",
            "max_regression": 0.03,
            "regression_unit": "absolute",
        },
    }


def plan_summary(plan: dict) -> str:
    """给 UI / 日志用的一行摘要。"""
    edits = plan.get("edits") or []
    if not edits:
        return f"noop 计划（目标：{plan.get('goal', '')[:60]}）"
    parts = []
    for e in edits:
        t = e.get("type")
        if t == "direction_ablation":
            parts.append(
                f"L{e.get('layer')} {e.get('module')} "
                f"{e.get('method')} α={e.get('magnitude')}"
            )
        elif t == "module_scale":
            parts.append(f"L{e.get('layer')} {e.get('module')} ×{e.get('factor')}")
        else:
            parts.append("noop")
    return f"{len(edits)} 条编辑：" + "；".join(parts)
