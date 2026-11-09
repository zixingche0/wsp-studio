"""工具定义（OpenAI function calling 格式）与提示片段。"""

from __future__ import annotations

import json

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_models",
            "description": "列出当前工作台里注册的模型及其能力（是否能编辑）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_model",
            "description": "查看模型结构：层数、张量名、某层有哪些模块、张量形状。",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "layer": {
                        "type": ["integer", "null"],
                        "description": "只看这一层；null 表示概览",
                    },
                },
                "required": ["model_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_probe",
            "description": (
                "运行内置探测任务，返回结构化摘要（各任务分数 + 失败样例）。"
                "会真的调用模型推理，可能较慢。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "math_short / ifeval_mini / loop_mini / refusal_mini / gen_basic",
                    },
                    "max_cases_per_task": {"type": ["integer", "null"]},
                },
                "required": ["model_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_plan",
            "description": (
                "提交一个编辑计划（不执行）。计划必须符合 DSL Schema。"
                "提交后系统会校验，不合法会返回错误让你修正。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "plan": {
                        "type": "object",
                        "description": "完整的编辑计划对象",
                    },
                },
                "required": ["project_id", "plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dry_run_plan",
            "description": (
                "对一个计划做干跑：不写任何文件，只计算会改动多少权重、"
                "范数变化、每个张量的前后 hash。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_plan",
            "description": (
                "真正应用计划，写出新 checkpoint。会先导出位级 diff。"
                "只有在用户明确同意、且 dry-run 结果合理时才调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"plan_id": {"type": "string"}},
                "required": ["plan_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eval",
            "description": (
                "对一次已应用的编辑跑评估门控。比较编辑前后的指标，"
                "决定通过或回滚。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"edit_id": {"type": "string"}},
                "required": ["edit_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback_edit",
            "description": "把一次编辑回滚到 base_revision，并校验文件 hash。",
            "parameters": {
                "type": "object",
                "properties": {"edit_id": {"type": "string"}},
                "required": ["edit_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_edits",
            "description": "列出编辑历史（含 dry-run 与实际应用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": ["string", "null"]},
                },
                "required": [],
            },
        },
    },
]


TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


def tool_result(payload: dict, *, ok: bool = True) -> str:
    """把工具结果序列化成给模型看的内容。"""
    body = {"ok": ok, **payload}
    return json.dumps(body, ensure_ascii=False, default=str)[:20000]
