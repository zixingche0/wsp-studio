"""Agent 循环：用户说话 -> 大模型调工具 -> 平台执行 -> 回传结果。

这是"用户直接和大模型说，大模型在平台上操作小模型"的核心。

安全约束：
* 工具调用有次数上限，防止死循环
* 每个工具都可能失败，失败信息原样回传给模型让它自我修正
* 所有副作用（建计划、应用、回滚）都落库，可追溯
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..editor import plan_schema as schema_mod
from ..llm import tools as tools_mod
from ..llm.client import LLMClient, LLMError
from ..llm.prompts import AGENT_SYSTEM
from ..service import Service, ServiceError

MAX_STEPS = 12


class AgentError(Exception):
    """Agent 运行失败。"""


@dataclass
class AgentEvent:
    """Agent 执行过程中的一个事件，用于流式展示。"""

    kind: str                 # text | tool_call | tool_result | plan | error | done
    data: dict = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "data": self.data, "at": self.at}


@dataclass
class AgentRun:
    """一次 Agent 运行的完整结果。"""

    conversation_id: str
    events: list[AgentEvent] = field(default_factory=list)
    final_text: str = ""
    steps: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "final_text": self.final_text,
            "steps": self.steps,
            "error": self.error,
            "events": [e.as_dict() for e in self.events],
        }


class Agent:
    """对话式权重编辑代理。"""

    def __init__(self, service: Service):
        self.svc = service

    # ------------------------------------------------------------------

    def run(
        self,
        conversation_id: str,
        user_message: str,
        *,
        project_id: str | None = None,
    ) -> AgentRun:
        db = self.svc.state.db
        conv = db.get_conversation(conversation_id)
        if not conv:
            raise AgentError(f"对话 {conversation_id!r} 不存在")
        project_id = project_id or conv["project_id"]
        project = self.svc.get_project(project_id)
        # 把当前对话 id 注入项目上下文，工具里要用（例如 propose_plan 需要归属）
        project = {**project, "_conversation_id": conversation_id}

        run = AgentRun(conversation_id=conversation_id)

        db.add_message(conversation_id, "user", user_message)

        try:
            client = self.svc.llm_client()
        except Exception as exc:
            run.error = str(exc)
            run.events.append(AgentEvent("error", {"message": str(exc)}))
            return run

        # 组装上下文
        history = db.list_messages(conversation_id)
        messages = self._build_messages(history, project)

        for step in range(MAX_STEPS):
            run.steps = step + 1
            try:
                resp = client.chat(
                    messages,
                    tools=tools_mod.TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                )
            except LLMError as exc:
                msg = str(exc)
                run.error = msg
                run.events.append(AgentEvent("error", {"message": msg}))
                db.add_message(conversation_id, "assistant", f"[错误] {msg}")
                return run

            if resp.content:
                run.events.append(AgentEvent("text", {"content": resp.content}))
                run.final_text = resp.content

            if not resp.wants_tools:
                db.add_message(conversation_id, "assistant", resp.content)
                run.events.append(AgentEvent("done", {"steps": run.steps}))
                return run

            # 把 assistant 的 tool_calls 记进上下文
            assistant_msg = {
                "role": "assistant",
                "content": resp.content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
            messages.append(assistant_msg)
            db.add_message(
                conversation_id,
                "assistant",
                resp.content or "",
                tool_calls=[
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in resp.tool_calls
                ],
            )

            for tc in resp.tool_calls:
                run.events.append(
                    AgentEvent("tool_call", {"name": tc.name, "arguments": tc.arguments})
                )
                result_text, payload, ok = self._dispatch(tc.name, tc.arguments, project)
                run.events.append(
                    AgentEvent(
                        "tool_result",
                        {"name": tc.name, "ok": ok, "summary": _summarize(payload)},
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                )
                db.add_message(
                    conversation_id, "tool", result_text, tool_call_id=tc.id
                )
                # 把生成计划这一动作单独发一个事件给 UI
                if tc.name == "propose_plan" and ok:
                    run.events.append(
                        AgentEvent(
                            "plan",
                            {
                                "plan_id": payload.get("plan_id"),
                                "summary": payload.get("summary", ""),
                            },
                        )
                    )

        run.error = f"超过最大步数 {MAX_STEPS}，已停止"
        run.events.append(AgentEvent("error", {"message": run.error}))
        return run

    # ------------------------------------------------------------------

    def _build_messages(self, history: list[dict], project: dict) -> list[dict]:
        """把数据库里的历史转成 OpenAI messages。"""
        system = AGENT_SYSTEM + "\n\n## 当前上下文\n"
        system += f"- 项目：{project['name']}（id={project['id']}）\n"
        system += f"- 模型：{project['model_id']}\n"
        if project.get("goal"):
            system += f"- 目标：{project['goal']}\n"
        entry = self.svc.state.registry.get(project["model_id"])
        if entry:
            system += (
                f"- 该模型能力：{'可编辑' if entry.can_edit else '只读'}"
                f"{'，可探测' if entry.can_probe else ''}\n"
            )
        messages = [{"role": "system", "content": system}]

        for msg in history[-40:]:
            role = msg["role"]
            if role == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                item: dict = {"role": "assistant", "content": msg["content"] or None}
                calls = msg.get("tool_calls")
                if calls:
                    item["tool_calls"] = [
                        {
                            "id": c.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": c.get("name", ""),
                                "arguments": json.dumps(
                                    c.get("arguments", {}), ensure_ascii=False
                                ),
                            },
                        }
                        for c in calls
                    ]
                messages.append(item)
            elif role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id") or "",
                        "content": msg["content"],
                    }
                )
        return messages

    # ------------------------------------------------------------------

    def _dispatch(
        self, name: str, args: dict, project: dict
    ) -> tuple[str, dict, bool]:
        """执行一个工具调用，返回 (给模型的文本, 结构化结果, 是否成功)。"""
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return (
                    tools_mod.tool_result({"error": f"未知工具 {name}"}, ok=False),
                    {"error": f"未知工具 {name}"},
                    False,
                )
            payload = handler(args, project)
            return tools_mod.tool_result(payload), payload, True
        except (ServiceError, ValueError, KeyError) as exc:
            msg = str(exc)
            return (
                tools_mod.tool_result({"error": msg}, ok=False),
                {"error": msg},
                False,
            )
        except Exception as exc:  # 兜底，避免一个工具崩溃整个对话
            msg = f"{type(exc).__name__}: {exc}"
            return (
                tools_mod.tool_result({"error": msg}, ok=False),
                {"error": msg},
                False,
            )

    # ------------------------------------------------------------------
    # 工具实现
    # ------------------------------------------------------------------

    def _tool_list_models(self, args: dict, project: dict) -> dict:
        models = self.svc.list_models()
        return {
            "models": [
                {
                    "id": m["id"],
                    "name": m["name"],
                    "capabilities": m["capabilities"],
                    "kind": m["kind"],
                }
                for m in models
            ]
        }

    def _tool_inspect_model(self, args: dict, project: dict) -> dict:
        model_id = args.get("model_id") or project["model_id"]
        return self.svc.inspect_model(model_id, args.get("layer"))

    def _tool_run_probe(self, args: dict, project: dict) -> dict:
        model_id = args.get("model_id") or project["model_id"]
        digest = self.svc.run_probe(
            model_id,
            tasks=args.get("tasks"),
            max_cases_per_task=args.get("max_cases_per_task"),
        )
        # 给模型看精简版，避免上下文爆炸
        return {
            "model_id": model_id,
            "scores": digest.get("scores"),
            "sampling": digest.get("sampling"),
            "failures": digest.get("failures", [])[:8],
            "task_errors": {
                k: v["error"] for k, v in digest.get("tasks", {}).items() if v.get("error")
            },
        }

    def _tool_propose_plan(self, args: dict, project: dict) -> dict:
        plan = args.get("plan")
        if not isinstance(plan, dict):
            raise ServiceError("plan 必须是一个 JSON 对象")
        conv_id = _current_conversation(project)
        row = self.svc.create_plan(
            project["id"], plan, conversation_id=conv_id, goal=plan.get("goal", "")
        )
        return {
            "plan_id": row["id"],
            "summary": schema_mod.plan_summary(plan),
            "note": "计划已保存，尚未执行。下一步请调用 dry_run_plan。",
        }

    def _tool_dry_run_plan(self, args: dict, project: dict) -> dict:
        plan_id = args["plan_id"]
        out = self.svc.dry_run_plan(plan_id)
        result = out["result"]
        return {
            "edit_id": out["edit"]["id"],
            "ok": result["ok"],
            "dry_run": True,
            "total_changed": result["total_changed"],
            "model_fraction": round(result["model_fraction"], 8),
            "outcomes": [
                {
                    "tensor": o.get("tensor_name"),
                    "ok": o["ok"],
                    "changed": o.get("stats", {}).get("changed_entries"),
                    "rows": o.get("stats", {}).get("changed_rows"),
                    "cols": o.get("stats", {}).get("changed_cols"),
                    "norm_change": round(
                        o.get("stats", {}).get("relative_norm_change", 0.0), 6
                    ),
                    "error": o.get("error", ""),
                }
                for o in result["outcomes"]
            ],
            "note": "干跑没有写任何文件。确认无误后再调用 apply_plan。",
        }

    def _tool_apply_plan(self, args: dict, project: dict) -> dict:
        plan_id = args["plan_id"]
        out = self.svc.apply_plan(plan_id)
        return {
            "edit_id": out["edit"]["id"] if out.get("edit") else "",
            "checkpoint": out.get("checkpoint"),
            "total_changed": out["result"]["total_changed"],
            "note": "已写出新 checkpoint。下一步请调用 run_eval。",
        }

    def _tool_run_eval(self, args: dict, project: dict) -> dict:
        edit_id = args["edit_id"]
        out = self.svc.run_eval(edit_id)
        return {
            "edit_id": edit_id,
            "passed": out["passed"],
            "gate": out["gate"],
            "before": out["before"],
            "after": out["after"],
            "rolled_back": out["rolled_back"],
        }

    def _tool_rollback_edit(self, args: dict, project: dict) -> dict:
        return self.svc.rollback_edit(args["edit_id"])

    def _tool_list_edits(self, args: dict, project: dict) -> dict:
        edits = self.svc.list_edits(args.get("project_id") or project["id"])
        return {
            "edits": [
                {
                    "id": e["id"],
                    "status": e["status"],
                    "dry_run": e["dry_run"],
                    "created_at": e["created_at"],
                }
                for e in edits[:20]
            ]
        }


def _current_conversation(project: dict) -> str | None:
    """从项目上下文里取当前对话 id（由 Service 注入）。"""
    return project.get("_conversation_id")


def _summarize(payload: dict) -> str:
    """给 UI 的一句话摘要。"""
    if "error" in payload:
        return f"失败：{payload['error']}"[:160]
    for key in ("summary", "note", "passed", "scores"):
        if key in payload:
            return f"{key}: {payload[key]}"[:160]
    return json.dumps(payload, ensure_ascii=False)[:160]
