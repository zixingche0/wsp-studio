"""Agent 对话全流程测试（真实 HTTP + mock LLM）。

验证：用户说话 -> 大模型调工具 -> 平台执行 -> 结果回传 的完整闭环。
用 httpx mock 拦截 LLM 请求，其余全部真实。
"""

from __future__ import annotations

import json
import sys

import httpx

BASE = "http://127.0.0.1:8765"
API = BASE + "/api"
client = httpx.Client(timeout=180.0)

passed, failed = [], []


def check(name, fn):
    try:
        v = fn()
        passed.append(name)
        print(f"  ✓ {name}" + (f" -> {v}" if v not in (True, None) else ""))
        return v
    except Exception as exc:
        failed.append((name, str(exc)))
        print(f"  ✗ {name}\n      {exc}")
        return None


def main() -> int:
    print("=" * 70)
    print("Agent 对话全流程")
    print("=" * 70)

    models = client.get(API + "/models").json()["models"]
    model = next((m for m in models if m["can_edit"]), None)
    if not model:
        print("没有可编辑模型")
        return 2

    proj = client.post(API + "/projects", json={
        "name": "Agent 测试", "model_id": model["id"], "goal": "测试"
    }).json()
    pid = proj["id"]
    conv = client.post(API + "/conversations", json={
        "project_id": pid, "title": "测试对话"
    }).json()
    cid = conv["id"]

    # 没有真实 LLM Key 时，直接用 Service 层 + 脚本化客户端
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from wsp.agent.loop import Agent
    from wsp.llm.client import LLMResponse, ToolCall
    from wsp.service import Service

    svc = Service()
    project = svc.get_project(pid)

    print("\n第 1 部分：工具逐个验证")

    def tool_call(name, args, cid_="c1"):
        return ToolCall(id=cid_, name=name, arguments=args,
                        raw_arguments=json.dumps(args))

    agent = Agent(svc)
    ctx = {**project, "_conversation_id": cid}

    def run_tool(name, args):
        text, payload, ok = agent._dispatch(name, args, ctx)
        return payload, ok

    check("list_models", lambda: (
        lambda r: f"{len(r[0]['models'])} 个模型" if r[1] else (_ for _ in ()).throw(
            AssertionError(r[0])))(run_tool("list_models", {})))

    check("inspect_model", lambda: (
        lambda r: f"{r[0]['num_layers']} 层" if r[1] else (_ for _ in ()).throw(
            AssertionError(r[0])))(run_tool("inspect_model", {"model_id": model["id"]})))

    check("inspect_model 指定层", lambda: (
        lambda r: f"{len(r[0].get('modules', []))} 个模块" if r[1] else
        (_ for _ in ()).throw(AssertionError(r[0])))(
            run_tool("inspect_model", {"model_id": model["id"], "layer": 15})))

    check("list_edits", lambda: (
        lambda r: f"{len(r[0]['edits'])} 条" if r[1] else (_ for _ in ()).throw(
            AssertionError(r[0])))(run_tool("list_edits", {})))

    check("未知工具被拒绝", lambda: (
        lambda r: True if not r[1] else (_ for _ in ()).throw(
            AssertionError("未知工具居然成功了")))(run_tool("rm_rf", {})))

    print("\n第 2 部分：计划相关工具")

    good_plan = {
        "goal": "测试",
        "edits": [{
            "type": "direction_ablation", "layer": 15, "module": "mlp.down_proj",
            "axis": 1, "index": {"kind": "range", "start": 0, "stop": 64},
            "method": "project_out", "magnitude": 0.5, "norm_preserve": True,
            "direction_source": {"kind": "random"},
            "expected_effect": "x", "risk": "y",
        }],
        "eval_gate": {"primary": "loop_mini", "min_delta": 0.05,
                      "max_regression": 0.03, "tasks": ["loop_mini"]},
    }

    proposed = check("propose_plan", lambda: (
        lambda r: r[0] if r[1] else (_ for _ in ()).throw(
            AssertionError(r[0])))(
        run_tool("propose_plan", {"project_id": pid, "plan": good_plan})))

    check("非法计划被拒", lambda: (
        lambda r: True if not r[1] else (_ for _ in ()).throw(
            AssertionError("非法计划通过了")))(run_tool("propose_plan", {
            "project_id": pid, "plan": {"goal": "x"}})))

    if proposed and "plan_id" in proposed:
        plan_id = proposed["plan_id"]

        dry = check("dry_run_plan", lambda: (
            lambda r: r[0] if r[1] else (_ for _ in ()).throw(
                AssertionError(r[0])))(run_tool("dry_run_plan", {"plan_id": plan_id})))

        if dry:
            check("干跑有输出", lambda: (
                dry["total_changed"] > 0 or (_ for _ in ()).throw(
                    AssertionError(f"改动 0"))))
            check("干跑符合预算", lambda: (
                dry["model_fraction"] < 0.01 or (_ for _ in ()).throw(
                    AssertionError(f"超预算 {dry['model_fraction']:.4%}"))))

    print("\n第 3 部分：完整对话循环（脚本化 LLM）")

    class Scripted:
        def __init__(self):
            self.step = 0
            self.seen_shapes = None

        def chat(self, messages, **kw):
            self.step += 1
            if self.step == 1:
                return LLMResponse(tool_calls=[tool_call(
                    "inspect_model", {"model_id": model["id"], "layer": 15}, "a")])
            if self.step == 2:
                tool_msg = [m for m in messages if m.get("role") == "tool"][-1]
                payload = json.loads(tool_msg["content"])
                self.seen_shapes = payload["shapes"]["mlp.down_proj"]["shape"]
                return LLMResponse(tool_calls=[tool_call(
                    "propose_plan", {"project_id": pid, "plan": good_plan}, "b")])
            if self.step == 3:
                tool_msg = [m for m in messages if m.get("role") == "tool"][-1]
                payload = json.loads(tool_msg["content"])
                return LLMResponse(tool_calls=[tool_call(
                    "dry_run_plan", {"plan_id": payload["plan_id"]}, "c")])
            return LLMResponse(content="分析完成，建议先干跑确认。")

    scripted = Scripted()
    orig = svc.llm_client
    svc.llm_client = lambda: scripted
    try:
        run = agent.run(cid, "这个模型数学题老是重复，帮我看看")
    finally:
        svc.llm_client = orig

    check("对话完成", lambda: (
        run.final_text or (_ for _ in ()).throw(AssertionError("没有最终回复"))))
    check("模型读到真实形状", lambda: (
        scripted.seen_shapes == [1024, 3584] or (_ for _ in ()).throw(
            AssertionError(f"形状 {scripted.seen_shapes}"))))
    check("产生了计划事件", lambda: (
        len([e for e in run.events if e.kind == "plan"]) == 1 or
        (_ for _ in ()).throw(AssertionError("没有 plan 事件"))))
    check("工具调用链完整", lambda: (
        len([e for e in run.events if e.kind == "tool_call"]) == 3 or
        (_ for _ in ()).throw(
            AssertionError(f"工具调用数 {len([e for e in run.events if e.kind=='tool_call'])}"))))

    print("\n第 4 部分：落库与可追溯")

    msgs = client.get(API + f"/conversations/{cid}/messages").json()["messages"]
    check("消息已落库", lambda: (
        len(msgs) > 0 or (_ for _ in ()).throw(AssertionError("无消息"))))
    check("含 user 消息", lambda: (
        any(m["role"] == "user" for m in msgs) or (_ for _ in ()).throw(
            AssertionError("缺 user"))))
    check("含 assistant 消息", lambda: (
        any(m["role"] == "assistant" for m in msgs) or (_ for _ in ()).throw(
            AssertionError("缺 assistant"))))
    check("含 tool 消息", lambda: (
        any(m["role"] == "tool" for m in msgs) or (_ for _ in ()).throw(
            AssertionError("缺 tool"))))
    check("assistant 记录了 tool_calls", lambda: (
        any(m.get("tool_calls") for m in msgs if m["role"] == "assistant") or
        (_ for _ in ()).throw(AssertionError("没记录 tool_calls"))))

    edits = client.get(API + f"/edits?project_id={pid}").json()["edits"]
    check("编辑已落库", lambda: (
        len(edits) >= 1 or (_ for _ in ()).throw(AssertionError("无编辑"))))
    check("编辑关联到项目", lambda: (
        all(e.get("project_id") == pid for e in edits) or (_ for _ in ()).throw(
            AssertionError("有孤儿编辑"))))

    # 清理
    client.delete(API + f"/projects/{pid}")

    print()
    print("=" * 70)
    print(f"通过 {len(passed)}，失败 {len(failed)}")
    print("=" * 70)
    if failed:
        for n, e in failed:
            print(f"  ✗ {n}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
