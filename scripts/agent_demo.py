"""用 mock LLM 走完整 Agent 对话，验证"用户说话 -> 大模型调工具"闭环。

不联网、不花 API 费用。用真实的 Service 和真实的 Qwen 模型（只做干跑）。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wsp.agent.loop import Agent  # noqa: E402
from wsp.llm.client import LLMResponse, ToolCall  # noqa: E402
from wsp.service import Service  # noqa: E402


class ScriptedClient:
    """按脚本扮演一个"会调工具的大模型"。"""

    def __init__(self):
        self.step = 0
        self.transcript = []

    def chat(self, messages, **kwargs):
        self.step += 1
        self.transcript.append(messages)
        if self.step == 1:
            return LLMResponse(tool_calls=[ToolCall(
                id="c1", name="inspect_model",
                arguments={"model_id": "qwen35-0.8b", "layer": 15},
                raw_arguments='{"model_id":"qwen35-0.8b","layer":15}')])
        if self.step == 2:
            # 从上一条 tool 结果里读 shape
            tool_msg = [m for m in messages if m.get("role") == "tool"][-1]
            payload = json.loads(tool_msg["content"])
            shape = payload["shapes"]["mlp.down_proj"]["shape"]
            self.transcript.append({"observed_shape": shape})
            plan = {
                "goal": "修复重复循环",
                "reasoning": f"down_proj 形状是 {shape}，沿 axis=1 选 64 列",
                "edits": [
                    {
                        "type": "direction_ablation",
                        "layer": 15,
                        "module": "mlp.down_proj",
                        "axis": 1,
                        "index": {"kind": "range", "start": 0, "stop": 64},
                        "method": "project_out",
                        "magnitude": 0.8,
                        "norm_preserve": True,
                        "direction_source": {"kind": "svd_top1"},
                        "expected_effect": "减少重复循环",
                        "risk": "可能影响通用生成",
                    }
                ],
                "eval_gate": {
                    "primary": "loop_mini",
                    "guards": ["math_short"],
                    "min_delta": 0.05,
                    "max_regression": 0.03,
                    "tasks": ["loop_mini", "math_short"],
                },
            }
            return LLMResponse(tool_calls=[ToolCall(
                id="c2", name="propose_plan",
                arguments={"project_id": PROJECT_ID, "plan": plan},
                raw_arguments=json.dumps({"project_id": PROJECT_ID, "plan": plan}))])
        if self.step == 3:
            tool_msg = [m for m in messages if m.get("role") == "tool"][-1]
            payload = json.loads(tool_msg["content"])
            return LLMResponse(tool_calls=[ToolCall(
                id="c3", name="dry_run_plan",
                arguments={"plan_id": payload["plan_id"]},
                raw_arguments=json.dumps({"plan_id": payload["plan_id"]}))])
        return LLMResponse(content="干跑完成。会改动约 0.013% 的权重，占比远低于 1% 上限。")


PROJECT_ID = ""


def main():
    global PROJECT_ID
    svc = Service()
    models = [m["id"] for m in svc.list_models()]
    if "qwen35-0.8b" not in models:
        print("请先导入 qwen35-0.8b")
        return 1

    proj = svc.create_project("Agent 演示", "qwen35-0.8b", "修复数学循环")
    PROJECT_ID = proj["id"]
    conv = svc.create_conversation(proj["id"], "第一次对话")

    scripted = ScriptedClient()
    svc.llm_client = lambda: scripted  # type: ignore

    agent = Agent(svc)
    run = agent.run(conv["id"], "这个模型做数学题老是重复，帮我看看能不能修。")

    print("=" * 60)
    print("对话事件流")
    print("=" * 60)
    for ev in run.events:
        d = ev.data
        if ev.kind == "tool_call":
            print(f"\n[调用] {d['name']}")
            print(f"       {json.dumps(d['arguments'], ensure_ascii=False)[:150]}")
        elif ev.kind == "tool_result":
            mark = "✓" if d["ok"] else "✗"
            print(f"[{mark}] {d['name']}: {d['summary'][:120]}")
        elif ev.kind == "plan":
            print(f"\n[计划] {d['summary']}")
        elif ev.kind == "text":
            print(f"\n[回复] {d['content']}")
        elif ev.kind == "error":
            print(f"[错误] {d['message']}")

    print("\n" + "=" * 60)
    print("验证")
    print("=" * 60)
    observed = [t for t in scripted.transcript if isinstance(t, dict)]
    if observed:
        print(f"模型读到的真实形状: {observed[0]['observed_shape']}")
        assert observed[0]["observed_shape"] == [1024, 3584], "形状不对！"

    plans = svc.state.db.list_plans(proj["id"])
    print(f"落库的计划数: {len(plans)}")
    assert len(plans) == 1

    edits = svc.list_edits(proj["id"])
    print(f"落库的编辑数: {len(edits)} (应为 1，来自干跑)")
    assert len(edits) == 1
    print(f"干跑改动: {edits[0]['diff']['total_changed']:,} 个权重")
    print(f"占模型比例: {edits[0]['diff']['model_fraction']:.6%}")

    # 确认没有写 checkpoint
    revs = list((svc.state.paths.revisions).rglob("*.safetensors"))
    print(f"checkpoint 文件数: {len(revs)} (应为 0，干跑不写文件)")
    assert len(revs) == 0

    print("\n✓ Agent 闭环验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
