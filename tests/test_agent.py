"""Agent 循环测试：用假的 LLM 客户端验证工具调度。

不联网、不调用真实模型。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from wsp.agent.loop import Agent, AgentError
from wsp.llm.client import LLMResponse, ToolCall


class FakeClient:
    """按脚本返回预设响应的假客户端。"""

    def __init__(self, script: list[LLMResponse]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.script:
            return LLMResponse(content="（脚本结束）")
        return self.script.pop(0)


def _tc(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args,
                    raw_arguments=json.dumps(args))


@pytest.fixture
def agent(service, monkeypatch):
    a = Agent(service)
    return a


def _patch_client(monkeypatch, service, fake):
    monkeypatch.setattr(service, "llm_client", lambda: fake)


def test_agent_calls_list_models(service, agent, project, monkeypatch):
    fake = FakeClient([
        LLMResponse(tool_calls=[_tc("list_models", {})]),
        LLMResponse(content="完成了"),
    ])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "看看有哪些模型")
    assert run.final_text == "完成了"
    kinds = [e.kind for e in run.events]
    assert "tool_call" in kinds and "tool_result" in kinds
    assert run.steps == 2


def test_agent_inspect_model_tool(service, agent, project, monkeypatch):
    fake = FakeClient([
        LLMResponse(tool_calls=[_tc("inspect_model", {"model_id": "tiny", "layer": 1})]),
        LLMResponse(content="ok"),
    ])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "看看第一层")
    results = [e for e in run.events if e.kind == "tool_result"]
    assert results and results[0].data["ok"] is True


def test_agent_propose_plan_then_dry_run(service, agent, project, monkeypatch, sample_plan):
    fake = FakeClient([
        LLMResponse(tool_calls=[_tc("propose_plan",
                                    {"project_id": project["id"], "plan": sample_plan})]),
        # 第二步：干跑（plan_id 是动态的，这里用任意值会被服务拒绝，
        # 所以脚本里先返回文本结束）
        LLMResponse(content="计划已提交"),
    ])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "给我一个计划")
    plan_events = [e for e in run.events if e.kind == "plan"]
    assert len(plan_events) == 1
    assert plan_events[0].data["plan_id"].startswith("plan_")


def test_agent_handles_invalid_plan(service, agent, project, monkeypatch):
    bad_plan = {"goal": "x"}  # 缺 edits / eval_gate
    fake = FakeClient([
        LLMResponse(tool_calls=[_tc("propose_plan",
                                    {"project_id": project["id"], "plan": bad_plan})]),
        LLMResponse(content="我会修正"),
    ])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "试试")
    results = [e for e in run.events if e.kind == "tool_result"]
    assert results[0].data["ok"] is False
    assert "schema" in results[0].data["summary"]


def test_agent_unknown_tool_is_reported(service, agent, project, monkeypatch):
    fake = FakeClient([
        LLMResponse(tool_calls=[_tc("rm_rf", {})]),
        LLMResponse(content="好"),
    ])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "删库")
    results = [e for e in run.events if e.kind == "tool_result"]
    assert results[0].data["ok"] is False


def test_agent_stops_at_max_steps(service, agent, project, monkeypatch):
    # 永远返回工具调用
    script = [LLMResponse(tool_calls=[_tc("list_models", {}, f"c{i}")])
              for i in range(20)]
    fake = FakeClient(script)
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "loop")
    assert "最大步数" in run.error


def test_agent_records_messages(service, agent, project, monkeypatch):
    fake = FakeClient([LLMResponse(content="你好")])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    agent.run(conv["id"], "打个招呼")
    msgs = service.list_messages(conv["id"])
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]


def test_agent_llm_error_is_captured(service, agent, project, monkeypatch):
    class BrokenClient:
        def chat(self, *a, **kw):
            from wsp.llm.client import LLMError
            raise LLMError("密钥错误")

    monkeypatch.setattr(service, "llm_client", lambda: BrokenClient())
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "hi")
    assert "密钥错误" in run.error


def test_agent_missing_conversation_raises(service, agent):
    with pytest.raises(AgentError):
        agent.run("conv_nope", "hi")


def test_agent_system_prompt_has_context(service, agent, project, monkeypatch):
    fake = FakeClient([LLMResponse(content="ok")])
    _patch_client(monkeypatch, service, fake)
    conv = service.create_conversation(project["id"], "t")
    agent.run(conv["id"], "hi")
    first_call = fake.calls[0]
    system = first_call[0]["content"]
    assert "WSP Studio" in system
    assert project["id"] in system
    assert "tiny" in system
    # 负面知识必须在 prompt 里
    assert "无法从权重读出" in system


def test_agent_full_pipeline_dry_run(service, agent, project, monkeypatch, sample_plan):
    """propose -> dry_run -> 结束（用真实 plan_id 串起来）。"""
    captured = {}

    class ChainedClient:
        def __init__(self):
            self.step = 0

        def chat(self, messages, **kwargs):
            self.step += 1
            if self.step == 1:
                return LLMResponse(tool_calls=[_tc(
                    "propose_plan",
                    {"project_id": project["id"], "plan": sample_plan}, "a")])
            if self.step == 2:
                # 从上一条 tool 结果里取 plan_id
                tool_msgs = [m for m in messages if m.get("role") == "tool"]
                payload = json.loads(tool_msgs[-1]["content"])
                captured["plan_id"] = payload["plan_id"]
                return LLMResponse(tool_calls=[_tc(
                    "dry_run_plan", {"plan_id": payload["plan_id"]}, "b")])
            return LLMResponse(content="干跑完成")

    monkeypatch.setattr(service, "llm_client", lambda: ChainedClient())
    conv = service.create_conversation(project["id"], "t")
    run = agent.run(conv["id"], "跑一遍")
    assert run.final_text == "干跑完成"
    assert captured["plan_id"].startswith("plan_")
    results = [e for e in run.events if e.kind == "tool_result"]
    assert results[-1].data["ok"] is True
