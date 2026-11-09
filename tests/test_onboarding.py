"""首次运行引导测试。"""

from __future__ import annotations

import pytest

from wsp import onboarding


def test_guide_always_has_storage():
    steps = onboarding.build_guide()
    keys = [s.key for s in steps]
    assert "storage" in keys


def test_guide_marks_missing_llm_as_todo():
    steps = onboarding.build_guide(has_llm_config=False)
    llm = next(s for s in steps if s.key == "llm")
    assert llm.status == "todo"
    assert llm.detail


def test_guide_marks_configured_llm_as_done():
    steps = onboarding.build_guide(has_llm_config=True)
    llm = next(s for s in steps if s.key == "llm")
    assert llm.status == "done"


def test_guide_model_step_has_download_command():
    steps = onboarding.build_guide(has_model=False)
    model = next(s for s in steps if s.key == "model")
    assert model.status == "todo"
    assert model.commands, "缺少模型时必须给出下载命令"
    # 命令里不能出现硬编码的本机路径
    joined = " ".join(model.commands)
    assert "F:\\" not in joined and "C:\\Users" not in joined


def test_guide_service_is_optional_when_down():
    steps = onboarding.build_guide(running_services=[])
    svc = next(s for s in steps if s.key == "service")
    assert svc.status == "optional"


def test_guide_service_done_when_running():
    steps = onboarding.build_guide(
        running_services=[{"name": "Ollama", "reachable": True, "base_url": "x"}]
    )
    svc = next(s for s in steps if s.key == "service")
    assert svc.status == "done"
    assert "Ollama" in svc.detail


def test_guide_summary_not_ready():
    steps = onboarding.build_guide(has_model=False, has_llm_config=False)
    summary = onboarding.guide_summary(steps)
    assert summary["ready"] is False
    assert summary["todo_count"] >= 2
    assert summary["next_step"] is not None


def test_guide_summary_edit_capable():
    """模型 + LLM + 存储齐了，就能编辑；缺推理运行时只影响评估。"""
    steps = onboarding.build_guide(has_model=True, has_llm_config=True,
                                   running_services=[])
    summary = onboarding.guide_summary(steps)
    assert summary["can_edit"] is True
    assert summary["edit_blockers"] == []
    # 本机可能缺 transformers，此时 can_evaluate 为 False 是正确行为
    assert isinstance(summary["can_evaluate"], bool)


def test_guide_summary_blocks_edit_without_model():
    steps = onboarding.build_guide(has_model=False, has_llm_config=True,
                                   running_services=[])
    summary = onboarding.guide_summary(steps)
    assert summary["can_edit"] is False
    assert "model" in summary["edit_blockers"]


def test_guide_summary_blocks_edit_without_llm():
    steps = onboarding.build_guide(has_model=True, has_llm_config=False,
                                   running_services=[])
    summary = onboarding.guide_summary(steps)
    assert summary["can_edit"] is False
    assert "llm" in summary["edit_blockers"]


def test_recommended_models_have_required_fields():
    for m in onboarding.recommend_models():
        assert "id" in m and "size_gb" in m and "why" in m
        assert "feasible" in m


def test_download_command_is_cross_platform():
    cmds = onboarding.download_command("Qwen/Qwen3.5-0.8B")
    assert cmds
    joined = " ".join(cmds)
    assert "Qwen/Qwen3.5-0.8B" in joined
    # 不能出现 Windows 专有路径
    assert "F:\\" not in joined


def test_download_command_respects_local_dir():
    cmds = onboarding.download_command("org/model", "/tmp/mymodel")
    assert any("/tmp/mymodel" in c for c in cmds)


def test_step_serializable():
    step = onboarding.Step(key="k", title="t", status="todo",
                           detail="d", commands=["c"], link="l")
    d = step.as_dict()
    assert d["key"] == "k"
    assert d["commands"] == ["c"]
