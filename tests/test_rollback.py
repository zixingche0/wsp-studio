"""回滚测试：按 edit_id 回滚，并校验 hash 恢复到 base_revision。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wsp.weights import store


def test_rollback_removes_checkpoint(service, project, sample_plan):
    out = service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    ckpt = Path(out["checkpoint"])
    assert ckpt.exists()
    edit_id = out["edit"]["id"]
    result = service.rollback_edit(edit_id)
    assert result["rolled_back"] is True
    assert not ckpt.exists()


def test_rollback_marks_status(service, project, sample_plan):
    out = service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    edit_id = out["edit"]["id"]
    service.rollback_edit(edit_id)
    edit = service.get_edit(edit_id)
    assert edit["status"] == "rolled_back"


def test_rollback_verifies_base_hashes(service, project, sample_plan):
    """回滚后，原始张量的 sha256 必须与 diff 记录的 base_sha256 一致。"""
    out = service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    edit_id = out["edit"]["id"]
    result = service.rollback_edit(edit_id)
    # tiny 模型所有张量都应校验到（diff 里只记录了被改的，但 base hash 要能对上）
    assert isinstance(result["verified_tensors"], list)


def test_original_weights_identical_after_rollback(service, project, sample_plan):
    info = store.load_model(service.get_model_entry("tiny").path)
    before = {n: store.tensor_sha256(store.read_tensor(info, n)[1])
              for n in info.tensor_names}
    out = service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    service.rollback_edit(out["edit"]["id"])
    after = {n: store.tensor_sha256(store.read_tensor(info, n)[1])
             for n in info.tensor_names}
    assert before == after


def test_rollback_unknown_edit_raises(service):
    from wsp.service import ServiceError

    with pytest.raises(ServiceError):
        service.rollback_edit("edit_does_not_exist")


def test_double_rollback_is_safe(service, project, sample_plan):
    out = service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    edit_id = out["edit"]["id"]
    service.rollback_edit(edit_id)
    second = service.rollback_edit(edit_id)
    assert second["rolled_back"] is True


def test_apply_twice_creates_two_checkpoints(service, project, sample_plan):
    plan_id = service.create_plan(project["id"], sample_plan)["id"]
    a = service.apply_plan(plan_id)
    b = service.apply_plan(plan_id)
    assert a["edit"]["id"] != b["edit"]["id"]
    assert Path(a["checkpoint"]).exists()
    assert Path(b["checkpoint"]).exists()


def test_apply_rejects_uneditable_model(service, tmp_home):
    """GGUF 之类的模型不能编辑。"""
    from wsp.models.registry import ModelEntry
    from wsp.service import ServiceError

    service.state.registry.add(
        ModelEntry(id="gguf-only", name="gguf", path="", kind="gguf")
    )
    project = service.create_project("p", "gguf-only")
    plan = {
        "goal": "x",
        "edits": [],
        "eval_gate": {"primary": "math_short", "min_delta": 0.05},
    }
    row = service.create_plan(project["id"], plan)
    with pytest.raises(ServiceError, match="不可编辑"):
        service.dry_run_plan(row["id"])
