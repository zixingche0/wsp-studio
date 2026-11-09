"""执行器测试：dry-run 不写文件，apply 写新 checkpoint，原始目录只读。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from wsp.config import Paths
from wsp.editor import executor, plan_schema
from wsp.weights import store


def _plan(**overrides):
    plan = {
        "goal": "测试",
        "edits": [
            {
                "type": "direction_ablation",
                "layer": 1,
                "module": "mlp.down_proj",
                "axis": 1,
                "index": {"kind": "range", "start": 0, "stop": 4},
                "method": "project_out",
                "magnitude": 0.5,
                "norm_preserve": True,
                "direction_source": {"kind": "random"},
                "expected_effect": "x",
                "risk": "y",
            }
        ],
        "eval_gate": {"primary": "math_short", "min_delta": 0.05},
    }
    plan.update(overrides)
    return plan


def test_dry_run_writes_nothing(service, project, sample_plan):
    result = service.dry_run_plan(
        service.create_plan(project["id"], sample_plan)["id"]
    )
    revisions = service.state.paths.revisions
    # 只允许存在空目录，不能有 checkpoint
    files = list(revisions.rglob("*.safetensors"))
    assert files == []
    assert result["result"]["dry_run"] is True


def test_apply_creates_checkpoint(service, project, sample_plan):
    plan_row = service.create_plan(project["id"], sample_plan)
    out = service.apply_plan(plan_row["id"])
    ckpt = Path(out["checkpoint"])
    assert ckpt.is_dir()
    assert (ckpt / "model.safetensors").is_file()
    assert (ckpt / "config.json").is_file()


def test_original_model_untouched(service, project, sample_plan):
    info = store.load_model(service.get_model_entry("tiny").path)
    before = {n: store.tensor_sha256(store.read_tensor(info, n)[1])
              for n in info.tensor_names}
    service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    after = {n: store.tensor_sha256(store.read_tensor(info, n)[1])
             for n in info.tensor_names}
    assert before == after


def test_apply_changes_only_selected_tensor(service, project, sample_plan):
    out = service.apply_plan(service.create_plan(project["id"], sample_plan)["id"])
    base = store.load_model(service.get_model_entry("tiny").path)
    edited = store.load_model(out["checkpoint"])
    target = "model.layers.1.mlp.down_proj.weight"
    for name in base.tensor_names:
        a = store.read_tensor(base, name)[1]
        b = store.read_tensor(edited, name)[1]
        if name == target:
            assert not np.array_equal(a, b), name
        else:
            assert np.array_equal(a, b), f"{name} 不该变"


def test_budget_violation_is_reported(service, project):
    """把整个张量都改掉，应当超过 1% 上限并被拒绝。"""
    plan = _plan()
    plan["edits"][0]["index"] = {"kind": "all"}
    plan["edits"][0]["magnitude"] = 1.0
    plan["edits"][0]["norm_preserve"] = False
    row = service.create_plan(project["id"], plan)
    out = service.dry_run_plan(row["id"])
    # tiny 模型只有 3 层，全改一个张量会超过 1%
    assert out["result"]["ok"] is False
    assert any("上限" in (o.get("error") or "") for o in out["result"]["outcomes"])


def test_noop_plan_produces_no_change(service, project):
    plan = plan_schema.empty_plan("什么都不做")
    row = service.create_plan(project["id"], plan)
    out = service.dry_run_plan(row["id"])
    assert out["result"]["ok"] is True
    assert out["result"]["total_changed"] == 0


def test_bad_tensor_name_reports_candidates(service, project):
    plan = _plan()
    plan["edits"][0]["layer"] = 999
    row = service.create_plan(project["id"], plan)
    out = service.dry_run_plan(row["id"])
    assert out["result"]["ok"] is False
    outcome = out["result"]["outcomes"][0]
    assert "候选" in outcome["error"] or "找不到" in outcome["error"]


def test_export_and_reimport_diff(service, project, sample_plan):
    """导出的 diff 能重建出完全一致的张量。"""
    from wsp.editor import executor as ex

    plan = sample_plan
    row = service.create_plan(project["id"], plan)
    service.apply_plan(row["id"])
    info = store.load_model(service.get_model_entry("tiny").path)
    import wsp.service as service_mod
    modified = service_mod._pairs_from_plan(plan, info)
    out = Path(service.state.paths.home) / "diff-test"
    manifest = ex.export_edit_diff(info, modified, out)
    assert manifest["changed_tensors"] >= 1
    rec = manifest["tensors"][0]
    assert len(rec["base_sha256"]) == 64
    assert len(rec["edited_sha256"]) == 64
    assert rec["base_sha256"] != rec["edited_sha256"]
    # npz 能被读回
    with np.load(out / "edit.npz") as npz:
        assert "t0_idx" in npz.files
        assert "t0_val" in npz.files


def test_diff_roundtrip_rebuilds_identical_tensor(service, project, sample_plan):
    """用 diff 重建，结果必须与 apply 出的 checkpoint 逐 bit 一致。"""
    import wsp.service as service_mod
    from wsp.editor import executor as ex

    info = store.load_model(service.get_model_entry("tiny").path)
    modified = service_mod._pairs_from_plan(sample_plan, info)
    out_dir = Path(service.state.paths.home) / "rt-diff"
    ex.export_edit_diff(info, modified, out_dir)

    with np.load(out_dir / "edit.npz") as npz:
        payload = {k: npz[k] for k in npz.files}
    import json
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    for i, rec in enumerate(manifest["tensors"]):
        _dt, base = store.read_tensor(info, rec["name"])
        rebuilt = np.array(base, copy=True).reshape(-1)
        rebuilt[payload[f"t{i}_idx"]] = payload[f"t{i}_val"]
        rebuilt = rebuilt.reshape(rec["shape"])
        assert store.tensor_sha256(rebuilt) == rec["edited_sha256"]
        assert store.tensor_sha256(rebuilt) == store.tensor_sha256(
            modified[rec["name"]][1]
        )
