"""存储清理（GC）测试。

核心保证：删除 checkpoint 之后，仍然可以从 diff 精确重建。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from wsp import storage
from wsp.weights import store


def _make_edits(service, project, plan, n: int = 3) -> list[str]:
    """制造 n 次编辑（混合干跑与应用）。"""
    ids = []
    for i in range(n):
        row = service.create_plan(project["id"], plan)
        out = service.apply_plan(row["id"])
        ids.append(out["edit"]["id"])
    return ids


def test_gc_removes_preview_rows(service, project, sample_plan):
    """干跑记录没有文件，只清数据库。"""
    for _ in range(3):
        row = service.create_plan(project["id"], sample_plan)
        service.dry_run_plan(row["id"])

    edits = service.list_edits(project["id"])
    previews = [e for e in edits if e["status"] == "preview"]
    assert len(previews) == 3

    result = service.run_storage_gc()
    assert result["removed_db_rows"] >= 3

    remaining = service.list_edits(project["id"])
    assert not any(e["status"] in storage.DISCARDABLE for e in remaining)


def test_gc_keeps_applied_by_default(service, project, sample_plan):
    ids = _make_edits(service, project, sample_plan, n=1)
    edit_id = ids[0]
    checkpoint = service.state.paths.revisions / "tiny" / "revisions" / edit_id
    assert checkpoint.is_dir()

    service.run_storage_gc()  # 默认不含 applied

    assert checkpoint.is_dir(), "已应用的 checkpoint 不该被删"
    assert service.get_edit(edit_id)["status"] == "applied"


def test_gc_removes_applied_when_asked(service, project, sample_plan):
    ids = _make_edits(service, project, sample_plan, n=1)
    edit_id = ids[0]
    checkpoint = service.state.paths.revisions / "tiny" / "revisions" / edit_id
    assert checkpoint.is_dir()

    result = service.run_storage_gc(include_applied=True)

    assert result["removed_checkpoints"] == 1
    assert not checkpoint.exists()
    assert result["freed_bytes"] > 0


def test_gc_keep_latest_protects_recent(service, project, sample_plan):
    ids = _make_edits(service, project, sample_plan, n=3)
    # 最新的应该被保护
    result = service.run_storage_gc(include_applied=True, keep_latest=1)
    assert result["removed_checkpoints"] == 2

    remaining = [
        p for p in (service.state.paths.revisions / "tiny" / "revisions").iterdir()
        if p.is_dir()
    ]
    assert len(remaining) == 1


def test_gc_never_deletes_without_diff(service, project, sample_plan):
    """没有 diff 的 checkpoint 不能删——删了就重建不了。"""
    ids = _make_edits(service, project, sample_plan, n=1)
    edit_id = ids[0]
    diff_dir = service.state.paths.edits / edit_id
    # 手动删掉 diff
    import shutil

    if diff_dir.exists():
        shutil.rmtree(diff_dir)

    plan = storage.plan_gc(
        service.state.db, service.state.paths, include_applied=True
    )
    kept_ids = [k["edit_id"] for k in plan.keep]
    assert edit_id in kept_ids
    assert "没有 diff" in next(
        k["reason"] for k in plan.keep if k["edit_id"] == edit_id
    )


def test_gc_refuses_paths_outside_workdir(service, project, sample_plan):
    """安全检查：不在 workdir 下的路径一律拒绝删除。"""
    ids = _make_edits(service, project, sample_plan, n=1)
    edit_id = ids[0]

    plan = service.plan_storage_gc(include_applied=True)
    # 篡改计划，指向 workdir 外面
    plan_obj = storage.GcPlan(
        delete_checkpoints=[{
            "edit_id": edit_id,
            "status": "applied",
            "path": str(service.state.paths.home.parent),  # 外面
            "bytes": 0,
        }]
    )
    result = storage.apply_gc(
        service.state.db, service.state.paths, plan_obj
    )
    assert result["removed_checkpoints"] == 0
    assert any("拒绝删除" in e for e in result["errors"])


def test_rebuild_from_diff_after_gc(service, project, sample_plan):
    """GC 之后，必须能从 diff + 原始模型精确重建编辑后的张量。"""
    import json

    import wsp.service as service_mod
    from wsp.editor import executor as ex

    ids = _make_edits(service, project, sample_plan, n=1)
    edit_id = ids[0]

    # 记录编辑后的张量 hash
    info = store.load_model(service.get_model_entry("tiny").path)
    modified = service_mod._pairs_from_plan(sample_plan, info)
    expected = {
        name: store.tensor_sha256(arr) for name, (_dt, arr) in modified.items()
    }

    service.run_storage_gc(include_applied=True)

    # 从 diff 重建
    diff_dir = service.state.paths.edits / edit_id
    manifest = json.loads((diff_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(diff_dir / "edit.npz") as npz:
        payload = {k: npz[k] for k in npz.files}

    for i, rec in enumerate(manifest["tensors"]):
        _dt, base = store.read_tensor(info, rec["name"])
        rebuilt = np.array(base, copy=True).reshape(-1)
        rebuilt[payload[f"t{i}_idx"]] = payload[f"t{i}_val"]
        rebuilt = rebuilt.reshape(rec["shape"])
        got = store.tensor_sha256(rebuilt)
        assert got == rec["edited_sha256"]
        assert got == expected[rec["name"]], f"{rec['name']} 重建结果不一致"


def test_usage_report(service, project, sample_plan):
    _make_edits(service, project, sample_plan, n=1)
    u = service.storage_usage()
    assert u["edits_total"] >= 1
    assert u["checkpoints_bytes"] > 0
    assert "home" in u


def test_plan_gc_does_not_touch_disk(service, project, sample_plan):
    """规划阶段不能有任何副作用。"""
    ids = _make_edits(service, project, sample_plan, n=1)
    checkpoint = service.state.paths.revisions / "tiny" / "revisions" / ids[0]
    assert checkpoint.is_dir()

    plan = service.plan_storage_gc(include_applied=True)
    assert plan["checkpoint_count"] == 1
    assert checkpoint.is_dir(), "只是规划，不该删东西"
    assert service.get_edit(ids[0]) is not None


def test_gc_idempotent(service, project, sample_plan):
    """连续清理两次不报错。"""
    _make_edits(service, project, sample_plan, n=2)
    first = service.run_storage_gc(include_applied=True)
    second = service.run_storage_gc(include_applied=True)
    assert first["removed_checkpoints"] == 2
    assert second["removed_checkpoints"] == 0
    assert second["errors"] == []
