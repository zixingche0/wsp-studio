"""存储清理（垃圾回收）。

解决的问题：每次 apply 都会写一份完整 checkpoint（1.7 GB / 个），
反复实验后磁盘很快被撑满，而旧记录早已没有保留价值。

清理策略（都是显式触发，不自动删）：
  * ``preview`` / ``preview_failed``：干跑记录，没有任何文件，只清数据库
  * ``rolled_back`` / ``eval_failed`` / ``apply_failed``：先把 diff 存成
    可复现的位级补丁（几百 KB），再删掉 1.7 GB 的 checkpoint
  * ``applied`` / ``eval_passed``：默认**不删**，除非显式要求

diff 永远保留，所以任何编辑都能从 diff + 原始模型精确重建。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import Paths
from .store import db as db_mod


@dataclass
class GcPlan:
    """一次清理计划。"""

    delete_checkpoints: list[dict] = field(default_factory=list)
    delete_db_rows: list[str] = field(default_factory=list)
    keep: list[dict] = field(default_factory=list)
    total_bytes: int = 0

    @property
    def checkpoint_count(self) -> int:
        return len(self.delete_checkpoints)

    def as_dict(self) -> dict:
        return {
            "delete_checkpoints": [
                {"edit_id": d["edit_id"], "status": d["status"],
                 "path": d["path"], "bytes": d["bytes"]}
                for d in self.delete_checkpoints
            ],
            "delete_db_rows": self.delete_db_rows,
            "keep": self.keep,
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_bytes / 1024**3, 2),
            "checkpoint_count": self.checkpoint_count,
        }


# 状态 -> 处理方式
DISCARDABLE = {"preview", "preview_failed"}          # 无文件，删数据库记录
RECLAIMABLE = {"rolled_back", "eval_failed", "apply_failed"}  # 可删 checkpoint
PRESERVE = {"applied", "eval_passed", "applying"}     # 默认保留


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def plan_gc(
    db: db_mod.Database,
    paths: Paths,
    *,
    include_applied: bool = False,
    keep_latest: int = 0,
) -> GcPlan:
    """算出可以清理什么，**不执行任何删除**。

    ``keep_latest``：无论如何保留最近 N 个 checkpoint（跨所有状态）。
    """
    plan = GcPlan()
    edits = db.list_edits()

    # 最近 N 个一律保留
    protected: set[str] = set()
    if keep_latest > 0:
        protected = {e["id"] for e in edits[:keep_latest]}

    for edit in edits:
        eid = edit["id"]
        status = edit["status"]
        ckpt = _checkpoint_path(paths, edit)

        if eid in protected:
            plan.keep.append({"edit_id": eid, "status": status,
                              "reason": "keep_latest"})
            continue

        if status in DISCARDABLE:
            plan.delete_db_rows.append(eid)
            continue

        if status in RECLAIMABLE:
            if ckpt and ckpt.is_dir():
                size = _dir_size(ckpt)
                # 必须有 diff 才能安全删 checkpoint
                diff_dir = paths.edits / eid
                if not (diff_dir / "manifest.json").is_file():
                    plan.keep.append({
                        "edit_id": eid, "status": status,
                        "reason": "没有 diff，删了就无法重建",
                    })
                    continue
                plan.delete_checkpoints.append({
                    "edit_id": eid, "status": status,
                    "path": str(ckpt), "bytes": size,
                })
                plan.total_bytes += size
            else:
                plan.delete_db_rows.append(eid)
            continue

        if status in PRESERVE:
            if include_applied and ckpt and ckpt.is_dir():
                diff_dir = paths.edits / eid
                if (diff_dir / "manifest.json").is_file():
                    size = _dir_size(ckpt)
                    plan.delete_checkpoints.append({
                        "edit_id": eid, "status": status,
                        "path": str(ckpt), "bytes": size,
                    })
                    plan.total_bytes += size
                    continue
                plan.keep.append({
                    "edit_id": eid, "status": status,
                    "reason": "没有 diff，删了就无法重建",
                })
                continue
            plan.keep.append({"edit_id": eid, "status": status,
                              "reason": "已应用且仍在用"})
            continue

        plan.keep.append({"edit_id": eid, "status": status, "reason": "未知状态"})

    return plan


def apply_gc(db: db_mod.Database, paths: Paths, plan: GcPlan) -> dict:
    """执行清理。"""
    removed_bytes = 0
    removed_paths: list[str] = []
    errors: list[str] = []

    for item in plan.delete_checkpoints:
        path = Path(item["path"])
        # 安全：只删 workdir 下的东西
        try:
            path.resolve().relative_to(paths.workdir.resolve())
        except ValueError:
            errors.append(f"{path} 不在 workdir 下，拒绝删除")
            continue
        try:
            shutil.rmtree(path, ignore_errors=False)
            removed_bytes += item["bytes"]
            removed_paths.append(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    removed_rows = 0
    for eid in plan.delete_db_rows:
        try:
            with db.cursor() as conn:
                conn.execute("DELETE FROM evals WHERE edit_id=?", (eid,))
                conn.execute("DELETE FROM edits WHERE id=?", (eid,))
            removed_rows += 1
        except Exception as exc:
            errors.append(f"{eid}: {exc}")

    return {
        "removed_checkpoints": len(removed_paths),
        "removed_db_rows": removed_rows,
        "freed_bytes": removed_bytes,
        "freed_gb": round(removed_bytes / 1024**3, 2),
        "errors": errors,
    }


def _checkpoint_path(paths: Paths, edit: dict) -> Path | None:
    """找一次编辑对应的 checkpoint 目录。"""
    diff = edit.get("diff") or {}
    exec_info = diff.get("execution") or {}
    p = exec_info.get("edit_dir")
    if p and Path(p).is_dir():
        return Path(p)
    candidate = paths.revisions / edit["model_id"] / "revisions" / edit["id"]
    return candidate if candidate.is_dir() else None


def usage_report(paths: Paths, db: db_mod.Database) -> dict:
    """磁盘使用情况。"""
    revs = paths.revisions
    diffs = paths.edits
    by_status: dict[str, int] = {}
    for e in db.list_edits():
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1

    return {
        "home": str(paths.home),
        "workdir_bytes": _dir_size(paths.workdir) if paths.workdir.is_dir() else 0,
        "checkpoints_bytes": _dir_size(revs) if revs.is_dir() else 0,
        "diffs_bytes": _dir_size(diffs) if diffs.is_dir() else 0,
        "edits_by_status": by_status,
        "edits_total": sum(by_status.values()),
    }
