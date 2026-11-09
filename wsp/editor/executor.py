"""编辑执行器：dry-run 与 apply。

关键行为：

* ``dry_run=True``  **绝不写任何文件**，只在内存里算 diff 并返回统计。
* ``apply`` 把结果写成**独立的 checkpoint 目录**，原始模型目录只读。
* 每次编辑都产出位级 diff（``edit.npz`` + ``manifest.json``），
  体积远小于整模型备份，且可以逐张量 sha256 校验。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config import Paths, assert_outside
from ..weights import names as name_utils
from ..weights import st, store
from . import ops, plan_schema


class ExecutorError(Exception):
    """执行失败。"""


@dataclass
class EditOutcome:
    """单条编辑的结果。"""

    index: int
    type: str
    tensor_name: str | None
    ok: bool
    stats: dict = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "type": self.type,
            "tensor_name": self.tensor_name,
            "ok": self.ok,
            "stats": self.stats,
            "error": self.error,
        }


@dataclass
class ExecutionResult:
    """整个计划的执行结果。"""

    dry_run: bool
    outcomes: list[EditOutcome] = field(default_factory=list)
    edit_dir: str = ""
    model_fraction: float = 0.0
    ok: bool = True
    error: str = ""
    plan: dict = field(default_factory=dict)

    @property
    def total_changed(self) -> int:
        return sum(o.stats.get("changed_entries", 0) for o in self.outcomes if o.ok)

    def as_dict(self) -> dict:
        return {
            "dry_run": self.dry_run,
            "ok": self.ok,
            "error": self.error,
            "model_fraction": self.model_fraction,
            "total_changed": self.total_changed,
            "edit_dir": self.edit_dir,
            "outcomes": [o.as_dict() for o in self.outcomes],
        }


def _direction_from_source(
    source: dict | None,
    *,
    rows: int,
    layer: int | None,
    module: str | None,
    seed: int = 0,
) -> np.ndarray | None:
    """把 direction_source 解析成向量。

    目前支持：
      * ``explicit``  直接给向量
      * ``random``    确定性伪随机（用于对照实验）
      * ``svd_top1``  需要传入张量本身，见 execute 里处理
    """
    if not source:
        return None
    kind = source.get("kind")
    if kind == "explicit":
        vec = source.get("vector")
        if not vec:
            return None
        return np.asarray(vec, dtype=np.float64)
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.standard_normal(rows)
    return None


def _svd_direction(
    arr: np.ndarray,
    dtype_str: str,
    axis: int,
    index_spec: dict | None = None,
) -> np.ndarray:
    """用**被选中的那部分**的最大奇异向量作为方向（离线兜底方案）。

    关键：必须在选中的切片上算 SVD。如果在整个张量上算，
    得到的全局主方向与局部切片几乎不对齐，编辑会弱到近乎无效。
    """
    flat = st.to_float(arr, dtype_str).astype(np.float64)

    if axis == 1:
        idx = ops._select_index(arr, 1, index_spec or {"kind": "all"})
        block = flat[:, idx]                 # [rows, k]
        u, _s, _vt = np.linalg.svd(block, full_matrices=False)
        return u[:, 0]
    idx = ops._select_index(arr, 0, index_spec or {"kind": "all"})
    block = flat[idx, :]                     # [k, cols]
    _u, _s, vt = np.linalg.svd(block, full_matrices=False)
    return vt[0]


def execute_plan(
    plan: dict,
    model_info: store.ModelInfo,
    paths: Paths,
    *,
    dry_run: bool = True,
    model_id: str = "",
    edit_id: str = "",
    progress: Callable[[str], None] | None = None,
) -> ExecutionResult:
    """执行一个已经校验过的计划。"""
    try:
        plan_schema.validate_plan(plan)
    except plan_schema.PlanValidationError as exc:
        return ExecutionResult(dry_run=dry_run, ok=False, error=str(exc), plan=plan)

    log = progress or (lambda _m: None)
    outcomes: list[EditOutcome] = []
    model_numel = sum(
        int(np.prod(info["shape"]))
        for shard in model_info.shards
        for k, info in st.read_header(shard)[0].items()
        if k != "__metadata__"
    )

    # 收集每个张量的修改后版本（内存里）
    modified: dict[str, tuple[str, np.ndarray]] = {}

    for i, edit in enumerate(plan.get("edits") or []):
        etype = edit.get("type", "noop")
        resolved = None
        if etype != "noop":
            resolved = _resolve(edit, model_info)
            if not resolved.ok:
                outcomes.append(
                    EditOutcome(
                        index=i,
                        type=etype,
                        tensor_name=None,
                        ok=False,
                        error=resolved.reason
                        + (
                            f"；候选: {resolved.candidates[:6]}"
                            if resolved.candidates
                            else ""
                        ),
                    )
                )
                continue

        try:
            if etype == "noop":
                outcomes.append(
                    EditOutcome(index=i, type="noop", tensor_name=None, ok=True,
                                stats={"changed_entries": 0})
                )
                continue

            tname = resolved.tensor_name
            assert tname is not None
            log(f"[{i+1}/{len(plan.get('edits') or [])}] {etype} on {tname}")

            if tname in modified:
                dtype_str, arr = modified[tname]
            else:
                dtype_str, arr = store.read_tensor(model_info, tname)

            if etype == "direction_ablation":
                direction = _direction_from_source(
                    edit.get("direction_source"),
                    rows=arr.shape[0],
                    layer=edit.get("layer"),
                    module=edit.get("module"),
                )
                if direction is None:
                    src = (edit.get("direction_source") or {}).get("kind", "svd_top1")
                    if src == "svd_top1":
                        direction = _svd_direction(
                            arr, dtype_str, edit.get("axis", 1), edit.get("index")
                        )
                new_arr, stats = ops.apply_direction_ablation(
                    arr,
                    dtype_str,
                    axis=edit["axis"],
                    index_spec=edit["index"],
                    direction=direction,
                    magnitude=edit["magnitude"],
                    method=edit.get("method") or "project_out",
                    norm_preserve=bool(edit.get("norm_preserve", True)),
                )
            elif etype == "rank1_subtract":
                src = edit.get("direction_source") or {}
                u_vec, v_vec = src.get("u"), src.get("v")
                if u_vec is None or v_vec is None:
                    raise ExecutorError(
                        "rank1_subtract 需要在 direction_source 里同时给出 u 和 v"
                    )
                new_arr, stats = ops.apply_rank1_subtract(
                    arr,
                    dtype_str,
                    axis=edit["axis"],
                    index_spec=edit["index"],
                    u=np.asarray(u_vec, dtype=np.float64),
                    v=np.asarray(v_vec, dtype=np.float64),
                    alpha=edit["magnitude"],
                )
            elif etype == "module_scale":
                new_arr, stats = ops.apply_module_scale(
                    arr, dtype_str, factor=edit["factor"]
                )
            else:
                raise ExecutorError(f"未知编辑类型 {etype!r}")

            ops.check_budget(stats, model_numel, tensor_numel=int(arr.size))
            modified[tname] = (dtype_str, new_arr)
            outcomes.append(
                EditOutcome(index=i, type=etype, tensor_name=tname, ok=True,
                            stats=stats)
            )
        except (ops.EditError, ExecutorError, store.ModelError) as exc:
            # 失败时也保留统计信息，让用户能看到"到底改了多少才会被拒"
            outcomes.append(
                EditOutcome(
                    index=i,
                    type=etype,
                    tensor_name=getattr(resolved, "tensor_name", None),
                    ok=False,
                    stats=locals().get("stats", {}),
                    error=str(exc),
                )
            )

    result = ExecutionResult(dry_run=dry_run, outcomes=outcomes, plan=plan)
    result.ok = all(o.ok for o in outcomes)
    if model_numel:
        result.model_fraction = result.total_changed / model_numel

    if not dry_run and result.ok and modified:
        result.edit_dir = _write_checkpoint(
            model_info, modified, paths, model_id=model_id, edit_id=edit_id
        )
        log(f"已写出新 checkpoint: {result.edit_dir}")

    return result


def _resolve(edit: dict, model_info: store.ModelInfo):
    """把编辑里的 layer/module 或 tensor_name 解析成真实张量名。"""
    return name_utils.resolve(
        model_info.tensor_names,
        layer=edit.get("layer"),
        module=edit.get("module"),
        name=edit.get("tensor_name"),
    )


def _write_checkpoint(
    model_info: store.ModelInfo,
    modified: dict[str, tuple[str, np.ndarray]],
    paths: Paths,
    *,
    model_id: str,
    edit_id: str,
) -> str:
    """写出新的 checkpoint 目录。

    做法：复制原始模型目录里**除被修改分片之外**的内容（硬链接优先），
    被修改的分片重写。
    """
    if not edit_id:
        raise ExecutorError("apply 需要 edit_id")
    revision_dir = paths.revisions / model_id / "revisions" / edit_id
    assert_outside(model_info.path, revision_dir)
    if revision_dir.exists():
        shutil.rmtree(revision_dir)
    revision_dir.mkdir(parents=True, exist_ok=True)

    # 按分片分组修改
    by_shard: dict[str, list[str]] = {}
    for tname in modified:
        shard = store.shard_of(model_info, tname)
        by_shard.setdefault(shard, []).append(tname)

    for shard in model_info.shards:
        target = revision_dir / os.path.basename(shard)
        if shard not in by_shard:
            _copy_or_link(shard, target)
            continue
        # 载入整个分片，替换被修改的张量，写回
        tensors = st.load(shard)
        for tname in by_shard[shard]:
            tensors[tname] = modified[tname]
        st.save(target, tensors, metadata={"format": "pt", "wsp_edit": edit_id})

    # 复制所有非 safetensors 的附属文件（配置、tokenizer 等）
    for entry in os.listdir(model_info.path):
        src = os.path.join(model_info.path, entry)
        if not os.path.isfile(src) or entry.endswith(".safetensors"):
            continue
        dst = revision_dir / entry
        if not dst.exists():
            shutil.copy2(src, dst)

    return str(revision_dir)


def _copy_or_link(src: str, dst: str) -> None:
    """优先硬链接（省空间），失败则复制。"""
    try:
        os.link(src, dst)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 位级 diff 导出
# ---------------------------------------------------------------------------

def export_edit_diff(
    model_info: store.ModelInfo,
    modified: dict[str, tuple[str, np.ndarray]],
    out_dir: str | os.PathLike,
) -> dict:
    """把改动导出成 ``edit.npz`` + ``manifest.json``。

    体积远小于整模型：一次 13 万权重的编辑大约 400 KB。
    """
    out = str(out_dir)
    os.makedirs(out, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records: list[dict] = []

    for i, tname in enumerate(sorted(modified)):
        dtype_str, new_arr = modified[tname]
        old_dtype, old_arr = store.read_tensor(model_info, tname)
        if old_dtype != dtype_str:
            raise ExecutorError(f"{tname}: dtype 不一致")
        mask = old_arr != new_arr
        changed = int(np.count_nonzero(mask))
        if changed == 0:
            continue
        idx = np.flatnonzero(mask).astype(np.int64)
        arrays[f"t{i}_idx"] = idx
        arrays[f"t{i}_val"] = new_arr.reshape(-1)[idx]
        old_f = st.to_float(old_arr, dtype_str)
        new_f = st.to_float(new_arr, dtype_str)
        records.append(
            {
                "name": tname,
                "shape": list(old_arr.shape),
                "dtype": "bfloat16" if dtype_str == "BF16" else dtype_str.lower(),
                "form": "sparse",
                "changed_entries": changed,
                "changed_rows": int(np.count_nonzero(mask.any(axis=1))),
                "changed_cols": int(np.count_nonzero(mask.any(axis=0))),
                "relative_change": float(
                    np.linalg.norm(new_f - old_f) / np.linalg.norm(old_f)
                )
                if np.linalg.norm(old_f)
                else 0.0,
                "base_sha256": store.tensor_sha256(old_arr),
                "edited_sha256": store.tensor_sha256(new_arr),
            }
        )

    np.savez_compressed(os.path.join(out, "edit.npz"), **arrays)
    manifest = {
        "format": "wsp.edit",
        "format_version": 1,
        "tensors": records,
        "changed_tensors": len(records),
        "changed_entries_total": sum(r["changed_entries"] for r in records),
        "base_model": model_info.name,
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def apply_edit_diff(
    model_info: store.ModelInfo,
    edit_dir: str | os.PathLike,
    out_dir: str | os.PathLike,
) -> dict:
    """从 diff 重建 checkpoint，并逐张量校验 sha256。

    这是回滚与"从 diff 复原"的共同底层。
    """
    src = str(edit_dir)
    with open(os.path.join(src, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "wsp.edit":
        raise ExecutorError(f"{edit_dir}: 不是 wsp.edit 格式")
    if manifest.get("format_version") != 1:
        raise ExecutorError(
            f"{edit_dir}: 不支持的格式版本 {manifest.get('format_version')}"
        )

    with np.load(os.path.join(src, "edit.npz")) as npz:
        payload = {k: npz[k] for k in npz.files}

    modified: dict[str, tuple[str, np.ndarray]] = {}
    for i, rec in enumerate(manifest["tensors"]):
        tname = rec["name"]
        dtype_str, base = store.read_tensor(model_info, tname)
        if store.tensor_sha256(base) != rec["base_sha256"]:
            raise ExecutorError(
                f"{tname}: 当前模型张量与 manifest 记录的 base_sha256 不一致，拒绝应用"
            )
        rebuilt = np.array(base, copy=True).reshape(-1)
        rebuilt[payload[f"t{i}_idx"]] = payload[f"t{i}_val"]
        rebuilt = rebuilt.reshape(rec["shape"])
        if store.tensor_sha256(rebuilt) != rec["edited_sha256"]:
            raise ExecutorError(f"{tname}: 重建结果与 edited_sha256 不一致")
        modified[tname] = (dtype_str, rebuilt)

    out = _write_checkpoint(model_info, modified, Paths(out_dir),  # type: ignore[arg-type]
                            model_id=model_info.name, edit_id=os.path.basename(out_dir))
    return {"ok": True, "edit_dir": out, "tensors": len(modified)}
