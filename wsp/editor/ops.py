"""编辑算子：真正修改权重的地方。

三个算子：

* ``direction_ablation``  沿一个方向做投影移除（或缩放）
* ``module_scale``        整体小幅缩放
* ``noop``                什么都不做（用于测试与对照）

设计原则（来自对官方编辑的实证）：

1. **切哪个维度必须显式写出**。官方 Qwen3.5 的编辑是 ``down_proj[:, 64列]``
   （沿 3584 维选 64 列，1024 行全覆盖），而 ``down_proj`` 形状是
   ``[1024, 3584]``。如果按"输出列"理解会完全算错。
2. **bf16 是位操作**。改动数按"bf16 位模式真的变了"统计，
   而不是按"浮点值变了"统计——官方 131072 个元素只有 130188 个真的变了。
3. **1% 上限硬编码**。超过就拒绝，不做"警告后继续"。
"""

from __future__ import annotations

import numpy as np

from ..weights import st

# 单次编辑允许改动的最大权重比例（超过即拒绝）
MAX_EDIT_FRACTION = 0.01

# 单个张量允许改动的最大比例（超过即拒绝）
# 模型级 1% 会放过"把一个张量改光"的情况：小张量在模型里占比小，
# 但改掉它 80% 的元素必然破坏功能。
MAX_TENSOR_FRACTION = 0.5


class EditError(Exception):
    """编辑本身不合法（不是运行时错误）。"""


def _select_index(arr: np.ndarray, axis: int, spec: dict) -> np.ndarray:
    """根据 spec 在 arr 的 axis 上选出要操作的下标。"""
    length = arr.shape[axis]
    kind = spec.get("kind", "all")

    if kind == "all":
        return np.arange(length)

    if kind == "range":
        start = int(spec.get("start", 0))
        stop = int(spec.get("stop", length))
        if not (0 <= start < stop <= length):
            raise EditError(f"range [{start}, {stop}) 超出维度 {length}")
        return np.arange(start, stop)

    if kind == "list":
        idx = np.asarray(spec.get("indices", []), dtype=np.int64)
        if idx.size == 0:
            raise EditError("index 列表为空")
        if idx.min() < 0 or idx.max() >= length:
            raise EditError(f"index 超出维度 {length}")
        return np.unique(idx)

    if kind == "top_norm":
        n = int(spec.get("n", 0))
        if n <= 0:
            raise EditError("top_norm 需要 n > 0")
        n = min(n, length)
        # 沿 axis 计算每一"条"的范数：把 axis 移到最前
        moved = np.moveaxis(arr, axis, 0).reshape(length, -1).astype(np.float32)
        norms = np.linalg.norm(moved, axis=1)
        return np.argsort(-norms)[:n]

    raise EditError(f"未知的 index.kind: {kind!r}")


def _get_direction(
    direction: np.ndarray | None,
    length: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """校验并返回单位方向向量。"""
    if direction is None:
        # 没有外部方向时，用一个确定性的伪随机单位向量。
        # 这**不是**真正有用的编辑，只是让算子可测试；planner 应当总给出方向。
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(length).astype(np.float64)
    else:
        v = np.asarray(direction, dtype=np.float64).reshape(-1)
        if v.size != length:
            raise EditError(f"方向维度 {v.size} 与目标维度 {length} 不一致")
    norm = np.linalg.norm(v)
    if norm == 0:
        raise EditError("方向向量为零向量")
    return v / norm


def apply_rank1_subtract(
    arr: np.ndarray,
    dtype_str: str,
    *,
    axis: int,
    index_spec: dict,
    u: np.ndarray,
    v: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict]:
    """减去一个 rank-1 分量：``W[sel] -= alpha * u ⊗ v``。

    这是官方 exp8 编辑的真实形状（实测：差分矩阵 σ₂/σ₁ = 1e-3，
    即严格 rank-1；影响 64 个**不连续**列，1024 行全覆盖）。

    与 ``project_out`` 的区别：
      * ``project_out`` 是"减掉切片自身在 d 上的投影"（自适应）
      * ``rank1_subtract`` 是"减掉一个给定的外部方向对"（精确复现）
    """
    if not -3 <= alpha <= 3 or alpha == 0:
        raise EditError(f"alpha={alpha} 超出合理范围 [-3, 3] 且不能为 0")
    if arr.ndim != 2:
        raise EditError(f"rank1_subtract 需要二维张量，实际是 {arr.shape}")

    axis = axis if axis >= 0 else arr.ndim + axis
    if axis not in (0, 1):
        raise EditError(f"axis 只能是 0 或 1，实际是 {axis}")

    idx = _select_index(arr, axis, index_spec)
    work = st.to_float(arr, dtype_str).astype(np.float64)

    if axis == 1:
        n_rows = work.shape[0]
        u_vec = _get_direction(u, n_rows)
        v_vec = np.asarray(v, dtype=np.float64).reshape(-1)
        if v_vec.size != idx.size:
            # 允许 v 是所有列的全长向量，只取选中部分
            if v_vec.size == work.shape[1]:
                v_vec = v_vec[idx]
            else:
                raise EditError(
                    f"v 维度 {v_vec.size} 与选中列数 {idx.size} 不匹配"
                )
        work[:, idx] -= alpha * np.outer(u_vec, v_vec)
    else:
        n_cols = work.shape[1]
        v_vec = _get_direction(v, n_cols)
        u_vec = np.asarray(u, dtype=np.float64).reshape(-1)
        if u_vec.size != idx.size:
            if u_vec.size == work.shape[0]:
                u_vec = u_vec[idx]
            else:
                raise EditError(
                    f"u 维度 {u_vec.size} 与选中行数 {idx.size} 不匹配"
                )
        work[idx, :] -= alpha * np.outer(u_vec, v_vec)

    new_arr = st.to_bits(work, dtype_str)
    stats = _diff_stats(arr, new_arr, axis=axis, idx=idx)
    stats.update(
        {
            "operator": "rank1_subtract",
            "axis": axis,
            "alpha": alpha,
            "selected": int(idx.size),
            "u_norm": float(np.linalg.norm(u)),
            "v_norm": float(np.linalg.norm(v)),
        }
    )
    return new_arr, stats


def apply_direction_ablation(
    arr: np.ndarray,
    dtype_str: str,
    *,
    axis: int,
    index_spec: dict,
    direction: np.ndarray | None,
    magnitude: float,
    method: str = "project_out",
    norm_preserve: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """沿 direction 对选中的切片做投影移除或缩放。

    ``project_out``：``s <- s - magnitude * (s · d) d``
        即把切片在 d 方向上的分量按比例去掉（magnitude=1 完全去掉）。
    ``scale``：``s <- s * magnitude``
        magnitude 是保留系数（1.0 不变，0.75 减弱）。

    ``norm_preserve`` 只在 project_out 时生效：缩放回原范数，避免整块塌缩。
    """
    if not 0 <= magnitude <= 2:
        raise EditError(f"magnitude={magnitude} 超出合理范围 [0, 2]")
    if method not in ("project_out", "scale"):
        raise EditError(f"未知的 method: {method!r}")
    if arr.ndim != 2:
        raise EditError(f"direction_ablation 需要二维张量，实际是 {arr.shape}")

    axis = axis if axis >= 0 else arr.ndim + axis
    if axis not in (0, 1):
        raise EditError(f"axis 只能是 0 或 1，实际是 {axis}")

    idx = _select_index(arr, axis, index_spec)

    work = st.to_float(arr, dtype_str).astype(np.float64)

    if axis == 1:
        block = work[:, idx]                      # [rows, k]
        length = block.shape[0]
        d = _get_direction(direction, length, seed=seed)
        orig_norms = np.linalg.norm(block, axis=0)
        if method == "project_out":
            proj = np.outer(d, d @ block)         # [rows, k]
            block = block - magnitude * proj
            if norm_preserve:
                new_norms = np.linalg.norm(block, axis=0)
                scale = np.divide(
                    orig_norms,
                    new_norms,
                    out=np.ones_like(orig_norms),
                    where=new_norms > 0,
                )
                block = block * scale
        else:
            block = block * magnitude
        work[:, idx] = block
    else:
        block = work[idx, :]                      # [k, cols]
        length = block.shape[1]
        d = _get_direction(direction, length, seed=seed)
        orig_norms = np.linalg.norm(block, axis=1)
        if method == "project_out":
            proj = np.outer(block @ d, d)         # [k, cols]
            block = block - magnitude * proj
            if norm_preserve:
                new_norms = np.linalg.norm(block, axis=1)
                scale = np.divide(
                    orig_norms,
                    new_norms,
                    out=np.ones_like(orig_norms),
                    where=new_norms > 0,
                )
                block = block * scale[:, None]
        else:
            block = block * magnitude
        work[idx, :] = block

    new_arr = st.to_bits(work, dtype_str)

    stats = _diff_stats(arr, new_arr, axis=axis, idx=idx)
    stats.update(
        {
            "operator": "direction_ablation",
            "axis": axis,
            "method": method,
            "magnitude": magnitude,
            "norm_preserve": norm_preserve,
            "selected": int(idx.size),
            "direction_norm": float(np.linalg.norm(d)),
        }
    )
    return new_arr, stats


def apply_module_scale(
    arr: np.ndarray,
    dtype_str: str,
    *,
    factor: float,
) -> tuple[np.ndarray, dict]:
    """整体缩放一个张量。factor 是保留系数。"""
    if not 0 < factor <= 3:
        raise EditError(f"factor={factor} 超出合理范围 (0, 3]")
    work = st.to_float(arr, dtype_str).astype(np.float64) * factor
    new_arr = st.to_bits(work, dtype_str)
    stats = _diff_stats(arr, new_arr, axis=0, idx=np.arange(arr.shape[0]))
    stats.update({"operator": "module_scale", "factor": factor})
    return new_arr, stats


def apply_noop(arr: np.ndarray, dtype_str: str) -> tuple[np.ndarray, dict]:
    """什么都不做，返回原数组的副本。"""
    stats = _diff_stats(arr, arr, axis=0, idx=np.arange(arr.shape[0]))
    stats.update({"operator": "noop"})
    return np.array(arr, copy=True), stats


def _diff_stats(
    before: np.ndarray,
    after: np.ndarray,
    *,
    axis: int,
    idx: np.ndarray,
) -> dict:
    """统计改动：按**位模式**判断是否真的变了。"""
    b = np.ascontiguousarray(before)
    a = np.ascontiguousarray(after)
    changed_mask = b != a
    changed = int(np.count_nonzero(changed_mask))
    total = int(b.size)

    fb = st.to_float(b, "BF16" if b.dtype == np.uint16 else "F32")
    fa = st.to_float(a, "BF16" if a.dtype == np.uint16 else "F32")
    nb = float(np.linalg.norm(fb)) if fb.size else 0.0
    na = float(np.linalg.norm(fa)) if fa.size else 0.0
    # 与官方 manifest 的 relative_change 口径一致：
    # ||new - old|| / ||old||，衡量"改了多少"，不是"范数变了多少"
    diff_norm = float(np.linalg.norm(fa.astype(np.float64) - fb.astype(np.float64)))

    return {
        "numel": total,
        "changed_entries": changed,
        "changed_fraction": (changed / total) if total else 0.0,
        "changed_rows": int(np.count_nonzero(changed_mask.any(axis=1))),
        "changed_cols": int(np.count_nonzero(changed_mask.any(axis=0))),
        "norm_before": nb,
        "norm_after": na,
        "relative_norm_change": ((na - nb) / nb) if nb else 0.0,
        "relative_change": (diff_norm / nb) if nb else 0.0,
        "selected_axis": axis,
        "selected_count": int(idx.size),
    }


def check_budget(stats: dict, model_numel: int, tensor_numel: int | None = None) -> None:
    """检查编辑是否超过上限。

    两道闸门，**任一超限即拒绝**：

    1. **模型级**：改动量 / 模型总权重 ≤ 1%
       防止一次改动太大，模型行为整体漂移。

    2. **张量级**：改动量 / 该张量元素数 ≤ 50%
       防止"把一个张量几乎改光"——即使这个张量在模型里占比很小，
       改掉 84% 的元素已经不是"精准编辑"了，而是重写。

    第二条来自实测：``index.kind="all"`` 时模型级只占 0.35%，
    看起来很安全，但该张量 84% 的元素都被改了，必然破坏功能。
    """
    if tensor_numel is None:
        tensor_numel = int(stats.get("numel") or 0)

    frac_model = (stats["changed_entries"] / model_numel) if model_numel else 0.0
    frac_tensor = (
        (stats["changed_entries"] / tensor_numel) if tensor_numel else 0.0
    )

    stats["model_fraction"] = frac_model
    stats["tensor_fraction"] = frac_tensor
    stats["within_model_budget"] = frac_model <= MAX_EDIT_FRACTION
    stats["within_tensor_budget"] = frac_tensor <= MAX_TENSOR_FRACTION
    stats["within_budget"] = (
        stats["within_model_budget"] and stats["within_tensor_budget"]
    )

    problems = []
    if not stats["within_model_budget"]:
        problems.append(
            f"改动 {frac_model:.4%} 的模型权重，超过 {MAX_EDIT_FRACTION:.0%} 上限"
        )
    if not stats["within_tensor_budget"]:
        problems.append(
            f"改动了张量的 {frac_tensor:.1%}，超过单张量 {MAX_TENSOR_FRACTION:.0%} 上限"
        )
    if problems:
        raise EditError(
            "；".join(problems)
            + "。请缩小 index 范围、减少 layer 数量，或降低 magnitude。"
        )
