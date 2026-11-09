"""编辑算子测试。重点验证几何形状与位级精确性。"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.editor import ops
from wsp.weights import st


def make_tensor(rows=32, cols=64, seed=0):
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
    return st.bf16_bits(arr)


def test_project_out_only_touches_selected_columns():
    """官方编辑的几何：axis=1 选 64 列，全部行覆盖，其余列逐 bit 不变。"""
    bits = make_tensor(16, 64)
    direction = np.random.default_rng(1).standard_normal(16)
    new, stats = ops.apply_direction_ablation(
        bits, "BF16", axis=1, index_spec={"kind": "range", "start": 0, "stop": 8},
        direction=direction, magnitude=1.0, method="project_out",
        norm_preserve=False,
    )
    mask = new != bits
    assert stats["changed_cols"] == 8
    assert stats["changed_rows"] == 16
    assert np.array_equal(new[:, 8:], bits[:, 8:])
    assert mask[:, :8].any()


def test_project_out_full_magnitude_removes_component():
    """magnitude=1 且不保范数时，选中列在 direction 上的投影应接近 0。"""
    bits = make_tensor(16, 32)
    d = np.random.default_rng(2).standard_normal(16)
    d /= np.linalg.norm(d)
    new, _ = ops.apply_direction_ablation(
        bits, "BF16", axis=1, index_spec={"kind": "range", "start": 0, "stop": 32},
        direction=d, magnitude=1.0, method="project_out", norm_preserve=False,
    )
    block = st.bf16_to_f32(new).astype(np.float64)
    residual = d @ block
    assert np.allclose(residual, 0, atol=1e-2)


def test_norm_preserve_keeps_column_norms():
    bits = make_tensor(16, 32)
    d = np.random.default_rng(3).standard_normal(16)
    new, _ = ops.apply_direction_ablation(
        bits, "BF16", axis=1, index_spec={"kind": "all"},
        direction=d, magnitude=1.0, method="project_out", norm_preserve=True,
    )
    before = np.linalg.norm(st.bf16_to_f32(bits).astype(np.float64), axis=0)
    after = np.linalg.norm(st.bf16_to_f32(new).astype(np.float64), axis=0)
    assert np.allclose(before, after, rtol=2e-2)


def test_scale_method_multiplies():
    bits = make_tensor(8, 8)
    new, stats = ops.apply_module_scale(bits, "BF16", factor=0.5)
    before = st.bf16_to_f32(bits)
    after = st.bf16_to_f32(new)
    assert np.allclose(after, before * 0.5, rtol=2e-2)
    assert stats["operator"] == "module_scale"


def test_noop_returns_identical_bits():
    bits = make_tensor(8, 8)
    new, stats = ops.apply_noop(bits, "BF16")
    assert np.array_equal(new, bits)
    assert stats["changed_entries"] == 0


def test_axis_0_selection():
    bits = make_tensor(32, 16)
    d = np.random.default_rng(4).standard_normal(16)
    new, stats = ops.apply_direction_ablation(
        bits, "BF16", axis=0, index_spec={"kind": "range", "start": 0, "stop": 4},
        direction=d, magnitude=0.5, method="project_out",
    )
    assert stats["changed_rows"] == 4
    mask = new != bits
    assert np.array_equal(new[4:, :], bits[4:, :])


def test_top_norm_selection_picks_largest():
    bits = make_tensor(8, 16)
    new, stats = ops.apply_direction_ablation(
        bits, "BF16", axis=1, index_spec={"kind": "top_norm", "n": 3},
        direction=np.ones(8), magnitude=0.1, method="project_out",
    )
    assert stats["selected"] == 3
    assert stats["changed_cols"] <= 3


def test_invalid_axis_raises():
    bits = make_tensor()
    with pytest.raises(ops.EditError):
        ops.apply_direction_ablation(
            bits, "BF16", axis=2, index_spec={"kind": "all"},
            direction=np.ones(32), magnitude=0.5,
        )


def test_dimension_mismatch_raises():
    bits = make_tensor(16, 8)
    with pytest.raises(ops.EditError, match="方向维度"):
        ops.apply_direction_ablation(
            bits, "BF16", axis=1, index_spec={"kind": "all"},
            direction=np.ones(99), magnitude=0.5,
        )


def test_zero_direction_raises():
    bits = make_tensor(8, 8)
    with pytest.raises(ops.EditError, match="零向量"):
        ops.apply_direction_ablation(
            bits, "BF16", axis=1, index_spec={"kind": "all"},
            direction=np.zeros(8), magnitude=0.5,
        )


def test_out_of_range_magnitude_raises():
    bits = make_tensor(8, 8)
    with pytest.raises(ops.EditError):
        ops.apply_direction_ablation(
            bits, "BF16", axis=1, index_spec={"kind": "all"},
            direction=np.ones(8), magnitude=5.0,
        )


def test_budget_cap_rejects_oversized_edit():
    bits = make_tensor(64, 128)
    stats = {"changed_entries": 10_000}
    with pytest.raises(ops.EditError, match="上限"):
        ops.check_budget(stats, model_numel=10_000)


def test_budget_allows_small_edit():
    stats = {"changed_entries": 10}
    ops.check_budget(stats, model_numel=1_000_000)
    assert stats["within_budget"] is True


def test_tensor_budget_blocks_rewriting_whole_tensor():
    """一个张量几乎被改光，即使模型级占比很低也必须拒绝。

    这是真实漏洞：8.7 亿参数的模型里，改光一个 367 万元的张量
    只占 0.35% 模型权重，看起来"合规"，但该张量 84% 被改了。
    """
    stats = {"changed_entries": 3_087_280, "numel": 3_670_016}
    with pytest.raises(ops.EditError, match="单张量"):
        ops.check_budget(
            stats, model_numel=873_000_000, tensor_numel=3_670_016
        )


def test_tensor_budget_reports_both_numbers():
    stats = {"changed_entries": 3_087_280, "numel": 3_670_016}
    try:
        ops.check_budget(stats, model_numel=873_000_000,
                         tensor_numel=3_670_016)
    except ops.EditError:
        pass
    # 失败后统计仍要可读
    assert stats["model_fraction"] < 0.01
    assert stats["tensor_fraction"] > 0.8
    assert stats["within_model_budget"] is True
    assert stats["within_tensor_budget"] is False
    assert stats["within_budget"] is False


def test_tensor_budget_allows_small_fraction():
    stats = {"changed_entries": 1000, "numel": 100_000}
    ops.check_budget(stats, model_numel=1_000_000, tensor_numel=100_000)
    assert stats["within_budget"] is True
    assert stats["tensor_fraction"] == pytest.approx(0.01)


def test_both_budgets_must_pass():
    """模型级通过但张量级不通过 -> 整体不通过。"""
    # 模型级 0.06%（通过），张量级 60%（不通过）
    stats = {"changed_entries": 60_000, "numel": 100_000}
    with pytest.raises(ops.EditError, match="单张量"):
        ops.check_budget(stats, model_numel=100_000_000, tensor_numel=100_000)
    assert stats["within_model_budget"] is True
    assert stats["within_tensor_budget"] is False


def test_both_budgets_fail_reports_both():
    """两个都超时要同时报出来。"""
    stats = {"changed_entries": 90_000, "numel": 100_000}
    with pytest.raises(ops.EditError) as exc:
        ops.check_budget(stats, model_numel=1_000_000, tensor_numel=100_000)
    msg = str(exc.value)
    assert "模型权重" in msg
    assert "单张量" in msg
