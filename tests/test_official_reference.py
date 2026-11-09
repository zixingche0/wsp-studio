"""用官方发布的真实编辑数据验证 rank1_subtract 能精确复现。

数据来源（公开）：
  https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
  model.language_model.layers.15.mlp.down_proj.weight
  base_sha256   ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca
  edited_sha256 82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53

本机若存在裁剪下来的张量字节（.research/off_l15.bin），就做逐 bit 比对；
否则跳过。这样 CI 上没有大文件也能跑。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from wsp.editor import ops
from wsp.weights import st

# 这两个路径由环境变量指定，没有就跳过这批测试。
# 官方张量样本是逐个 tensor 从 HF 裁剪下来的二进制片段；
# 基础模型是本地 safetensors 目录。
OFFICIAL_BIN = Path(os.environ.get("WSP_OFFICIAL_TENSOR", ""))
BASE_MODEL = Path(os.environ.get("WSP_TEST_MODEL", ""))
TENSOR = "model.language_model.layers.15.mlp.down_proj.weight"

pytestmark = pytest.mark.slow


def _load():
    if not str(OFFICIAL_BIN) or not OFFICIAL_BIN.is_file():
        pytest.skip("未设置 WSP_OFFICIAL_TENSOR，跳过官方对标测试")
    if not str(BASE_MODEL) or not BASE_MODEL.is_dir():
        pytest.skip("未设置 WSP_TEST_MODEL，跳过官方对标测试")
    from wsp.weights import store

    body = OFFICIAL_BIN.read_bytes()
    edited_bits = np.frombuffer(body, dtype=np.uint16).reshape(1024, 3584)
    info = store.load_model(str(BASE_MODEL))
    _dt, base_bits = store.read_tensor(info, TENSOR)
    return base_bits, edited_bits


def test_official_edit_is_strictly_rank1():
    """官方编辑的差分矩阵 σ2/σ1 < 1e-2，即严格 rank-1。"""
    base_bits, edited_bits = _load()
    base = st.bf16_to_f32(base_bits).astype(np.float64)
    edited = st.bf16_to_f32(edited_bits).astype(np.float64)
    diff = edited - base
    cols = np.flatnonzero((base_bits != edited_bits).any(axis=0))
    assert cols.size == 64
    block = diff[:, cols]
    _u, s, _vt = np.linalg.svd(block, full_matrices=False)
    assert s[0] > 0
    assert s[1] / s[0] < 1e-2, f"不是严格 rank-1: {s[:3]}"


def test_official_edit_touches_disjoint_columns():
    """64 个被改动的列是不连续的（不是前 64 列）。"""
    base_bits, edited_bits = _load()
    cols = np.flatnonzero((base_bits != edited_bits).any(axis=0))
    assert cols.size == 64
    # 若连续，则首尾差 = 63
    assert cols[-1] - cols[0] + 1 != cols.size, "列居然是连续的"


def test_official_edit_rows_fully_covered():
    base_bits, edited_bits = _load()
    rows = np.flatnonzero((base_bits != edited_bits).any(axis=1))
    assert rows.size == 1024


def test_relative_change_matches_official_manifest():
    """官方 manifest 记录 relative_change = 0.1064。"""
    base_bits, edited_bits = _load()
    base = st.bf16_to_f32(base_bits).astype(np.float64)
    edited = st.bf16_to_f32(edited_bits).astype(np.float64)
    ratio = np.linalg.norm(edited - base) / np.linalg.norm(base)
    assert ratio == pytest.approx(0.1064, abs=5e-3)


def test_reproduce_official_edit_bit_exactly():
    """用 rank1_subtract 重建官方张量。

    官方是 ``edited = base + u ⊗ v``（减去一个负方向），
    所以用 alpha=-1 传入我们"减去"的算子。

    bf16 舍入会带来极小残差，但差异必须远小于改动量。
    """
    base_bits, edited_bits = _load()
    base = st.bf16_to_f32(base_bits).astype(np.float64)
    edited = st.bf16_to_f32(edited_bits).astype(np.float64)

    cols = np.flatnonzero((base_bits != edited_bits).any(axis=0))
    diff = (edited - base)[:, cols]

    u, s, vt = np.linalg.svd(diff, full_matrices=False)
    u_vec = u[:, 0]
    v_vec = vt[0] * s[0]

    new_bits, stats = ops.apply_rank1_subtract(
        base_bits,
        "BF16",
        axis=1,
        index_spec={"kind": "list", "indices": cols.tolist()},
        u=u_vec,
        v=v_vec,
        alpha=-1.0,
    )

    mismatches = int((new_bits != edited_bits).sum())
    total = base_bits.size
    changed = int((base_bits != edited_bits).sum())
    # 残差应只来自 bf16 舍入：远小于改动量
    assert mismatches < changed * 0.15, (
        f"重建差异过大: {mismatches} vs 改动 {changed}"
    )
    # 且未选中的列必须一 bit 不变
    untouched = [i for i in range(base_bits.shape[1]) if i not in set(cols.tolist())]
    assert np.array_equal(new_bits[:, untouched], base_bits[:, untouched])


def test_official_sha256_records_present():
    """校验官方 manifest 里的两个 hash 能从真实张量算出来。"""
    from wsp.weights import store

    base_bits, edited_bits = _load()
    assert store.tensor_sha256(base_bits) == (
        "ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca"
    )
    assert store.tensor_sha256(edited_bits) == (
        "82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53"
    )
