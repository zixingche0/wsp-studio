"""safetensors 读写与 bf16 位操作测试。"""

from __future__ import annotations

import numpy as np
import pytest

from wsp.weights import st


def test_roundtrip_all_dtypes(tmp_path):
    path = tmp_path / "x.safetensors"
    tensors = {
        "a": ("F32", np.arange(6, dtype=np.float32).reshape(2, 3)),
        "b": ("F16", np.ones(4, dtype=np.float16)),
        "c": ("I32", np.arange(5, dtype=np.int32)),
        "d": ("BF16", np.array([[0x3F80, 0x3F00]], dtype=np.uint16)),
    }
    st.save(path, tensors, {"format": "pt"})
    loaded = st.load(path)
    assert set(loaded) == set(tensors)
    for name, (dtype, arr) in tensors.items():
        got_dtype, got_arr = loaded[name]
        assert got_dtype == dtype
        assert np.array_equal(got_arr, arr), name


def test_read_header_without_data(tmp_path):
    path = tmp_path / "x.safetensors"
    st.save(path, {"t": ("F32", np.zeros((8, 8), dtype=np.float32))})
    header, offset = st.read_header(path)
    assert "t" in header
    assert header["t"]["shape"] == [8, 8]
    assert offset > 0


def test_missing_tensor_raises(tmp_path):
    path = tmp_path / "x.safetensors"
    st.save(path, {"t": ("F32", np.zeros(4, dtype=np.float32))})
    with pytest.raises(st.SafetensorsError):
        st.load(path, ["nope"])


def test_corrupt_header_raises(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"not-a-real-header")
    with pytest.raises(st.SafetensorsError):
        st.read_header(path)


def test_bf16_roundtrip_is_exact_for_representable():
    values = np.array([1.0, -2.0, 0.5, 0.0, 128.0], dtype=np.float32)
    bits = st.bf16_bits(values)
    back = st.bf16_to_f32(bits)
    assert np.array_equal(values, back)


def test_bf16_round_to_nearest_even():
    """1.0 和 2.0 之间 bf16 只有有限精度，检查舍入方向。"""
    x = np.array([1.0 + 0.001], dtype=np.float32)
    bits = st.bf16_bits(x)
    back = st.bf16_to_f32(bits)
    # bf16 有 8 位尾数，1.0 附近的间距是 2^-8
    assert abs(float(back[0]) - 1.0) < 2 ** -8


def test_bf16_zero_and_negative_zero():
    assert st.bf16_bits(np.array([0.0], dtype=np.float32))[0] == 0
    neg = st.bf16_bits(np.array([-0.0], dtype=np.float32))[0]
    assert neg == 0x8000


def test_to_float_and_to_bits_consistency():
    arr = np.array([0x3F80, 0x4000], dtype=np.uint16)  # 1.0, 2.0
    floats = st.to_float(arr, "BF16")
    assert list(floats) == [1.0, 2.0]
    back = st.to_bits(floats, "BF16")
    assert np.array_equal(back, arr)
