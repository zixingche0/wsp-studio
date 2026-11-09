"""纯 numpy 的 safetensors 读写。

bfloat16 以原始 uint16 位模式保存，全程不做浮点运算，
因此可以做到逐 bit 精确复现（这是权重编辑可校验的前提）。

参考了 safetensors 官方格式与 lm_understand-logs 的 numpy 实现思路，
但代码为独立编写。
"""

from __future__ import annotations

import json
import os
import struct
from typing import Iterable, Mapping

import numpy as np

# safetensors dtype 字符串 -> numpy dtype
# BF16 没有对应的 numpy dtype，用 uint16 承载原始位模式
DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

# numpy dtype -> safetensors dtype 字符串（用于写回）
NUMERIC_TO_ST = {
    np.dtype(np.float64): "F64",
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.uint16): "BF16",
    np.dtype(np.int64): "I64",
    np.dtype(np.int32): "I32",
    np.dtype(np.int16): "I16",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}

MAX_HEADER_BYTES = 100 * 1024 * 1024


class SafetensorsError(Exception):
    """safetensors 文件格式错误。"""


def read_header(path: str | os.PathLike) -> tuple[dict, int]:
    """读取 safetensors 头，返回 (header_dict, data_start_offset)。"""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SafetensorsError(f"{path}: 文件太短，不是合法的 safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > MAX_HEADER_BYTES:
            raise SafetensorsError(f"{path}: header 长度异常 ({n})")
        blob = f.read(n)
        if len(blob) != n:
            raise SafetensorsError(f"{path}: header 被截断")
        try:
            header = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise SafetensorsError(f"{path}: header 不是合法 JSON") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path}: header 必须是对象")
    return header, 8 + n


def tensor_names(path: str | os.PathLike) -> list[str]:
    """只读 header，列出张量名（不加载数据）。"""
    header, _ = read_header(path)
    return [k for k in header if k != "__metadata__"]


def metadata(path: str | os.PathLike) -> dict:
    header, _ = read_header(path)
    return header.get("__metadata__", {}) or {}


def load(
    path: str | os.PathLike,
    names: Iterable[str] | None = None,
) -> dict[str, tuple[str, np.ndarray]]:
    """读取张量。

    返回 ``{name: (dtype_str, ndarray)}``；BF16 张量的 ndarray 是 uint16 位模式。

    只读 header 再 seek，所以可以只取需要的张量而不载入整个文件。
    """
    want = set(names) if names is not None else None
    header, data_start = read_header(path)
    out: dict[str, tuple[str, np.ndarray]] = {}
    with open(path, "rb") as f:
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if want is not None and name not in want:
                continue
            dtype_str = info.get("dtype")
            shape = info.get("shape")
            offsets = info.get("data_offsets")
            if dtype_str not in DTYPES:
                raise SafetensorsError(f"{path}:{name}: 不支持的 dtype {dtype_str!r}")
            if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                raise SafetensorsError(f"{path}:{name}: header 字段缺失")
            begin, end = int(offsets[0]), int(offsets[1])
            np_dtype = DTYPES[dtype_str]
            expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np_dtype).itemsize
            if end - begin != expected:
                raise SafetensorsError(
                    f"{path}:{name}: 数据长度与 shape 不符 ({end - begin} != {expected})"
                )
            f.seek(data_start + begin)
            buf = f.read(end - begin)
            if len(buf) != end - begin:
                raise SafetensorsError(f"{path}:{name}: 数据被截断")
            arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
            out[name] = (dtype_str, arr)
    if want is not None:
        missing = want - set(out)
        if missing:
            raise SafetensorsError(f"{path}: 缺少张量 {sorted(missing)[:5]}")
    return out


def save(
    path: str | os.PathLike,
    tensors: Mapping[str, tuple[str, np.ndarray]],
    metadata: Mapping[str, str] | None = None,
) -> str:
    """写出 safetensors。``tensors`` 形如 ``{name: (dtype_str, ndarray)}``。

    返回写入的路径。调用方负责保证不会覆盖原始模型。
    """
    head: dict[str, dict] = {}
    offset = 0
    blobs: list[bytes] = []
    for name in sorted(tensors):
        dtype_str, arr = tensors[name]
        if dtype_str not in DTYPES:
            raise SafetensorsError(f"{name}: 不支持的 dtype {dtype_str!r}")
        arr = np.ascontiguousarray(arr)
        if arr.dtype != np.dtype(DTYPES[dtype_str]):
            raise SafetensorsError(
                f"{name}: 数组 dtype {arr.dtype} 与声明的 {dtype_str} 不一致"
            )
        blob = arr.tobytes()
        head[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    if metadata:
        head["__metadata__"] = dict(metadata)
    encoded = json.dumps(head, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for blob in blobs:
            f.write(blob)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# bfloat16 位操作
# ---------------------------------------------------------------------------

def bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 位模式，round-to-nearest-even。

    与 torch 的 ``.to(torch.bfloat16)`` 对有限值一致。
    """
    arr = np.ascontiguousarray(x, dtype=np.float32)
    u = arr.view(np.uint32).astype(np.uint64)
    # RNE: 加上 0x7FFF + 最低保留位
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """bfloat16 位模式 -> float32。"""
    return (np.ascontiguousarray(bits).astype(np.uint32) << 16).view(np.float32)


def to_float(arr: np.ndarray, dtype_str: str) -> np.ndarray:
    """把任意支持的张量转换成 float32 便于统计。"""
    if dtype_str == "BF16":
        return bf16_to_f32(arr)
    return np.asarray(arr, dtype=np.float32)


def to_bits(values: np.ndarray, dtype_str: str) -> np.ndarray:
    """float 值 -> 目标 dtype 的存储表示。"""
    if dtype_str == "BF16":
        return bf16_bits(values)
    return np.asarray(values, dtype=DTYPES[dtype_str])
