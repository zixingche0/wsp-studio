"""模型仓库：定位 safetensors 分片、读取配置、计算张量 hash。

这一层不加载全部权重，只读 header。真正需要张量时才由上层按名读取。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import names as name_utils
from . import st


class ModelError(Exception):
    """模型目录不可用。"""


@dataclass
class ModelInfo:
    """一个本地模型的元信息。"""

    path: str
    shards: list[str] = field(default_factory=list)
    tensor_names: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    weight_map: dict[str, str] = field(default_factory=dict)
    total_bytes: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.path))

    @property
    def num_layers(self) -> int:
        return int(
            self.config.get("num_hidden_layers")
            or self.config.get("text_config", {}).get("num_hidden_layers")
            or len(name_utils.list_layers(self.tensor_names))
        )

    @property
    def num_tensors(self) -> int:
        return len(self.tensor_names)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "num_layers": self.num_layers,
            "num_tensors": self.num_tensors,
            "total_bytes": self.total_bytes,
            "total_gb": round(self.total_bytes / 1024**3, 3),
            "architectures": self.config.get("architectures", []),
            "model_type": self.config.get("model_type"),
            "shards": [os.path.basename(s) for s in self.shards],
        }


def _find_index_files(path: str) -> list[str]:
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".safetensors.index.json")
    )


def _find_shards(path: str) -> list[str]:
    """找到 safetensors 分片。

    Qwen3.5 的文件名是 ``model.safetensors-00001-of-00001.safetensors``，
    不是常见的 ``model-00001-of-00001.safetensors``，所以两种都要认。
    """
    found = []
    for f in sorted(os.listdir(path)):
        if not f.endswith(".safetensors"):
            continue
        full = os.path.join(path, f)
        if not os.path.isfile(full):
            continue
        found.append(full)
    return found


def load_model(path: str) -> ModelInfo:
    """扫描模型目录，建立索引（不加载张量）。"""
    if not os.path.isdir(path):
        raise ModelError(f"模型目录不存在: {path}")

    shards = _find_shards(path)
    if not shards:
        raise ModelError(f"目录里没有 .safetensors 文件: {path}")

    config: dict = {}
    cfg_path = os.path.join(path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                config = json.load(f)
        except json.JSONDecodeError as exc:
            raise ModelError(f"config.json 不是合法 JSON: {cfg_path}") from exc

    # 用 index.json 里的 weight_map 建立 张量名 -> 分片 的映射（如果存在）
    weight_map: dict[str, str] = {}
    for idx in _find_index_files(path):
        try:
            with open(idx, encoding="utf-8") as f:
                data = json.load(f)
            for tensor, shard in (data.get("weight_map") or {}).items():
                candidate = os.path.join(path, shard)
                if os.path.isfile(candidate):
                    weight_map[tensor] = candidate
        except json.JSONDecodeError:
            continue

    # 补上 index 里没有声明的分片
    names: list[str] = list(weight_map)
    for shard in shards:
        try:
            for t in st.tensor_names(shard):
                if t not in weight_map:
                    weight_map[t] = shard
                    names.append(t)
        except st.SafetensorsError as exc:
            raise ModelError(f"读取 {shard} 失败: {exc}") from exc

    if not names:
        raise ModelError(f"模型里没有任何张量: {path}")

    total = sum(info["data_offsets"][1] - info["data_offsets"][0]
                for shard in shards
                for k, info in st.read_header(shard)[0].items() if k != "__metadata__")

    return ModelInfo(
        path=os.path.abspath(path),
        shards=shards,
        tensor_names=sorted(set(names)),
        config=config,
        weight_map=weight_map,
        total_bytes=total,
    )


def shard_of(info: ModelInfo, tensor_name: str) -> str:
    """返回张量所在分片路径。"""
    shard = info.weight_map.get(tensor_name)
    if shard is None:
        raise ModelError(f"张量 {tensor_name} 不在任何分片里")
    return shard


def read_tensor(info: ModelInfo, tensor_name: str) -> tuple[str, np.ndarray]:
    """读取单个张量（dtype_str, ndarray）。"""
    return st.load(shard_of(info, tensor_name), [tensor_name])[tensor_name]


def tensor_sha256(arr: np.ndarray) -> str:
    """张量内容的 sha256（C 序、连续内存）。"""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def file_sha256(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    """文件 sha256（分块读取，适合大文件）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def model_fingerprint(info: ModelInfo) -> dict:
    """模型的轻量指纹：每个分片的大小 + 张量数。

    用于判断"是不是同一个模型"，比整文件 hash 快得多。
    """
    shards = []
    for shard in info.shards:
        item = {
            "file": os.path.basename(shard),
            "bytes": os.path.getsize(shard),
            "tensors": len(st.tensor_names(shard)),
        }
        shards.append(item)
    return {
        "name": info.name,
        "num_tensors": info.num_tensors,
        "num_layers": info.num_layers,
        "shards": shards,
    }


def stat_tensor(info: ModelInfo, tensor_name: str) -> dict:
    """张量统计：形状、dtype、范数。只读 header + 该张量。"""
    dtype_str, arr = read_tensor(info, tensor_name)
    flat = st.to_float(arr, dtype_str)
    return {
        "name": tensor_name,
        "dtype": dtype_str,
        "shape": list(arr.shape),
        "numel": int(arr.size),
        "norm": float(np.linalg.norm(flat)),
        "abs_mean": float(np.mean(np.abs(flat))),
        "sha256": tensor_sha256(arr),
    }
