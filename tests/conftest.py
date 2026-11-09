"""测试夹具：一个很小的 safetensors 模型，形状模仿 Qwen3.5。

用它做全链路测试，不需要 torch、不需要联网、不需要真实模型。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wsp.weights import st  # noqa: E402

TINY_LAYERS = 3
TINY_HIDDEN = 32
TINY_INTER = 64


def build_tiny_model(path: Path) -> Path:
    """造一个 tiny 模型目录。"""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    tensors: dict[str, tuple[str, np.ndarray]] = {}

    def bf(arr: np.ndarray) -> tuple[str, np.ndarray]:
        return ("BF16", st.bf16_bits(arr.astype(np.float32)))

    tensors["model.embed_tokens.weight"] = bf(
        rng.standard_normal((128, TINY_HIDDEN)) * 0.02
    )
    for layer in range(TINY_LAYERS):
        prefix = f"model.layers.{layer}"
        tensors[f"{prefix}.mlp.down_proj.weight"] = bf(
            rng.standard_normal((TINY_HIDDEN, TINY_INTER)) * 0.02
        )
        tensors[f"{prefix}.mlp.gate_proj.weight"] = bf(
            rng.standard_normal((TINY_INTER, TINY_HIDDEN)) * 0.02
        )
        tensors[f"{prefix}.mlp.up_proj.weight"] = bf(
            rng.standard_normal((TINY_INTER, TINY_HIDDEN)) * 0.02
        )
        tensors[f"{prefix}.self_attn.o_proj.weight"] = bf(
            rng.standard_normal((TINY_HIDDEN, TINY_HIDDEN)) * 0.02
        )
        tensors[f"{prefix}.input_layernorm.weight"] = bf(
            np.ones(TINY_HIDDEN, dtype=np.float32)
        )
    tensors["model.norm.weight"] = bf(np.ones(TINY_HIDDEN, dtype=np.float32))
    tensors["lm_head.weight"] = bf(
        rng.standard_normal((128, TINY_HIDDEN)) * 0.02
    )

    st.save(path / "model.safetensors", tensors)
    with open(path / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_hidden_layers": TINY_LAYERS,
                "hidden_size": TINY_HIDDEN,
                "architectures": ["TinyForCausalLM"],
                "model_type": "tiny",
            },
            f,
        )
    return path


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """隔离的 WSP_HOME。"""
    home = tmp_path / "wsp-home"
    home.mkdir()
    monkeypatch.setenv("WSP_HOME", str(home))
    monkeypatch.delenv("WSP_LLM_API_KEY", raising=False)
    return home


@pytest.fixture
def tiny_model_path(tmp_path) -> Path:
    return build_tiny_model(tmp_path / "tiny-model")


@pytest.fixture
def service(tmp_home):
    """一个全新的 Service（每次测试用独立数据库）。"""
    from wsp.service import Service

    return Service()


@pytest.fixture
def project(service, tiny_model_path):
    """导入 tiny 模型并建一个项目。"""
    entry = service.import_model(str(tiny_model_path), model_id="tiny")
    return service.create_project("测试项目", entry["id"], "修复重复循环")


@pytest.fixture
def sample_plan() -> dict:
    """一个合法的最小计划。"""
    return {
        "goal": "减少重复循环",
        "reasoning": "测试用",
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
                "expected_effect": "减少重复",
                "risk": "可能影响通用生成",
            }
        ],
        "eval_gate": {
            "primary": "math_short",
            "guards": ["ifeval_mini"],
            "min_delta": 0.05,
            "max_regression": 0.03,
            "delta_unit": "absolute",
            "regression_unit": "absolute",
            "tasks": ["math_short", "ifeval_mini"],
        },
    }
