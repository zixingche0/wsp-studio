"""张量名解析测试（跨模型家族）。"""

from __future__ import annotations

from wsp.weights import names

QWEN35 = [
    "model.language_model.layers.15.mlp.down_proj.weight",
    "model.language_model.layers.15.mlp.gate_proj.weight",
    "model.language_model.layers.19.mlp.down_proj.weight",
    "model.language_model.embed_tokens.weight",
    "model.visual.merger.linear_fc1.weight",
]

LLAMA = [
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.embed_tokens.weight",
]

GPT2 = [
    "transformer.h.0.mlp.c_proj.weight",
    "transformer.h.1.mlp.c_proj.weight",
    "transformer.wte.weight",
]


def test_resolve_exact_qwen35():
    res = names.resolve(QWEN35, layer=15, module="mlp.down_proj")
    assert res.ok
    assert res.tensor_name == "model.language_model.layers.15.mlp.down_proj.weight"


def test_resolve_exact_llama():
    res = names.resolve(LLAMA, layer=1, module="mlp.down_proj")
    assert res.ok
    assert res.tensor_name == "model.layers.1.mlp.down_proj.weight"


def test_resolve_with_mlp_alias_on_gpt2():
    """用户说 mlp.down_proj，GPT-2 里叫 mlp.c_proj。"""
    res = names.resolve(GPT2, layer=0, module="mlp.down_proj")
    assert res.ok
    assert res.tensor_name == "transformer.h.0.mlp.c_proj.weight"


def test_resolve_by_full_name():
    res = names.resolve(QWEN35, name="model.language_model.layers.19.mlp.down_proj.weight")
    assert res.ok


def test_resolve_by_suffix():
    res = names.resolve(QWEN35, name="layers.19.mlp.down_proj.weight")
    assert res.ok
    assert "layers.19" in res.tensor_name


def test_missing_layer_returns_candidates():
    res = names.resolve(QWEN35, layer=99, module="mlp.down_proj")
    assert not res.ok
    assert res.candidates
    assert res.reason


def test_ambiguous_module_without_layer():
    res = names.resolve(LLAMA, module="mlp.down_proj")
    assert not res.ok
    assert "LLAMA" or len(res.candidates) == 2


def test_infer_layer_and_module():
    assert names.infer_layer(QWEN35[0]) == 15
    assert names.infer_layer("model.embed_tokens.weight") is None
    assert names.infer_module(QWEN35[0]) == "mlp.down_proj"
    assert names.infer_module("transformer.h.0.mlp.c_proj.weight") == "mlp.c_proj"


def test_list_layers_and_modules():
    assert names.list_layers(QWEN35) == [15, 19]
    mods = names.list_modules(QWEN35, layer=15)
    assert "mlp.down_proj" in mods
    assert "mlp.gate_proj" in mods


def test_empty_tensor_list():
    res = names.resolve([], layer=0, module="mlp.down_proj")
    assert not res.ok
