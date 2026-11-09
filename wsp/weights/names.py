"""张量名解析：把用户/LLM 说的 "layer 15, module mlp.down_proj"
映射到模型里真实的张量名。

不同模型家族命名差异很大：

  Qwen3.5   model.language_model.layers.15.mlp.down_proj.weight
  Llama     model.layers.15.mlp.down_proj.weight
  GPT-2     transformer.h.15.mlp.c_proj.weight
  Qwen2     model.layers.15.mlp.down_proj.weight
  Gemma     model.layers.15.mlp.down_proj.weight

所以按 精确 -> 后缀 -> 归一化别名 三级匹配，并且永远返回候选列表而不是
抛异常，让上层（LLM 或用户）能自我修正。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 常见模块别名：用户说法 -> 可能的真实片段
MODULE_ALIASES: dict[str, tuple[str, ...]] = {
    "mlp.down_proj": ("mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "mlp.fc2"),
    "mlp.up_proj": ("mlp.up_proj", "mlp.dense_h_to_4h", "mlp.fc1", "mlp.c_fc"),
    "mlp.gate_proj": ("mlp.gate_proj", "mlp.w1"),
    "attn.o_proj": ("self_attn.o_proj", "attn.c_proj", "attention.wo", "self_attention.dense"),
    "attn.q_proj": ("self_attn.q_proj", "attn.c_attn", "attention.wq", "self_attention.query"),
    "attn.k_proj": ("self_attn.k_proj", "attention.wk", "self_attention.key"),
    "attn.v_proj": ("self_attn.v_proj", "attention.wv", "self_attention.value"),
    "embed": ("embed_tokens", "wte", "word_embeddings"),
    "lm_head": ("lm_head", "output", "embed_out"),
}

# 归一化时去掉的架构前缀（不携带语义）
_PREFIX_STRIP = (
    "model.language_model.",
    "language_model.model.",
    "language_model.",
    "model.",
    "transformer.",
    "gpt_neox.",
    "base_model.model.",
)

_LAYER_RE = re.compile(r"(?:layers?|h|blocks?|block)\.(\d+)\.")


@dataclass
class ResolveResult:
    """一次解析的结果。"""

    ok: bool
    tensor_name: str | None = None
    candidates: list[str] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tensor_name": self.tensor_name,
            "candidates": self.candidates,
            "reason": self.reason,
        }


def _strip_prefix(name: str) -> str:
    for prefix in _PREFIX_STRIP:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _module_variants(module: str) -> list[str]:
    """把 "mlp.down_proj" 展开成所有可能的写法。"""
    module = module.strip().strip(".")
    variants = [module]
    if module in MODULE_ALIASES:
        variants.extend(MODULE_ALIASES[module])
    # 也允许用户只写 "down_proj"
    if "." in module:
        tail = module.rsplit(".", 1)[-1]
        if tail not in variants:
            variants.append(tail)
    # 去重保序
    seen, ordered = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def infer_layer(name: str) -> int | None:
    """从张量名里推断层号。"""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def infer_module(name: str) -> str:
    """从张量名里推断模块名（去掉前缀、层号和 .weight 后缀）。"""
    base = _strip_prefix(name)
    base = re.sub(r"\.weight$", "", base)
    base = _LAYER_RE.sub("", base)
    return base


def resolve(
    tensor_names: list[str],
    layer: int | None = None,
    module: str | None = None,
    name: str | None = None,
) -> ResolveResult:
    """把 (layer, module) 或完整 name 解析成真实张量名。

    优先级：
      1. 给了完整 name -> 精确 / 后缀匹配
      2. 给了 layer + module -> 归一化后按 ``layers.{L}.{module}.weight`` 匹配
      3. 匹配不到 -> 返回同类候选（同层或同模块）供上层修正
    """
    if not tensor_names:
        return ResolveResult(False, reason="模型没有任何张量")

    # --- 1. 完整名 ---
    if name:
        if name in tensor_names:
            return ResolveResult(True, tensor_name=name)
        suffix = "." + name.lstrip(".")
        hits = [t for t in tensor_names if t.endswith(suffix)]
        if len(hits) == 1:
            return ResolveResult(True, tensor_name=hits[0])
        if len(hits) > 1:
            return ResolveResult(
                False,
                candidates=hits[:20],
                reason=f"名称 {name!r} 匹配到 {len(hits)} 个张量，请指定完整路径",
            )
        return ResolveResult(False, reason=f"找不到张量 {name!r}")

    # --- 2. layer + module ---
    if module is None:
        return ResolveResult(False, reason="必须提供 module 或完整张量名")

    variants = _module_variants(module)

    for variant in variants:
        if layer is None:
            # 没有层号：模块可能出现在所有层，全部返回
            hits = [
                t
                for t in tensor_names
                if _strip_prefix(t) == f"{variant}.weight"
                or _strip_prefix(t).endswith(f".{variant}.weight")
            ]
            if len(hits) == 1:
                return ResolveResult(True, tensor_name=hits[0])
            if len(hits) > 1:
                return ResolveResult(
                    False,
                    candidates=sorted(hits)[:40],
                    reason=f"模块 {variant!r} 出现在 {len(hits)} 个层，请指定 layer",
                )
            continue

        # 有层号：先找精确层
        pattern = re.compile(rf"(?:^|\.)layers?\.{layer}\.{re.escape(variant)}\.weight$")
        hits = [t for t in tensor_names if pattern.search(t)]
        if not hits:
            # 也允许 "h.{layer}" 这类写法
            pattern_h = re.compile(rf"(?:^|\.)h\.{layer}\.{re.escape(variant)}\.weight$")
            hits = [t for t in tensor_names if pattern_h.search(t)]
        if not hits:
            # 后缀宽松匹配
            tail = f".{layer}.{variant}.weight"
            hits = [t for t in tensor_names if t.endswith(tail)]
        if len(hits) == 1:
            return ResolveResult(True, tensor_name=hits[0])
        if len(hits) > 1:
            return ResolveResult(
                False,
                candidates=hits[:20],
                reason=f"layer={layer} module={variant!r} 有歧义",
            )

    # --- 3. 全部失败，给候选 ---
    return ResolveResult(
        False,
        candidates=_suggest(tensor_names, layer, module),
        reason=f"找不到 layer={layer} module={module!r}；请从 candidates 里选",
    )


def _suggest(tensor_names: list[str], layer: int | None, module: str | None) -> list[str]:
    """给出最接近的候选：优先同层，其次同模块。"""
    same_layer: list[str] = []
    if layer is not None:
        same_layer = sorted(t for t in tensor_names if infer_layer(t) == layer)
    if same_layer:
        return same_layer[:40]
    if module:
        tail = module.rsplit(".", 1)[-1]
        hits = sorted(t for t in tensor_names if tail in t)
        if hits:
            return hits[:40]
    return sorted(tensor_names)[:40]


def list_layers(tensor_names: list[str]) -> list[int]:
    """列出模型里出现过的层号。"""
    layers = {infer_layer(t) for t in tensor_names}
    return sorted(x for x in layers if x is not None)


def list_modules(tensor_names: list[str], layer: int | None = None) -> list[str]:
    """列出模块名（可按层过滤）。"""
    mods = set()
    for t in tensor_names:
        if layer is not None and infer_layer(t) != layer:
            continue
        mods.add(infer_module(t))
    return sorted(mods)
