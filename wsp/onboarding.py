"""首次运行引导：检测环境、给出针对当前平台的建议。

新用户上手最大的障碍不是代码，是"我该下哪个模型、放哪儿、怎么让推理服务跑起来"。
所以这里把每一步都变成可执行的建议，而不是文档里的一句话。
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field

from . import platform as plat


@dataclass
class Step:
    """一个引导步骤。"""

    key: str
    title: str
    status: str               # done | todo | optional | blocked
    detail: str = ""
    commands: list[str] = field(default_factory=list)
    link: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "commands": self.commands,
            "link": self.link,
        }


# 推荐模型：显存/内存友好，且有官方编辑数据可对比
RECOMMENDED_MODELS = [
    {
        "id": "Qwen/Qwen3.5-0.8B",
        "size_gb": 1.63,
        "why": "最小、纯语言；有官方编辑数据可逐 bit 对比",
        "min_ram_gb": 4,
    },
    {
        "id": "Qwen/Qwen3-0.6B",
        "size_gb": 1.5,
        "why": "更早的架构，社区实测最多（wsp-kit 有真实运行记录）",
        "min_ram_gb": 4,
    },
    {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "size_gb": 0.7,
        "why": "极小，CPU 上也能秒回，适合先跑通流程",
        "min_ram_gb": 2,
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "size_gb": 2.5,
        "why": "架构标准，命名规整（验证解析器用）",
        "min_ram_gb": 6,
    },
]


def recommend_models() -> list[dict]:
    """按当前机器的内存筛出能跑的模型。"""
    ram = plat.total_memory_gb()
    out = []
    for m in RECOMMENDED_MODELS:
        feasible = ram == 0 or ram >= m["min_ram_gb"]
        out.append({**m, "feasible": feasible})
    return out


def _hf_endpoint_hint() -> str:
    """国内网络下必须提示镜像。"""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        return f"已设置 HF_ENDPOINT={endpoint}"
    return "提示：如果你在中国大陆，先设置 HF_ENDPOINT=https://hf-mirror.com"


def download_command(model_id: str, local_dir: str | None = None) -> list[str]:
    """生成跨平台的下载命令。"""
    target = local_dir or f"./models/{model_id.replace('/', '--')}"
    if shutil.which("hf"):
        return [f"hf download {model_id} --local-dir {target}"]
    if shutil.which("huggingface-cli"):
        return [f"huggingface-cli download {model_id} --local-dir {target}"]
    # 都没有就给出用 hf_hub 的 python 单行
    return [
        f"pip install -U huggingface_hub",
        f'python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{model_id}', local_dir=r'{target}')\"",
    ]


def build_guide(
    *,
    has_model: bool = False,
    has_llm_config: bool = False,
    running_services: list[dict] | None = None,
) -> list[Step]:
    """生成引导步骤。"""
    steps: list[Step] = []
    os_name = plat.os_name()
    device = plat.detect_device()

    # 1. 推理运行时
    has_torch = _has_module("torch")
    has_transformers = _has_module("transformers")
    if has_torch and has_transformers:
        steps.append(Step(
            key="runtime",
            title="本地推理运行时",
            status="done",
            detail=f"torch + transformers 已安装；设备：{device.name}",
        ))
    else:
        missing = []
        if not has_torch:
            missing.append("torch")
        if not has_transformers:
            missing.append("transformers")
        steps.append(Step(
            key="runtime",
            title="本地推理运行时",
            status="todo",
            detail=f"缺少 {', '.join(missing)}。"
                   "只有评估编辑后的模型才需要它；纯编辑只用 numpy。",
            commands=[f'pip install -e ".[model]"'],
        ))

    # 2. 推理服务（可选：任何 OpenAI 兼容服务都能加速探测）
    up = [s for s in (running_services or []) if s.get("reachable")]
    if up:
        names = "、".join(s["name"] for s in up)
        steps.append(Step(
            key="service",
            title="本地推理服务",
            status="done",
            detail=f"运行中：{names}（探测会更快）",
        ))
    else:
        steps.append(Step(
            key="service",
            title="本地推理服务（可选）",
            status="optional",
            detail="没有检测到运行中的本地推理服务。不装也行——"
                   "会直接用本地 torch 推理，只是慢一些。"
                   "任何 OpenAI 兼容服务都可以（LM Studio / Ollama / llama.cpp / vLLM / LocalAI…），"
                   "也可以用 WSP_INFERENCE_URLS 指定。",
            commands=[
                "Ollama: ollama serve          # 默认 11434",
                "llama.cpp: llama-server -m <模型> --port 8080",
                "或任意 OpenAI 兼容端点，在网页「设置」里填 Base URL",
            ],
        ))

    # 3. 可编辑模型
    if has_model:
        steps.append(Step(
            key="model",
            title="可编辑的模型",
            status="done",
            detail="已经注册了至少一个 safetensors 模型",
        ))
    else:
        best = next((m for m in recommend_models() if m["feasible"]), None)
        steps.append(Step(
            key="model",
            title="可编辑的模型",
            status="todo",
            detail=(
                (f"推荐 {best['id']}（{best['size_gb']} GB）：{best['why']}"
                 if best else "没有找到适合本机内存的推荐模型")
                + "。注意：GGUF 只能探测，不能编辑——必须是 safetensors 目录。"
            ),
            commands=download_command(best["id"]) if best else [],
            link=_hf_endpoint_hint(),
        ))

    # 4. 大模型 API
    if has_llm_config:
        steps.append(Step(
            key="llm",
            title="大模型 API",
            status="done",
            detail="已配置",
        ))
    else:
        steps.append(Step(
            key="llm",
            title="大模型 API",
            status="todo",
            detail="在网页右上角「设置」里填 Base URL 和 API Key。"
                   "任何 OpenAI 兼容端点都行——云端 API、或本机任意推理服务。",
            link="https://platform.openai.com/api-keys",
        ))

    # 5. 数据目录
    from .config import get_paths

    paths = get_paths()
    writable = os.access(paths.home, os.W_OK)
    steps.append(Step(
        key="storage",
        title="数据目录",
        status="done" if writable else "blocked",
        detail=f"{paths.home}"
               + ("" if writable else "（不可写！请设置 WSP_HOME）"),
        commands=[] if writable else [f"set WSP_HOME=<可写目录>  # {os_name}"],
    ))

    return steps


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def guide_summary(steps: list[Step]) -> dict:
    """引导的整体状态。

    区分两个层次：
      * ``can_edit``    能编辑权重（只需要模型 + 数据目录 + LLM 配置）
      * ``can_evaluate`` 能评估编辑效果（额外需要本地推理运行时）
    """
    by_key = {s.key: s for s in steps}
    todo = [s for s in steps if s.status == "todo"]
    blocked = [s for s in steps if s.status == "blocked"]

    edit_blockers = [
        k for k in ("model", "storage", "llm")
        if by_key.get(k) and by_key[k].status in ("todo", "blocked")
    ]
    eval_blockers = edit_blockers + (
        ["runtime"] if by_key.get("runtime") and by_key["runtime"].status != "done" else []
    )

    return {
        "ready": not todo and not blocked,
        "can_edit": not edit_blockers,
        "can_evaluate": not eval_blockers,
        "edit_blockers": edit_blockers,
        "eval_blockers": eval_blockers,
        "todo_count": len(todo),
        "blocked_count": len(blocked),
        "next_step": (blocked or todo or [None])[0].as_dict()
        if (blocked or todo)
        else None,
    }
