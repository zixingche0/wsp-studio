"""推理后端抽象。

探测与评估都需要"跑一遍模型"。这里定义统一接口，允许两种实现：

* ``OpenAICompatBackend``  通过 OpenAI 兼容 HTTP 接口（LM Studio / Ollama /
  vLLM / 任意云端），**不需要 torch**，只要求模型已被服务化。
* ``HFLocalBackend``       本地 transformers 加载 safetensors（需要 torch），
  用于编辑后立刻评估——这是唯一能读到"刚改完的权重"的方式。

默认优先用 HTTP 后端（快、省内存）；本地后端在安装了 torch 时才可用。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx


class BackendError(Exception):
    """后端不可用或调用失败。"""


@dataclass
class Generation:
    """一次生成的结果。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def hit_token_limit(self) -> bool:
        return self.finish_reason in ("length", "max_tokens")


class Backend:
    """推理后端接口。"""

    name: str = "base"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system: str | None = None,
    ) -> Generation:
        raise NotImplementedError

    def available(self) -> bool:
        return False

    def describe(self) -> dict:
        return {"backend": self.name, "available": self.available()}


class OpenAICompatBackend(Backend):
    """OpenAI 兼容的 /chat/completions 后端。"""

    name = "openai-compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("WSP_LLM_API_KEY") or "not-needed"
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/models", headers=self._headers())
                return resp.status_code < 500
        except httpx.HTTPError:
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system: str | None = None,
    ) -> Generation:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise BackendError(f"请求 {self.base_url} 失败: {exc}") from exc

        if resp.status_code >= 400:
            raise BackendError(
                f"后端返回 {resp.status_code}: {resp.text[:300]}"
            )

        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise BackendError("后端返回的不是 JSON") from exc

        choices = data.get("choices") or []
        if not choices:
            raise BackendError(f"后端没有返回 choices: {str(data)[:200]}")

        choice = choices[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return Generation(
            text=message.get("content") or "",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "",
            raw=data,
        )

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "available": self.available(),
        }


class HFLocalBackend(Backend):
    """本地 transformers 后端（需要 torch）。

    用于评估**刚刚被编辑过**的 checkpoint：这是 HTTP 后端做不到的，
    因为 LM Studio 加载的是原始 GGUF。

    设备自动选择：CUDA > MPS > CPU，可用 ``device`` 参数或
    ``WSP_DEVICE`` 环境变量强制指定。
    """

    name = "hf-local"

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        dtype: str = "auto",
        load_in_4bit: bool = False,
    ):
        self.model_path = model_path
        self.device = device or ""
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._import_error: str = ""
        self._resolved_device: str = ""

    def _resolve_device(self, torch) -> str:
        """决定用哪个设备。"""
        if self._resolved_device:
            return self._resolved_device
        if self.device:
            self._resolved_device = self.device
            return self.device
        import os

        env = os.environ.get("WSP_DEVICE")
        if env:
            self._resolved_device = env
            return env
        try:
            if torch.cuda.is_available():
                self._resolved_device = "cuda"
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._resolved_device = "mps"
                return "mps"
        except Exception:
            pass
        self._resolved_device = "cpu"
        return "cpu"

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self._import_error = (
                "本地推理需要 torch + transformers：pip install 'wsp-studio[model]'"
            )
            raise BackendError(self._import_error) from exc

        device = self._resolve_device(torch)

        # 设备与 dtype 的匹配：CPU 上 float16 会极慢甚至不支持
        if self.dtype == "auto":
            if device.startswith("cuda"):
                torch_dtype = torch.bfloat16
            elif device == "mps":
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            torch_dtype = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(self.dtype, torch.float32)
            if device == "cpu" and torch_dtype == torch.float16:
                torch_dtype = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=False
        )

        # transformers 5.x 用 dtype，4.x 用 torch_dtype
        try:
            import transformers as _tf
            _major = int(str(_tf.__version__).split(".")[0])
        except Exception:
            _major = 4
        dtype_key = "dtype" if _major >= 5 else "torch_dtype"
        kwargs: dict = {
            dtype_key: torch_dtype,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit and device.startswith("cuda"):
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                kwargs["device_map"] = "auto"
            except ImportError:
                pass  # 没装 bitsandbytes 就退化为正常加载

        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if "device_map" not in kwargs:
            self._model = self._model.to(device)
        self._model.eval()

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            return os.path.isdir(self.model_path)
        except ImportError:
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system: str | None = None,
    ) -> Generation:
        self._lazy_load()
        import torch

        assert self._tokenizer is not None and self._model is not None

        if system:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = (system + "\n\n" if system else "") + prompt

        inputs = self._tokenizer(text, return_tensors="pt").to(
            self._resolved_device or self.device or "cpu"
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id
            or self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][prompt_tokens:]
        text_out = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        depth = new_tokens.shape[-1]
        return Generation(
            text=text_out,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(depth),
            finish_reason="length" if depth >= max_tokens else "stop",
        )

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_path": self.model_path,
            "device": self._resolved_device or self.device or "auto",
            "dtype": self.dtype,
            "load_in_4bit": self.load_in_4bit,
            "available": self.available(),
            "import_error": self._import_error,
        }
