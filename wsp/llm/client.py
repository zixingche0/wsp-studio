"""LLM 客户端：OpenAI 兼容 + 重试 + JSON 修复。

配置来源优先级（从高到低）：

1. 页面/数据库里保存的配置（用户显式输入）
2. 环境变量 ``WSP_LLM_BASE_URL`` / ``WSP_LLM_API_KEY`` / ``WSP_LLM_MODEL``

API Key 永远不明文落库，见 ``store.secrets``。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx


class LLMError(Exception):
    """LLM 调用失败。"""


@dataclass
class LLMConfig:
    """一次调用的完整配置。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 180.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.environ.get("WSP_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("WSP_LLM_API_KEY", ""),
            model=os.environ.get("WSP_LLM_MODEL", "gpt-4o-mini"),
        )

    def masked(self) -> dict:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key_set": bool(self.api_key),
        }


@dataclass
class ToolCall:
    """模型请求调用一个工具。"""

    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""


@dataclass
class LLMResponse:
    """一次 LLM 调用结果。"""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    attempts: int = 1
    repaired: bool = False

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """OpenAI 兼容客户端。"""

    def __init__(self, config: LLMConfig):
        self.config = config

    # ---------------- 基础调用 ----------------

    def _post(self, payload: dict) -> dict:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"无法连接 {url}: {exc}") from exc
        if resp.status_code == 401:
            raise LLMError("API Key 无效或已过期（401）")
        if resp.status_code == 404:
            raise LLMError(f"接口不存在（404）：{self.config.base_url}，请检查 base_url")
        if resp.status_code >= 400:
            raise LLMError(f"LLM 返回 {resp.status_code}: {resp.text[:500]}")
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise LLMError("LLM 返回的不是 JSON") from exc

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> LLMResponse:
        """发起对话，自动重试。"""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                data = self._post(payload)
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                calls = []
                for i, tc in enumerate(msg.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    raw = fn.get("arguments") or "{}"
                    try:
                        args = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        args = {}
                    calls.append(
                        ToolCall(
                            id=tc.get("id") or f"call_{i}",
                            name=fn.get("name") or "",
                            arguments=args,
                            raw_arguments=raw if isinstance(raw, str) else "",
                        )
                    )
                return LLMResponse(
                    content=msg.get("content") or "",
                    tool_calls=calls,
                    finish_reason=(data.get("choices") or [{}])[0].get("finish_reason", ""),
                    usage=data.get("usage") or {},
                    attempts=attempt + 1,
                )
            except LLMError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))

        raise LLMError(str(last_error))

    # ---------------- JSON 生成 + 修复 ----------------

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        schema_hint: str = "",
        retries: int = 2,
    ) -> tuple[dict, LLMResponse]:
        """要求模型只返回 JSON，并自动修复常见问题。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self.chat(messages, temperature=0.0, retries=retries)
        text = resp.content
        try:
            return extract_json(text), resp
        except ValueError:
            pass

        # 一次修复尝试
        repair_messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    "你上一条回复不是合法 JSON。请只返回修正后的 JSON，"
                    "不要任何解释、不要 markdown 代码块。"
                    + (f"\n必须满足的结构：{schema_hint}" if schema_hint else "")
                ),
            },
        ]
        resp2 = self.chat(repair_messages, temperature=0.0, retries=retries)
        data = extract_json(resp2.content)
        resp2.repaired = True
        return data, resp2

    # ---------------- 连通性测试 ----------------

    def test_connection(self) -> dict:
        """测试配置是否可用，返回诊断信息。"""
        started = time.time()
        try:
            resp = self.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=8,
                temperature=0.0,
                retries=0,
            )
            return {
                "ok": True,
                "latency_s": round(time.time() - started, 2),
                "reply": resp.content[:100],
                "model": self.config.model,
            }
        except LLMError as exc:
            return {
                "ok": False,
                "latency_s": round(time.time() - started, 2),
                "error": str(exc),
                "model": self.config.model,
            }


# ---------------------------------------------------------------------------
# JSON 提取与修复
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """从模型回复里抽出 JSON 对象。

    处理：纯 JSON、代码块包裹、前后有解释文字、尾随逗号、单引号。
    """
    if not text or not text.strip():
        raise ValueError("回复为空")

    candidates: list[str] = []
    stripped = text.strip()
    candidates.append(stripped)

    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())

    # 第一个 { 到最后一个 }
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first:last + 1])

    for cand in candidates:
        for attempt in (cand, _fix_json(cand)):
            try:
                data = json.loads(attempt)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

    raise ValueError(f"无法从回复中解析出 JSON: {text[:200]!r}")


def _fix_json(text: str) -> str:
    """修复常见 JSON 问题：尾随逗号、单引号、注释。"""
    # 去掉 // 和 /* */ 注释（不处理字符串内部，够用）
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # 尾随逗号
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # 单引号键名/字符串 -> 双引号（简单场景）
    if '"' not in text:
        text = text.replace("'", '"')
    # 未加引号的键
    text = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', text)
    return text
