"""LLM 客户端测试：JSON 提取/修复、错误处理、工具解析。

用 httpx 的 MockTransport，不联网。
"""

from __future__ import annotations

import json

import httpx
import pytest

from wsp.llm import client as llm_client


def _response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _chat_payload(content: str = "", tool_calls=None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _client_with(handler) -> llm_client.LLMClient:
    cfg = llm_client.LLMConfig(
        base_url="http://test.local/v1", api_key="sk-x", model="m"
    )
    client = llm_client.LLMClient(cfg)
    transport = httpx.MockTransport(handler)

    # 让 _post 用 mock transport
    original = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    httpx.Client = patched  # type: ignore
    client._restore = lambda: setattr(httpx, "Client", original)  # type: ignore
    return client


def test_extract_json_plain():
    assert llm_client.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = '```json\n{"a": 1}\n```'
    assert llm_client.extract_json(text) == {"a": 1}


def test_extract_json_with_prose():
    text = 'Here is the plan:\n{"a": 1}\nHope that helps!'
    assert llm_client.extract_json(text) == {"a": 1}


def test_extract_json_trailing_comma():
    assert llm_client.extract_json('{"a": 1,}') == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        llm_client.extract_json("no json here at all")


def test_extract_json_empty_raises():
    with pytest.raises(ValueError):
        llm_client.extract_json("")


def test_chat_parses_tool_calls():
    def handler(request):
        return _response(
            _chat_payload(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "inspect_model",
                            "arguments": '{"model_id": "tiny"}',
                        },
                    }
                ]
            )
        )

    client = _client_with(handler)
    try:
        resp = client.chat([{"role": "user", "content": "hi"}])
        assert resp.wants_tools
        assert resp.tool_calls[0].name == "inspect_model"
        assert resp.tool_calls[0].arguments == {"model_id": "tiny"}
    finally:
        client._restore()


def test_chat_raises_on_401():
    def handler(request):
        return _response({"error": "bad key"}, status=401)

    client = _client_with(handler)
    try:
        with pytest.raises(llm_client.LLMError, match="API Key"):
            client.chat([{"role": "user", "content": "hi"}], retries=0)
    finally:
        client._restore()


def test_chat_raises_on_404():
    def handler(request):
        return _response({"error": "no route"}, status=404)

    client = _client_with(handler)
    try:
        with pytest.raises(llm_client.LLMError, match="base_url"):
            client.chat([{"role": "user", "content": "hi"}], retries=0)
    finally:
        client._restore()


def test_chat_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _response({"error": "boom"}, status=500)
        return _response(_chat_payload("ok"))

    client = _client_with(handler)
    try:
        resp = client.chat([{"role": "user", "content": "hi"}], retries=1)
        assert resp.content == "ok"
        assert resp.attempts == 2
    finally:
        client._restore()


def test_chat_json_repairs_once():
    """第一次返回非 JSON，第二次返回合法 JSON。"""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        body = json.loads(request.content)
        if calls["n"] == 1:
            return _response(_chat_payload("这不是 JSON"))
        # 第二次应当带有修复指令
        last = body["messages"][-1]["content"]
        assert "合法 JSON" in last
        return _response(_chat_payload('{"goal": "x", "edits": [], "eval_gate": {}}'))

    client = _client_with(handler)
    try:
        data, resp = client.chat_json("sys", "user")
        assert data["goal"] == "x"
        assert resp.repaired is True
    finally:
        client._restore()


def test_test_connection_success():
    def handler(request):
        return _response(_chat_payload("ok"))

    client = _client_with(handler)
    try:
        result = client.test_connection()
        assert result["ok"] is True
        assert result["reply"] == "ok"
    finally:
        client._restore()


def test_test_connection_failure():
    def handler(request):
        return _response({"error": "denied"}, status=401)

    client = _client_with(handler)
    try:
        result = client.test_connection()
        assert result["ok"] is False
        assert "error" in result
    finally:
        client._restore()


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("WSP_LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("WSP_LLM_API_KEY", "k")
    monkeypatch.setenv("WSP_LLM_MODEL", "m1")
    cfg = llm_client.LLMConfig.from_env()
    assert cfg.base_url == "http://x/v1"
    assert cfg.model == "m1"


def test_masked_config_hides_key():
    cfg = llm_client.LLMConfig(
        base_url="http://x", api_key="sk-secret", model="m"
    )
    masked = cfg.masked()
    assert "sk-secret" not in str(masked)
    assert masked["api_key_set"] is True
