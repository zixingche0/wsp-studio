"""API Key 加密存储测试。"""

from __future__ import annotations

import pytest

from wsp.store import secrets


def test_encrypt_decrypt_roundtrip(tmp_home):
    original = "sk-proj-abcdef1234567890"
    stored = secrets.encrypt_secret(original, tmp_home)
    assert original not in stored, "密文里不能出现明文"
    assert secrets.decrypt_secret(stored, tmp_home) == original


def test_mask_never_leaks_full_key():
    masked = secrets.mask_secret("sk-verysecretvalue-1234")
    assert "verysecretvalue" not in masked
    assert masked.endswith("1234")


def test_mask_empty():
    assert secrets.mask_secret("") == ""


def test_short_key_fully_masked():
    masked = secrets.mask_secret("abc")
    assert set(masked) == {"*"}


def test_tampered_ciphertext_rejected(tmp_home):
    stored = secrets.encrypt_secret("sk-test", tmp_home)
    # 破坏最后几个字符
    tampered = stored[:-6] + "AAAAAA"
    if tampered == stored:
        pytest.skip("没有变化")
    with pytest.raises(secrets.SecretsError):
        secrets.decrypt_secret(tampered, tmp_home)


def test_empty_string_stays_empty(tmp_home):
    assert secrets.encrypt_secret("", tmp_home) == ""
    assert secrets.decrypt_secret("", tmp_home) == ""


def test_scheme_reported():
    assert secrets.scheme_name() in ("dpapi", "localkey")


def test_service_saves_encrypted_key(service):
    """通过 Service 保存后，数据库里必须是密文；取出必须是明文。"""
    service.save_llm_config(
        base_url="https://example.com/v1", api_key="sk-secret-9876", model="my-model"
    )
    raw = service.state.db.get_setting("llm.api_key")
    assert raw
    assert "sk-secret-9876" not in raw
    cfg = service.get_llm_config()
    assert cfg.api_key == "sk-secret-9876"
    assert cfg.base_url == "https://example.com/v1"
    assert cfg.model == "my-model"


def test_public_config_masks_key(service):
    service.save_llm_config(api_key="sk-secret-9876")
    public = service.llm_config_public()
    assert public["api_key_set"] is True
    assert "sk-secret-9876" not in str(public)


def test_saving_without_key_preserves_existing(service):
    service.save_llm_config(api_key="sk-first")
    service.save_llm_config(model="another-model")
    cfg = service.get_llm_config()
    assert cfg.api_key == "sk-first"
    assert cfg.model == "another-model"
