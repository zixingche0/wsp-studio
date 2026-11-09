"""跨平台与通用性测试。

这些测试保证：换一台机器、换一个操作系统、换一种本地推理工具，
代码仍然能工作——不依赖任何本机特定路径。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from wsp import platform as plat
from wsp.models import discovery


# ---------------------------------------------------------------------------
# 平台探测
# ---------------------------------------------------------------------------

def test_os_name_is_known():
    assert plat.os_name() in ("windows", "macos", "linux")


def test_platform_report_shape():
    report = plat.platform_report()
    for key in ("os", "python", "device", "devices", "model_roots", "tools"):
        assert key in report
    assert report["device"]["kind"]


def test_devices_always_has_cpu():
    devices = plat.list_devices()
    assert any(d["value"] == "cpu" for d in devices)


def test_device_env_override(monkeypatch):
    monkeypatch.setenv("WSP_DEVICE", "cuda:2")
    info = plat.detect_device()
    assert info.kind == "forced"
    assert info.name == "cuda:2"


def test_total_memory_positive():
    assert plat.total_memory_gb() >= 0


def test_tool_availability_is_dict():
    tools = plat.tool_availability()
    assert isinstance(tools, dict)
    assert "git" in tools


# ---------------------------------------------------------------------------
# 模型目录发现：跨平台
# ---------------------------------------------------------------------------

def test_known_model_roots_all_exist():
    for source, path in plat.known_model_roots():
        assert path.is_dir(), f"{source} 返回了不存在的目录 {path}"


def test_known_model_roots_no_duplicates():
    roots = plat.known_model_roots()
    keys = [str(p.resolve()).lower() for _s, p in roots]
    assert len(keys) == len(set(keys))


def test_env_model_dirs_is_respected(monkeypatch, tmp_path):
    extra = tmp_path / "my-models"
    extra.mkdir()
    monkeypatch.setenv("WSP_MODEL_DIRS", str(extra))
    plat.invalidate_roots_cache()
    try:
        roots = plat.known_model_roots()
        assert any(p == extra for _s, p in roots)
    finally:
        plat.invalidate_roots_cache()


def test_roots_cache_can_be_invalidated(monkeypatch, tmp_path):
    """缓存必须能被清掉，否则改了环境变量看不到变化。"""
    first = tmp_path / "a"
    first.mkdir()
    monkeypatch.setenv("WSP_MODEL_DIRS", str(first))
    plat.invalidate_roots_cache()
    assert any(p == first for _s, p in plat.known_model_roots())

    second = tmp_path / "b"
    second.mkdir()
    monkeypatch.setenv("WSP_MODEL_DIRS", str(second))
    plat.invalidate_roots_cache()
    roots = plat.known_model_roots()
    assert any(p == second for _s, p in roots)
    plat.invalidate_roots_cache()


def test_lmstudio_home_pointer(monkeypatch, tmp_path):
    """LM Studio 新版用 .lmstudio-home-pointer 指向真正的 home。"""
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()
    real_lm = tmp_path / "real-lmstudio"
    real_lm.mkdir()
    monkeypatch.setattr(plat, "home_dir", lambda: fake_home)
    (fake_home / ".lmstudio-home-pointer").write_text(str(real_lm), encoding="utf-8")
    assert plat._lmstudio_home() == real_lm  # noqa: SLF001


def test_lmstudio_home_missing(monkeypatch, tmp_path):
    fake_home = tmp_path / "empty-home"
    fake_home.mkdir()
    monkeypatch.setattr(plat, "home_dir", lambda: fake_home)
    assert plat._lmstudio_home() is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# 模型发现：各来源格式
# ---------------------------------------------------------------------------

def _make_hf_model(root: Path, name: str = "m1", size: int = 1024) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * size)
    return d


def test_scan_hf_dir_recognized(tmp_path):
    d = _make_hf_model(tmp_path)
    model = discovery.scan_hf_dir(d)
    assert model is not None
    assert model.kind == "safetensors"
    assert model.editable is True


def test_scan_gguf_dir_not_editable(tmp_path):
    d = tmp_path / "gguf-model"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.gguf").write_bytes(b"\0" * 512)
    model = discovery.scan_hf_dir(d)
    assert model is not None
    assert model.kind == "gguf"
    assert model.editable is False


def test_scan_dir_without_weights_is_none(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert discovery.scan_hf_dir(d) is None


def test_walk_for_models_finds_nested(tmp_path):
    _make_hf_model(tmp_path / "a" / "b", "nested")
    found = discovery.walk_for_models(tmp_path, depth=4)
    assert len(found) == 1
    assert found[0].editable is True


def test_walk_stops_at_model_dir(tmp_path):
    """找到模型目录后不再深入，避免把分片当成独立模型。"""
    d = _make_hf_model(tmp_path / "outer", "model")
    (d / "sub").mkdir()
    (d / "sub" / "config.json").write_text("{}", encoding="utf-8")
    (d / "sub" / "model.safetensors").write_bytes(b"\0" * 10)
    found = discovery.walk_for_models(tmp_path, depth=5)
    assert len(found) == 1


def test_scan_flat_skips_partial_and_mmproj(tmp_path):
    (tmp_path / "downloading_x.gguf.part").write_bytes(b"\0")
    (tmp_path / "mmproj-model.gguf").write_bytes(b"\0")
    (tmp_path / "good.gguf").write_bytes(b"\0" * 100)
    found = discovery.scan_flat_models(tmp_path)
    names = [m.name for m in found]
    assert "good" in names
    assert not any("downloading" in n for n in names)
    assert not any(n.startswith("mmproj") for n in names)


def test_make_id_stable_and_safe():
    a = discovery.make_id("LM Studio", "Qwen/Qwen3.5-0.8B")
    b = discovery.make_id("LM Studio", "Qwen/Qwen3.5-0.8B")
    assert a == b
    assert "/" not in a and "\\" not in a and " " not in a


def test_ollama_discovery_on_fake_layout(tmp_path, monkeypatch):
    """构造一个 Ollama 目录结构，验证能识别。"""
    from wsp.models import sources

    models = tmp_path / "ollama"
    manifests = models / "manifests" / "registry.ollama.ai" / "library" / "llama3"
    manifests.mkdir(parents=True)
    (manifests / "latest").write_text(
        '{"layers": [{"digest": "sha256:abc"}]}', encoding="utf-8"
    )
    blobs = models / "blobs"
    blobs.mkdir()
    (blobs / "sha256-abc").write_bytes(b"\0" * 2048)
    monkeypatch.setenv("OLLAMA_MODELS", str(models))
    src = sources.OllamaSource()
    assert src.available()
    found = src.discover()
    assert any(m.source == "ollama" for m in found)


def test_hf_cache_discovery_on_fake_layout(tmp_path, monkeypatch):
    from wsp.models import sources

    hub = tmp_path / "hub"
    snap = hub / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}", encoding="utf-8")
    (snap / "model.safetensors").write_bytes(b"\0" * 4096)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    src = sources.HuggingFaceCacheSource()
    assert src.available()
    found = src.discover()
    assert any("Qwen3.5-0.8B" in m.name for m in found)


# ---------------------------------------------------------------------------
# 服务端点扫描
# ---------------------------------------------------------------------------

def test_known_services_list_nonempty():
    assert len(plat.KNOWN_SERVICES) >= 4
    names = [n for n, _u, _k in plat.KNOWN_SERVICES]
    assert "LM Studio" in names
    assert "Ollama" in names


def test_scan_services_returns_all_known():
    """即使都没在跑，也要返回完整列表（带 reachable=False）。"""
    found = plat.scan_services(timeout=0.2)
    assert len(found) == len(plat.KNOWN_SERVICES)
    assert all(hasattr(e, "reachable") for e in found)


def test_service_url_override(monkeypatch):
    monkeypatch.setenv("WSP_OLLAMA_URL", "http://127.0.0.1:9999/v1")
    found = plat.scan_services(timeout=0.2)
    ollama = next(e for e in found if e.name == "Ollama")
    assert ollama.base_url == "http://127.0.0.1:9999/v1"


# ---------------------------------------------------------------------------
# 数据目录回退链
# ---------------------------------------------------------------------------

def test_home_env_wins(monkeypatch, tmp_path):
    from wsp import config

    target = tmp_path / "custom-home"
    monkeypatch.setenv("WSP_HOME", str(target))
    assert config._default_home() == target.resolve()  # noqa: SLF001


def test_home_falls_back_without_env(monkeypatch):
    from wsp import config

    monkeypatch.delenv("WSP_HOME", raising=False)
    monkeypatch.delenv("WSP_DATA_DIR", raising=False)
    home = config._default_home()  # noqa: SLF001
    assert home.is_absolute()


def test_config_dir_per_platform():
    d = plat.config_dir()
    assert d.is_absolute()
    if sys.platform == "win32":
        assert "wsp-studio" in str(d).lower()


# ---------------------------------------------------------------------------
# 密钥方案：跨平台
# ---------------------------------------------------------------------------

def test_scheme_is_one_of_known():
    from wsp.store import secrets

    assert secrets.scheme_name() in ("dpapi", "keychain", "secretservice", "localkey")


def test_scheme_description_exists():
    from wsp.store import secrets

    desc = secrets.scheme_description()
    assert len(desc) > 5


def test_roundtrip_works_on_any_platform(tmp_home):
    from wsp.store import secrets

    stored = secrets.encrypt_secret("sk-portability-test", tmp_home)
    assert secrets.decrypt_secret(stored, tmp_home) == "sk-portability-test"
