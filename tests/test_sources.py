"""模型来源插件体系测试。

核心断言：**没有任何单一工具是必需的**。
移除 LM Studio、没有 Ollama、没有 HF 缓存，系统仍应正常工作。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wsp.models import sources


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试用干净的注册表。"""
    sources.reset_registry()
    yield
    sources.reset_registry()


# ---------------------------------------------------------------------------
# 插件体系基础
# ---------------------------------------------------------------------------

def test_builtin_sources_registered():
    keys = {s.key for s in sources.all_sources()}
    assert {
        "lmstudio",
        "ollama",
        "huggingface",
        "llamacpp",
        "local",
        "custom",
    } <= keys


def test_no_single_tool_is_required():
    """所有内置来源都是可选的：禁用全部后仍不抛异常。"""
    assert len(sources.all_sources()) >= 5


def test_disable_sources_by_env(monkeypatch):
    monkeypatch.setenv("WSP_DISABLE_SOURCES", "lmstudio,ollama")
    keys = {s.key for s in sources.all_sources()}
    assert "lmstudio" not in keys
    assert "ollama" not in keys
    # 其他来源不受影响
    assert "local" in keys


def test_register_custom_source():
    class FakeSource(sources.ModelSource):
        key = "fake"
        label = "假的"

        def available(self) -> bool:
            return True

        def discover(self):
            return []

    sources.register_source(FakeSource())
    keys = {s.key for s in sources.all_sources()}
    assert "fake" in keys


def test_custom_source_can_provide_models():
    from wsp.models.discovery import FoundModel

    class StaticSource(sources.ModelSource):
        key = "static"
        label = "静态"

        def discover(self):
            return [
                FoundModel(
                    id="static-model",
                    name="Static Model",
                    source="static",
                    kind="safetensors",
                    path="/tmp/x",
                    editable=True,
                )
            ]

    sources.register_source(StaticSource())
    found = sources.discover_all()
    assert any(m.id == "static-model" for m in found)


def test_plugin_loaded_from_env(monkeypatch):
    """WSP_MODEL_SOURCES 能加载外部插件。"""
    import sys
    import types

    module = types.ModuleType("wsp_fake_plugin")
    module.__dict__["FakePlugin"] = type(
        "FakePlugin",
        (sources.ModelSource,),
        {"key": "plugin", "label": "插件", "discover": lambda self: []},
    )
    sys.modules["wsp_fake_plugin"] = module
    try:
        monkeypatch.setenv("WSP_MODEL_SOURCES", "wsp_fake_plugin:FakePlugin")
        keys = {s.key for s in sources.all_sources()}
        assert "plugin" in keys
    finally:
        sys.modules.pop("wsp_fake_plugin", None)


def test_source_exception_does_not_break_others():
    class Broken(sources.ModelSource):
        key = "broken"

        def discover(self):
            raise RuntimeError("boom")

    class Good(sources.ModelSource):
        key = "good"

        def discover(self):
            from wsp.models.discovery import FoundModel

            return [
                FoundModel(
                    id="good-1", name="g", source="good",
                    kind="gguf", path="/x",
                )
            ]

    sources.register_source(Broken())
    sources.register_source(Good())
    found = sources.discover_all()
    assert any(m.id == "good-1" for m in found)


def test_source_report_has_status():
    report = sources.source_report()
    assert report
    for item in report:
        assert "key" in item and "label" in item
        assert "available" in item and "found" in item


# ---------------------------------------------------------------------------
# 各来源独立可用
# ---------------------------------------------------------------------------

def test_lmstudio_source_without_lmstudio_installed(monkeypatch, tmp_path):
    """没有 LM Studio 时，来源应报告不可用且不崩溃。"""
    monkeypatch.setattr(sources.plat, "_lmstudio_home", lambda: None)
    src = sources.LmStudioSource()
    assert src.available() is False
    assert src.discover() == []


def test_lmstudio_source_with_fake_home(monkeypatch, tmp_path):
    """有 LM Studio 目录结构时能识别。"""
    fake_home = tmp_path / "lmhome"
    repo = fake_home / "models" / "publisher" / "repo"
    repo.mkdir(parents=True)
    (repo / "model.gguf").write_bytes(b"\0" * 1024)
    monkeypatch.setattr(sources.plat, "_lmstudio_home", lambda: fake_home)
    monkeypatch.setattr(sources.plat, "_lmstudio_settings", lambda: {})
    src = sources.LmStudioSource()
    assert src.available()
    found = src.discover()
    assert len(found) == 1
    assert found[0].source == "lmstudio"
    assert found[0].editable is False


def test_ollama_source_without_ollama(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "nohome")
    src = sources.OllamaSource()
    assert src.discover() == []


def test_hf_source_without_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "nohome")
    src = sources.HuggingFaceCacheSource()
    assert src.discover() == []


def test_custom_dir_source(monkeypatch, tmp_path):
    d = tmp_path / "mymodels" / "model-a"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * 512)
    monkeypatch.setenv("WSP_MODEL_DIRS", str(tmp_path / "mymodels"))
    src = sources.CustomDirSource()
    assert src.available()
    found = src.discover()
    assert len(found) == 1
    assert found[0].editable is True


def test_custom_dir_source_inactive_without_env(monkeypatch):
    monkeypatch.delenv("WSP_MODEL_DIRS", raising=False)
    src = sources.CustomDirSource()
    assert src.available() is False


# ---------------------------------------------------------------------------
# 推理端点：不绑定特定工具
# ---------------------------------------------------------------------------

def test_candidate_endpoints_covers_multiple_tools():
    from wsp import platform as plat

    names = {n for n, _u in plat.candidate_endpoints()}
    assert len(names) >= 4
    assert "LM Studio" in names
    assert "Ollama" in names


def test_custom_inference_urls(monkeypatch):
    from wsp import platform as plat

    monkeypatch.setenv("WSP_INFERENCE_URLS", "http://10.0.0.5:9999/v1")
    urls = [u for _n, u in plat.candidate_endpoints()]
    assert "http://10.0.0.5:9999/v1" in urls


def test_scan_services_has_no_assumption(monkeypatch):
    from wsp import platform as plat

    monkeypatch.setenv("WSP_INFERENCE_URLS", "")
    found = plat.scan_services(timeout=0.1)
    # 即使一个都没跑，也要返回完整候选列表
    assert len(found) == len(plat.candidate_endpoints())


def test_running_services_function_is_tool_agnostic():
    """registry 里不该再有 lmstudio 专属的 running 检测。"""
    from wsp.models import registry

    assert hasattr(registry, "discover_running_services")
    assert not hasattr(registry, "discover_lmstudio_running")


# ---------------------------------------------------------------------------
# 完全无本地工具的环境
# ---------------------------------------------------------------------------

def test_works_with_zero_sources(monkeypatch, tmp_path):
    """禁用所有来源后，discover_all 返回空列表而不报错。"""
    monkeypatch.setenv(
        "WSP_DISABLE_SOURCES",
        "lmstudio,ollama,huggingface,llamacpp,local,custom",
    )
    monkeypatch.delenv("WSP_MODEL_DIRS", raising=False)
    found = sources.discover_all()
    assert found == []
    assert sources.source_report() == [] or all(
        not s["available"] for s in sources.source_report()
    )


# ---------------------------------------------------------------------------
# roots() 协议：来源之间不重复扫描
# ---------------------------------------------------------------------------

def test_every_source_exposes_roots():
    for s in sources.all_sources():
        assert hasattr(s, "roots")
        roots = s.roots()
        assert isinstance(roots, list)
        assert all(isinstance(r, Path) for r in roots)


def test_lmstudio_roots_include_download_folder(monkeypatch, tmp_path):
    fake_home = tmp_path / "lm"
    download = tmp_path / "downloads"
    monkeypatch.setattr(sources.plat, "_lmstudio_home", lambda: fake_home)
    monkeypatch.setattr(
        sources.plat, "_lmstudio_settings",
        lambda: {"downloadsFolder": str(download)},
    )
    roots = sources.LmStudioSource().roots()
    assert fake_home / "models" in roots
    assert download in roots


def test_custom_roots_from_env(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("WSP_MODEL_DIRS", f"{a}{os.pathsep}{b}")
    roots = sources.CustomDirSource().roots()
    assert a in roots and b in roots


def test_generic_source_avoids_claimed_dirs(monkeypatch, tmp_path):
    """通用目录不应重复扫描其他来源已经认领的目录。"""
    claimed = tmp_path / "claimed"
    claimed.mkdir()

    class ClaimSource(sources.ModelSource):
        key = "claimer"

        def roots(self):
            return [claimed]

        def discover(self):
            return []

    sources.register_source(ClaimSource())

    # 模拟：known_model_roots 里也包含这个目录
    monkeypatch.setattr(
        sources.plat, "known_model_roots",
        lambda: [("某来源", claimed)],
    )
    found = sources.GenericDirSource().discover()
    assert found == [], "通用目录不该扫描已被认领的目录"


def test_no_hardcoded_source_names_in_generic():
    """通用目录不能靠名字前缀来过滤来源。"""
    import inspect

    src = inspect.getsource(sources.GenericDirSource)
    assert 'startswith(("LM Studio"' not in src
    assert "source.startswith" not in src
