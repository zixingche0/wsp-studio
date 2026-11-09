"""平台探测：操作系统、计算设备、常见模型目录。

这一层存在的唯一目的，是让上层代码里**不出现任何机器特定的常量**。

所有"这台机器上有什么"的判断都收敛到这里，且每一项都可以被环境变量覆盖：

    WSP_HOME            数据目录
    WSP_DEVICE          强制指定设备（cuda / cuda:1 / mps / cpu）
    WSP_MODEL_DIRS      额外模型目录，用 os.pathsep 分隔
    WSP_LMSTUDIO_URL    LM Studio 端点
    WSP_OLLAMA_URL      Ollama 端点
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 操作系统
# ---------------------------------------------------------------------------

def os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def home_dir() -> Path:
    """用户主目录（跨平台）。"""
    return Path.home()


def config_dir() -> Path:
    """各平台的惯例配置目录。"""
    override = os.environ.get("WSP_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os_name() == "windows":
        base = os.environ.get("APPDATA") or str(home_dir())
        return Path(base) / "wsp-studio"
    if os_name() == "macos":
        return home_dir() / "Library" / "Application Support" / "wsp-studio"
    base = os.environ.get("XDG_CONFIG_HOME") or str(home_dir() / ".config")
    return Path(base) / "wsp-studio"


def data_dir() -> Path:
    """默认数据目录。"""
    override = os.environ.get("WSP_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os_name() == "windows":
        base = os.environ.get("LOCALAPPDATA") or str(home_dir())
        return Path(base) / "wsp-studio"
    return config_dir()


# ---------------------------------------------------------------------------
# 计算设备
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """一次探测得到的计算设备信息。"""

    kind: str = "cpu"          # cpu | cuda | mps | unknown
    name: str = ""
    total_memory_gb: float = 0.0
    available: bool = True
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "total_memory_gb": round(self.total_memory_gb, 2),
            "available": self.available,
            "reason": self.reason,
        }


def _probe_cuda() -> DeviceInfo | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        index = 0
        count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(index)
        return DeviceInfo(
            kind="cuda",
            name=f"{props.name} (共 {count} 张)",
            total_memory_gb=props.total_memory / 1024**3,
            reason=f"cuda:{index}",
        )
    except Exception as exc:
        return DeviceInfo(kind="cpu", name="CPU", reason=f"CUDA 探测失败: {exc}")


def _probe_mps() -> DeviceInfo | None:
    if sys.platform != "darwin":
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        if torch.backends.mps.is_available():
            return DeviceInfo(kind="mps", name="Apple Silicon GPU", reason="mps")
    except Exception:
        pass
    return None


def _cpu_info() -> DeviceInfo:
    name = "CPU"
    try:
        import platform

        name = platform.processor() or platform.machine() or "CPU"
    except Exception:
        pass
    mem = total_memory_gb()
    return DeviceInfo(kind="cpu", name=name, total_memory_gb=mem)


def total_memory_gb() -> float:
    """物理内存（GB）。跨平台尽力而为。"""
    try:
        if os_name() == "linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024**2
        elif os_name() == "windows":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / 1024**3
        elif os_name() == "macos":
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0:
                return int(out.stdout.strip()) / 1024**3
    except Exception:
        pass
    return 0.0


_device_cache: DeviceInfo | None = None


def detect_device(use_cache: bool = True) -> DeviceInfo:
    """推荐的计算设备。

    优先级：环境变量 > CUDA > MPS > CPU。

    结果会缓存：``import torch`` 本身就要 1~2 秒，
    每次请求都重新探测会让界面明显卡顿。
    """
    global _device_cache
    forced = os.environ.get("WSP_DEVICE")
    if forced:
        return DeviceInfo(kind="forced", name=forced, reason="来自 WSP_DEVICE")

    if use_cache and _device_cache is not None:
        return _device_cache

    info: DeviceInfo | None = None
    for probe in (_probe_cuda, _probe_mps):
        info = probe()
        if info is not None:
            break
    if info is None:
        info = _cpu_info()
    _device_cache = info
    return info


def invalidate_device_cache() -> None:
    global _device_cache
    _device_cache = None


def list_devices() -> list[dict]:
    """列出所有可用设备，给设置页用。"""
    out = [{"value": "cpu", "label": "CPU", "detail": f"{total_memory_gb():.1f} GB 内存"}]
    cuda = _probe_cuda()
    if cuda:
        out.append({
            "value": "cuda",
            "label": cuda.name,
            "detail": f"{cuda.total_memory_gb:.1f} GB 显存",
        })
    mps = _probe_mps()
    if mps:
        out.append({"value": "mps", "label": "Apple Silicon (MPS)", "detail": "统一内存"})
    return out


# ---------------------------------------------------------------------------
# 常见模型目录发现
# ---------------------------------------------------------------------------

def _env_dirs() -> list[Path]:
    raw = os.environ.get("WSP_MODEL_DIRS", "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


_roots_cache: tuple[float, list[tuple[str, Path]]] | None = None
_ROOTS_TTL = 30.0  # 秒


def known_model_roots(use_cache: bool = True) -> list[tuple[str, Path]]:
    """返回 (来源名, 目录) 的列表，只包含实际存在的目录。

    覆盖主流本地推理工具的默认位置，跨三个操作系统。

    结果缓存 30 秒：``Path.is_dir()`` 在断开的网络盘上可能阻塞数秒，
    每次请求都重扫会让界面卡住。
    """
    global _roots_cache
    import time as _time

    if use_cache and _roots_cache is not None:
        ts, cached = _roots_cache
        if _time.time() - ts < _ROOTS_TTL:
            return cached

    result = _compute_model_roots()
    _roots_cache = (_time.time(), result)
    return result


def invalidate_roots_cache() -> None:
    """清缓存（测试或用户新增目录后调用）。"""
    global _roots_cache
    _roots_cache = None


def _compute_model_roots() -> list[tuple[str, Path]]:
    home = home_dir()
    candidates: list[tuple[str, Path]] = []

    # --- LM Studio ---
    lm_home = _lmstudio_home()
    if lm_home:
        candidates.append(("LM Studio", lm_home / "models"))
        settings = _lmstudio_settings()
        if settings:
            folder = settings.get("downloadsFolder")
            if folder:
                candidates.append(("LM Studio (下载目录)", Path(folder)))

    # --- Ollama ---
    if os.environ.get("OLLAMA_MODELS"):
        candidates.append(("Ollama", Path(os.environ["OLLAMA_MODELS"])))
    candidates.append(("Ollama", home / ".ollama" / "models"))
    if os_name() == "macos":
        candidates.append(("Ollama", home / ".ollama" / "models"))
    elif os_name() != "windows":
        candidates.append(("Ollama", Path("/usr/share/ollama/.ollama/models")))
    if os_name() == "windows":
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates.append(("Ollama", Path(local) / "Ollama" / "models"))

    # --- HuggingFace 缓存 ---
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(("HuggingFace", Path(hf_home) / "hub"))
    candidates.append(("HuggingFace", home / ".cache" / "huggingface" / "hub"))
    candidates.append(("HuggingFace", home / ".cache" / "huggingface"))

    # --- llama.cpp ---
    candidates.append(("llama.cpp", home / ".cache" / "llama.cpp"))
    candidates.append(("llama.cpp", home / "llama.cpp" / "models"))

    # --- 通用 ---
    candidates.append(("通用", home / "models"))
    candidates.append(("通用", home / "Models"))
    if os_name() == "windows":
        for drive in "CDEFG":
            candidates.append((f"通用 ({drive}:)", Path(f"{drive}:/models")))
    else:
        candidates.append(("通用", Path("/models")))
        candidates.append(("通用", Path("/data/models")))

    # --- 用户显式指定 ---
    for p in _env_dirs():
        candidates.append(("自定义", p))

    seen: set[str] = set()
    out: list[tuple[str, Path]] = []
    for name, path in candidates:
        try:
            if not path.is_dir():
                continue
            key = str(path.resolve()).lower()
        except (OSError, RuntimeError):
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append((name, path))
    return out


def _lmstudio_home() -> Path | None:
    """LM Studio 的数据目录。

    新版会写一个 ``.lmstudio-home-pointer`` 指向真正的 home，
    所以要优先读它。
    """
    env = os.environ.get("LMSTUDIO_HOME")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    default = home_dir() / ".lmstudio"
    pointer = home_dir() / ".lmstudio-home-pointer"
    if pointer.is_file():
        try:
            target = Path(pointer.read_text(encoding="utf-8").strip())
            if target.is_dir():
                return target
        except OSError:
            pass
    return default if default.is_dir() else None


def _lmstudio_settings() -> dict:
    home = _lmstudio_home()
    if not home:
        return {}
    settings = home / "settings.json"
    if not settings.is_file():
        return {}
    try:
        import json

        with open(settings, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# 本地推理服务的默认端点
# ---------------------------------------------------------------------------

@dataclass
class ServiceEndpoint:
    """一个本地推理服务的端点。"""

    name: str
    base_url: str
    kind: str = "openai"        # openai | ollama
    reachable: bool = False
    models: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "kind": self.kind,
            "reachable": self.reachable,
            "models": self.models,
            "note": self.note,
        }


KNOWN_SERVICES = [
    ("LM Studio", "http://127.0.0.1:1234/v1", "openai"),
    ("Ollama", "http://127.0.0.1:11434/v1", "openai"),
    ("llama.cpp server", "http://127.0.0.1:8080/v1", "openai"),
    ("vLLM", "http://127.0.0.1:8000/v1", "openai"),
    ("text-generation-webui", "http://127.0.0.1:5000/v1", "openai"),
    ("KoboldCpp", "http://127.0.0.1:5001/v1", "openai"),
    ("LocalAI", "http://127.0.0.1:8081/v1", "openai"),
    ("Jan", "http://127.0.0.1:1337/v1", "openai"),
]


def candidate_endpoints() -> list[tuple[str, str]]:
    """所有候选端点（内置 + 环境变量覆盖 + 用户追加）。"""
    overrides = {
        "LM Studio": os.environ.get("WSP_LMSTUDIO_URL"),
        "Ollama": os.environ.get("WSP_OLLAMA_URL"),
    }
    out: list[tuple[str, str]] = []
    for name, url, _kind in KNOWN_SERVICES:
        out.append((name, (overrides.get(name) or url).rstrip("/")))
    for extra in os.environ.get("WSP_INFERENCE_URLS", "").split(","):
        extra = extra.strip()
        if extra:
            out.append(("自定义", extra.rstrip("/")))
    return out


def scan_services(timeout: float = 0.8, parallel: bool = True) -> list[ServiceEndpoint]:
    """扫描常见本地推理服务端口，看哪些在跑。

    不假设你装了哪一个——所有端点平等对待，只是显示名字不同。

    默认**并行**扫描。串行扫 8 个端点、每个超时 0.8 秒，最坏要 6.4 秒，
    会把 HTTP 请求卡住。
    """
    import concurrent.futures

    import httpx

    targets = candidate_endpoints()
    results: dict[str, ServiceEndpoint] = {
        base: ServiceEndpoint(name=name, base_url=base) for name, base in targets
    }

    def probe(base: str, name: str) -> tuple[str, bool, list[str]]:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(
                    base + "/models", headers={"Authorization": "Bearer x"}
                )
                if resp.status_code >= 500:
                    return base, False, []
                try:
                    data = resp.json()
                    models = [
                        m.get("id") for m in (data.get("data") or []) if m.get("id")
                    ]
                except ValueError:
                    models = []
                return base, True, models
        except Exception:
            return base, False, []

    if parallel and len(targets) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, len(targets))
        ) as pool:
            for base, ok, models in pool.map(lambda t: probe(t[1], t[0]), targets):
                ep = results[base]
                ep.reachable = ok
                ep.models = models
    else:
        for name, base in targets:
            _b, ok, models = probe(base, name)
            results[base].reachable = ok
            results[base].models = models

    return list(results.values())


def scan_services_fast(timeout: float = 0.35) -> list[ServiceEndpoint]:
    """快速扫描：更短超时，用于健康检查这种不该阻塞的场景。"""
    return scan_services(timeout=timeout)


# ---------------------------------------------------------------------------
# 工具可用性
# ---------------------------------------------------------------------------

def tool_availability() -> dict:
    """检查外部工具是否可用（用于提示用户如何补齐）。"""
    return {
        "git": bool(shutil.which("git")),
        "git_lfs": bool(shutil.which("git-lfs")),
        "huggingface-cli": bool(shutil.which("huggingface-cli"))
        or bool(shutil.which("hf")),
        "ollama": bool(shutil.which("ollama")),
    }


def platform_report(scan_services: bool = True) -> dict:
    """一份完整的平台报告，给 /api/health 和设置页用。

    ``scan_services=False`` 时跳过端口探测（约 1 秒），
    用于不该阻塞的路径。
    """
    report = {
        "os": os_name(),
        "python": sys.version.split()[0],
        "home": str(home_dir()),
        "config_dir": str(config_dir()),
        "memory_gb": round(total_memory_gb(), 1),
        "device": detect_device().as_dict(),
        "devices": list_devices(),
        "model_roots": [{"source": s, "path": str(p)} for s, p in known_model_roots()],
        "tools": tool_availability(),
    }
    if scan_services:
        report["services"] = [e.as_dict() for e in scan_services_fn()]
    return report


# 允许测试替换
scan_services_fn = scan_services
