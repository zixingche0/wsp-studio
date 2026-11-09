# 跨平台与通用性说明

这个项目**不是为某一台机器写的**。这份文档列出所有与机器相关的点，
以及它们各自如何被隔离或覆盖。

## 1. 没有任何硬编码路径

源码里不存在硬编码的盘符、用户主目录或桌面路径。
所有路径都从下面的优先级推导：

```
WSP_HOME  >  WSP_DATA_DIR  >  源码旁的 ./wsp-data  >  平台惯例目录
```

平台惯例目录：

| 系统 | 默认位置 |
|---|---|
| Windows | `%LOCALAPPDATA%\wsp-studio` |
| macOS | `~/Library/Application Support/wsp-studio` |
| Linux | `$XDG_CONFIG_HOME/wsp-studio` 或 `~/.config/wsp-studio` |

`wsp platform` 可以看到当前实际用的路径。

## 2. 模型发现是**插件体系**，不绑定任何工具

`wsp discover` / `GET /api/models/discover` 会跑所有启用的来源插件：

| 来源 | key | 识别方式 |
|---|---|---|
| **LM Studio** | `lmstudio` | 读 `.lmstudio-home-pointer` 和 `settings.json` 的 `downloadsFolder` |
| **Ollama** | `ollama` | 解析 `manifests/` + `blobs/`；respect `OLLAMA_MODELS` |
| **HuggingFace 缓存** | `huggingface` | `models--org--name/snapshots/<rev>/`；respect `HF_HOME` |
| **llama.cpp** | `llamacpp` | `~/.cache/llama.cpp`、`~/llama.cpp/models` |
| **通用目录** | `local` | `~/models`、`X:/models`、`/models`、`/data/models` |
| **自定义目录** | `custom` | `WSP_MODEL_DIRS` |

**没有任何一个来源是必需的。** 全部禁用后系统仍可运行，只是发现不到模型
（你仍然可以用 `wsp import <路径>` 手动导入）。

### 禁用某个来源

```bash
WSP_DISABLE_SOURCES=lmstudio,ollama wsp discover
```

### 加上你自己的来源

写一个插件，然后让 WSP 加载它：

```python
# my_source.py
from pathlib import Path
from wsp.models.sources import ModelSource
from wsp.models.discovery import FoundModel, make_id, scan_hf_dir

class MySource(ModelSource):
    key = "mysource"
    label = "我的模型仓库"

    def roots(self) -> list[Path]:
        return [Path("/mnt/nas/models")]

    def discover(self) -> list[FoundModel]:
        out = []
        for d in self.roots()[0].iterdir():
            m = scan_hf_dir(d, source=self.key)
            if m:
                out.append(m)
        return out
```

```bash
WSP_MODEL_SOURCES=my_source:MySource wsp discover
```

也可以在进程内注册：

```python
from wsp.models import register_source
register_source(MySource())
```

### 自定义目录（最简单的方式）

不想写插件时，直接给目录：

```bash
export WSP_MODEL_DIRS=/mnt/nas/models:~/my-models
wsp discover
```

### 来源之间的避让

每个插件实现 `roots()` 声明自己扫描目录，通用目录据此避让，
**不靠名字硬编码**。所以新加的插件不会和内置来源重复扫描。

## 3. 推理端点：同样不绑定

自动探测的本地推理服务（全部平等对待）：

| 服务 | 默认端口 |
|---|---|
| LM Studio | 1234 |
| Ollama | 11434 |
| llama.cpp server | 8080 |
| vLLM | 8000 |
| text-generation-webui | 5000 |
| KoboldCpp | 5001 |
| LocalAI | 8081 |
| Jan | 1337 |

**任何 OpenAI 兼容端点都能用**，在设置页填 Base URL 即可。
也可以用环境变量追加：

```bash
export WSP_INFERENCE_URLS=http://192.168.1.50:8000/v1,http://gpu-box:1234/v1
```

覆盖内置端点：

```bash
export WSP_LMSTUDIO_URL=http://localhost:1234/v1   # 可选，不影响其他
export WSP_OLLAMA_URL=http://localhost:11434/v1
```

> 注意：这些只是**候选扫描列表**。一个都没装也不影响使用——
> 探测会用本地 torch 后端直接加载模型。

## 4. 计算设备自动选择

```
WSP_DEVICE  >  CUDA  >  Apple MPS  >  CPU
```

dtype 也跟着设备走：

| 设备 | 默认 dtype |
|---|---|
| CUDA | bfloat16 |
| MPS | float16 |
| CPU | float32（float16 在 CPU 上极慢） |

可选 4-bit 量化：`HFLocalBackend(load_in_4bit=True)`，仅 CUDA + bitsandbytes 时生效。

## 5. API Key 按平台选最强方案

| 平台 | 方案 | 说明 |
|---|---|---|
| Windows | DPAPI | `CryptProtectData`，绑定当前用户 |
| macOS | Keychain | 通过 `security` 命令 |
| Linux | Secret Service | 通过 `secret-tool`（GNOME Keyring / KWallet） |
| 兜底 | 本机密钥文件 | 较弱，会在 UI 上明确标注 |

```bash
wsp health   # 看 "encryption" 和 "encryption_detail"
```

## 6. 首次运行引导

`GET /api/onboarding` 会返回当前环境还缺什么：

```jsonc
{
  "steps": [
    {"key": "runtime", "status": "todo",      "commands": ["pip install -e \".[model]\""]},
    {"key": "service", "status": "optional",  "detail": "没有检测到本地理服务"},
    {"key": "model",   "status": "todo",      "commands": ["hf download Qwen/Qwen3.5-0.8B ..."]},
    {"key": "llm",     "status": "todo",      "detail": "在设置里填 Base URL 和 API Key"},
    {"key": "storage", "status": "done"}
  ],
  "summary": {
    "can_edit": false,
    "can_evaluate": false,
    "edit_blockers": ["model", "llm"],
    "eval_blockers": ["model", "llm", "runtime"]
  },
  "recommended_models": [...]
}
```

**`can_edit` 和 `can_evaluate` 是分开的**：
- 编辑只需要 numpy + 模型 + LLM 配置
- 评估额外需要 torch + transformers

这是有意的：只想试试编辑的人不必装 2GB 的 torch。

## 7. 依赖分层

```toml
dependencies = [numpy, fastapi, uvicorn, pydantic, jsonschema, httpx, pyyaml]
optional-dependencies:
  model = [torch, transformers, safetensors]   # 只有本地推理需要
  dev   = [pytest]
```

`pip install -e .` 就能编辑权重、跑 API、开界面。
`pip install -e ".[model]"` 才加上本地推理。

## 8. 已知的平台差异

| 项 | 影响 |
|---|---|
| 硬链接 | `_copy_or_link()` 优先硬链接，跨文件系统会失败并自动回退到复制 |
| 文件锁 | 没有用文件锁；Windows 上并发写同一 checkpoint 可能失败 |
| 路径大小写 | Windows 不区分大小写，模型 id 统一转小写 |
| 控制台编码 | Windows 控制台默认 GBK，中文可能乱码；设 `PYTHONUTF8=1` 解决 |
| 长路径 | Windows 260 字符限制；建议数据目录不要嵌套太深 |

## 9. 如何验证可移植性

```bash
pytest tests/test_portability.py -v
```

这批测试覆盖：
- 各平台路径推导
- LM Studio home 指针 / 缺失时的回退
- Ollama manifest 解析
- HF 缓存结构识别
- 环境变量覆盖（`WSP_DEVICE` / `WSP_MODEL_DIRS` / `OLLAMA_MODELS` / `HF_HOME`）
- 密钥方案在任意平台都能往返

## 10. 如果你要接到自己的环境

```bash
# 自定义数据目录
export WSP_HOME=/data/wsp

# 添加模型目录
export WSP_MODEL_DIRS=/data/models:/mnt/nas/models

# 指定设备
export WSP_DEVICE=cuda:1

# 指定本地服务
export WSP_OLLAMA_URL=http://192.168.1.10:11434/v1
```

然后 `wsp serve`，所有发现和探测都会走这些设置。
