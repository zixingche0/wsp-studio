# WSP Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](docs/PORTABILITY.md)
[![Tests](https://github.com/wsp-studio/wsp-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/wsp-studio/wsp-studio/actions/workflows/tests.yml)
[![Dependencies](https://img.shields.io/badge/core%20deps-numpy%20only-green.svg)](pyproject.toml)

本地小模型权重编辑工作台。**用户直接和大模型对话，大模型通过这个平台操作小模型。**

不是复现论文，是一个可操作的产品：探测 → 规划 → 干跑 → 应用 → 门控 → 回滚。

> 核心引擎只依赖 **numpy**。torch / transformers 只在"跑模型推理"时才需要。

---

## 它能做什么

你对大模型说："这个模型做数学题老是重复，帮我看看。" 然后：

1. 平台跑探测，测出它在哪些任务上失败、失败成什么样
2. 大模型读探测摘要，给出一个编辑计划（JSON）
3. 平台先**干跑**：告诉你会改多少权重、占多少比例、范数怎么变——**不写任何文件**
4. 你确认后才**应用**，写出新 checkpoint（原始模型目录永远不动）
5. 跑评估门控，指标不达标就**自动回滚**

## 它能做什么之外的（重要）

这些是**实测结论**，不是保守估计：

- **编辑只能移除障碍，不能新增能力。** 论文里 Gemma 3 270M 本身不循环，怎么改都提升不了。
- **从权重读不出语义。** Cribl 用顶尖模型盲测 10 次，猜模型领域 0 次命中。
- **预测不了编辑效果。** 相关系数只有 0.06~0.09。所以必须干跑 + 门控。
- **多数编辑会被拒绝。** wsp-kit 7 次真实运行只有 2 次过门控；OpenWSP 用同样条件复跑，只拿到原文 1/4 的分数。

所以这个产品的诚实定位是：**让"试一试"变得安全、可回滚、可审计**，而不是"保证改好"。

---

## 安装

需要 Python 3.11+。**跨平台**：Windows / macOS / Linux 都可用，无需改任何配置。

```bash
# 核心（编辑引擎，只需 numpy；不带推理）
pip install -e .

# 完整（加本地推理：torch + transformers）
pip install -e ".[model]"

# 开发
pip install -e ".[dev]"
```

> 编辑引擎本身**只用 numpy**，不需要 torch。torch 只在"跑模型推理"时才需要。

**换一台机器要先看环境：**

```bash
wsp platform     # 系统、设备（CUDA/MPS/CPU）、内存、已知模型目录
wsp discover     # 自动扫描 LM Studio / Ollama / HF 缓存 / 通用目录
wsp health       # 还缺什么
```

## 启动

```bash
wsp serve
# 浏览器打开 http://127.0.0.1:8765
```

数据默认存在平台惯例目录，可用 `WSP_HOME` 覆盖。首次打开会自动检测环境，
在对话区列出"还差几步"。

## 移植性

这个项目**不绑定任何特定的本地推理工具**，也没有任何机器特定路径。

**模型发现是插件体系**——LM Studio / Ollama / HF 缓存 / llama.cpp / 通用目录
都只是内置插件，**没有一个是必需的**。可以禁用、可以加自己的。

**推理端点同样开放**——任何 OpenAI 兼容服务都能接（LM Studio / Ollama /
llama.cpp / vLLM / LocalAI / Jan…），也可以用环境变量追加。

所有与机器相关的点都在 `wsp/platform.py` 和 `wsp/models/sources.py` 里收敛：

| 变量 | 作用 | 默认 |
|---|---|---|
| `WSP_HOME` | 数据目录 | 源码旁 `./wsp-data`，否则平台惯例目录 |
| `WSP_MODEL_DIRS` | 额外模型目录（无需写插件） | — |
| `WSP_MODEL_SOURCES` | 加载来源插件 `module:Class` | — |
| `WSP_DISABLE_SOURCES` | 禁用内置来源 | — |
| `WSP_INFERENCE_URLS` | 追加推理端点 | — |
| `WSP_DEVICE` | 强制设备 | 自动：CUDA > MPS > CPU |
| `HF_ENDPOINT` | HF 镜像 | 官方 |

API Key 按平台选最强方案：Windows DPAPI / macOS Keychain / Linux Secret Service，
都没有时回退到本机密钥并**在 UI 上明确标注强度较弱**。

详见 [docs/PORTABILITY.md](docs/PORTABILITY.md)。

---

## 第一次使用

### 1. 准备模型

**可编辑的模型必须是 safetensors 格式。** GGUF（LM Studio / Ollama 里那些）只能用来探测。

```bash
# 先看看本机已经有什么
wsp discover

# 国内网络建议先设镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载（命令会自动适配你装了哪个 CLI）
hf download Qwen/Qwen3.5-0.8B --local-dir ./models/Qwen3.5-0.8B
```

或者用页面：**模型 → 扫描本机所有模型来源**，可编辑的会直接列出导入按钮。

### 2. 配置大模型 API

页面右上角 **设置**：

- Base URL：`https://api.openai.com/v1`，或本地 `http://127.0.0.1:1234/v1`
- API Key：直接填，保存后**用 Windows DPAPI 加密写进 SQLite**，界面永远只显示 `****abcd`
- 模型名、温度

点**测试连接**验证。

### 3. 建项目、开始对话

选模型 → 新建项目 → 新对话 → 直接说需求。

---

## 支持的编辑类型

| 类型 | 作用 | 适用场景 |
|---|---|---|
| `direction_ablation` | 沿方向 d 把选中切片在 d 上的**投影**减掉 | 去除某个行为方向 |
| `rank1_subtract` | 减去一个**外部给定**的 rank-1 分量 `u⊗v` | 精确复现某次编辑 |
| `module_scale` | 整体缩放 | 实验性，多数会被门控拒绝 |
| `noop` | 什么都不做 | 对照组，用来证明"不是噪声" |

### 关于方向：官方编辑的真实形状

我们抓取了官方发布的编辑后张量做比对，实测结论（**推翻了"选前 64 列"的猜测**）：

| 项 | 官方实际 |
|---|---|
| 改动列 | **64 个，完全不连续**（0, 2, 11, 25, … 3003） |
| 差分矩阵 | **严格 rank-1**（σ₂/σ₁ ≈ 1e-3） |
| 行覆盖 | 1024 行全覆盖 |
| relative_change | 0.1064（是"差异范数/原范数"，不是"范数变化"） |

**这 64 列不是按范数选的**，是按语义选的（Astra 分析了激活）。所以规划器必须能**显式列出列索引**，`index.kind: "list"` 是主力，`range` 只用于测试。

`tests/test_official_reference.py` 用官方真实数据做了逐 bit 校验，包括两个张量的 sha256 比对。

---

## 评估门控

```jsonc
"eval_gate": {
  "primary": "math_short",        // 唯一判定"提升了"的指标
  "guards": ["ifeval_mini"],      // 不许退化的
  "min_delta": 0.05,              // 提升阈值
  "delta_unit": "absolute",       // absolute | relative
  "max_regression": 0.03,
  "regression_unit": "absolute",
  "ppl_worsen_ratio": 1.10        // 困惑度恶化上限（可选）
}
```

不通过 → 自动回滚 → 记录原因。

---

## 磁盘管理

每次 apply 会写一份完整 checkpoint（1.7 GB / 个），反复实验会很快占满磁盘。

```bash
wsp storage              # 看占用
wsp gc                   # 预览能清理什么（不删）
wsp gc --apply           # 清理已回滚/失败的 checkpoint
wsp gc --apply --include-applied   # 连已应用的也删（diff 保留）
wsp gc --apply --keep-latest 2     # 保留最近 2 个
```

清理是安全的：**diff 永远保留**（每次几百 KB），任何编辑都能从
「原始模型 + diff」精确重建，`pytest tests/test_storage_gc.py` 验证了这一点。

## CLI

```bash
wsp platform                  # 平台信息（系统/设备/目录）
wsp discover                  # 扫描本机所有模型来源
wsp health                    # 环境体检：还缺什么
wsp models                    # 列出已注册模型
wsp import <path> --model-id x
wsp inspect qwen35-0.8b --layer 15
wsp probe qwen35-0.8b --tasks math_short,loop_mini
wsp project "修复循环" qwen35-0.8b --goal "..."
wsp plan <project_id> "减少重复循环"
wsp apply <plan_id> --dry-run # 只预览
wsp apply <plan_id>           # 真正应用
wsp eval <edit_id>
wsp rollback <edit_id>
wsp edits
wsp storage                   # 磁盘占用
wsp gc --apply                # 清理旧 checkpoint
```

## HTTP API

完整列表见 `wsp/api/routes.py`。核心几个：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/models/import` | 导入 safetensors |
| GET | `/api/models/sources` | 各模型来源状态 |
| GET | `/api/inference/services` | 运行中的推理服务 |
| POST | `/api/probes/run` | 跑探测 |
| POST | `/api/conversations/{id}/chat` | **Agent 对话（核心）** |
| POST | `/api/plans/generate` | 让 LLM 生成计划 |
| POST | `/api/plans/{id}/apply` | `{"dry_run": true/false}` |
| POST | `/api/evals/run` | 评估门控 |
| POST | `/api/rollback` | 回滚 |
| GET | `/api/edits` | 编辑历史 |
| PUT | `/api/settings/llm` | 保存 LLM 配置 |

---

## 安全边界

- 原始模型目录**只读**，代码里 `assert_outside()` 强制检查
- API Key 用 DPAPI 加密，密文不可跨机器解密；UI 只显示掩码
- 单次编辑**硬上限 1% 权重**，超过直接拒绝
- 权重**永不上传**到外部 API，只发结构化摘要
- 拒绝"绕过安全对齐 / 让模型变聪明 / 上传权重"类目标

---

## 数据布局

```
wsp-data/
  models/registry.json        # 模型注册表
  workdir/
    models/<id>/revisions/<edit_id>/   # 编辑后的 checkpoint
    edits/<edit_id>/                    # 位级 diff（edit.npz + manifest.json）
  wsp.db                      # SQLite：projects/conversations/plans/edits/evals
```

编辑存的是**位级 diff** 而不是整模型备份：13 万权重的改动约 400 KB，对比整模型 1.6 GB。

---

## 测试

```bash
pytest                    # 144 个，不需要 torch / 不需要联网 / < 15s
pytest -m slow            # 额外 6 个，用官方真实数据做逐 bit 校验
```

---

## 目录结构

```
wsp-studio/
  wsp/
    config.py          路径与只读保护
    service.py         统一门面
    cli.py             命令行
    weights/
      st.py            纯 numpy safetensors 读写（bf16 位操作）
      names.py         跨模型家族张量名解析
      store.py         模型索引、hash
    editor/
      plan_schema.py   DSL + JSON Schema
      ops.py           编辑算子
      executor.py     干跑 / 应用 / diff 导出
    probe/
      tasks.py         内置探测任务
      backends.py      HTTP / 本地推理后端
      runner.py        探测运行器
    eval/gate.py       门控
    llm/
      client.py        OpenAI 兼容客户端 + JSON 修复
      prompts.py       system prompt（含实证负面知识）
      tools.py         function calling 定义
    agent/loop.py      Agent 循环
    api/               FastAPI
    web/index.html     单页前端
  tests/
```

---

## 已知限制

- 编辑算子只在 `mlp.down_proj` 上验证过；Qwen3.5 的 hybrid linear attention 未测。
- 探测任务只有 5 个内置任务，每个几道题；不是完整 benchmark。
- 本地推理后端依赖 transformers 对模型架构的支持。Qwen3.5 需要 `transformers>=5.20`。
- 本地推理服务加载的是**原始权重**，读不到刚改完的 checkpoint；
  评估编辑后的模型必须用本地 torch 后端。
- `direction_source` 没有实现"从激活差估计方向"（论文里的做法），
  目前只有 `explicit` / `random` / `svd_top1`。
- 无 GPU 时 CPU 推理较慢（2B 模型约 9~14 tok/s）。

---

## 参考与致谢

编辑算子的设计参考并验证自以下公开工作：

- **[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)**
  — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL), 2026-11-02。
  这是本项目 `rank1_subtract` 和编辑几何的直接来源。
- **[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)**
  — 官方实验日志与编辑 diff（代码 MIT / 日志 CC BY 4.0）。
- **[RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)**
  — 经 sha256 校验的编辑后模型，本项目用它做逐 bit 对标。

设计思路参考（**未使用其代码**）：

- [wsp-kit](https://github.com/AuremHQ/weight-space-programming)（MIT）— 1% 上限与极简门控的取舍
- [OpenWSP](https://github.com/openwsp/openwsp)（MIT）— diff manifest 字段设计
- [Heretic](https://github.com/p-e-w/heretic)（AGPL-3.0）— Modifier/Scorer 插件二分思路

完整依赖与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## 参与

- [贡献指南](CONTRIBUTING.md)
- [行为准则](CODE_OF_CONDUCT.md)
- [安全策略](SECURITY.md)（**漏洞请走私密报告，不要开公开 issue**）
- [更新日志](CHANGELOG.md)

## 许可证

[MIT](LICENSE) © 2026 WSP Studio contributors

### 关于"不许商业化"

这个项目采用 MIT —— 它**允许**商业使用。选 MIT 的理由：

- 同领域项目（wsp-kit、OpenWSP、di37 的 skill）都用 MIT
- MIT 能被所有包管理器、镜像站、awesome 列表正常收录
- 如果你想限制商业使用，可以换成
  [PolyForm Noncommercial](https://polyformproject.org/licenses/noncommercial/1.0.0/)，
  但那样它就不再是 OSI 认可的开源软件

如果你想加商业限制，改 `LICENSE` 即可 —— 代码里没有任何地方依赖许可证类型。

