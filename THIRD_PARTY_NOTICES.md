# 第三方声明

本项目使用以下开源软件。完整依赖列表见 `pyproject.toml`。

## 运行时依赖

| 项目 | 许可证 | 用途 |
|---|---|---|
| [NumPy](https://numpy.org/) | BSD-3-Clause | 张量运算、bf16 位操作 |
| [FastAPI](https://fastapi.tiangolo.com/) | MIT | HTTP API |
| [Uvicorn](https://www.uvicorn.org/) | BSD-3-Clause | ASGI 服务器 |
| [Pydantic](https://docs.pydantic.dev/) | MIT | 请求校验 |
| [jsonschema](https://github.com/python-jsonschema/jsonschema) | MIT | 编辑计划 Schema 校验 |
| [httpx](https://www.python-httpx.org/) | BSD-3-Clause | HTTP 客户端 |
| [PyYAML](https://pyyaml.org/) | MIT | 配置文件 |

## 可选依赖

| 项目 | 许可证 | 用途 |
|---|---|---|
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | 本地推理（评估用） |
| [Transformers](https://huggingface.co/docs/transformers) | Apache-2.0 | 模型加载与生成 |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | 格式参考（本项目自带解析器） |

## 开发依赖

| 项目 | 许可证 |
|---|---|
| [pytest](https://pytest.org/) | MIT |

## 研究成果与数据

本项目的编辑算子设计参考并验证自以下公开发布：

- **Sparks of Weight-Space Programming: LLMs can natively read & write
  neural network weights** — Rikiya Takehi, Jayden Teoh, Alex Zhang,
  Omar Khattab (MIT CSAIL), 2026-11-02.
  <https://rikiyat.github.io/blog/weight-space-programming/>
  （博客文章，非同行评审论文）

- **RikiyaT/lm_understand-logs** — 官方实验日志与编辑 diff（代码 MIT，
  日志 CC BY 4.0）。<https://github.com/RikiyaT/lm_understand-logs>

- **RikiyaT/Qwen3.5-0.8B-weight-space-programming** — Hugging Face 上
  经 sha256 校验的编辑后模型（Apache-2.0）。
  <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>

`tests/test_official_reference.py` 用上述公开数据做逐 bit 校验。
这些数据**不随本项目分发**，需要用户自行下载。

## 设计参考（未使用其代码）

以下项目的**设计思路**对本项目有影响，但未复制其代码：

- [wsp-kit (AuremHQ)](https://github.com/AuremHQ/weight-space-programming) —
  MIT。参考了"1% 权重上限"和"极简门控"的设计取舍。

- [OpenWSP](https://github.com/openwsp/openwsp) — MIT。
  参考了编辑 diff 的 manifest 字段设计（`base_sha256` / `edited_sha256`）。

- [Heretic](https://github.com/p-e-w/heretic) — **AGPL-3.0**。
  参考了 Modifier/Scorer 插件二分的架构思路。
  **本项目未使用其任何代码**（AGPL 与本项目 MIT 不兼容）。

## 兼容性说明

本项目的许可证是 [MIT](LICENSE)。上述设计参考均未产生代码级依赖，
因此不受其许可证约束。

## 模型权重

本项目**不分发任何模型权重**。使用到的模型（如 Qwen3.5-0.8B，
Apache-2.0）需由用户自行下载，并遵守其各自的许可证。
