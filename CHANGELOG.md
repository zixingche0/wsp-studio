# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-11-09

首个可用版本。

### 新增

**核心**
- 纯 numpy 的 safetensors 读写，bf16 按位模式处理（不做浮点转换），
  可逐 bit 复现编辑结果
- 跨模型家族的张量名解析（Qwen / Llama / GPT-2 / Gemma 的命名差异）
- 四种编辑算子：`direction_ablation` / `rank1_subtract` /
  `module_scale` / `noop`
- 编辑计划 JSON Schema，严格校验，多余字段直接拒绝
- 位级 diff 导出（`edit.npz` + `manifest.json`），
  13 万权重的改动约 400 KB，对比整模型 1.6 GB

**预算与安全**
- 双重预算闸门：模型级 ≤1% **且** 单张量 ≤50%
- 原始模型目录只读（`assert_outside`），防路径穿越
- 默认 dry-run，不写任何文件
- 编辑前后逐张量 sha256 校验，可精确重建

**Agent**
- 对话式操作：用户说需求，大模型调工具
- 9 个工具：探查模型、跑探测、提计划、干跑、应用、评估、回滚等
- system prompt 内置实证负面知识，抑制"编辑能新增能力"这类幻觉
- 目标合规过滤：拒绝绕过安全对齐 / 注入能力 / 上传权重

**探测**
- 5 个内置任务：`math_short` / `ifeval_mini` / `loop_mini` /
  `refusal_mini` / `gen_basic`
- 记录采样配置（thinking / temperature 等），
  防止把"解码问题"误判为"权重问题"

**评估**
- 门控：主指标提升 ≥ `min_delta`，guard 退化 ≤ `max_regression`
- 可选困惑度恶化上限
- 未通过自动回滚并记录原因

**平台适配**
- 无任何机器特定路径，全部走 `WSP_HOME` / `WSP_MODEL_DIRS` 等环境变量
- 模型来源插件体系：LM Studio / Ollama / HuggingFace 缓存 / llama.cpp /
  通用目录 / 自定义，**没有哪一个是必需的**
- 计算设备自动选择：CUDA > Apple MPS > CPU，dtype 跟着设备走
- 推理端点不绑定工具：任何 OpenAI 兼容服务都能接
- API Key 按平台选最强方案：DPAPI / Keychain / Secret Service / 本机密钥

**界面**
- 单页前端，零构建（原生 HTML/JS）
- 对话内嵌卡片：探测结果 / 编辑计划 / 干跑 diff / 门控结论
- 首次运行引导：检测环境并给出针对当前平台的补齐命令

**存储管理**
- SQLite 保存项目 / 对话 / 计划 / 编辑 / 评估，五层外键，无孤儿记录
- `wsp gc` 清理旧 checkpoint（diff 永远保留，可重建）

### 已验证

- 用官方发布的真实编辑张量做了逐 bit 校验，
  两个张量的 sha256 与官方 manifest 一致
- 真实 Qwen3.5-0.8B 上跑通 探测 → 编辑 → 应用 → 回滚 全链路
- 233 个单元测试 + 123 项系统级检查（HTTP / 前端渲染 / 并发 / 安全）

### 已知限制

- 编辑算子只在 `mlp.down_proj` 上验证过；
  Qwen3.5 的 hybrid linear attention 部分未测
- `direction_source` 只支持 `explicit` / `random` / `svd_top1`，
  **没有**实现"从激活差估计方向"
- 评估编辑后的模型必须用本地 torch 后端
  （HTTP 服务加载的是原始权重，读不到新 checkpoint）
- 探测任务只有 5 个、每任务几道题，不是完整 benchmark
- 无 GPU 时 CPU 推理较慢（2B 模型约 9~14 tok/s）

[0.1.0]: https://github.com/wsp-studio/wsp-studio/releases/tag/v0.1.0
