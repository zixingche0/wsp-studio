# 安全策略

## 报告漏洞

**不要**开公开 issue。发邮件给仓库维护者，或使用 GitHub 的
[私密漏洞报告](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)。

请包含：复现步骤、影响范围、你的环境（OS / Python 版本）。

## 这个项目处理什么敏感数据

| 数据 | 位置 | 保护方式 |
|---|---|---|
| 大模型 API Key | 本地 SQLite | Windows DPAPI / macOS Keychain / Linux Secret Service |
| 模型权重 | 本地磁盘 | 不离开本机，永不上传 |
| 对话记录 | 本地 SQLite | 明文（本地文件，非加密） |
| 编辑 diff | 本地磁盘 | 明文，含张量 sha256 |

**API Key 永远不会**：
- 出现在 HTTP 响应里（只返回掩码 `${'*'}****abcd`）
- 写进日志
- 上传到任何地方

**权重永远不会**上传。发送给大模型 API 的只有结构化探测摘要
（分数、失败样例的 prompt/output 片段）。

## 攻击面

### 1. 本地 HTTP 服务（127.0.0.1:8765）

默认只监听 `127.0.0.1`。如果改成 `0.0.0.0`，同网段任何人都能调用你的
API Key（通过 `/api/settings/llm/test` 触发请求）和修改权重。

```bash
# 不要这样做，除非你清楚后果
WSP_HOST=0.0.0.0 wsp serve
```

### 2. 模型目录

应用会读写模型目录。已实现的保护：

- `assert_outside()` 确保写入目标不在原始模型目录内
- 路径穿越（`../`）被 `Path.resolve()` 拦截
- GC 只删 `WSP_HOME/workdir` 下的内容

### 3. 大模型返回的编辑计划

LLM 输出不直接执行。流程是：

1. JSON Schema 严格校验（多余字段直接拒绝）
2. 预算检查（模型级 ≤1% 且单张量 ≤50%）
3. 默认 dry-run，必须显式 `dry_run=false` 才写文件
4. 每次编辑记录 diff + sha256，可回滚

### 4. 恶意模型文件

`wsp/weights/st.py` 是手写的 safetensors 解析器，会校验：

- header 长度上限（100 MB）
- 每个张量的数据长度与 shape 一致
- dtype 白名单
- 用 `np.frombuffer` 而不是 `pickle`，**不执行任意代码**

但如果有人在模型里放超大张量，仍可能 OOM。加载前请确认模型来源。

### 5. 依赖

`torch` 的 `torch.load` 默认使用 pickle。本项目**不调用** `torch.load`，
而是用自己的 safetensors 解析器。如果你扩展代码时要用 `torch.load`，
务必加 `weights_only=True`。

## 已知的非问题

- **没有认证**：这是单用户本地工具。多用户场景请自行加反代 + 认证。
- **CORS 全开**：因为前端和后端同源部署。如果前后端分离需要收紧。
- **对话记录明文**：本地文件，不加密。敏感场景请自行加密磁盘。

## 支持版本

只有 `main` 分支的最新代码接受安全修复。
