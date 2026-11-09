# 贡献指南

感谢你愿意参与。这个项目很小，规则也很简单。

## 开发环境

```bash
git clone https://github.com/<your-name>/wsp-studio.git
cd wsp-studio

# 只跑编辑引擎（不需要 torch）
pip install -e .

# 加上本地推理（评估需要）
pip install -e ".[model]"

# 开发依赖
pip install -e ".[dev]"
```

## 跑测试

```bash
# 单元测试：不需要 torch、不需要联网、不需要真实模型
pytest

# 官方数据对标（需要先准备样本，见下）
WSP_TEST_MODEL=/path/to/Qwen3.5-0.8B \
WSP_OFFICIAL_TENSOR=/path/to/off_l15.bin \
pytest -m slow

# 全量（含 HTTP / 前端 / 并发 / 安全，需要先 wsp serve）
python scripts/run_all_tests.py
```

### 官方对标测试的准备

`tests/test_official_reference.py` 会对官方的真实编辑做逐 bit 校验。
需要两个文件：

1. `WSP_TEST_MODEL`：本地 `Qwen/Qwen3.5-0.8B` 的 safetensors 目录
2. `WSP_OFFICIAL_TENSOR`：从官方 checkpoint 裁剪出的
   `model.language_model.layers.15.mlp.down_proj.weight` 原始字节
   （来自 [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming)）

没有这两个文件时这批测试会自动跳过，不影响普通开发。

## 代码约定

- **中文注释**，解释"为什么"而不是"是什么"
- 类型注解尽量写全（`from __future__ import annotations`）
- 新增编辑类型必须在 `wsp/editor/plan_schema.py` 加 schema 分支
- 涉及权重的函数**不能**有副作用地修改传入数组
- 任何写磁盘的函数都要能被 `dry_run` 短路

## 提交前自查

```bash
pytest                              # 必须全绿
grep -rn "C:\\\\Users\|/home/" wsp/  # 不能有机器特定路径
grep -rniE "sk-[a-z0-9]{20,}" .     # 不能有真实密钥
```

## 新增一个模型来源

见 `docs/PORTABILITY.md` 第 2 节。要点：

```python
from wsp.models.sources import ModelSource

class MySource(ModelSource):
    key = "mysource"
    label = "我的仓库"

    def roots(self):        # 声明扫哪些目录，通用来源会据此避让
        return [Path("/data/models")]

    def discover(self):     # 返回 FoundModel 列表，异常会被吞掉
        ...
```

然后 `WSP_MODEL_SOURCES=mymodule:MySource wsp discover` 就能加载。

## 新增一种编辑算子

1. 在 `wsp/editor/ops.py` 写纯函数，签名参考
   `apply_direction_ablation`：输入 `(arr, dtype_str, **params)`，
   返回 `(new_arr, stats)`
2. 在 `plan_schema.py` 的 `EDIT_SCHEMA` 加类型和必填字段
3. 在 `executor.py` 的 `execute_plan` 加分支
4. 在 `service.py` 的 `_pairs_from_plan`（用于导出 diff）加同样的分支
5. 补测试，**必须验证未选中部分逐 bit 不变**

## 隐私

提交前确认没有把你的 API Key、模型路径、用户名带进来。
`.gitignore` 已经覆盖了常见情况，但**不要依赖它**。

## 许可证

提交即表示你同意以 [MIT](LICENSE) 授权你的贡献。
