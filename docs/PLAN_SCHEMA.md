# 编辑计划 JSON Schema（生成态）

这份文件是给你看"大模型应该吐什么"的。程序里的权威定义在
`wsp/editor/plan_schema.py`，可通过 `GET /api/schema/plan` 获取。

## 完整示例

```jsonc
{
  "goal": "修复数学推理中的重复循环",
  "reasoning": "探测显示数学题输出重复同一段，撞 token 上限率 90%；失败样例集中在多步算术。考虑对 MLP 输出通道做方向移除。",

  "edits": [
    {
      "type": "rank1_subtract",
      "layer": 15,
      "module": "mlp.down_proj",
      "axis": 1,
      "index": {
        "kind": "list",
        "indices": [0, 2, 11, 25, 34, 35, 38, 39]
      },
      "magnitude": 1.0,
      "direction_source": {
        "kind": "explicit",
        "u": [0.12, -0.03, 0.44],
        "v": [0.02, 0.11, -0.07]
      },
      "expected_effect": "打断 MLP 写出的循环方向",
      "risk": "可能影响需要长推理的题目"
    },
    {
      "type": "direction_ablation",
      "layer": 19,
      "module": "mlp.down_proj",
      "axis": 1,
      "index": { "kind": "top_norm", "n": 64 },
      "method": "project_out",
      "magnitude": 0.8,
      "norm_preserve": true,
      "direction_source": { "kind": "svd_top1" },
      "expected_effect": "减弱同方向的残余",
      "risk": "通用文本生成可能轻微退化"
    }
  ],

  "eval_gate": {
    "primary": "math_short",
    "guards": ["ifeval_mini"],
    "tasks": ["math_short", "ifeval_mini", "loop_mini"],
    "min_delta": 0.05,
    "delta_unit": "absolute",
    "max_regression": 0.03,
    "regression_unit": "absolute",
    "ppl_worsen_ratio": 1.10
  }
}
```

## 字段说明

### 顶层

| 字段 | 必填 | 说明 |
|---|---|---|
| `goal` | ✅ | 一句话目标，最长 2000 字 |
| `reasoning` | | 规划依据，最长 4000 字 |
| `edits` | ✅ | 0~8 条。空数组 = noop |
| `eval_gate` | ✅ | 门控配置 |
| `expected_effect` | | 整体预期 |

### 每条 edit（公共）

| 字段 | 必填 | 说明 |
|---|---|---|
| `type` | ✅ | `direction_ablation` / `rank1_subtract` / `module_scale` / `noop` |
| `layer` | 视类型 | 层号 |
| `module` | 视类型 | 如 `mlp.down_proj` |
| `tensor_name` | | 也可直接给完整张量名（优先于 layer+module） |
| `expected_effect` | ✅ | 预期效果（说明书要求） |
| `risk` | ✅ | 风险说明（说明书要求） |

### `direction_ablation`

| 字段 | 必填 | 说明 |
|---|---|---|
| `axis` | ✅ | `0` 或 `1`，**切哪个维度** |
| `index` | ✅ | 选哪些位置 |
| `method` | ✅ | `project_out` 或 `scale` |
| `magnitude` | ✅ | 0~2。`project_out` 时是移除强度（1.0 全移除）；`scale` 时是保留系数 |
| `norm_preserve` | | 是否保范数，建议 `true` |
| `direction_source` | | `{"kind": "svd_top1"}` / `{"kind": "random"}` / `{"kind":"explicit","vector":[...]}` |

### `rank1_subtract`

| 字段 | 必填 | 说明 |
|---|---|---|
| `axis` | ✅ | `0` 或 `1` |
| `index` | ✅ | 选哪些位置 |
| `magnitude` | ✅ | 即 α。正数=减去，负数=加上 |
| `direction_source.u` | ✅ | 行方向向量（会被归一化） |
| `direction_source.v` | ✅ | 列方向向量（长度需匹配选中数） |

### `module_scale`

| 字段 | 必填 | 说明 |
|---|---|---|
| `factor` | ✅ | 保留系数，(0, 3] |

> ⚠️ 已证实缩放无法新增能力，多数会被门控拒绝。

### `index`

| kind | 字段 | 说明 |
|---|---|---|
| `all` | — | 全部位置（几乎必超预算） |
| `list` | `indices: [int]` | 明确索引（**主力**） |
| `range` | `start`, `stop` | 半开区间 |
| `top_norm` | `n` | 按范数取 top-N |

### `eval_gate`

| 字段 | 必填 | 说明 |
|---|---|---|
| `primary` | ✅ | 唯一判定提升的指标名 |
| `min_delta` | ✅ | 提升阈值 |
| `delta_unit` | | `absolute`（默认）/ `relative` |
| `max_regression` | | 允许的最大退化，默认 0.03 |
| `regression_unit` | | `absolute` / `relative` |
| `guards` | | 不许退化的指标列表 |
| `tasks` | | 要跑的探测任务 |
| `ppl_worsen_ratio` | | 困惑度恶化上限 |

## 内置探测任务

| 名称 | 指标 | 说明 |
|---|---|---|
| `math_short` | `accuracy` | 6 道短数学题 |
| `ifeval_mini` | `follow_rate` | 5 条格式约束 |
| `loop_mini` | `loop_rate` | 长枚举任务，测重复 |
| `refusal_mini` | `refusal_rate` | 中性请求的拒答率 |
| `gen_basic` | `none` | 只采样，不评分 |

## 校验失败会怎样

`POST /api/plans` 返回 400，`detail` 里列出具体路径和原因，例如：

```
编辑计划不符合 schema：
  - edits.0: 'axis' is a required property
  - edits.0.magnitude: 9.0 is greater than the maximum of 2
```

Agent 模式下，这个错误会原样回传给大模型，让它自己修正后重试。
