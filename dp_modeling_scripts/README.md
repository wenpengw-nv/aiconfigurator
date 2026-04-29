# DP Imbalance Modeling

Compare measured vs predicted decode step time for DeepSeek-R1 wide-EP under
DP=16 attention. One script, two plots (static + dynamic).

```
dp_modeling_scripts/
├── dp_imbalance_modeling.py        # 唯一脚本
├── data/decode_step_aligned.csv # 实测 per-bucket 数据（输入）
└── output/                         # 运行后自动生成
    ├── static.csv     static.png   # G0–G2 mean + G3/G4 每 180s
    └── dynamic.csv    dynamic.png  # G3/G4/G5 每 30s 时序
```

## 依赖

```bash
cd /path/to/aiconfigurator && pip install .
```

## 用法

```bash
cd dp_modeling_scripts
python dp_imbalance_modeling.py
```

可选参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data-csv` | `data/decode_step_aligned.csv` | 实测 per-bucket CSV |

输出：`output/{static,dynamic}.{csv,png}`。

## 模型

每个 decode step 分两段：

```
predicted_ms = attention(batch=max_rank_batch, isl=max_rank_ctx//max_rank_batch)
             + MoE(batch=round(total_batch/16),  isl=8192)
```

- **Attention**：DP global barrier 决定整步等于最重 rank 的耗时
- **MoE**：DeepEP A2A 之后所有 token 一起算，输入 `total_batch/DP_SIZE` 是因为
  AIC 内部会乘回 `attention_dp_size`

**static** 模式：G0–G2 取整段 warmup 后均值，G3/G4 切 180s 窗口，每窗口对
`(max_rank_batch, max_rank_ctx, total_batch, global_step_itl_max)` 取均值后跑一次
预测。

**dynamic** 模式：G3/G4/G5 切 30s 窗口，每窗口跑一次预测，画时序对比。

## 输入 CSV schema

`data/decode_step_aligned.csv` 必含的列：

| 列 | 含义 |
|---|---|
| `scenario`, `group` | 场景名 / 组（G0…G5） |
| `time_bucket_center_s` | 桶中心时刻（秒） |
| `num_active_ranks` | 桶内活跃 rank 数 |
| `max_rank_batch` | 16 rank 中 batch 最大值 |
| `max_rank_ctx` | 16 rank 中 ctx 最大值 |
| `total_batch` | 16 rank batch 之和 |
| `global_step_itl_max` | 该桶 16 rank ITL 的最大值（实测基准 — barrier 下木桶效应取最慢 rank） |

## 输出 CSV 列

`static.csv` / `dynamic.csv`:

| 列 | 含义 |
|---|---|
| `scenario`, `group` | 场景名 / 组 |
| `marker` | `mean`（G0–G2）或 `30-210s`（窗口） |
| `max_rank_batch`, `max_rank_ctx`, `total_batch` | 模型输入 |
| `measured_ms` | 实测 step time |
| `predicted_ms` | 模型预测 step time |
| `error_pct` | `(predicted - measured) / measured * 100` |
| `time_s` | 仅 dynamic：窗口中心时刻 |

## 换平台

aiconfigurator 仓库里需要放好两份数据，再改脚本里 `get_database` 的第一个参数：

```
aiconfigurator/src/aiconfigurator/systems/
├── <gpu>.yaml                       # 系统注册（GPU 规格、NCCL、节点配置）
└── data/<gpu>/sglang/0.5.9/         # sglang 0.5.9 性能采样数据
```

参考现有的 `h200.yaml` 和 `data/h200/sglang/0.5.9/` 即可。
