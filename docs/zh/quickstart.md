# HistoWeave 快速入门（中文）

约 **10 分钟**：从安装到第一份 HTML 报告，再到方法推荐与真实数据入口。

> HistoWeave 是空间转录组的 **编排 + 评测 + 可复现报告层**，不是“又一个最优 domain 算法”。
> 包名：`histoweave-spatial` · Python **≥ 3.11** · 许可 BSD-3-Clause

---

## 0. 选哪条路径？

| 你是谁 | 安装 | 下一步 |
|--------|------|--------|
| 学生 / 第一次接触 | `pip install "histoweave-spatial[scanpy]"` | §2 Demo |
| Visium 实验室 | 加上 `[io,spatial]` | §5 读入 outs |
| 方法开发者 | `[scanpy,dev]` | §7 插件与验证 |
| 要 SpaGCN/GraphST/STAGATE | 隔离环境 + 官方包 | §8 SOTA |

---

## 1. 安装

### 1.1 推荐：最小可用

```bash
conda create -n histoweave python=3.12 -y
conda activate histoweave

pip install "histoweave-spatial[scanpy]"

histoweave --version
histoweave list-methods
```

### 1.2 真实数据读写（Visium / Xenium 等）

```bash
pip install "histoweave-spatial[io,spatial,scanpy]"
```

### 1.3 按需 extra

```bash
# 去卷积 / 注释 / SVG 等
pip install "histoweave-spatial[cell2location,scanvi,celltypist,spatialde,liana]"

# 深度学习栈（重）
pip install "histoweave-spatial[deep-learning]"
```

### 1.4 从本仓库源码（开发）

```bash
cd path/to/histoweave
pip install -e ".[scanpy,io,spatial,dev]"
pytest -q
```

---

## 2. 60 秒 Demo（零数据）

### CLI

```bash
histoweave run --demo --out demo_report.html
# 用浏览器打开 demo_report.html
```

### Python

```python
import histoweave as ts

data = ts.datasets.make_synthetic(seed=0)   # 合成数据 + 种植 domain
result = ts.run_pipeline(data, verbose=True)  # QC → 归一化 → 域检测 → 注释
ts.build_report(result, "demo_report.html")   # 自包含 HTML 报告

for step in result.uns["run_manifest"]["steps"]:
    logger = __import__("logging").getLogger(__name__)
    logger.info("%s %s %s", step["category"], step["method"], step["version"])
```

或：

```bash
python examples/quickstart.py
```

---

## 3. 浏览方法与成熟度

```bash
histoweave list-methods
histoweave list-methods --category domain_detection --json
```

| 成熟度 | 含义 |
|--------|------|
| **validated** | 有多数据集证据包（见 `docs/methods/validation/`） |
| production | 运维可靠、工程可用 |
| beta | 真实上游封装，证据未齐 |
| experimental | 基线 / 研究孵化器 |

优先试用 validated 域方法示例：`spectral` · `gaussian_mixture` · `banksy_py` · `spagcn` · `stagate`。

完整目录：[方法指南](../methods/index.md) · [验证报告](../methods/validation/index.md)

---

## 4. 对比两个域检测方法（Workshop 核心）

```python
from histoweave.plugins import MethodCategory, create_method
from histoweave._math import adjusted_rand_index
import histoweave as ts

data = ts.datasets.make_synthetic(n_cells=600, n_genes=40, n_domains=3, seed=1)
norm = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(data)

for name in ("kmeans", "spectral"):
    out = create_method(
        MethodCategory.DOMAIN_DETECTION,
        name,
        n_domains=3,
        random_state=0,
    ).run(norm.copy())
    ari = adjusted_rand_index(
        out.obs["domain_truth"].to_numpy(),
        out.obs["domain"].to_numpy(),
    )
    logger = __import__("logging").getLogger(__name__)
    logger.info("%-12s ARI=%.3f", name, ari)
```

**要点：** 有 ground truth 时用 ARI；真实分析常没有 domain GT，应看多方法一致性与 uncertainty，而不是硬编一个 Leiden 当 domain 标签。

---

## 5. 读入真实 Visium / Xenium

```bash
# Visium Space Ranger outs 目录
histoweave ingest --input /path/to/spaceranger/outs --assay visium --out sample.ttab

# Xenium
histoweave ingest --input /path/to/xenium/output --assay xenium --out sample.ttab
```

```python
from histoweave.io import read

data = read("visium", "/path/to/spaceranger/outs")
```

没有真实数据时，生成格式对齐的 fixture：

```python
from histoweave.datasets import write_visium_fixture, write_xenium_fixture

write_visium_fixture("demo_visium")
write_xenium_fixture("demo_xenium")
```

---

## 6. 方法推荐（差异化功能）

推荐**不需要** query 上的 domain 标签；会相对 global-best 报告是否“真的更优”。

```bash
# 若仓库内有预计算 landscape：
histoweave recommend --in sample.ttab \
  --knowledge-base figure3_results/landscape.json \
  --json --out recommendation.json
```

```python
from histoweave.benchmark import MethodRecommender, AnalysisTask
import histoweave as ts

data = ts.datasets.make_synthetic(seed=0)
rec = MethodRecommender("figure3_results/landscape.json").recommend(
    data,
    task=AnalysisTask.SPATIAL_DOMAIN,
    platform="visium",
    spatial_context_policy="high",
)
logger = __import__("logging").getLogger(__name__)
logger.info("%s", rec.summary())
logger.info("beats global-best baseline? %s", getattr(rec, "beats_global_best_baseline", None))
logger.info("%s", getattr(rec, "warnings", None))
```

若 `landscape.json` 不存在，可先：

```bash
histoweave benchmark --task domain_detection --out landscape.json
```

---

## 7. 自定义流水线

```python
from histoweave import run_pipeline
from histoweave.workflow import PipelineStep
import histoweave as ts

data = ts.datasets.make_synthetic(seed=0)
steps = [
    PipelineStep("qc", "basic_qc", {"n_mads": 3.0}),
    PipelineStep("normalization", "log1p_cp10k"),
    PipelineStep("domain_detection", "spectral", {"n_domains": 4, "random_state": 0}),
    PipelineStep("annotation", "marker_score"),
]
result = run_pipeline(data, steps)
ts.build_report(result, "custom_report.html")
```

CLI 逐步执行（每步写入 provenance）：

```bash
histoweave step --in sample.ttab --category normalization --method log1p_cp10k
histoweave step --in sample.ttab --category domain_detection --method spectral \
  --param n_domains=7
```

---

## 8. SOTA 与隔离环境（可选）

| 方法 | 说明 |
|------|------|
| SpaGCN | `pip install SpaGCN==1.2.7 scikit-misc`；多切片真实 ARI 见 validation |
| GraphST | 官方包 + 常需 `pot`；可用 `run_real_graphst_stagate_ari.py` |
| STAGATE | 官方 `STAGATE_pyG`；Windows 上 `torch-sparse` 易不匹配 |
| RCTD / BayesSpace | 需 R / Rscript，无 backend 时 **fail-closed** |

```bash
# 仓库内真实 ARI 复现（需已装官方后端）
python research/method_validation/run_real_graphst_stagate_ari.py \
  --methods graphst,stagate --max-obs 1000
```

**原则：** 缺官方包时 HistoWeave **不会**静默换成玩具算法。

---

## 9. 发现 / 第二组织（高级）

```bash
# DLPFC cryptic-niche 发现
histoweave discovery run

# Xenium 淋巴结第二组织
histoweave discovery xenium-lymph
histoweave discovery xenium-lymph --gc-deep-dive
```

---

## 10. 30 分钟 Workshop

一键 notebook：

- 仓库路径：[`examples/workshop_30min.ipynb`](https://github.com/HERRY423/Histoweave/blob/main/examples/workshop_30min.ipynb)
- 脚本等价：`python examples/quickstart.py`

建议时间表：

| 分钟 | 内容 |
|------|------|
| 0–5 | 定位：编排层 vs 单一算法 |
| 5–10 | 安装 + Demo 报告 |
| 10–18 | `kmeans` vs `spectral` ARI |
| 18–25 | `list-methods` + validated 证据 |
| 25–30 | recommend / 下一步真实 Visium |

---

## 11. 常见问题

**Q: `ModuleNotFoundError: SpaGCN`？**
A: 正常。未装官方包时应报错；`pip install SpaGCN==1.2.7` 或改用 `spectral` / `banksy_py`。

**Q: 推荐结果没超过 global-best？**
A: 这是**合法科学输出**，表示应保留多方法 ensemble，而不是假装“一定有个性化最优”。

**Q: 能把 Leiden 当 domain 真值做基准吗？**
A: Task contract **拒绝** 自监督标签充当 spatial-domain GT。

**Q: Windows 上 STAGATE / torch-sparse？**
A: 优先 edge_index 路径或隔离 py3.12 + 匹配 torch 的 sparse 轮子；见 `GRAPHST_STAGATE_REAL_ARI.md`。

---

## 12. 下一步阅读

| 文档 | 内容 |
|------|------|
| [英文 Quickstart](../quickstart.md) | 与本文对应的英文版 |
| [方法选择](../method-selection.md) | 何时用哪类方法 |
| [验证报告索引](../methods/validation/index.md) | 多数据集证据 |
| [CONTRIBUTING](https://github.com/HERRY423/Histoweave/blob/main/CONTRIBUTING.md) | 贡献插件与成熟度 |
| [科学 claim 边界](https://github.com/HERRY423/Histoweave#scientific-claim-read-this-first) | 我们声称什么 / 不声称什么 |

---

**一句话记住：** 先 `run --demo` 出报告，再对比 2 个 validated 方法，最后在真实 Visium 上用 `recommend` + 多方法一致性，而不是找“唯一冠军算法”。
