# HistoWeave 深度审阅 — 新一轮优化后的改进方案

> 审阅日期：2026-07-11 | 基线：**122 passed / 4 skipped / 0 failed / 0 mypy errors / 2 ruff issues**

---

## 一、本轮优化状态总结

自上次 REVIEW_2026_07_11.md 以来，核心改进已全部落地。当前状态：

| 维度 | 7月10日基线 (DEFECT_ANALYSIS) | 7月11日当前 |
|------|-----|-----|
| 测试 | 74 passed | **122 passed** (+48) |
| 方法数 | 8 个 toy | **12 个定义 / 10 个有真实实现** |
| Nextflow | ❌ 不存在 | ✅ DSL2 完整管道 (7 进程) |
| Bundle 完整性 | 无校验 | ✅ SHA-256 + 原子提交 + 路径穿越防护 |
| 参数验证 | 无 | ✅ 类型检查 + bounds + choices |
| 标记基因解析 | 简单索引查找 | ✅ 双命名空间 + 大小写折叠 + fail-closed |
| 错误处理 | try/except | ✅ 结构化异常层次 + 部分成功回执 |
| 管道容错 | 无 | ✅ `on_error="continue"` |
| CI/CD | ❌ | ✅ 3-job × 9-matrix CI |
| ruff | 5 个 | **2 个** |
| mypy | 0 错误 | **0 错误** |

新增模块：`logging.py`（已写但未集成）、`spatial_svg.py`（Moran's I）、`spatial_graph.py`（图指标）、`sklearn_clustering.py`（DBSCAN/层次/谱聚类）、`integration.py`（ComBat）

---

## 二、当前问题：按投入产出比排序

以下 8 项是 **本轮深度审阅的核心输出**——它们不是之前报告的复述，而是基于当前代码的新发现或未被充分重视的问题。

---

### 🔴 [N1] `logging.py` 已完整实现但零集成 — 投入极低，产出极高

**发现：** `logging.py` 是一个完整的结构化日志模块：JSON 格式化器（`JsonFormatter`）、上下文关联（`log_context(run_id=, step_id=)`）、密钥脱敏（`redact()`/`redact_text()`）、事件 API（`log_event()`）——但它**未被任何生产代码调用**。`MATURITY_ROADMAP.md` 将日志覆盖率标为 0%，但问题不是"没有日志模块"，而是"有模块但没用"。

**文件证据：**
- `pipeline.py` — 19 处 `print()` / 无 `logging` 调用
- `cli.py` — 全部使用 `print()` 和 `sys.stderr.write()`
- `bundle.py` — `warnings.warn()` 但无结构化日志
- 所有插件 `run()` 方法 — 零日志输出

**修复方案（2-3 小时）：**
1. 在 `pipeline.py` 中：`execute_step()` 调用 `log_event(logger, INFO, "step_start", ...)` / `"step_ok"` / `"step_failed"`
2. 在 `bundle.py` 中：写入/读取时记录 artifact 校验事件
3. 在 `cli.py` 中：`main()` 入口调用 `configure_logging(level=...)`，新增 `--log-level` / `--log-format` 标志
4. 在 `pyproject.toml` 的 `[tool.ruff.lint]` 中启用 `T20`（禁止 `print`）以强制执行

---

### 🔴 [N2] `_tenx.py` CSC→dense 用 Python for-loop 逐列重建 — 性能硬伤

**文件/行号：** `io/_tenx.py:91-93`

```python
dense = np.zeros((n_features, n_barcodes), dtype=data.dtype if data.size else float)
for col in range(n_barcodes):                    # ← 真实数据 10 万列
    start, stop = int(indptr[col]), int(indptr[col + 1])
    dense[indices[start:stop], col] = data[start:stop]
```

10x `.h5` 本身存储为 **CSC 稀疏格式**（`data + indices + indptr`），这正是 `scipy.sparse.csc_matrix` 的构造函数签名。当前代码用一个 Python 循环把稀疏矩阵还原为 dense——100k barcodes × 每次 Python 切片 = 数十秒到数分钟。而 `scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)` 是零拷贝的 C 级构造，然后 `.toarray()` 也走 C 循环。

**修复：**
```python
from scipy.sparse import csc_matrix
sparse = csc_matrix((data, indices, indptr), shape=(n_features, n_barcodes))
dense = sparse.toarray().T  # 与当前行为一致
```

**同等问题：** `model.py:247` (`from_anndata`) 无条件 `.toarray()` 对稀疏 AnnData 是 OOM 陷阱。应保留稀疏：`X = adata.X`，直接传给 `SpatialTable.__post_init__`（需 [N6] 稀疏支持）。

---

### 🟡 [N3] `_combat_nonparametric` 逐基因 Python 循环 — 2 万次 scipy 调用

**文件/行号：** `integration.py:192-204`

```python
for g in range(n_genes):            # ← 20,000 iterations
    ...
    ranks = rankdata(within_batch, method="average")
    ...
```

参数化 ComBat (`_combat_parametric`) 是完全向量化的；非参数版本则逐基因调用 `scipy.stats.rankdata`。对 2 万基因的真实数据集，这是 ~20,000 次 Python→C→Python 往返。应在文档/`assumptions` 中标注此为慢路径，或在方法描述中推荐参数化模式。`rankdata` 的 `axis` 参数（SciPy ≥1.9）支持按列操作，可完全向量化。

---

### 🟡 [N4] `_spatial_embedding()` coords=None 时静默退化 — 需显式警告

**文件/行号：** `sklearn_clustering.py:54-58`

```python
coords = data.spatial
if coords is not None and w > 0:
    nbr = neighborhood_mean(scores, coords, n_neighbors)
    return (1 - w) * zscore(scores) + w * zscore(nbr)
return zscore(scores)   # ← 静默退化：spatial_weight 被忽略
```

当用户设置了 `spatial_weight=0.3`（默认值）但 `obsm['spatial']` 缺失时，该方法**静默**退化为纯表达聚类。用户不会知道空间信息被忽略了，结果与预期不符。这是 REVIEW_2026_07_11.md 中已标记但未修复的问题。

**修复：** `if coords is None and w > 0:` 时发出 `warnings.warn("spatial_weight > 0 but obsm['spatial'] is missing; falling back to expression-only clustering")`

---

### 🟡 [N5] 报告模板 `templates/` 目录未在 wheel 中包含 — 安装后运行时崩溃

**文件/行号：** `report/report.py:14` — `_TEMPLATE_DIR = Path(__file__).parent / "templates"`

`pyproject.toml` 中 `[tool.hatch.build.targets.wheel]` 仅配置 `packages = ["src/histoweave"]`，**未声明** `package_data` 或 `include` 来包含 Jinja2 模板。当前 `pip install -e .`（开发安装）通过目录链接能找到模板，但 `pip install`（非编辑模式 / wheel 安装）**不会**包含 `src/histoweave/report/templates/report.html.j2`，`build_report()` 将抛出 `TemplateNotFound`。

**修复：** 在 `pyproject.toml` 中添加：
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/histoweave"]
include = ["src/histoweave/report/templates/**/*.j2"]
```

---

### 🟢 [N6] `ruf` ContextVar 可变默认值 — 一行修复

**文件/行号：** `logging.py:22`

```python
_CORRELATION: ContextVar[dict[str, str]] = ContextVar(
    "histoweave_log_correlation", default={}   # ← 可变默认值
)
```

`ContextVar` 的 `default` 参数不接受可变对象（语义与函数默认参数相同）。ruff B039 已报告。修复：`default=None`，在访问时做 `_CORRELATION.get() or {}`。

---

### 🟢 [N7] `ruff` import 排序 — 一行命令

**文件/行号：** `workflow/pipeline.py:8`

`import platform` 和 `import os` 应排在 `from __future__` 之后、`from dataclasses` 之前（stdlib → third-party → first-party 顺序）。`ruff check --fix` 自动修复。

---

### 🟢 [N8] `Annotation` 的 Z-score 跨全转录组计算 — 科学语义偏差

**文件/行号：** `annotate.py:42` — `Z = zscore(data.X)`

`zscore(data.X)` 对**全转录组**（所有基因）做标准化，然后仅取标记基因的均值。这意味着标记基因的 "score" 受到非标记基因表达量的间接影响（通过全转录组的均值和标准差）。正确的做法是：仅对标记基因子集做 z-score，或者 z-score 后按标记基因索引取子集。

这是 DEFECT_ANALYSIS.md [M4] 中标记的问题。影响等级取决于数据：在全转录组 z-score 下，高表达标记基因的相对优势被稀释。对科学结论有微妙但真实的影响。

---

## 三、中等投入的架构改进（1-3 天）

这些是之前报告已识别但优先级可调整的项目：

### [A1] `SpatialTable.X` 稀疏支持 — 数据规模的硬瓶颈

当前 `X: np.ndarray` 硬编码。10 万细胞 × 2 万基因 = 16 GB dense / ~1.6 GB sparse。这是 10× 的内存差距。

**方案（分阶段）：**
1. 第一阶段：`__post_init__` 接受 `np.ndarray | scipy.sparse.spmatrix`，内部存储保持原样
2. 第二阶段：在所有 `X.mean(axis=0)` 等操作处使用适配器（`np.asarray` / `.toarray()` 仅必要时调用）
3. 第三阶段：`from_anndata` 中移除强制 `.toarray()`

### [A2] KMeans 空簇处理 — 已知算法边界情况

`_math.py:76-79`：当某簇在迭代中变为空集时，旧中心被保留但远离数据，永远无法恢复。应重新初始化为离当前中心最远的点。

### [A3] `MethodSpec` 添加 `deprecated` 字段 — 为向后兼容做准备

当方法从 v0.1.0 升级到 v0.2.0 且结果不兼容时，需要通知用户。在 `MethodSpec` 中添加：
```python
deprecated: bool = False
replaced_by: str | None = None
```

---

## 四、迭代路线图

```
┌──────────────────────────────────────────────────────────────────┐
│ 本周（2-3 天）—— 高投入产出比的快速修复                           │
├──────────────────────────────────────────────────────────────────┤
│  □ N1: logging.py 集成到 pipeline + CLI（+ --log-level 标志）     │
│  □ N2: _tenx.py 用 scipy.sparse.csc_matrix 替换 Python for-loop │
│  □ N4: _spatial_embedding coords=None 添加警告                   │
│  □ N5: pyproject.toml 包含 Jinja2 模板（防止 wheel 崩溃）        │
│  □ N6+N7: 修复 2 个 ruff 问题（ContextVar + import 排序）        │
│  □ 在 pyproject.toml 的 ruff.lint.select 中添加 "T20"（禁 print） │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 下周（2-3 天）—— 性能与科学正确性                                 │
├──────────────────────────────────────────────────────────────────┤
│  □ N3: _combat_nonparametric 文档化性能警告 + 向量化优化          │
│  □ N8: Annotation zscore 改为仅对标记基因子集计算                 │
│  □ A2: KMeans 空簇重新初始化                                     │
│  □ A1 第一阶段: SpatialTable 接受稀疏 X（含 __post_init__ 适配）  │
│  □ 添加 SECURITY.md + CITATION.cff + .dockerignore + CODEOWNERS  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 两周内（1-2 周）—— 运营成熟度                                     │
├──────────────────────────────────────────────────────────────────┤
│  □ CI 构建 Docker 镜像并推送 ghcr.io                             │
│  □ 在 CI 中集成 pip-audit / safety check                         │
│  □ 为 mkdocstrings 配置 mkdocs.yml（API 参考文档）                │
│  □ 添加 1 个真实公共数据集 benchmark fixture（缓存 + SHA-256）    │
│  □ A3: MethodSpec deprecated/replaced_by 字段                    │
│  □ 在 ruff.lint.select 中添加 "T20" print 禁令                  │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ 一月内（3-4 周）—— 方法扩展                                       │
├──────────────────────────────────────────────────────────────────┤
│  □ 封装 Cellpose 2（细胞分割）→ 填补 SEGMENTATION 空缺           │
│  □ 封装 LIANA+（细胞间通讯）→ 填补 CCC 空缺                       │
│  □ 注册 ingestion 方法（visium_reader, xenium_reader）           │
│  □ 属性测试（Hypothesis/PBT）→ 提高测试有效性                     │
│  □ CLI argcomplete shell 补全                                    │
│  □ 至少 3 个完整示例（Visium pipeline, custom plugin, batch）     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 五、与之前审阅的差异

与 DEFECT_ANALYSIS.md / MATURITY_ROADMAP.md / REVIEW_2026_07_11.md 相比，本轮审阅的核心增量：

| 发现 | 之前审阅覆盖？ | 本轮判断 |
|------|:---:|------|
| `logging.py` 已实现零集成 | 模糊提到"无 logging" | **未发现模块已写好但没用** |
| `_tenx.py` Python for-loop CSC 重建 | ❌ 未发现 | **新发现 — 性能硬伤** |
| `_combat_nonparametric` 逐基因循环 | ❌ 未发现 | **新发现 — 2 万次 scipy 调用** |
| Jinja2 模板未入 wheel | ❌ 未发现 | **新发现 — 安装后运行时崩溃** |
| `_spatial_embedding` 静默退化 | ✅ 已标记 | 再次标记（仍未修复）|
| `Annotation` zscore 语义 | ✅ 已标记 | 补充了影响分析 |
| ContextVar 可变默认值 | ❌ 未发现 | **新发现 — ruff B039** |
| CSC→dense 在 `from_anndata` 中 | ✅ 已标记 | 补充了 `_tenx.py` 中的同等问题 |

---

## 六、结论

这轮优化将 HistoWeave 提升到了**"Phase-1 生产级工程基础"**。122 个测试通过、零 mypy 错误、完整的 CI 矩阵、结构化的错误处理和版本化 bundle 都是实实在在的质量信号。

**当前最紧迫的三件事（按顺序）：**
1. **logging.py 集成** — 模块已写好但未接入，这是纯"连接"工作（最高投入产出比）
2. **修复 _tenx.py + from_anndata 的稀疏→dense 强制转换** — 这是阻碍真实数据加载的性能/内存瓶颈
3. **修复 pyproject.toml 模板包含** — 否则 `pip install`（非编辑模式）的用户打开 HTML 报告会直接崩溃

这三项共需 4-6 小时，完成后 HistoWeave 即可安全地分发给 alpha 用户。
