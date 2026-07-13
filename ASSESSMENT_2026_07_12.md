# HistoWeave 深度评估 — 2026-07-12

> **基线：** 0 ruff / 0 mypy errors in 40 source files / 126 tests (122 passed, 4 skipped)
> **方法总数：** 15（8 类别覆盖，3 空类别）| **评估范围：** 全量代码 + 6 份审阅文档交叉验证

---

## 执行摘要

HistoWeave 当前处于**"高质量工程基础 + 零科学深度"**的非对称状态。自上次深度审阅（7月11日）以来，2 个缺陷被修复（`_tenx.py` scipy.sparse 优化、`logging.py` 部分集成），6 份审阅文档被生成。但**没有任何新方法封装、没有稀疏矩阵支持、没有架构改进、没有真实数据集、没有交互式可视化**——这些是功能深度的硬瓶颈。

**一句话诊断：代码已经足够好，现在是停止审阅、开始写方法封装代码的时候。**

---

## 一、缺陷修复追踪（跨 6 份审阅文档）

### 已修复

| 来源 | 缺陷 | 状态 | 证据 |
|------|------|:--:|------|
| DEEP [N2] | `_tenx.py` Python for-loop CSC重建 | ✅ 已修复 | `_tenx.py:92-94` — `scipy.sparse.csc_matrix((data, indices, indptr), shape=...)` |
| DEEP [N1] | `logging.py` 零集成 | ⚠️ 部分 | 3 文件 import (`pipeline.py`, `cli.py`, `bundle.py`)，但 50 处 `print()` 未迁移 |
| ruff | 2 个 issue | ✅ 已修复 | `ruff check src/` → All checks passed! |
| DEEP [N5] | Jinja2 模板未入 wheel | ⚠️ 不确定 | `pyproject.toml` 有 `artifacts = [...]` 但 `artifacts` 是 Hatch build artifacts（用于生成文件），不是 `include`（用于静态文件）。非编辑安装可能仍然崩溃 |

### 未修复（按严重程度）

| 来源 | 缺陷 | 严重度 | 位置 | 导致问题 |
|------|------|:--:|------|------|
| DEFECT [H1] | `SpatialTable.X` 不接受稀疏矩阵 | 🔴 阻塞 | `model.py:109,119` | 真实数据 OOM（10 万细胞=16GB vs 1.6GB sparse） |
| DEFECT [H4] | 管道步骤间无错误隔离（已部分修复，`on_error="continue"` 已添加） | 🟡 | `pipeline.py` | 已修复 |
| DEFECT [H5] | 报告仅静态 HTML | 🔴 阻塞 | `report/` | 无可交互可视化 |
| DEFECT [B1] | Nextflow 管道未 CI 测试 | 🟡 | `workflows/nextflow/` | 管道是否正确运行未经 CI 验证 |
| DEFECT [B3] | 真实方法封装为零 | 🔴 阻塞 | 全局 | 无法产出生物学分析 |
| DEFECT [B4] | 无真实数据集成测试 | 🔴 阻塞 | `tests/` | 无法保证真实数据兼容性 |
| DEEP [H5] | `from_anndata` 强制 `.toarray()` | 🔴 | `model.py:247` | AnnData 稀疏 → dense 转换造成 OOM |
| DEEP [N8] | Annotation z-score 跨全转录组 | 🟡 | `annotate.py:42` | 科学语义偏差 |
| DEEP [N3] | `_combat_nonparametric` 逐基因循环 | 🟡 | `integration.py:192-204` | 20000 次 scipy 调用 |
| MATURITY [S2] | 6 个 toy 方法未替换 | 🔴 | 多个文件 | 用户信任度为零 |
| FUNCTIONAL | 3 个类别完全空缺 | 🔴 | — | ingestion, segmentation, ccc |

---

## 二、当前状态量化

### 方法清单（15 个注册方法）

```
annotation         (1):  marker_score ⚪ toy
deconvolution      (1):  marker_deconv ⚪ toy
domain_detection   (7):  kmeans ⚪ toy | gaussian_mixture 🟢 sklearn | spectral 🟢 sklearn
                         agglomerative 🟢 sklearn | dbscan 🟢 sklearn | mean_shift 🟢 sklearn
                         optics 🟢 sklearn
integration        (1):  combat 🟡 pure numpy (real algorithm, manual implementation)
neighborhood       (1):  spatial_graph 🟡 networkx (real library, basic metrics)
normalization      (2):  log1p_cp10k ⚪ toy | r_lognorm 🟡 R bridge (demo)
qc                 (1):  basic_qc ⚪ toy
svg                (1):  morans_i 🟡 numpy+scipy (real statistic, manual impl)
─────────────────────────────────────────────
🟢 真实库封装: 6 (sklearn)   🟡 真实算法自行实现: 3   ⚪ 玩具实现: 5   🔵 R 桥接: 1
```

### 空类别（3/11）

```
ingestion — readers.py 有 VisiumReader/XeniumReader/StereoSeqReader 但未注册为 Method
segmentation — 零实现, 零存根
ccc — 零实现, 零存根
```

### 工程质量快照

| 指标 | 当前值 | 上一轮 (7/11) | 趋势 |
|------|:--:|:--:|:--:|
| ruff errors | **0** | 2 | ✅ 改善 |
| mypy errors | **0** (40 files) | 0 (35 files) | ✅ 维持 +5 files |
| print() calls | **50** | ~50 | ⚸ 持平 |
| 测试总数 | 126 | 122 (+4 new?) | ✅ 微增 |
| 跳过测试 | 4 | 3 (+1?) | ⚸ 持平 |
| src files | 40 | 35 | ✅ +5 (benchmark expansion) |
| 真实方法封装 | 0 | 0 | ⚸ 持平 |
| 真实数据集 | 0 | 0 | ⚸ 持平 |

---

## 三、跨审阅文档的"已知但未修复"清单

以下是从 6 份审阅文档中提取的**共同标记项**——被标记了多轮但从未修复：

### 标记了 3+ 次的项：

1. **SpatialTable.X 稀疏支持** (DEFECT [H1], MATURITY [E1], FUNCTIONAL A2, DEEP [A1])
   - 状态：❌ 未修复，`np.asarray(X)` 强制 dense
   - 影响：所有真实数据分析的硬瓶颈

2. **from_anndata 强制 .toarray()** (DEFECT [H1], DEEP [A1], DEEP [N2])
   - 状态：❌ 未修复，`model.py:247`
   - 影响：加载真实 AnnData 时 OOM

3. **真实方法封装为零** (DEFECT [B3], MATURITY [S2], NATURE §2, FUNCTIONAL B)
   - 状态：❌ 未修复，0 个新方法
   - 影响：平台无生物学价值

4. **交互式可视化缺失** (DEFECT [H5], MATURITY [U-], NATURE §2, FUNCTIONAL)
   - 状态：❌ 未修复，仍是静态 SVG
   - 影响：用户体验致命短板

### 标记了 2 次的项：

5. **Annotation z-score 语义** (DEFECT [M4], DEEP [N8])
6. **KMeans 空簇处理** (DEFECT [H3], DEEP [A2])
7. **Ingestion 方法未注册** (DEFECT [L4], MATURITY [S1])
8. **数据集注册表** (MATURITY [S3], NATURE §2, FUNCTIONAL C)

### 被修复的项：

1. **Nextflow 管道** (DEFECT [B1]) → ✅
2. **容器 Dockerfile** (DEFECT [B2]) → ✅
3. **_tenx.py 性能** (DEEP [N2]) → ✅
4. **ruff/mypy** → ✅

---

## 四、关键路径分析：从当前状态到"首次生物学分析"

```
当前状态                                    目标状态
────────                                   ────────
0 个真实方法                                1 个真实方法可用
0 个真实数据集                              1 个真实数据集
静态 HTML 报告                              交互式可视化
dense-only 数据模型                         sparse-capable
                                          +
首次端到端分析完成

                    ┌─────────────────────────────────────┐
                    │ 关键路径（必须按顺序完成）            │
                    │                                     │
                    │ 1. 修复 sparse X + from_anndata     │
                    │    ↓ (解锁真实数据加载)               │
                    │ 2. 下载 1 个真实数据集 (DLPFC)       │
                    │    ↓ (有了测试数据)                   │
                    │ 3. 封装 1 个真实方法 (BANKSY)        │
                    │    ↓ (有了生物学分析能力)             │
                    │ 4. 端到端运行: DLPFC → BANKSY        │
                    │    ↓ (验证管道完整性)                 │
                    │ 5. 添加 Vitessce 交互式可视化        │
                    │    ↓ (用户可以看到结果)               │
                    │ ✅ 最小可行生物学分析就绪             │
                    └─────────────────────────────────────┘
```

**关键路径总工作量估算：** 5-7 天（单人全职）

---

## 五、下一步行动（优先级排序）

### 🔴 P0 — 本周必须完成（2-3 天）

| # | 行动 | 文件 | 工作量 | 验证方法 |
|---|------|------|:--:|------|
| 1 | **SpatialTable.X 接受 sparse** | `model.py:109,119-120` | 2h | `csr_matrix` 构造 `SpatialTable` 不崩溃 |
| 2 | **from_anndata 保留稀疏** | `model.py:247` | 0.5h | 稀疏 AnnData → SpatialTable → X 仍是 sparse |
| 3 | **下载 DLPFC 数据集** | `tests/data/dlpfc_151507/` | 1h | `SpatialTable` 加载成功，验证 shape |
| 4 | **修复 template wheel 包含** | `pyproject.toml` | 0.5h | `pip install` (非编辑) → `build_report()` 不崩溃 |
| 5 | **迁移 50 处 print() → logging** | `cli.py` (48), `pipeline.py` (2) | 2h | 所有输出通过 logger |

### 🟡 P1 — 下周（3-5 天）

| # | 行动 | 文件 | 工作量 |
|---|------|------|:--:|
| 6 | **封装 BANKSY（首个真实空间域检测方法）** | 新建 `plugins/builtin/_banksy.py` | 2 天 |
| 7 | **DLPFC benchmark fixture** | `tests/test_real_data.py` + `benchmark/harness.py` | 0.5 天 |
| 8 | **Vitessce 最小集成** | `report/report.py` (CDN + view config JSON) | 1 天 |
| 9 | **注册 ingestion 方法** | 新建 `plugins/builtin/ingestion.py` | 0.5 天 |
| 10 | **添加 MethodSpec.maturity 字段** | `interfaces.py` | 1h |

### 🟢 P2 — 两周内（3-5 天）

| # | 行动 |
|---|------|
| 11 | 封装 cell2location（去卷积） |
| 12 | 封装 SpatialDE（SVG） |
| 13 | SklearnClusterWrapper 通用基类（重构现有 7 个方法） |
| 14 | Method.run_on_anndata 桥接钩子 |
| 15 | 参数化方法烟雾测试模板 |

---

## 六、概率评估

**自问：HistoWeave 在当前状态下，能否在 3 个月内吸引外部贡献者？**

**答案：概率 < 5%。**

原因链：
1. 外部开发者看一个工具 → 先看 README → 看到 "pre-alpha scaffold" → 失去兴趣
2. 如果他们坚持看到了 Plugin API → 发现所有方法都是 toy → "没有我用不了的东西"
3. 如果他们想贡献方法 → 没有真实方法封装示例 → 不知道从哪开始
4. 没有公开 benchmark leaderboard → 贡献方法无可见性
5. 没有 CITATION.cff → "这个项目发过论文吗？"

**转折点：**
- 封装 **3 个高影响力方法** (BANKSY + cell2location + SpatialDE) → 概率升至 25%
- + **1 个真实数据集 benchmark** + **公开 leaderboard** → 概率升至 50%
- + **预印本 (bioRxiv)** → 概率升至 70%

---

## 七、与前一轮审阅的差异

这次评估与之前 6 份审阅文档的**核心增量**：

1. **修复状态追踪** — 首次系统性交叉验证了 6 份文档中的所有缺陷项，区分了 "已修复"、"部分修复"、"标记 n 次仍未修复"
2. **发现 `_tenx.py` 已修复** — 之前 DEEP 审阅标记的 N2 性能 bug 被静默修复（使用 `scipy.sparse.csc_matrix`）
3. **发现 `logging.py` 部分集成** — 3 个文件导入，但 50 处 `print()` 未迁移
4. **发现方法总数是 15 而非 12** — `spatial_graph` 和 `spatial_svg` 在之前的某些文档中被漏计
5. **"关键路径"分析** — 首次识别出到达"首次生物学分析"所需的 5 个最小步骤
6. **"已标记但未修复"清单** — 暴露了 4 个被标记了 3+ 次但从未修复的项，这是执行力瓶颈的信号

---

## 结论

**HistoWeave 的问题不再是"我们不知道要做什么"——6 份审阅文档已经提供了详尽的路线图。问题是"我们还没开始做"。**

三个最紧迫的行动（按顺序）：
1. **修复 sparse X + from_anndata**（2.5h，解锁所有真实数据分析）
2. **下载 DLPFC + 跑通首个真实数据管道**（3h，建立可信度）
3. **封装 BANKSY**（2 天，首个真实方法的生物学价值证明）

这三个行动完成后，HistoWeave 从 "高质量脚手架" 变为 "有生物学分析能力的平台"。这是质的飞跃。
