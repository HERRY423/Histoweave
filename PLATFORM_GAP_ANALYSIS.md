# HistoWeave 对标成熟科研平台 — 深度缺陷分析

> **分析日期：** 2026-07-12
> **基线：** 126 tests / 0 failures / 0 mypy errors / 2 ruff nit / ~5500 LOC Python + Nextflow DSL2 + 2 Dockerfiles
> **分析范围：** 全量代码 + 前五轮审阅记录 + 对 7 个成熟竞品平台的深度比较
> **目标：** 回答"HistoWeave 距离一个能被领域采用的成熟平台还有多远？"

---

## 执行摘要

HistoWeave 当前处于一个**不对称的成熟状态**——工程纪律维度（类型安全、测试、CI、错误处理、数据完整性）已达到甚至超过多数已发表学术软件的水平，但**科学功能深度**和**用户体验完整性**与成熟平台之间存在系统性差距。

**对标 7 个成熟平台后，识别出 5 个维度共 28 项具体差距。**

关键判断：**HistoWeave 能否成功不取决于代码质量（已经足够好），而取决于 (1) 能否快速封装 15+ 真实方法 (2) 能否在方法推荐/性能地形图上建立不可替代的科学叙事 (3) 能否被非开发者（生物学家）独立使用。**

---

## 一、对标平台概览

### 1.1 对标矩阵

| 平台 | 发表期刊 | 年 | 核心定位 | HistoWeave 与之关系 |
|------|----------|---|---------|-----------------|
| **Squidpy** | Nature Methods | 2022 | 空间数据分析工具包（Python） | **互补** — HistoWeave 编排 Squidpy 方法 |
| **SpatialData** | Nature Methods | 2025 | 空间多模态数据标准 | **依赖** — 目标后端 |
| **Giotto Suite** | Genome Biology | 2021/2024 | 全栈空间分析平台（R/Python） | **最直接竞品** |
| **Seurat v5** | Nature Biotechnology | 2024 | 单细胞+空间分析（R） | **生态标准** — 方法的主要来源 |
| **scvi-tools** | Nature Biotechnology | 2022 | 概率模型框架（Python） | **深度竞品** — 方法深度基准 |
| **Open Problems** | Nature Methods | 2022 | 单细胞 benchmark | **模式对标** — HistoWeave 的空间版本 |
| **nf-core** | Nature Biotechnology | 2020 | 社区管道生态（Nextflow） | **运营对标** — 社区治理模型 |

### 1.2 各平台在六个维度的能力简表

| 维度 | Squidpy | SpatialData | Giotto | Seurat | scvi-tools | Open Problems | nf-core | **HistoWeave 当前** |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 方法数量 | ~15 | 0 (仅格式) | ~40+ | ~30+ | ~15 | 0 (仅评测) | 0 (仅管道) | **12 (6 toy)** |
| 数据标准 | ✅ AnnData | ✅ OME-Zarr | ❌ 自定义 | ✅ SeuratObj | ✅ AnnData | ❌ h5ad | ❌ 无 | ⚠️ SpatialTable (过渡) |
| 容器化执行 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ Docker/Singularity | ✅ Nextflow + Docker |
| 跨语言(R/Py) | ❌ Python | ❌ | ⚠️ v2 GiottoPy | ❌ R only | ❌ Python | ❌ | ✅ 任何语言 | ✅ R 桥接已设计 |
| Benchmark | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **核心功能** | ❌ | ✅ 3 任务 + 地形图 |
| 互动可视化 | ✅ napari | ✅ Vitessce | ✅ 自带 viewer | ⚠️ ggplot2 | ❌ | ❌ | ❌ | ❌ 静态 HTML |
| 方法推荐 | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ 仅排名 | ❌ | ✅ **独特能力** |
| 社区插件 | ❌ | ⚠️ IO only | ✅ modules | ⚠️ packages | ✅ external | ❌ | ✅ **核心模式** | ✅ entry_points |
| 真实数据教程 | ✅ ≥5 | ⚠️ notebooks | ✅ ≥10 | ✅ ≥10 | ✅ ≥5 | ✅ 多个 | ✅ ≥20 | ❌ 0 |
| CI/CD | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 二、维度一：科学功能完整性 — 🔴 最严重差距

### 2.1 方法封装深度：12 vs 竞争者的 30-40+

```
HistoWeave 方法构成（当前 12 个）：
├── 🟢 真实封装 (6): dbscan, agglomerative, spectral, gaussian_mixture,
│                     mean_shift, optics (均来自 sklearn)
├── 🟡 基础实现 (1): combat (参数化+非参数化，来自 scipy)
├── 🟡 分析框架 (1): morans_i (基于 numpy 的空间自相关)
├── 🟡 图分析   (1): spatial_graph (基于 networkx)
├── 🔴 玩具实现 (4): basic_qc, log1p_cp10k, kmeans, marker_score, marker_deconv
└── 🔴 R 桥接   (1): r_lognorm (仅 demo，依赖容器)
```

**对标差距：**

| 分析类别 | HistoWeave 当前 | Squidpy | Giotto | Seurat | **差距严重度** |
|----------|:--:|:--:|:--:|:--:|:--:|
| 域检测 | 7 (6 sklearn + 1 toy) | 3 (spatial neighbors, graph) | 10+ (HMRF, kmeans, Louvain, Leiden...) | 5+ | 🟡 数量足够，缺深度方法 |
| 去卷积 | 1 (toy marker) | 0 | 4 (RCTD, SPOTlight, stereoscope, DWLS) | 3 (RCTD, CARD, SPOTlight) | 🔴 严重不足 |
| SVG 检测 | 1 (Moran's I) | 1 (Moran's I) | 4 (SpatialDE, nnSVG, SPARK-X, Trendsceek) | 3 (SCT, MarkVariogram) | 🔴 严重不足 |
| 细胞分割 | **0** | 0 | 0 | 0 | 🔴 完全缺失 |
| 细胞通讯 | **0** | 1 (ligand-receptor) | 5+ (CellChat, NicheNet, COMMOT...) | 0 | 🔴 完全缺失 |
| 注释 | 1 (toy marker) | 0 | 3 (PAGE, ScType, RCTD) | 5+ (Azimuth, scmap...) | 🔴 严重不足 |
| 整合 | 1 (ComBat) | 0 | 3 (Harmony, scVI, LIGER) | 5+ (CCA, RPCA, Harmony...) | 🟡 中等差距 |
| 归一化 | 2 (toy, R demo) | 1 | 3 (sctransform, scran, SCnorm) | 3 (SCT, scran, LogNormalize) | 🟡 中等差距 |
| QC | 1 (toy) | 1 (calculate_qc_metrics) | 3 (filter, doublet, mito) | 3+ | 🟡 中等差距 |

### 2.2 核心缺失清单（16 个方法，按优先级排序）

**🔴 P0 — 必须在 Phase-1 完成（6 个）：**

| # | 方法 | 来源 | 类别 | 理由 |
|---|------|------|------|------|
| 1 | **cell2location** | Pyro/Bayesian | deconvolution | 空间去卷积的事实标准，Nature Biotech 2022 |
| 2 | **BANKSY** | R/Python | domain_detection | 首个被大规模 benchmark 验证为最佳的空间域方法 |
| 3 | **Cellpose 2** | PyTorch | segmentation | 细胞分割唯一选择，Nat Methods 2022 |
| 4 | **scANVI** | scvi-tools | annotation | 半监督注释，可处理未见细胞类型 |
| 5 | **SpatialDE** | Python | svg | 空间变异基因检测的开创性方法 |
| 6 | **SCT (sctransform)** | R/Seurat | normalization | Seurat 默认归一化，40k+ 引用 |

**🟡 P1 — Phase-1 后期或 Phase-2 早期（5 个）：**

| # | 方法 | 理由 |
|---|------|------|
| 7 | **LIANA+** | 细胞间通讯的统一框架，整合 16 种方法 |
| 8 | **RCTD** | 单细胞分辨率的去卷积，Robust 模式 |
| 9 | **STAGATE** | 图注意力自编码器，空间域检测深度学习代表 |
| 10 | **Harmony** | 批次效应校正，单细胞+空间通用 |
| 11 | **nnSVG** | 基于神经网络的 SVG，可扩展到大尺度数据 |

**🟢 P2 — Phase-2（5 个）：**

| # | 方法 | 理由 |
|---|------|------|
| 12-16 | SpaGCN, Tangram, CellChat, SPARK-X, CytoSPACE | 完善各类别的覆盖 |

### 2.3 数据集注册表：0 vs 竞争者的 5-15 个

**当前状态：** 所有 benchmark 使用合成数据（Voronoi blob），0 个真实公开数据集。

**对标差距：**

| 平台 | 真实 benchmark 数据集 | 覆盖平台 |
|------|:--:|------|
| Open Problems | 15+ | 10x, Drop-seq, Smart-seq2, CITE-seq |
| Squidpy | 5+ (自带示例) | Visium, MERFISH |
| Giotto | 10+ (tutorial) | Visium, MERFISH, CosMx, Xenium |
| **HistoWeave** | **0** | — |

**关键缺失：**
- DLPFC (Maynard et al., 2021) — 12 个人脑切片，手动标注的层结构（空间域检测的标准 benchmark）
- 10x Xenium 人乳腺癌 (Janesick et al., 2023) — 单细胞分辨率，377 基因 panel
- Visium 人淋巴结 (10x 官方) — 免疫微环境 benchmark
- Vizgen MERFISH 小鼠脑 (Allen Brain Atlas) — ~500 基因，细胞类型 ground truth
- Stereo-seq 小鼠胚胎 (Chen et al., 2022) — 亚细胞分辨率，发育轨迹

---

## 三、维度二：数据模型与互操作性 — 🟡 设计正确，执行滞后

### 3.1 SpatialTable vs SpatialData：过渡期已持续过长

**当前实现：** `SpatialTable` 是一个自定义容器（兼容 AnnData 但不兼容 SpatialData），包含 `X/obs/var/obsm/layers/images/shapes/uns`。

**架构设计是正确的：** 下游代码全部通过 `SpatialTable` API 编程，后端切换为 SpatialData 不破坏接口。

**但过渡期风险在累积：**
1. `from_anndata()` 强制 `.toarray()` 丢失稀疏性 — 真实数据的 OOM 风险
2. `shapes` 字段是 `dict[str, Any]` — 不兼容 GeoDataFrame/GeoParquet
3. `images` 字段是 `dict[str, np.ndarray]` — 不支持多尺度/金字塔图像
4. 没有 SpatialData 桥接（设计中有占位，无实现）
5. 没有 OME-Zarr 导入/导出

**对标差距：**

| 能力 | SpatialData (2025) | HistoWeave 当前 |
|------|:--:|:--:|
| 多模态（RNA+Protein+ATAC） | ✅ 原生支持 | ❌ 仅 RNA |
| 多尺度图像（金字塔） | ✅ OME-TIFF/OME-Zarr | ❌ 仅 dense array |
| 坐标变换矩阵 | ✅ 原生支持 | ❌ 无 |
| 惰性加载 | ✅ Dask + Zarr | ❌ 全量内存 |
| AnnData ↔ SpatialData 往返 | ✅ | ❌ 单向 (AnnData → SpatialTable) |
| 数据校验 | ✅ JSON Schema | ⚠️ 运行时 `__post_init__` |
| 云存储 (S3/GCS) | ✅ 原生支持 | ❌ 本地文件系统 |

### 3.2 稀疏矩阵支持：彻底缺失

**代码证据：**
- `SpatialTable.X` 类型注解是 `np.ndarray`（`model.py:109`）
- `__post_init__` 调用 `np.asarray(self.X)` 将稀疏转换为 dense
- `from_anndata` 调用 `.toarray()`（`model.py:247`）
- `_tenx.py` 用 Python for-loop 从 CSC 格式逐列重建 dense（`_tenx.py:91-93`）

**影响定量化：**
- 10 万细胞 × 2 万基因：dense ~16 GB vs sparse ~1.6 GB（10:1 内存差距）
- 50 万细胞（Xenium 全片）：dense ~80 GB（超出多数工作站 RAM）vs sparse ~8 GB

### 3.3 跨平台兼容性

| 平台 | HistoWeave 可读？ | 方法兼容？ |
|------|:--:|:--:|
| Squidpy (AnnData) | ✅ 通过 `from_anndata` | ✅ 插件系统设计即为此 |
| SpatialData | ❌ (未实现桥接) | ⚠️ 设计预留 |
| Seurat (R) | ❌ (需 R 桥接 + 容器) | ⚠️ R 桥接 demo 仅归一化 |
| Giotto | ❌ | ❌ |
| CELLxGENE / CZ CELLxGENE | ❌ | ❌ |
| HuBMAP / HTAN | ❌ | ❌ |

---

## 四、维度三：用户体验与可访问性 — 🔴 致命短板

### 4.1 可视化：无交互性

**当前状态：** 静态 HTML 报告 + 内联 SVG（`report/` 模块）。

**竞争者的可视化能力：**

| 平台 | 空间散点图 | 基因表达叠加 | 互动选择 | 3D 视图 | 多视图联动 | 图像配准 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| **Squidpy** | ✅ napari | ✅ | ✅ | ⚠️ | ✅ | ✅ napari |
| **Giotto** | ✅ 自带 WebGL | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Seurat** | ⚠️ ggplot2 静态 | ⚠️ 静态 | ❌ | ❌ | ❌ | ❌ |
| **SpatialData** | ✅ Vitessce | ✅ | ✅ | ❌ | ✅ | ✅ napari |
| **Loupe Browser** | ✅ 桌面应用 | ✅ | ✅ | ❌ | ⚠️ | ✅ 原生 |
| **HistoWeave** | ❌ 静态 SVG | ❌ | ❌ | ❌ | ❌ | ❌ |

**修复优先级：**
1. Vitessce 嵌入式集成（CDN 加载 + JSON view config）— 可在 1 周内完成最小可行版本
2. Plotly 交互式 HTML 报告 — 作为 Vitessce 的降级方案
3. napari-spatialdata 集成 — 更深度但需原生应用

### 4.2 文档：不完整的 4 页概念文档

**当前文档：**
- ✅ `index.md` — 项目概述
- ✅ `quickstart.md` — 快速开始
- ✅ `architecture.md` — 六层架构
- ✅ `concepts.md` — 核心概念
- ❌ 无 API 参考（`mkdocstrings` 已是依赖但未配置）
- ❌ 无插件开发教程（`plugin-template/` 有 README 但缺少从零到一教程）
- ❌ 无方法选择指南（"我的 Xenium 脑切片应该用什么方法？"）
- ❌ 无故障排除页面
- ❌ 无真实数据教程

**竞品文档对比：**

| 平台 | 文档类型 | 页面数 | API 参考 | 教程数 | 视频 |
|------|---------|:--:|:--:|:--:|:--:|
| Squidpy | ReadTheDocs | 50+ | ✅ 自动生成 | 10+ | 3+ |
| Seurat | 独立网站 | 100+ | ✅ 手工编写 | 20+ | 5+ |
| Giotto | 独立网站 | 80+ | ✅ | 15+ | 2+ |
| SpatialData | ReadTheDocs | 60+ | ✅ | 8+ | 0 |
| **HistoWeave** | MkDocs | **4** | ❌ | **1 (quickstart.py)** | **0** |

### 4.3 示例代码：1 个 quickstart.py

**当前：** `examples/quickstart.py` — 合成数据 → 管道 → 报告（~30 行）。

**需要的示例（按用户旅程排列）：**
1. ❌ 真实 Visium 数据端到端分析
2. ❌ 真实 Xenium 数据分析
3. ❌ 自定义插件开发
4. ❌ Benchmark + 方法选择
5. ❌ 批效应校正
6. ❌ Nextflow 云部署

### 4.4 CLI 可用性

| 功能 | HistoWeave | Squidpy | Giotto |
|------|:--:|:--:|:--:|
| 子命令 | ✅ 8 个 | ❌ (无 CLI) | ❌ (无 CLI) |
| Shell 补全 | ❌ | ❌ | ❌ |
| 进度条 | ❌ | ❌ | ⚠️ |
| 彩色输出 | ❌ | ❌ | ⚠️ |
| 错误恢复建议 | ⚠️ (部分) | ❌ | ⚠️ |
| `--dry-run` | ❌ | ❌ | ❌ |
| 配置文件 (YAML) | ❌ | ❌ | ✅ |

---

## 五、维度四：计算可扩展性 — 🟡 设计完整，生产未验证

### 5.1 管道执行器

**强项：**
- ✅ Nextflow DSL2 完整管道（7 进程，11 参数验证函数）
- ✅ 多 profile（local, docker, singularity, slurm, kubernetes, aws, conda）
- ✅ 进程内 runner 支持开发调试（`run_pipeline`）
- ✅ 结构化错误处理 + 部分成功/失败语义
- ✅ RunManifest 版本化完整重现信息

**差距：**
| 能力 | nf-core 标准 | HistoWeave 当前 |
|------|-------------|-------------|
| 多步骤并行 | ✅ DAG 自动 | ⚠️ 线性 DAG（无并行分支） |
| 失败恢复 (resume) | ✅ `-resume` | ✅ Nextflow 原生 |
| 云端执行 (AWS/GCP) | ✅ 完整支持 | ✅ profile 已配置，未测试 |
| 资源标签 (label) | ✅ 基于实际测试 | ⚠️ 基本标签（low_mem, domain_detection） |
| 输入校验 | ✅ nf-validation | ⚠️ 手动参数校验函数 |
| 多平台 CI 测试 | ✅ nf-core CI | ❌ Nextflow 管道无 CI 测试 |
| 模块化 (subworkflows) | ✅ DSL2 modules | ❌ 整体式 main.nf |
| 社区管道注册 | ✅ nf-core/modules | ❌ |

### 5.2 数据规模验证

**当前 benchmark 数据：**
- 合成数据：`n_cells=600, n_genes=50`（`make_synthetic` 默认参数）
- 真实数据：0 个公开数据集被测试

**规模测试金字塔（完全缺失）：**
```
❌ 单元级：500 cells × 50 genes（当前基准，耗时 ~0.1s）
❌ 中等：10k cells × 3k genes（Visium 典型，耗时 ~5s）
❌ 大规模：100k cells × 20k genes（Xenium 全片，耗时 ~60s）
❌ 图谱级：500k cells × 20k genes（多切片整合，耗时 ~5min）
❌ 生产级：1M+ cells（HuBMAP，需要 out-of-core）
```

### 5.3 容器镜像

**强项：**
- ✅ 两个 Dockerfile 存在且设计合理（histoweave-python, histoweave-r）
- ✅ R 容器已预装 scran, scater, SpatialExperiment, sctransform
- ✅ Python 容器包含完整空间依赖

**缺失：**
- ❌ 镜像从未构建/推送到 ghcr.io（CI 中无 docker build 步骤）
- ❌ 无 `.dockerignore`（已列在 MATURITY_ROADMAP 但未执行）
- ❌ 无版本标签策略（Docker tag 应跟随 PyPI 版本）
- ❌ 无镜像大小优化（多阶段构建、distroless 基础镜像）

---

## 六、维度五：社区与生态 — 🟡 基础框架齐全，生态为零

### 6.1 社区基础设施完成度

| 项目 | 状态 | 对标 (nf-core) |
|------|:--:|------|
| CODE_OF_CONDUCT.md | ✅ | ✅ |
| CONTRIBUTING.md | ✅ | ✅ |
| ROADMAP.md | ✅ | ✅ |
| CHANGELOG.md | ✅ | ✅ |
| CODEOWNERS | ❌ | ✅ |
| SECURITY.md | ❌ | ✅ |
| CITATION.cff | ❌ | ✅ |
| .dockerignore | ❌ | ✅ |
| Dependabot / Renovate | ❌ | ✅ |
| 预印本 (bioRxiv) | ❌ | ✅（nf-core 2019） |
| Twitter/社交平台 | ❌ | ✅ |
| 公开 leaderboard | ❌ | ✅ (Open Problems) |
| 培训工作坊/教程视频 | ❌ | ✅ (nf-core 有多次) |

### 6.2 外部贡献吸引力

**HistoWeave 的优势（对开发者）：**
- 清晰的插件接口 + `plugin-template/`
- entry_points 发现机制
- BSD-3 许可

**HistoWeave 的劣势（对开发者）：**
- 0 个外部插件 → 无社会证明
- 无真实方法封装 → "为什么不用 Squidpy 直接调？"
- 无公开 leaderboard → 贡献方法无可见性
- 无文档教程 → 无法独立上手

### 6.3 质量控制信号

**HistoWeave 当前的质量信号：**

| 信号 | 水平 | 竞品参考 |
|------|------|---------|
| 测试用例数 | 126 (优秀) | Squidpy ~150, Giotto ~200 |
| 测试/代码行比 | ~1:43 | 学术软件平均 1:80-200 |
| 类型覆盖 | 0 mypy 错误 | Squidpy 有 type hints 但不严格 |
| CI 矩阵 | 3 OS × 3 Python | 已超过多数竞品 |
| 性能回归 | ❌ 无 | ❌ 几乎所有竞品都没有 |
| 依赖审计 | ❌ | ❌ |
| 基准测试门禁 | ✅ `--min-score 0.90` | 竞品无此功能 |

---

## 七、综合性差距热力图

```
                        功能深度  数据互操作  用户体验  可扩展性  社区生态  工程质量
                        ────────  ─────────  ────────  ────────  ──────  ──────
Squidpy (Nat Meth 22)   ████░░    ████░░     ████░░    ██░░░░    ████░░  ███░░░
SpatialData (NatM 25)   ██░░░░    ██████     ████░░    ██░░░░    ████░░  ████░░
Giotto (Genome Biol)    ██████    ██░░░░     ████░░    ██░░░░    ████░░  ███░░░
Seurat v5 (Nat Bio 24)  █████░    ██░░░░     ███░░░    ██░░░░    ██████  ████░░
scvi-tools (Nat Bio 22) █████░    ████░░     ██░░░░    ███░░░    ████░░  █████░
Open Problems (NatM 22) ██░░░░    ██░░░░     ████░░    ██░░░░    ████░░  █████░
nf-core (Nat Bio 20)    ██░░░░    ██░░░░     ██░░░░    ██████    ██████  ██████
═══════════════════════════════════════════════════════════════════════════════
HistoWeave 当前            ██░░░░    ███░░░     █░░░░░    ████░░    ██░░░░  █████░
HistoWeave 目标 (12月)     ████░░    █████░     ████░░    █████░    ████░░  ██████
```

### 热力图解读

- **HistoWeave 的唯一超标领域：** 工程质量（测试/类型/CI）+ 可扩展性（Nextflow 设计）— 两个"防御性"维度远超多数竞品
- **HistoWeave 的致命短板：** 用户体验（可视化、文档、示例）— 比所有竞品都差
- **HistoWeave 的关键追赶方向：** 功能深度（方法数量/质量）— 需要 4-6 个月密集封装

---

## 八、最关键的风险

### 风险 1：平台在到达功能临界点之前就失去动力

**现状：** 12 个方法中 6 个是 sklearn wrapper（包装成本 ~每人日），剩下 4 个 toy 实现无法产出生物学见解。生物学家用户打开 HistoWeave 跑一遍 pipeline → 看到 kmeans 聚类结果 → "就这？我用 Seurat 点几下就出来了。" → 流失。

**缓解：** Phase-1 前必须封装至少 cell2location + BANKSY + SpatialDE 三个"杀手级"方法。这不是工程问题——是纯粹的资源投入。

### 风险 2：别人抢先发表"空间 GLUE/SuperGLUE"

**当前 HistoWeave 的独特叙事：** 空间方法性能地形图 + 集成推荐引擎。

**威胁：** Open Problems 团队或 scverse 社区可能扩展他们的 benchmark 到空间数据（他们已有所有基础设施）。如果他们在 12 个月内发布而 HistoWeave 还没预印本，HistoWeave 将失去最关键的差异化点。

**缓解：** 立即开始 5 数据集 × 10 方法的性能地形图试点，并在 arXiv/bioRxiv 上放预印本确立时间戳。

### 风险 3：SpatialData 成熟后架空 HistoWeave 的数据模型

**当前 SpatialTable 存在的原因：** SpatialData 的 OME-Zarr 对快速开发太重。

**威胁：** SpatialData 正在快速成熟（Nature Methods 2025）。如果 HistoWeave 在 6-9 个月内不切换到 SpatialData 后端，将面临两种困境：(a) 维护独立的数据模型，增加与生态的摩擦 (b) 匆忙切换导致 API 破坏。

**缓解：** Phase-1 中期完成 `SpatialTable` → `SpatialData` 后端切换（策略 A：薄包装器）。

---

## 九、建议：基于"资源杠杆率"的优先级排序

排序原则：每一项工作的**投入产出比** = 影响力 / 所需时间。

### 🔴 P0 — 如果只做 5 件事（3-6 个月）

| # | 工作项 | 所需时间 | 投入产出比 | 为什么 |
|---|--------|---------|-----------|--------|
| 1 | **封装 6 个核心真实方法** (cell2location, BANKSY, Cellpose, scANVI, SpatialDE, SCT) | 3-4 月 | ⭐⭐⭐⭐⭐ | 消除"toy"标签，产出生物学价值 |
| 2 | **Vitessce 交互式可视化集成** | 1-2 周 | ⭐⭐⭐⭐⭐ | 最显著的用户体验鸿沟 |
| 3 | **5 个真实公开数据集注册表** (DLPFC, Xenium Breast, Visium Lymph, MERFISH Brain, Stereo-seq Mouse) | 2-3 周 | ⭐⭐⭐⭐⭐ | Benchmark 的科学可信度基础 |
| 4 | **SpatialTable.X 稀疏支持** | 1 周 | ⭐⭐⭐⭐⭐ | 解除 16 GB 数据硬限制 |
| 5 | **完整文档 + 3 个教程** (API 参考, 真实 Visium 教程, 插件开发教程) | 2-3 周 | ⭐⭐⭐⭐ | 降低用户上手门槛 |

### 🟡 P1 — 下一步（6-9 个月）

| # | 工作项 | 依赖 P0 |
|---|--------|:--:|
| 6 | 5 数据集 × 10 方法性能地形图试点 → 预印本 | 需 #1, #3 |
| 7 | SpatialTable → SpatialData 后端切换 | 需 #4 |
| 8 | Docker 镜像 CI 构建+发布 ghcr.io | 无 |
| 9 | 方法版本淘汰机制 (deprecated/replaced_by) | 无 |
| 10 | 社区建设 (SECURITY.md, CITATION.cff, Dependabot, 外部合作) | 无 |

### 🟢 P2 — 长期完善（9-18 个月）

| # | 工作项 |
|---|--------|
| 11 | 15 数据集 × 30 方法完整地形图 |
| 12 | 大规模 benchmark (AWS spot, 100 万细胞) |
| 13 | 生物学案例研究（肿瘤微环境/脑组织） |
| 14 | 外部插件贡献 ≥5 |
| 15 | Nature Methods 投稿 |

---

## 十、与之前的审阅文件的增量

本分析**不是**对 DEFECT_ANALYSIS.md / REVIEW_2026_07_11_DEEP.md / MATURITY_ROADMAP.md / NATURE_LEVEL_STRATEGY.md 的重复。它们分别关注：

- **DEFECT_ANALYSIS.md** — 代码级缺陷（Phase-0 脚手架内部质量）
- **REVIEW_2026_07_11_DEEP.md** — 本轮新发现的代码问题（logging 集成、性能 bug、wheel 打包）
- **MATURITY_ROADMAP.md** — 从 Phase-1 到成熟产品的 5 维度 22 项优化
- **NATURE_LEVEL_STRATEGY.md** — 如何发 Nature 论文的科学创新路线图

**本分析的核心增量：**
1. **竞品量化对比表** — 7 个平台在 10 个维度上的能力矩阵（此前无）
2. **方法封装缺口清单** — 16 个缺失方法，每个有来源和优先级排序（此前是"需要 5 个真实方法"的模糊陈述）
3. **数据集注册表缺口** — 5 个具体公共数据集及用途（此前是"需要真实数据"的笼统要求）
4. **用户体验热力图** — 可视化/文档/示例与竞品的系统性差距量化
5. **风险矩阵** — 3 个具体风险及缓解策略
6. **SpatialData 过渡的紧迫性分析** — 给出 6-9 个月窗口期

---

## 结论

**HistoWeave 离"成熟科研平台"的距离 = 6 个核心方法 + 5 个真实数据集 + 1 个交互式可视化 + 1 套完整文档。**

这些不是技术难题——每一项都有明确的技术路径。它们是资源投入和执行力问题。

HistoWeave 的独特优势（工程纪律 + Nextflow 编排 + 方法推荐引擎 + 跨语言统一接口）在现有竞品中不存在。但优势窗口期有限——估计 12-18 个月内，Open Problems 或 scverse 社区就会填补"空间 benchmark"的空缺。

**建议的下一步行动（本周内）：**
1. 启动 cell2location 封装（最高影响力的单个方法）
2. 下载 DLPFC 数据集并创建 benchmark fixture
3. Vitessce CDN + JSON config 的最小集成
4. 配置 `mkdocstrings` 生成 API 参考

### 5.2 Update — computational scalability proof completed (2026-07-15)

The missing scale pyramid is now implemented as `histoweave scale` and backed by a
single-node 16-vCPU/64-GB execution over 1k, 10k, 100k, 500k and 1M cells (2,000 genes,
5% CSR density) across 30 pure-compute methods. The checked-in `scalability_proof/` artifacts
record all 150 method-scale cells, including 104 successful runs, 20 memory ceilings and 26
intentional post-ceiling skips. Ten methods reached 1M cells; full methodology, per-method
boundaries, CSV measurements and editable SVG/PNG figures are included rather than inferred.
