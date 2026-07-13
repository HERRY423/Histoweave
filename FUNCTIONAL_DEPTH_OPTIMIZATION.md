# HistoWeave 功能深度 — 优化方向与执行路线图

> **日期：** 2026-07-12
> **当前基线：** 12 方法（6 sklearn wrapper + 1 ComBat + 1 Moran's I + 4 toy + 1 R demo）
> **目标：** 达到 20+ 真实方法、11/11 类别覆盖、≥3 真实数据集验证
> **核心原则：** 每项优化必须可量化、有明确完成标准、能独立验证

---

## 一、核心瓶颈诊断

在写任何代码之前，先识别**限制方法封装速度的瓶颈是什么**。

### 当前写一个新方法的成本分解

以 `SpectralDomains`（`sklearn_clustering.py:177-225`，49 行）为例：

```
def run(self, data):
    data = data.copy()                                    # 1 行 — 模板
    embedding = _spatial_embedding(data, ...)              # 6 行 — 共享辅助函数
    labels = SpectralClustering(...).fit_predict(...)      # 3 行 — 实际逻辑 ⭐
    data.obs[...] = _categorical_labels(labels)            # 1 行 — 模板
    data.obsm["X_pca"] = embedding                        # 1 行 — 模板
    return self.finalize(data, step="domain_detection")    # 1 行 — 模板

+ 13 行 spec 声明
+ 25 行 class + docstring 模板
──────────
实际差异化逻辑: 3 行 (6%)
模板/仪式代码:     46 行 (94%)
```

**核心发现：94% 的代码是模板。** 这意味着当前的方法封装效率极低——每增加一个方法，大部分工作是在重复粘贴。

### 优化后的目标成本

| 方法类型 | 当前成本 | 优化后目标 | 杠杆手段 |
|----------|:--:|:--:|------|
| sklearn 聚类 | ~50 行/方法 | **~15 行/方法** | 通用 wrapper 基类 |
| Python 库方法 | 未建立 | **~40 行/方法** | `SpatialTable ↔ AnnData` 自动桥接 |
| R/Bioconductor | ~120 行 (r_demo) | **~45 行/方法** | R 桥接模板基类 |
| 深度学习 | 未建立 | **~80 行/方法** | GPU 检测 + checkpoint 模板 |

---

## 二、优化方向 A：方法封装效率（架构改进）

### A1. 🔴 建立 `SklearnClusterWrapper` 通用基类（投入：2 小时，收益：6 方法 → 永久收益）

**现状问题：** 7 个 sklearn 域检测方法（kmeans, dbscan, agglomerative, spectral, gmm, mean_shift, optics）各自独立实现，共享 90% 的代码（`_spatial_embedding` → 聚类器 → `_categorical_labels`）。

**优化方案：** 创建一个参数化的工厂基类，将具体 sklearn 聚类器的差异压缩为两个参数：聚类器类 + kwargs 映射。

```python
# 优化前: 每个方法 ~50 行
@register
class SpectralDomains(Method):
    spec = MethodSpec(name="spectral", ...13 行...)
    def run(self, data): ...36 行...

# 优化后: 每个方法 ~12 行
@register_sklearn_clusterer(
    name="spectral",
    clusterer_cls="sklearn.cluster.SpectralClustering",
    static_kwargs={"affinity": "nearest_neighbors", "assign_labels": "kmeans"},
    param_mapping={"n_domains": "n_clusters"},
    summary="Spectral clustering on spatial k-NN graph.",
    wraps="sklearn.cluster.SpectralClustering",
)
class SpectralDomains(SklearnClusterMethod):
    pass  # 所有逻辑由基类提供
```

**执行步骤：**
1. 创建 `plugins/builtin/_sklearn_base.py`，实现 `SklearnClusterMethod` 基类 + `register_sklearn_clusterer` 工厂装饰器
2. 将 7 个现有方法迁移到新基类（每个 ~5 分钟）
3. 验证：所有现有测试通过，无行为变化

**收益量化：**
- 现有方法维护成本降低 70%
- 新增 sklearn 方法成本从 ~50 行降到 ~12 行
- 参数验证、错误处理、provenance 由基类统一保证

### A2. 🔴 建立 `SpatialDataBridge` 自动转换层（投入：4 小时，收益：所有 Python 方法）

**现状问题：** 每个封装 Python 库的方法都需要手动写 `SpatialTable → AnnData` 转换和反向转换。`from_anndata` 强制 `.toarray()` 丢失稀疏性。`obsm['spatial']` 坐标需要手动迁移。

**优化方案：** 在 `Method` 基类上添加一个 `run_on_anndata` 可选钩子 + 自动桥接。

```python
class Method(ABC):
    """..."""

    def run(self, data: SpatialTable) -> SpatialTable:
        """默认实现：尝试调用 run_on_anndata 钩子"""
        if hasattr(self, 'run_on_anndata'):
            adata = data.to_anndata()
            result_adata = self.run_on_anndata(adata)
            result = SpatialTable.from_anndata(result_adata)
            # 保留空间层
            result.images = data.images
            result.shapes = data.shapes
            result.obsm.setdefault('spatial', data.spatial)
            return self.finalize(result, step=self.spec.category.value)
        raise NotImplementedError("子类必须实现 run 或 run_on_anndata")

    # run_on_anndata 是可选的 — 如果不实现，回退到 run
```

**收益：** 封装 scanpy/squidpy/scvi-tools 的方法只需实现 `run_on_anndata(self, adata: AnnData) -> AnnData`，所有桥接逻辑由基类处理。从 ~80 行降到 ~40 行。

**同时修复 `from_anndata` 的稀疏强制转换：**
```python
# model.py:247 — 当前
X = adata.X
X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

# 优化后
X = adata.X
if hasattr(X, "toarray"):
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X  # 保留稀疏，由 __post_init__ 处理
    else:
        X = np.asarray(X)
else:
    X = np.asarray(X)
```

### A3. 🔴 建立 `RContainerMethod` 基类（投入：3 小时，收益：所有 R 方法）

**现状问题：** `r_demo.py` 的 119 行中，仅 3 行是与 R 脚本的差异化逻辑；其余 116 行是 h5ad 读写 + subprocess 调用 + 路径查找 + 错误处理。每个新 R 方法需要复制这 116 行。

**优化方案：** 提取 `RContainerMethod` 基类：

```python
class RContainerMethod(Method, ABC):
    """Base for containerized R/Bioconductor methods.

    子类只需定义:
      - spec (包含 R 脚本路径)
      - _build_r_args(data, params) -> list[str]  # CLI 参数
      - _validate_r_output(adata: AnnData) -> None  # 可选的后置校验
    """

    r_script: str  # 容器内路径，如 "/usr/local/bin/histoweave-banksy.R"
    r_packages: tuple[str, ...] = ()  # 预检清单

    def run(self, data: SpatialTable) -> SpatialTable:
        # 116 行模板逻辑：h5ad 读写 + subprocess + 错误处理
        ...
```

**收益：** 每个 R 方法从 ~120 行降到 ~35 行（仅 spec + R 参数构建）。

### A4. 🟡 建立参数化测试模板（投入：2 小时，收益：每个新方法自动获得测试）

**现状问题：** 每个新方法需要手动写测试。大部分测试是相同的模式（"方法是否返回 SpatialTable？"、"结果是否包含预期列？"、"方法是否确定性？"）。

**优化方案：** 在 `conftest.py` 中添加参数化 fixture：

```python
# conftest.py
METHOD_SMOKE_TESTS = [
    ("qc", "basic_qc", {}),
    ("normalization", "log1p_cp10k", {}),
    ("domain_detection", "kmeans", {"n_domains": 3}),
    # 新增方法只需在此列表添加一行
]

@pytest.mark.parametrize("category,method,params", METHOD_SMOKE_TESTS)
def test_method_smoke(category, method, params, synthetic_data):
    """每个注册方法必须通过烟雾测试"""
    m = create_method(category, method, **params)
    result = m.run(synthetic_data.copy())
    assert isinstance(result, SpatialTable)
    assert len(result.provenance) > 0
```

---

## 三、优化方向 B：方法优先级队列（18 个方法，按投入产出比排序）

### B1. 第一批（本月，4 周）— 6 个方法，每个 ~2-4 天

这些是**用户可见价值最高 + 技术路径最清晰**的方法。完成后 HistoWeave 即可产出生物学有意义的分析。

| # | 方法 | 类别 | 语言 | 难度 | 工作量 | 输入 | 输出 |
|---|------|------|------|:--:|:--:|------|------|
| 1 | **cell2location** | deconvolution | Python | 🔴 高 | 4 天 | `run_on_anndata` 桥接 | `obsm['proportions']` |
| 2 | **BANKSY** | domain_detection | R/Python | 🟡 中 | 2 天 | `RContainerMethod` | `obs['domain']` |
| 3 | **SpatialDE** | svg | Python | 🟡 中 | 2 天 | `run_on_anndata` 桥接 | `var['spatialde_pval']` |
| 4 | **scANVI** | annotation | Python | 🟡 中 | 3 天 | `RunOnAnnData` 桥接 | `obs['cell_type']` |
| 5 | **SCT (sctransform)** | normalization | R | 🟢 低 | 1 天 | `RContainerMethod` | 替换 `X` |
| 6 | **Cellpose 2** | segmentation | Python | 🟡 中 | 3 天 | 直接调用 | `obs['cell_mask']` |

**完成标准：** 全部 6 个方法通过烟雾测试 + 至少 1 个方法在真实 DLPFC 数据集上验证。

### B2. 第二批（第 2-3 月）— 6 个方法，每个 ~1-3 天

| # | 方法 | 类别 | 难度 | 工作量 | 理由 |
|---|------|------|:--:|:--:|------|
| 7 | **RCTD** | deconvolution | 🟡 中 | 2 天 | 单细胞分辨率去卷积，Robust 模式 |
| 8 | **Tangram** | annotation+deconv | 🟡 中 | 2 天 | 深度学习空间映射，一石二鸟 |
| 9 | **LIANA+** | ccc | 🟡 中 | 3 天 | 统一 16 种 CCC 方法的框架 |
| 10 | **nnSVG** | svg | 🟡 中 | 2 天 | 可扩展的神经网络 SVG |
| 11 | **Harmony** | integration | 🟢 低 | 1 天 | 最广泛使用的整合方法 |
| 12 | **STAGATE** | domain_detection | 🔴 高 | 3 天 | 图注意力自编码器，领域 SOTA |

### B3. 第三批（第 4-6 月）— 6 个方法

| # | 方法 | 类别 | 备注 |
|---|------|------|------|
| 13 | **SpaGCN** | domain_detection | 图卷积空间域检测 |
| 14 | **DestVI** | deconvolution | scvi-tools 生态 |
| 15 | **COMMOT** | ccc | 最优传输 CCC |
| 16 | **SPARK-X** | svg | 非参数 SVG，速度快 |
| 17 | **CellTypist** | annotation | 免疫细胞注释标准 |
| 18 | **BBKNN** | integration | 图基整合 |

### 各阶段里程碑

```
第 1 月末 ─── 12 → 18 方法, 9/11 类别覆盖
              cell2location + BANKSY + SpatialDE 产出首个生物学见解
              A1/A2/A3 架构优化完成（后续方法封装速度翻倍）

第 3 月末 ─── 18 → 24 方法, 10/11 类别覆盖
              首个 DLPFC 真实数据 benchmark
              推荐引擎可用于真实用户数据

第 6 月末 ─── 24 → 30 方法, 11/11 类别覆盖
              3 个真实数据集注册表
              社区贡献插件 ≥2
```

---

## 四、优化方向 C：数据集注册表（从 0 到 5 个真实数据集）

### C1. 最小可行数据集注册表设计（投入：3 天）

```python
# datasets/registry.py — 版本化、可缓存、可校验的数据集注册表

@dataclass
class DatasetEntry:
    """One registered benchmark dataset."""
    name: str                    # "dlpfc_151507"
    description: str             # "DLPFC slice 151507 (Maynard et al. 2021)"
    url: str                     # 下载 URL
    sha256: str                  # 校验和
    assay: str                   # "visium", "xenium", ...
    tissue: str                  # "brain", "tumor", ...
    species: str                 # "human", "mouse"
    n_obs: int                   # spot/cell 数量
    n_vars: int                  # 基因数量
    ground_truth: dict[str, str] # {"domain_truth": "obs column", ...}
    license: str                 # "CC-BY 4.0"
    paper_doi: str               # 原始发表 DOI

    def download(self, cache_dir: Path) -> SpatialTable:
        """下载（如果未缓存）→ 校验 → 加载为 SpatialTable"""
        ...
```

### C2. 5 个优先数据集

| # | 数据集 | 平台 | 组织 | 用途 | 大小 | 优先级 |
|---|--------|------|------|------|------|:--:|
| 1 | **DLPFC** (Maynard 2021) | Visium | 人脑前额叶 | 空间域检测 gold standard (12 slices × manual layer annotation) | ~50MB/slice | 🔴 P0 |
| 2 | **10x Xenium Breast** (Janesick 2023) | Xenium | 人乳腺癌 | 单细胞分辨率域检测 + 分割 | ~500MB | 🔴 P0 |
| 3 | **Visium Human Lymph Node** (10x demo) | Visium | 人淋巴结 | 免疫微环境 deconvolution benchmark | ~200MB | 🟡 P1 |
| 4 | **MERFISH Mouse Brain** (Allen Institute) | MERFISH | 小鼠脑 | 跨平台比较 + cell type annotation | ~1GB | 🟡 P1 |
| 5 | **Stereo-seq Mouse Embryo** (Chen 2022) | Stereo-seq | 小鼠胚胎 | 发育轨迹 + 亚细胞分辨率 scaling test | ~2GB | 🟢 P2 |

### C3. 数据集特征提取的集成

每个数据集入库时自动调用 `benchmark/features.py:extract_features()` 提取 19 维特征向量，存入注册表。这确保：
- 推荐引擎有新数据集时无需重新提取特征
- 特征向量版本化（数据集重新处理时特征自动更新）
- `feature_dataframe()` 可直接从注册表生成

---

## 五、优化方向 D：空类别补全

当前 3 个类别完全无实现：

| 类别 | 现状 | 最低交付 | 所需工作量 |
|------|------|---------|:--:|
| **SEGMENTATION** | 0 方法 | Cellpose 2 wrapper | 3 天 |
| **CELL_CELL_COMMUNICATION** | 0 方法 | LIANA+ wrapper | 3 天 |
| **INGESTION** | `io/readers.py` 有代码但未注册为 Method | 注册 3 个 reader 为 Method | 1 天 |

**INGESTION 的修复最简单（1 天）：**

```python
# 在 io/readers.py 或新建 plugins/builtin/ingestion.py

@register
class VisiumIngestion(Method):
    spec = MethodSpec(
        name="visium_reader",
        category=MethodCategory.INGESTION,
        version="0.1.0",
        summary="Ingest 10x Visium Space Ranger output.",
        params=(
            ParamSpec("path", "str", "", "Space Ranger output directory."),
            ParamSpec("engine", "str", "native", "native or spatialdata."),
        ),
        assays=("visium",),
    )
    def run(self, data: SpatialTable) -> SpatialTable:
        # 这是一个特殊情况 — Ingestion 不接收 SpatialTable 输入
        # 需要特殊处理或改造接口
        ...
```

**注意：** Ingestion 类方法与其他方法有根本区别——它们不需要输入 `SpatialTable`（数据源是文件系统），这是插件接口设计的边缘情况。短期方案：在 CLI 中直接调用 `io.read()` 而非通过 plugin 接口；长期方案：在 `Method` 接口中增加 `is_source: bool = False` 标记。

---

## 六、优化方向 E：方法质量分级体系

当前所有方法平铺在注册表中，用户无从区分"cell2location（Nature Biotech, 2000+ 引用）"和"kmeans（教学实现）"。

### E1. 为 `MethodSpec` 添加质量标记

```python
@dataclass(frozen=True)
class MethodSpec:
    # ... 现有字段 ...

    # 新增字段：
    maturity: str = "toy"         # "toy" | "community" | "reference" | "validated"
    citation_count: int = 0       # Scopus/Google Scholar 引用数
    publication: str = ""         # "Nature Biotechnology (2022)"
    peer_reviewed: bool = False   # 方法本身是否经过同行评审
```

**分级定义：**
- **toy** — 教学实现，仅用于架构验证（basic_qc, kmeans, marker_score）
- **community** — 由社区贡献，未经过 HistoWeave 团队严格验证
- **reference** — HistoWeave 团队封装并验证的标准实现
- **validated** — 在 ≥3 个真实数据集上与原始实现对比验证

### E2. 在 `list-methods` 中展示质量分级

```
$ histoweave list-methods --category domain_detection

NAME              VERSION   MATURITY    WRAPS                        BENCHMARK
banksy            0.1.0     reference   BANKSY (Nat Biotech 2024)    ARI=0.78 ★
gaussian_mixture  0.1.0     validated   sklearn.mixture.Gaussian     ARI=0.99
spectral          0.1.0     community   sklearn.cluster.Spectral     ARI=0.98
kmeans            0.1.0     toy         _math.kmeans                 ARI=0.99
```

---

## 七、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|:--:|:--:|------|
| cell2location 封装遇到 Pyro 版本兼容问题 | 中 | 阻塞 deconvolution | 备选 RCTD（纯 R，更稳定）|
| BANKSY 原实现依赖特定 R 包版本 | 中 | 容器构建复杂化 | 锁定 renv.lock 或使用 Python reimplementation |
| Cellpose 2 GPU 依赖阻碍 CI 测试 | 高 | CI 无法运行 | 添加 `--cpu` 模式 + 小图像 smoke test |
| R 桥接在 CI Windows 上不可用 | 高 | 测试跳过过多 | 在 CI 中添加 1 个 Linux+R 矩阵条目 |
| 外部依赖 API 变化破坏封装 | 中 | 维护负担 | 每个方法 pin 主依赖版本 + Dependabot 监控 |

---

## 八、执行检查清单

### 第 1 周 — 架构优化
- [ ] A1: 创建 `_sklearn_base.py` + 迁移 7 个 sklearn 方法
- [ ] A2: `Method.run_on_anndata` 钩子 + 修复 `from_anndata` 稀疏转换
- [ ] A3: `RContainerMethod` 基类 + 重构 `r_demo.py`
- [ ] A4: 参数化方法烟雾测试模板
- [ ] 验证: 现有 126 个测试全部通过

### 第 2-3 周 — 第一批方法
- [ ] cell2location wrapper + AnnData 桥接测试
- [ ] BANKSY wrapper（使用 RContainerMethod）
- [ ] SpatialDE wrapper
- [ ] 验证：每个新方法在合成数据 + DLPFC 数据上运行

### 第 4 周 — 数据集 + 剩余方法
- [ ] C1: 数据集注册表基础实现
- [ ] DLPFC 数据集下载 + 校验 + 注册
- [ ] scANVI wrapper + Cellpose 2 wrapper
- [ ] INGESTION 方法注册

### 第 2 月 — 质量体系建设
- [ ] E1: MethodSpec 质量标记字段
- [ ] 第二批 6 个方法
- [ ] 第 2 个真实数据集入库 (Xenium Breast)
- [ ] `histoweave list-methods` 展示质量分级

---

## 九、关键指标

| 指标 | 当前值 | 1 月末目标 | 3 月末目标 | 6 月末目标 |
|------|:--:|:--:|:--:|:--:|
| 方法总数 | 12 | 18 | 24 | 30 |
| 真实方法（非 toy） | 6 | 12 | 18 | 25 |
| 类别覆盖 | 8/11 | 9/11 | 10/11 | 11/11 |
| 真实数据集 | 0 | 1 (DLPFC) | 3 | 5 |
| 每个方法的平均代码行 | ~50 | ~30 | ~25 | ~25 |
| 有 benchmark 评分的方法 | 7 | 12 | 18 | 24 |
| R 方法数 | 1 (demo) | 3 | 5 | 8 |

---

## 结论

功能深度的优化不是"写更多代码"——而是**先降低写代码的成本，再规模化生产**。

三个杠杆按优先顺序：
1. **A1+A2+A3（架构改进）** → 每方法成本降低 60-70%
2. **B1+B2（优先方法）** → 12 个高影响力方法覆盖核心用户场景
3. **C1+E1（质量基础设施）** → 数据集 + 质量分级让方法可信赖

完成后，HistoWeave 的方法深度将达到 Giotto 的 60-70%（方法数）+ 独家的推荐引擎 + 更好的工程纪律，这是一个有竞争力的定位。
