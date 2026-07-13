# HistoWeave 缺陷分析与迭代路线图

> **分析日期：** 2026-07-10  
> **分析范围：** Phase-0 脚手架全量代码（30 个源文件、11 个测试文件、文档、配置）  
> **验证状态：** 全量测试通过（74 passed, 2 skipped），mypy 零错误，ruff 5 个轻度告警  
> **版本：** histoweave v0.0.1.dev0

---

## 一、总体评价

HistoWeave 的 Phase-0 脚手架完成度超过预期。六层架构清晰、API 设计一致、测试覆盖率扎实（76 个用例覆盖数据模型、数值核心、插件注册、管道运行、基准测试、CLI、IO 往返、报告生成）。代码质量基线良好（mypy 零错误）。

**核心矛盾：** 当前代码是 "可运行的架构声明"——架构设计正确、接口定义完整，但距离 "生物学家可用的最小产品" 仍有决定性缺口。以下按严重程度列出所有发现。

---

## 二、严重缺陷（Blockers — 阻塞 MVP 交付）

### [B1] Nextflow 管道定义完全缺失

**现状：** `workflows/nextflow/` 目录不存在。CLI 提供了 `ingest → step → report` 的分步命令来模拟 Nextflow DAG，但没有一行 `.nf` 代码。ROADMAP.md 和 architecture.md 都将 Nextflow 列为标准执行器，但 Phase-0 脚手架仅交付了进程内 `run_pipeline()`。

**影响：** 无法在笔记本之外运行。多步骤并行、容器化隔离、HPC/云部署全部阻塞于此。

**修复方案：**
1. 创建 `workflows/nextflow/main.nf`，将 `default_pipeline()` 的四个步骤翻译为 DSL2 进程
2. 每个进程使用 `histoweave-r` 容器镜像
3. 输入/输出通过 `bundle/` 目录传递
4. 添加 `nextflow run` 集成测试

**优先级：** Phase-1 启动前必须完成。

---

### [B2] 容器镜像不存在

**现状：** `r_demo.py` 引用 `/usr/local/bin/histoweave-sc-transform.R` 作为容器内路径，但项目中没有任何 Dockerfile、`workflows/containers/` 目录或容器构建脚本。R 桥接测试在 CI 中必然跳过（Windows 和 CI 环境均无 Rscript）。

**影响：** R 互操作性是 HistoWeave 的核心卖点之一（"R↔Python divide becomes an implementation detail"）。没有容器镜像，这一承诺无法兑现。

**修复方案：**
1. 创建 `workflows/containers/histoweave-r/Dockerfile`（基于 rocker/r-base，安装 anndata R 包 + histoweave-sc-transform.R）
2. 添加 GitHub Actions 工作流构建并推送镜像到 ghcr.io
3. `r_demo.py` 中的路径查找逻辑保持在 CI 中通过挂载卷可用

**优先级：** Phase-1 早期（与 B1 并行）。

---

### [B3] 真实方法封装为零

**现状：** 8 个内置方法全部是刻意简化的教学实现（`kmeans` 不是 BANKSY，`marker_score` 不是 scANVI，`marker_deconv` 不是 cell2location）。`MethodSpec.wraps` 字段被设计用于声明 "底层封装了哪个真实方法"，但所有内置方法的 `wraps` 均为 `None`。`MethodSpec.language` 除 `r_lognorm` 外全为 `"python"`。

**影响：** 当前平台无法产出有生物学意义的分析结果。Phase-0 以架构验证为目标，这是合理的；但 Phase-1 必须交付至少 5 个真实方法封装。

**修复方案：**
1. **域检测：** 封装 BANKSY（R 容器化）或 STAGATE（Python）
2. **注释：** 封装 scANVI 或 CellTypist
3. **去卷积：** 封装 cell2location 或 RCTD
4. **SVG 检测：** 封装 SpatialDE 或 nnSVG
5. **分割：** 封装 Cellpose 2
6. 更新 `plugin-template/` 提供真实封装示例

**优先级：** Phase-1 核心交付物。

---

### [B4] 无真实数据集成测试

**现状：** 所有 76 个测试仅使用合成数据（每代 <1 秒）。`write_visium_fixture` / `write_xenium_fixture` 能写出格式正确的文件，但没有测试用真实的 Visium/Xenium 公开数据集（如 10x Genomics 官方数据集）验证端到端流程。

**影响：** IO 读取器虽通过了夹具往返测试，但无法保证能正确解析真实 Space Ranger / Xenium 输出。差异可能来自：版本差异、非标准字段、编码问题。

**修复方案：**
1. 选取 1 个公开 Visium 数据集（如 human lymph node, ~1GB）和 1 个 Xenium 数据集
2. 添加 `tests/test_real_data.py`，在 CI 中通过缓存下载
3. 验证：形状正确、坐标范围合理、关键元数据字段存在

**优先级：** Phase-1 早期。

---

## 三、高优先级缺陷（High — 影响可维护性与可扩展性）

### [H1] `SpatialTable` 不支持稀疏矩阵

**代码位置：** `data/model.py:105` — `X: np.ndarray`

空间转录组数据通常有 90%+ 零值。当前将 `X` 硬编码为 dense `np.ndarray`，这意味着一个 10 万细胞的 Visium 数据集（~2 万基因）将消耗 ~16GB 内存（float64），而稀疏存储仅需 ~1.6GB。

**修复方案：** 将 `X` 的类型放宽为 `np.ndarray | scipy.sparse.spmatrix`，在 `__post_init__` 中通过 `safely_dense()` 适配器统一访问。

---

### [H2] `subset_obs` 使用 `.loc[mask]` 对重复索引不安全

**代码位置：** `data/model.py:194` — `obs=self.obs.loc[mask].copy()`

当 `obs.index` 有重复值时（虽然不常见但合法），`.loc[boolean_array]` 会返回多行，导致结果长度不匹配。应改用 `.iloc` 或 `np.where(mask)`。

---

### [H3] `KMeans` 空簇不更新中心导致收敛失败

**代码位置：** `_math.py:76-79`

```python
for c in range(k):
    members = X[labels == c]
    if len(members):
        centers[c] = members.mean(axis=0)
```

当某簇在迭代中变为空集（k-means 的已知边界情况），旧中心被保留但可能已远离任何数据点，永远无法重新获得成员。修复：空簇应重新初始化为离当前中心最远的点。

---

### [H4] 管道无步骤间错误隔离

**代码位置：** `workflow/pipeline.py:77-94`

`run_pipeline()` 中任何一个步骤抛出异常都会导致整个管道崩溃。基准测试框架 (`benchmark/harness.py:97`) 是更好的模式——捕获异常、记录错误、继续下一个方法。管道应提供 `on_error: "skip" | "stop"` 选项。

---

### [H5] 报告仅输出静态 HTML，无交互性

**代码位置：** `report/report.py` + `report/svg.py`

当前报告是纯静态 HTML + 内联 SVG。architecture.md 承诺的 Vitessce / napari-spatialdata 集成完全未实现。对于空间数据，可交互的可视化是用户的核心需求。

**修复方案：** 在报告中嵌入 Vitessce 的 CDN 加载 + JSON 配置，至少对空间散点图提供缩放/平移/悬停功能。

---

### [H6] Bundle 序列化对 `shapes` 是丢失式的

**代码位置：** `io/bundle.py:83-87`

```python
try:
    json.dumps(geometry, cls=_NumpyJSONEncoder)
    shapes_serializable[name] = geometry
except TypeError:
    warnings.append(f"shapes[{name!r}] not JSON-serializable; dropped from bundle")
```

当 shapes 包含复杂几何（GeoDataFrame、多边形）时，它被静默丢弃。这对 Visium spot 边界或 Xenium 细胞分割轮廓来说是不可接受的。应使用 GeoParquet 或 WKT 序列化。

---

### [H7] `Provenance` 和 `RunManifest` 重复记录步骤信息

**代码位置：** `data/model.py:159-161` vs `workflow/pipeline.py:83-93`

每个步骤的方法、版本、参数在两个结构中独立记录，增加不一致风险。应考虑 `RunManifest.steps` 引用 `Provenance` 条目而非复制。

---

## 四、中等优先级缺陷（Medium — 代码质量与健壮性）

### [M1] Ruff 告警（5 处）

```
cli.py:134 — isinstance 应使用 X | Y 语法 (UP038)
cli.py:134 — 行宽超过 100 字符 (E501)
r_demo.py:13 — 未使用的 import shutil (F401)
r_demo.py:15 — 未使用的 import sys (F401)
r_demo.py:98 — 行宽超过 100 字符 (E501)
```

建议修复后加入 pre-commit 钩子。

---

### [M2] `SpatialTable.copy()` 是 Pandas 浅拷贝

`pd.DataFrame.copy()` 默认是深拷贝，但索引在某些 Pandas 版本中是共享的。应显式调用 `.copy(deep=True)`。

---

### [M3] PCA Gram 矩阵路径在极端条件数下不稳定

`_math.py:39-44`：当 `n_features >> n_obs` 且条件数很大时，`Xc @ Xc.T` 的浮点误差会被放大。实际使用中罕见（经过归一化+log1p 的数据条件数通常可控），但在未来处理原始计数时可能成为问题。可降级为截断 SVD。

---

### [M4] `MarkerScoreAnnotation` Z-score 全转录组后取标记基因均值

`annotate.py:40-46`：正确的标记基因评分应该是跨标记基因的标准化，而非跨全转录组。当前实现受非标记基因表达水平影响，可能产生偏差。

---

### [M5] `_now()` 函数在 `pipeline.py` 中内联定义

`workflow/pipeline.py:101-104` 在模块顶层定义了 `_now()`，但 `Provenance` dataclass 中有相同的 lambda。应统一为一个来源。

---

### [M6] CLI `--version` 作为全局标志和子命令同时存在

`cli.py:34` 添加全局 `--version`，`cli.py:37` 又添加 `version` 子命令。argparse 处理全局标志可能有歧义。建议仅保留子命令。

---

### [M7] R 桥接测试在 Windows / 无 R 环境下静默跳过

2 个 R 桥接测试 (`test_r_normalize_preserves_structure`, `test_r_normalize_is_deterministic`) 在环境中 Rscript 不可用时跳过。这是正确的，但应至少在一个 CI 矩阵条目中运行（Linux + 安装 R）。

---

### [M8] 基准测试的 score 写回操作修改 frozen dataclass

`benchmark/harness.py:118` — `cls.spec = replace(cls.spec, benchmark=bench)` — 使用 `dataclasses.replace` 修改 frozen dataclass 是正确的，但如果多个基准测试并发运行，存在竞态条件。应在文档中标注 "非线程安全"。

---

## 五、低优先级改进（Low — 完善体验）

### [L1] 无 `workflows/` 目录的占位 README

architecture.md 频繁引用 `workflows/nextflow/` 和 `workflows/containers/`，但该目录不存在，新贡献者会困惑。

---

### [L2] 无 CI/CD 配置

pyproject.toml 引用 GitHub Actions，但 `.github/workflows/` 不存在。建议添加：lint (ruff+mypy)、test (pytest 多 Python 版本)、build (wheel/sdist)。

---

### [L3] docs/ 缺少 API 参考

文档覆盖概念和架构，但没有自动生成的 API 参考（如 `mkdocstrings` 虽已列为文档依赖，但未配置）。

---

### [L4] `MethodCategory` 枚举中 6 个类别无实现

`SEGMENTATION`、`SPATIALLY_VARIABLE_GENES`、`NEIGHBORHOOD`、`CELL_CELL_COMMUNICATION`、`INTEGRATION`、`INGESTION` 在设计中有定义，但零实现。应在相关目录中放置 `NOT_IMPLEMENTED_YET` 存根文件，包含目标封装的方法列表。

---

### [L5] 合成数据生成器的领域形状过于简单

Voronoi 分区产生的领域边界是直线，而真实组织区域是不规则的。这是刻意简化，但可能导致对空间平滑方法的过度乐观评估。

---

### [L6] 报告模板使用硬编码色板

`report.html.j2` 和 `svg.py` 各自维护独立的色板定义。应统一为单一来源（或可配置主题）。

---

### [L7] 无版本化数据集管理

`datasets/` 模块在调用时始终生成新数据。应支持命名版本的数据集（如 `make_synthetic(version="v1")`）以确保基准测试的可重复性。

---

## 六、架构前瞻性问题

这些不是当前代码的"缺陷"，而是需要在后续迭代中做出的设计决策：

### [A1] 数据模型何时切换到真实 SpatialData？

当前 `SpatialTable` 是与 SpatialData 平行运行的自定义容器。切换策略有两种：
- **策略 A：** `SpatialTable` 成为 `SpatialData` 的薄包装器（内部存储 `SpatialData` 对象）
- **策略 B：** 保持 `SpatialTable` 作为 HistoWeave 的规范 API，通过 `to_spatialdata()` / `from_spatialdata()` 桥接

建议策略 A（Phase-1 中期），减少双写维护成本。

---

### [A2] 插件隔离与安全

当第三方插件（通过 `histoweave.plugins` 入口点注册）可以被 `run_pipeline` 执行时，插件代码可以访问文件系统、网络和进程。需要决定：沙箱级别是什么？Phase-2 将引入容器化执行（每个方法在独立容器中运行），这天然提供隔离，但在那之前需明确声明 "插件运行在与用户相同权限的进程中"。

---

### [A3] 基准测试数据集格式标准化

当前基准测试任务 (`domain_detection_task`, `deconvolution_task`) 硬编码合成数据集。Phase-2 需要标准化的公开数据集注册表——包括数据来源、预处理步骤、ground truth 类型和许可信息。

---

## 七、迭代优先级路线图

```
┌─────────────────────────────────────────────────────────────────┐
│  近期（1-2 周）— Phase-0 收尾                                    │
├─────────────────────────────────────────────────────────────────┤
│  ✓ 修复 5 个 ruff 告警                                          │
│  ✓ 修复 H2（subset_obs + 重复索引）                              │
│  ✓ 修复 H3（kmeans 空簇处理）                                    │
│  ✓ 添加 .github/workflows/ci.yml（lint + test）                  │
│  ✓ 创建 workflows/ 占位 README                                  │
│  ✓ 在 docs/ 中启用 mkdocstrings API 参考                        │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  中期（1-3 月）— Phase-1 启动                                    │
├─────────────────────────────────────────────────────────────────┤
│  ■ B1：编写 Nextflow DSL2 管道定义（main.nf + nextflow.config）  │
│  ■ B2：构建并发布 histoweave-r 容器镜像                             │
│  ■ B3：封装首批 5 个真实方法（BANKSY, scANVI, cell2location,    │
│         Cellpose, SpatialDE）                                    │
│  ■ B4：添加 1-2 个真实公开数据集的集成测试                       │
│  ■ H4：管道步骤错误隔离（on_error="skip"）                       │
│  ■ H5：Vitessce 嵌入式交互可视化                                 │
│  ■ H1：X 矩阵支持 scipy.sparse                                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  远期（3-9 月）— Phase-1 完成 → Phase-2                          │
├─────────────────────────────────────────────────────────────────┤
│  ■ A1：SpatialTable → SpatialData 后端切换                       │
│  ■ A3：标准化公开数据集注册表                                    │
│  ■ H6：shapes 序列化升级为 GeoParquet                            │
│  ■ H7：统一 Provenance / RunManifest 数据模型                    │
│  ■ 基准测试任务扩展到 4-5 个类别                                 │
│  ■ 外部贡献者插件 ≥ 3 个                                         │
│  ■ Phase-2 alpha 发布 + 首篇预印本                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、测试覆盖缺口一览

| 覆盖维度 | 状态 | 缺口 |
|----------|------|------|
| 数据模型 | ✅ 9 个测试 | 重复索引边缘情况 |
| 数值核心 | ✅ 21 个测试 | PCA 数值稳定性（大条件数） |
| 内置方法 | ✅ 9 个测试 | 多参数组合 |
| 管道运行 | ✅ 4 个测试 | 错误恢复路径 |
| 报告生成 | ✅ 1 个测试 | 缺失字段/空数据的报告 |
| IO 读取器 | ✅ 5 个测试 | 真实数据、格式变体 |
| Bundle 往返 | ✅ 2 个测试 | 大 shape 序列化失败 |
| CLI | ✅ 8 个测试 | 缺失 --assay 等错误路径 |
| 插件注册 | ✅ 7 个测试 | 并发注册 |
| 基准测试 | ✅ 4 个测试 | 方法失败的基准 |
| R 桥接 | ⚠️ 3 个测试(2 skip) | 仅在容器 CI 中运行 |
| 真实数据 | ❌ 0 个测试 | 整个维度缺失 |
| 性能/内存 | ❌ 0 个测试 | 无回归保护 |

---

## 九、结论

HistoWeave 的 Phase-0 脚手架是 **高质量的架构验证**。它在 ~3000 行 Python 中展示了：清晰的六层分离、一致的插件接口、可用的管道/基准测试/报告闭环、以及扎实的测试覆盖。代码质量基线（mypy 零错误、76 个测试全通过）表明项目有良好的工程纪律。

**从脚手架到产品的关键路径清晰：**
1. Nextflow 管道 + 容器镜像（运营基础）
2. 封装 5+ 真实方法（生物学价值）
3. 真实数据集成测试（可信度）
4. 交互式可视化（用户体验）

前两个是阻塞项，必须在 Phase-1 的前三分之一时间内解决。按当前代码质量和架构设计估算，一个 3 人团队可在 6-9 个月内完成 Phase-1 交付。
