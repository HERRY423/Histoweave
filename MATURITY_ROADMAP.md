# HistoWeave 成熟产品路线图

> 评估日期：2026-07-11 | 基线：122 tests / 0 failures / 0 mypy errors / 35 source files / 4600 LOC  
> 当前阶段：Phase-1 early（高质量工程基础确立，功能覆盖不完整）

---

## 执行摘要

HistoWeave 目前在**工程纪律**维度已达到生产级水平（零类型错误、结构化错误处理、数据完整性校验、CI/CD 管道），但**科学功能覆盖**和**用户体验**维度仍有显著差距。从当前状态到可发布的成熟产品，需要跨越 5 个维度共 22 项优化，按依赖关系分为短期（1-4 周）→ 中期（1-3 月）→ 长期（3-6 月）。

---

## 维度一：科学功能完整性（最关键的差距）

### 当前状态
```
方法类别：11 个定义 / 8 个有实现 / 3 个完全空缺
真实方法：12 个方法中 4 个封装真实工具（sklearn, scipy, networkx, ComBat）
          6 个是玩具实现（basic_qc, log1p_cp10k, r_lognorm, kmeans, marker_score, marker_deconv）
空缺类别：SEGMENTATION, CELL_CELL_COMMUNICATION, INGESTION
```

### [S1] 🔴 高 — 填补空缺方法类别

**segmentation（细胞分割）：** 这是单细胞分辨率空间数据（Xenium, CosMx, MERSCOPE）的基础步骤。需要封装 Cellpose 2 或 StarDist。

**ccc（细胞间通讯）：** 空间转录组学的核心应用之一。需要封装 LIANA+、CellChat 或 NicheNet。

**ingestion：** 虽然 `io/readers.py` 已实现读取逻辑，但未注册为 `Method`。应在 `MethodCategory.INGESTION` 下注册 `visium_reader`, `xenium_reader` 方法。

### [S2] 🔴 高 — 将核心玩具方法替换为真实封装

| 类别 | 当前玩具 | 应封装 | 依赖 |
|------|----------|--------|------|
| annotation | marker_score | scANVI / CellTypist / TACCO | PyTorch / scvi-tools |
| deconvolution | marker_deconv | cell2location / RCTD / SPOTlight | Pyro / R |
| normalization | log1p_cp10k | scran (R容器) / SCTransform | R / anndata R |
| qc | basic_qc | scanpy.pp.calculate_qc_metrics + scrublet | scanpy |

**注意：** kmeans 有 3 个 sklearn 替代方案（dbscan, agglomerative, spectral），但域检测仍缺真正的空间域检测方法（如 BANKSY, STAGATE, SpaGCN）。

### [S3] 🟡 中 — 添加结构化基准数据集注册表

当前 `benchmark/harness.py` 的 `domain_detection_task()` 硬编码合成数据。需要：
- 版本化数据集注册表（`histoweave.datasets.registry`）
- 至少 3 个公开的真实空间数据集与已知 ground truth
- 数据集应可缓存、可校验（SHA-256）

---

## 维度二：生产运营能力

### [P1] 🔴 高 — 构建并发布容器镜像到公开注册表

`workflows/containers/` 中的 Dockerfile 存在但从未构建/发布。GitHub Container Registry 是标准选择。
- 在 CI 中自动构建 `histoweave-python` 和 `histoweave-r` 镜像
- 版本标签跟随 PyPI 版本
- 添加 `.dockerignore`

### [P2] ✅ 已完成 — 结构化日志

运行时代码、示例、基准脚本、文档代码样例及容器健康检查均已统一使用
标准库 `logging`：
- 分层日志（DEBUG/INFO/WARNING/ERROR）
- 每个模块独立的 logger
- 管道运行/基准测试/IO 操作的日志上下文（run_id）
- AST 回归契约阻止直接标准输出调用重新进入 Python 源码

### [P3] 🟡 中 — 依赖审计与安全策略

- 无 `SECURITY.md`
- 无 Dependabot / Renovate 配置
- 无 `pip-audit` 或 `safety check` 在 CI 中
- 建议：添加 `pip-audit` 到 CI；创建 `SECURITY.md`；开启 Dependabot

### [P4] 🟡 中 — PyPI / conda-forge 发布流水线

`pyproject.toml` 已配置完整，但 CI 中缺少发布步骤：
- 在 git tag 推送时自动发布到 PyPI（Trusted Publisher）
- 创建 conda-forge recipe（`recipe/meta.yaml`）
- 添加 `CITATION.cff` 以便学术引用

### [P5] 🟢 低 — 资源消耗回归测试

当前基准测试衡量方法准确性，但不衡量内存/时间。需要：
- 在 benchmark harness 中限制每个方法的最大内存和时间
- 在 CI 中设置固定的资源预算进行回归

---

## 维度三：用户体验

### [U1] 🔴 高 — 文档补齐

当前 4 个 Markdown 文档（index, quickstart, architecture, concepts）仅覆盖基本概念。缺失：
- **API 参考：** `mkdocstrings` 已在 `pyproject.toml` 的 docs extra 中，但未配置。需要 `mkdocs.yml` 配置 + 自动生成。
- **插件开发指南：** 如何写一个封装 BANKSY 的插件？需要完整教程。
- **故障排除：** 常见错误和解决方案。
- **方法选择指南：** "我的数据是 Xenium 人脑切片，我应该用哪个域检测方法？"

### [U2] 🟡 中 — CLI shell 补全

`histoweave` CLI 有 8 个子命令但无自动补全。添加 `argcomplete` 支持：
```bash
eval "$(register-python-argcomplete histoweave)"
```

### [U3] 🟡 中 — 错误消息可操作性

当前错误消息大多良好（如 `"No markers for label 'invented' matched"`），但需系统性审查：
- 每个异常应包含 1) 问题描述 2) 可能的原因 3) 建议的解决方法
- 添加 `histoweave doctor` 的输出作为错误报告的推荐附件

### [U4] 🟢 低 — 添加更多示例

当前 `examples/quickstart.py` 只有一个示例。需要：
- `examples/real_visium_pipeline.py` — 真实数据的完整工作流
- `examples/custom_plugin.py` — 完整插件开发教程
- `examples/benchmark_and_select.py` — 基准测试 + 方法选择
- `examples/batch_integration.py` — 批效应校正

---

## 维度四：工程健壮性

### [E1] 🟡 中 — `SpatialTable.X` 不支持稀疏矩阵

当前 `X: np.ndarray` 硬编码为稠密。10 万细胞 × 2 万基因的双精度矩阵 = 16 GB，而稀疏存储仅需 ~1.6 GB。这是数据规模的硬瓶颈。

**修复方案：** 将 `X` 的类型放宽为 `np.ndarray | scipy.sparse.spmatrix`，在关键点使用适配器函数统一访问。`_tenx.py` 已读取 CSC 稀疏矩阵但立即 `dense.T`。

### [E2] 🟡 中 — 无方法版本淘汰机制

当方法升级到 v0.2.0 且结果与 v0.1.0 不兼容时，如何通知用户？需要：
- `MethodSpec` 添加 `deprecated: bool = False` 和 `replaced_by: str | None`
- `_resolve_params` 在调用已淘汰方法时发出 `DeprecationWarning`
- 文档中的每个方法页面标注可用版本

### [E3] 🟡 中 — Bundle schema 迁移策略

当前 `BUNDLE_SCHEMA_VERSION = 1`。当升级到 v2 时，旧 bundle 无法读取且仅给出错误消息。需要：
- `read_bundle` 支持 v1→v2 迁移（至少对最近 2 个版本）
- 在 `bundle.json` 中记录 creator 版本以便追溯

### [E4] 🟢 低 — 增加属性测试和变异测试

当前测试全是基于示例的测试。成熟的科学软件需要：
- **属性测试（Hypothesis/PBT）：** "归一化后所有值应 ≥ 0"，"subset_obs 后 n_obs 应 ≤ 原始 n_obs"
- **变异测试（mutmut）：** 验证测试套件的有效性（检测被测代码的变异）

### [E5] 🟢 低 — 插件发现机制的健壮性

`entry_points` 加载失败时记录到 `_PLUGIN_FAILURES`（已实现）。但缺少：
- 插件加载超时保护（恶意/挂起的插件不应阻塞整个平台）
- 插件版本约束（"此方法需要 histoweave >= 0.1.0"）
- 插件有效性烟雾测试（加载后调用 `MethodSpec` 验证而非信任声明）

---

## 维度五：生态与社区

### [C1] 🟡 中 — 社区基础设施补齐

- `.github/CODEOWNERS` — 代码审查自动分配
- `SECURITY.md` — 安全漏洞报告流程
- `CITATION.cff` — 学术界引用格式
- `.dockerignore` — 容器构建优化
- `Dependabot` — 依赖自动更新

### [C2] 🟢 低 — 插件市场/索引

当前 `histoweave.plugins` 入口点可以注册第三方方法，但无处发现它们。Phase-2 需要：
- 一个 JSON schema 定义方法元数据
- 一个中心化的方法索引（GitHub Pages? 类似 nf-core 的模块列表）
- 方法质量徽章（测试通过、性能基准、容器可用）

### [C3] 🟢 低 — 国际化

当前所有消息硬编码为英文。科学工具的核心用户群是全球性的。不需要立即翻译，但应：
- 将所有用户可见的消息提取到可翻译的字符串
- 至少支持英文和中文（项目的中文背景）

---

## 优先级路线图

```
┌────────────────────────────────────────────────────────────────────┐
│  短期（1-4 周）— 收尾 Phase-1                                      │
├────────────────────────────────────────────────────────────────────┤
│  □ S1: 注册 ingestion 方法 + 为 SEGMENTATION/CCC 添加 TODO 存根    │
│  □ S3: 创建 datasets.registry（版本化 + SHA-256 校验）             │
│  □ P1: CI 中构建并推送容器镜像（ghcr.io）                          │
│  □ U1: API 文档自动生成（mkdocstrings）                            │
│  □ C1: SECURITY.md + .dockerignore + Dependabot + CODEOWNERS      │
│  □ E1: SpatialTable.X 稀疏矩阵支持                                │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  中期（1-3 月）— 方法扩展 + 可用性                                  │
├────────────────────────────────────────────────────────────────────┤
│  □ S2: 封装 scANVI + cell2location + Cellpose (3 个核心真实方法)  │
│  □ P2: 结构化日志（logging 模块）                                  │
│  □ P3: CI 集成 pip-audit + 依赖审计                                │
│  □ P4: PyPI 自动发布 + conda-forge recipe + CITATION.cff          │
│  □ U2: CLI shell 补全 (argcomplete)                                │
│  □ U3: 错误消息审查 + 可操作性改进                                 │
│  □ U4: 3+ 完整示例（Visium pipeline, custom plugin, batch correction）│
│  □ E2: 方法版本淘汰机制（deprecated + replaced_by）                │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  长期（3-6 月）— 规模化 + 生态                                     │
├────────────────────────────────────────────────────────────────────┤
│  □ S2 完成: 10+ 真实方法封装（含 R/Bioconductor 容器化方法）      │
│  □ E3: Bundle schema 迁移策略（v1→v2 兼容性）                     │
│  □ E4: 属性测试 + 变异测试                                        │
│  □ E5: 插件加载超时 + 版本约束 + 烟雾测试                          │
│  □ C2: 方法索引/市场（类似 nf-core modules）                      │
│  □ C3: 消息国际化（en + zh）                                       │
│  □ P5: 资源消耗回归测试                                           │
│  □ 交互式可视化（Vitessce / napari-spatialdata）                  │
│  □ 首个公开发布 v0.1.0                                            │
└────────────────────────────────────────────────────────────────────┘
```

---

## 七个关键数字

| 指标 | 当前值 | 成熟产品目标 | 差距 |
|------|--------|-------------|------|
| 分析方法数 | 12（6 toy + 6 real-ish） | 20+（≥15 real） | -8 |
| 覆盖类别 | 8 / 11 | 11 / 11 | -3 |
| 真实数据集验证 | 0（仅合成） | ≥3 公开数据集 | -3 |
| 日志覆盖率 | 0% | 100% 模块 | -100% |
| API 文档页面 | 0 | 全模块自动生成 | -37 files |
| 容器镜像可用 | 0（仅 Dockerfile） | 2（py + r）已发布 | -2 |
| CI 平台覆盖率 | Linux + macOS + Windows (计划) | 全部运行中 | 待验证 |

---

## 结论

HistoWeave 离"成熟产品"的距离**并非代码质量**——在这方面它已经领先多数学术软件——而是**功能广度和运营成熟度**。一个生物学用户打开 HistoWeave 后能否完成从数据到结论的完整分析，取决于是否有真实的方法封装（S2）、清晰的文档（U1）、可靠的安装路径（P1/P4）。这些是当前最紧迫的差距，按建议的路线图可在 3-6 个月内系统性地解决。
