# 跨组织空间邻域发现证据审计

**冻结日期：** 2026-07-15
**研究问题：** 空间邻域、SCTransform（SCT）、scVI 与跨组织/跨平台类别泛化能否支持 Nature Methods（NM）级生物学发现？
**总判定：** **NO-GO。当前数据与结果不支持 NM 级跨组织生物学发现。**

目前最强的正向信号是 DLPFC 的 scVI `vascular_barrier` 分支，但其效应很小（β=0.0268）、三供体 95% CI 跨 0，而且在 log-normalization 与 SCT 中方向反转，因此已被预设的跨预处理稳健性检验否定。GEI、状态条件化和保守尺度层级三个跨域假设也都未通过各自门槛。Slide-seqV2 原始计数恢复与真实 SCT 成功属于重要的**技术结果**，不是生物学发现。

本报告使用四类标签：

- **结果**：有文件级产物，可从原始/审计输入复核。
- **弱候选**：局部门槛通过，但主门槛失败；仅可用于设计下一实验。
- **NO-GO**：预设判定失败，不允许升级为发现。
- **待核验**：只有对话中的数字，没有落盘脚本/结果，不能视为可复现证据。

## 1. 数据与计数合同

| 数据域 | 当前规模 | 独立生物学 n | 计数合同 | SCT/scVI 合法性 | 可主张范围 |
|---|---:|---:|---|---|---|
| 人 DLPFC / Visium | 47,338 spots × 33,538 genes；12 sections | 3 donors | 12/12 文件 `X == layers['counts']`，有限、非负、整数 UMI | 合法 | 供体感知的 DLPFC 内部分析 |
| 小鼠下丘脑 / MERFISH 缓存 | 73,655 cells × 161 genes；12 Bregma batches | 1 animal | 99.9936% 已存值非整数；`X` 与 `counts` 是同一体积归一化矩阵 | **不合法** | 仅标签/拓扑描述 |
| 小鼠海马 / Slide-seqV2 缓存 | 41,786 beads × 4,000 genes | 1 puck | 已存值 100% 非整数；缓存 `counts` 不是原始计数 | **不合法** | 仅标签/拓扑描述 |
| 小鼠海马 / 恢复的 Slide-seqV2 原始矩阵 | 53,208 beads × 23,264 genes | 1 puck | int32 CSR；22,396,657 nnz；32,311,360 UMI；合同全部通过 | 合法 | 技术验证与单 puck 候选 |

证据文件：[缓存计数审计](results/count_audit.csv)、[DLPFC 12 切片审计](dlpfc_sct_scvi/results/audit_all_12_sections.csv)、[Slide-seq 原始计数合同](slideseq_raw/results/count_contract.json)、[Slide-seq 注释匹配](slideseq_raw/results/annotation_match.json)。

恢复的 Slide-seq 原始矩阵与缓存注释实现 41,786/41,786 精确 barcode 匹配，坐标最大偏差为 0，4,000/4,000 缓存基因均存在于原始矩阵。它修复了输入合法性，但仍只有一个 puck，不能提供独立生物学重复。

### 不可识别的混杂

当前设计中，组织、物种、技术、空间分辨率和预处理完全共线：

- DLPFC = 人 / 皮层 / Visium / 多细胞 spot；
- MERFISH = 小鼠 / 下丘脑 / 161 基因靶向成像 / 单细胞；
- Slide-seqV2 = 小鼠 / 海马 / 全转录组测序 / bead。

因此，任何“跨组织差异”都同样可能是物种、平台、分辨率、基因面板或处理差异。现有三域最多支持“联合域迁移的探索性试验”，不能识别组织效应，也不能证明跨组织保守机制。

## 2. 预设判定门槛

| 分支 | 预设/冻结门槛 | 实际判定 |
|---|---|---|
| GEI 跨域假设 | 每域 OR≥1.5；空间置换 p≤0.01；LODO AUC≥0.70 且 ΔAUC≥0.10 | 全部失败 |
| 状态条件化 | 空间 FDR、方向一致，并且留一域 Spearman 中位数≥0.10 | 12 个候选均失败；该分支尚无落盘证据 |
| 保守尺度层级 | 两过程正向；跨尺度 max-T p≤0.05；半衰尺度可解析；神经元同型尺度长于星形胶质–血管尺度；独立重复 | 三域均未通过域级门槛 |
| DLPFC 三预处理稳健性 | lognorm/SCT/scVI 同号；每分支 |β|≥0.05；每分支≥2/3 供体同向、空间 q≤0.10、LODO ΔR²>0 的供体≥2/3 | 无模块通过 |

以上是本项目为阻止事后挑选而冻结的分析门槛，不是期刊官方标准。若要升级到 NM 级候选，本项目进一步建议：每域至少 3 个独立受试者；同组织跨平台和同平台跨组织的桥接设计；预注册 held-out-domain AUC≥0.70 且 ΔAUC≥0.10；meta OR≥1.5、95% CI 不跨 1、≥80% 受试者同向、q<0.05、I²≤50%；最后用独立成像/蛋白及扰动实验验证。

## 3. 已完成假设检验

### 3.1 胶质富集界面（GEI）：NO-GO

预设假设是解剖/神经元身份转换边界富集星形胶质、少突胶质和血管支持程序。三域结果均不支持：

| 域 | OR | 区间 | 空间 p（199 shifts） | 判定 |
|---|---:|---:|---:|---|
| DLPFC | 0.9538 | 3-donor CI [0.7154, 1.2718] | 0.960 | NO-GO |
| MERFISH | 0.6638 | 描述性 cell CI [0.6346, 0.6944] | 1.000 | NO-GO；方向与假设相反 |
| Slide-seqV2 | 1.0106 | 描述性 bead CI [0.9578, 1.0663] | 0.535 | NO-GO |

LODO 中，DLPFC、MERFISH、Slide-seqV2 的完整模型 AUC 分别为 0.4991、0.6663、0.7302；相对基线 ΔAUC 分别为 −0.0020、−0.0103、−0.0053。即使 Slide-seqV2 的绝对 AUC 超过 0.70，邻域熵没有带来增量预测价值。完整证据见 [域效应](results/domain_effects.csv)、[LODO](results/leave_one_domain_out.csv)、[机器可读判定](results/pilot_results.json) 和 [主图](results/figure1_gei_pilot.png)。

模块层面也不构成跨域机制：效应小且方向不一致；MERFISH 的 `astro_ion` 和 `oligo_myelin` 实际分别只有 AQP4 与 MBP 单基因，血管/免疫模块为 0 基因，不能与全转录组多基因模块等价比较（[模块覆盖](results/module_gene_coverage.csv)）。

### 3.2 状态条件化：方向线索存在，但仍为 NO-GO；文件级复核待完成

仓库中未找到该分支的脚本、CSV、JSON 或置换分布。以下数字仅来自分析消息，应视为**待核验**，不能作为最终论文结果。

- DLPFC 中，`neuron_fraction` 对 astro/oligo 四个终点的 β 均为负（−0.0738、−0.1052、−0.1313、−0.1020；空间 q≈0.0106），方向为 3/3 供体一致；但四个供体 t 区间均跨 0。
- MERFISH 中，AQP4 与 MBP 的 neuron β 分别为 −0.1372 和 −0.0820（q≈0.0106）；这两个“模块”均为单基因且生物学 n=1。
- 缓存 Slide-seqV2 中，astro、AQP4、oligo、MBP 的 neuron β 分别约为 −0.0387、−0.1078、−0.1246、−0.0493；但输入为归一化缓存且生物学 n=1。
- 四个终点的 LODO Spearman 中位数仅 0.0195、0.0482、0.0853、0.0877，全部低于预设 0.10；12 个 `candidate_go` 和总 `NM_GO` 均为 false。

因此，“神经元占比越高，胶质支持状态越低”只能列为最高排序的弱方向线索，不能称为跨组织泛化规律。它也可能由细胞组成、区域解剖或平台捕获效率造成。

### 3.3 保守空间尺度层级：NO-GO

预设假设是神经元同型组织的空间尺度应普遍长于星形胶质–血管耦合尺度。结果如下：

| 域 | 神经元 peak radius / effect / max-T p | 星形胶质–血管 peak radius / effect / max-T p | 关键失败原因 |
|---|---|---|---|
| DLPFC | 6.118 / 0.1723 / 0.01 | 1.000 / 0.0416 / 0.01 | 两个 half-decay 都右删失，尺度差不可解析 |
| MERFISH | 1.000 / 0.1500 / 0.01 | 4.030 / 0.0115 / 0.01 | peak 排序与假设相反；half-decay 均右删失 |
| Slide-seqV2 | 1.084 / 0.1177 / 0.01 | 4.173 / 0.0052 / 0.31 | peak 排序相反；胶质–血管不显著；至少一个 half-decay 未解析 |

三域 `domain_pass` 均为 false，最终判定为 `NO_GO_FOR_CONSERVED_SCALE_HIERARCHY`。见 [尺度摘要](scale_hierarchy/results/scale_summary.csv)、[预设方法与判定](scale_hierarchy/results/method_and_decision.json) 和 [尺度图](scale_hierarchy/results/figure_scale_hierarchy.png)。

### 3.4 DLPFC 真计数 lognorm/SCT/scVI 三分支：NO-GO

三个供体各选择一个与供体中位 spot 数最接近的切片（151508、151669、151673），选择规则不看表达结果。scVI 使用 11,637 spots × 1,927 genes、3 section batches、80 CPU epochs；空间检验使用其中 11,532 个具有有效 layer 标签的 spots。SCT v2 Pearson residual 直接来自相同整数 UMI，且从未输入 scVI。

最关键结果是方法分支反转：

| 模块 | lognorm β / q | SCT β / q | scVI β / q | 判定 |
|---|---:|---:|---:|---|
| GEI | −0.0241 / 0.030 | −0.0334 / 0.030 | +0.0106 / 0.360 | 方向反转，NO-GO |
| vascular_barrier | −0.0115 / 0.276 | −0.0157 / 0.150 | +0.0268 / 0.030 | scVI 单分支弱信号被另两分支否定 |
| oligo_myelin | −0.0160 / 0.070 | −0.0148 / 0.104 | −0.0172 / 0.148 | 唯一三分支同号，但效应远低于 0.05 且 FDR 门槛失败 |

scVI `vascular_barrier` 虽为 3/3 供体正向，LODO ΔR² 为 0.00121、0.00215、0.00108，空间 q=0.030，但 β=0.0268、供体 CI [−0.0309, 0.0845] 且跨预处理反向，因此不是稳健候选。lognorm 与 SCT 的 12 个 donor×module 效应高度一致（Pearson r=0.908；Spearman ρ=0.888），而 scVI 与 lognorm、SCT 的 Pearson r 仅 0.200、0.124。housekeeping control 在 scVI 中也显著（β=0.0360，CI [0.0157, 0.0563]，q=0.030），进一步削弱胶质特异性解释。

完整证据见 [结果说明](dlpfc_sct_scvi/RESULTS_README.md)、[三分支效应](dlpfc_sct_scvi/results/overall_effects.csv)、[稳健性门槛](dlpfc_sct_scvi/results/module_branch_concordance.csv)、[跨分支相关](dlpfc_sct_scvi/results/branch_pair_correlations.csv)、[LODO](dlpfc_sct_scvi/results/leave_one_donor_out_prediction.csv) 和 [主图](dlpfc_sct_scvi/results/figure1_dlpfc_robustness.png)。

### 3.5 Slide-seqV2 原始计数与 SCT：技术结果，非生物学发现

真实 `sctransform::vst` v2 Pearson smoke test 在 4,000 个分层抽样 beads × 2,024 genes 上完成，残差全部有限。基因与 log-depth 的中位绝对相关从 raw log1p 的 0.1260、library-normalized 的 0.0940 降至 SCT 的 0.01333，说明恢复的计数可用于真实的计数模型。见 [SCT 技术指标](slideseq_raw/results/sct_technical_metrics.json)、[抽样合同](slideseq_raw/results/pilot_selection.json) 与 [运行摘要](slideseq_raw/results/run_summary.json)。

该结果只证明输入合法与技术去深度相关有效。单 puck、同一公开数据的标签转移和无独立验证，禁止将其表述为海马生物学发现或跨组织保守程序。Slide-seq scVI 单 puck 技术烟测在本冻结时点尚未完成；即使完成，也不会改变 biological n=1 的主张边界。

## 4. Nature Methods 新颖性缺口

当前三个数据集都是高度饱和的公开基准，重复发现层结构、下丘脑核团、CA/DG、白质/灰质少突胶质状态或提高 ARI 都不足以构成新生物学：

- [BANKSY（Nature Genetics, 2024）](https://www.nature.com/articles/s41588-024-01664-3) 已在 Moffitt 下丘脑 MERFISH 中利用邻域转录组分出白质与灰质成熟少突胶质状态，并在同一 DLPFC 12 切片上做域分割。
- [CytoCommunity（Nature Methods, 2024）](https://www.nature.com/articles/s41592-023-02124-2) 已系统化组织细胞邻域发现。
- [INSPIRE（Nature Genetics, 2026）](https://www.nature.com/articles/s41588-026-02579-x) 已分析同一 DLPFC 三供体数据及同一 53,208×23,264 Slide-seqV2 海马原始矩阵，并进行跨来源空间转录组整合。
- [Nicheformer（Nature Methods, 2025）](https://www.nature.com/articles/s41592-025-02814-z) 已在 73 种组织、上亿规模的单细胞/空间语料上学习邻域表征并进行跨组织任务；[Novae（Nature Methods, 2025）](https://www.nature.com/articles/s41592-025-02899-6) 也已做图基础的跨数据集空间表征。
- [多测序型空间技术系统比较（Nature Methods, 2024）](https://www.nature.com/articles/s41592-024-02325-3) 显示平台特异 marker 多于跨平台共享 marker，且不同细胞通讯方法难以得到一致结果；这直接要求当前项目使用正交平台桥接，而不是把平台差异解释为组织机制。

SCT 与 scVI 本身是计数模型/表征工具，而不是物理邻域证据；二者都应从原始 UMI 计数建模（[SCTransform](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1874-1)，[scVI](https://www.nature.com/articles/s41592-018-0229-2)）。跨数据映射也已有 [scArches](https://www.nature.com/articles/s41587-021-01001-7) 等成熟先例。要超越这些工作，需要的是新的、可扰动的生物机制和独立跨平台验证，不是再次证明可以整合这些基准。

## 5. 下一步最小实验设计

1. **建立可识别的桥接队列。** 最低限度每个组织×平台单元 ≥3 个独立动物/供体；同一动物的皮层、下丘脑、海马取相邻切片，在至少两种技术上测量。这样同时具备“同组织跨平台”和“同平台跨组织”比较，才能分离组织与平台效应。NM 级主张更稳妥的目标是每单元 5–6 个独立样本并平衡性别/批次。
2. **统一原始输入。** 所有测序数据保留整数 UMI；MERFISH 面板需预先覆盖完整 astro/oligo/vascular/neuronal 模块及阴性控制，禁止用 AQP4 或 MBP 单基因冒充模块。
3. **冻结一个机制假设。** 当前最高排序但仍很弱的线索是“神经元组成升高伴随胶质支持程序降低”。新队列中应预注册效应方向、模块、空间半径、协变量、mask-preserving spatial null 和停止规则；不得继续在这三个旧基准上事后换题。
4. **以受试者为统计单位。** 使用组织×平台固定效应、动物/供体随机效应的层级模型；采用 leave-one-subject-and-domain-out 与空间块留出。spots/cells 只能增加测量精度，不能增加生物学 n。
5. **正交与扰动验证。** 在全新样本上以 RNAscope/IF 同时验证神经元标志、AQP4/离子稳态、MBP/髓鞘、CLDN5/血管屏障；随后用神经活动增强/抑制或切片培养扰动检验胶质程序是否按预设方向改变。没有这一步只能称关联。

## 6. 可复现性与未关闭事项

- DLPFC 最终结果已落盘，但一键 Python driver 仍指向需要缺失 R `anndata` 的旧 bridge；实际有效结果由 [run_sct_scores.R](dlpfc_sct_scvi/run_sct_scores.R) 的 Matrix Market 路径生成。交付前必须修补 driver，否则 fresh run 会中止。
- 状态条件化分支没有落盘脚本、表格或 null 分布；应在任何论文使用前重新运行并固化。
- `pilot_gei.py` 与 `scale_hierarchy/run_scale_hierarchy.py` 的动态 figure-helper 加载需在 `exec_module` 前注册 `sys.modules[spec.name]`；`audit_h5ad.py` 的 JSON 输出需兼容 NumPy scalar。临时 `.patch_probe` 也应删除。当前分析产物不受影响，但 fresh run 兼容性尚未完全关闭。
- [GEI 结果 JSON](results/pilot_results.json) 含非标准 `NaN` token；严格 JSON 消费者可能拒绝，应改为 `null`。

## 最终可防守结论

**没有发现一个可同时满足独立重复、原始计数合法、空间零模型、跨预处理稳健、跨域预测增益和文献新颖性的生物学规律。** 当前最重要的产出是：三个预设假设被诚实否定；DLPFC scVI 弱信号被另两种计数处理方向反转；Slide-seqV2 原始计数与 SCT 技术路径被修复。下一步应停止在饱和基准上继续挖掘，并转向有桥接、重复、正交和扰动验证的新队列。
