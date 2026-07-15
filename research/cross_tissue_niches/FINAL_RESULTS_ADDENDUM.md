# 最终结果勘误与补充

**日期：** 2026-07-15
**适用文件：** [DISCOVERY_REPORT.md](DISCOVERY_REPORT.md)
**权威性：** 本附录覆盖主报告中“Slide-seq scVI 单 puck 技术烟测尚未完成”的旧状态；其余主报告结论不变。

## 1. Slide-seqV2 scVI 技术烟测已完成

scVI 1.3.3 在恢复的整数原始计数上完成了 40-epoch CPU 烟测：固定 4,000 beads × 2,024 genes、20 维 latent；raw-count gate、latent 有限性、normalized expression 有限且非负均通过。数据只有一个真实 puck，因此 `batch_key=null`，没有创建伪 batch，`biological_n=1`，`generalization_claim=false`。

| 表征 | 与 log-depth 的中位绝对相关 |
|---|---:|
| raw log1p | 0.1260 |
| library-normalized log1p | 0.0940 |
| scVI normalized expression | 0.1265 |
| scVI latent | 0.1206 |

因此，scVI 只通过“真计数可训练、可产生有限输出”的技术门；其 normalized expression/latent 的该项深度相关诊断未优于普通 library normalization，不能据此声称更好的归一化、生物学发现或跨组织泛化。证据见 [scVI 状态](slideseq_raw/results/scvi_status.json)、[scVI 技术指标](slideseq_raw/results/scvi_technical_metrics.json) 和 [最终运行摘要](slideseq_raw/results/run_summary.json)。

## 2. DLPFC vascular 弱候选的 Slide-seq 外部方向检验

为避免事后换模块，验证固定使用 DLPFC 预声明的 9 个 `vascular_barrier` 基因；固定 2,024-gene Slide-seq pilot 中只覆盖 5/9（CLDN5、PECAM1、KDR、SLC2A1、MFSD2A），未做旁系同源或其他基因替代。暴露为六近邻海马 cluster Shannon entropy；模型控制 log library depth、局部 spacing、局部血管丰度和 cluster 标签；空间主检验使用 999 次二维 toroidal shifts。

| 分支 | β | analytic p | spatial-shift p | BH q | 与相应 DLPFC 分支同向？ |
|---|---:|---:|---:|---:|---|
| lognorm | +0.02819 | 0.0881 | 0.055 | 0.0825 | 否；DLPFC 为 −0.01149 |
| SCT | +0.02975 | 0.0749 | 0.051 | 0.0825 | 否；DLPFC 为 −0.01569 |
| scVI | +0.04325 | 0.00344 | 0.090 | 0.0900 | 是；DLPFC 为 +0.02681 |

三个 Slide-seq 分支内部都为正，lognorm–SCT score 的相关为 r=0.956；但三个空间 shift p 都大于预设 0.05，scVI 的显著 analytic p=0.00344 在保留空间结构的零模型下变为 p=0.090。更关键的是，Slide-seq 的 lognorm/SCT 方向与 DLPFC 相反，scVI score 与 lognorm/SCT 的相关也只有 0.233/0.253。该检验只有一个 puck、标签来自同一公开数据，且模块覆盖仅 55.6%。机器判定为 `all_branch_directions_match_dlpfc=false`、`scvi_shift_p_le_0.05=false`、`biological_validation=false`。

结论：**Slide-seq 没有外部验证 DLPFC vascular 弱候选；相反，它再次显示结果依赖模型分支。** 证据见 [冻结假设与最终判定](slideseq_raw/results/vascular_external_hypothesis.json)、[分支效应](slideseq_raw/results/vascular_external_hypothesis_effects.csv)、[分支相关](slideseq_raw/results/vascular_external_hypothesis_branch_correlations.csv)、[基因覆盖](slideseq_raw/results/vascular_external_hypothesis_gene_coverage.csv) 和 [999-shift null](slideseq_raw/results/vascular_external_hypothesis_shift_nulls.npz)。

## 3. 可复现性债务的最终状态

- DLPFC 的表格、null、LODO、图和模型结果已经完成并通过文件级 QA；但主 Python driver 仍指向需要缺失 R `anndata` 的旧 bridge。实际 SCT 结果由 [run_sct_scores.R](dlpfc_sct_scvi/run_sct_scores.R) 通过 Matrix Market 真计数路径生成。fresh one-command run 在 driver 修复前仍会中止，不能称为已关闭的一键复现流程。
- `pilot_gei.py` 与 `scale_hierarchy/run_scale_hierarchy.py` 的动态 figure helper 仍需在 `exec_module` 前写入 `sys.modules[spec.name]`；`audit_h5ad.py` 的 JSON 输出仍需加入 NumPy scalar serializer；临时 `.patch_probe` 尚未删除。多次最小 `apply_patch` 更新被本机 Windows sandbox helper 的 `helper_unknown_error` 阻断。这些问题不改变已落盘统计结果，但必须在对外发布前修复并 fresh-run。
- 状态条件化分支仍只有消息级数值，没有落盘脚本、CSV/JSON 与空间 null；它继续保持“待核验/NO-GO”，不得写入论文 Results。

## 4. 最终判定

加入最终 scVI 与 vascular 外部检验后，证据链更加明确：

1. Slide-seq 原始计数、SCT 和 scVI 的技术路径可运行；
2. 单 puck scVI 没有展示优于简单 library normalization 的该项深度相关诊断；
3. DLPFC scVI vascular 信号在 DLPFC 内被 lognorm/SCT 方向反转；
4. Slide-seq 中虽三分支为正，但空间主检验未通过，且 lognorm/SCT 又与 DLPFC 相反；
5. biological n=1 与 5/9 基因覆盖进一步阻止外部生物学验证。

**最终仍为 NO-GO：不存在可主张的 NM 级跨组织空间邻域生物学发现。**
