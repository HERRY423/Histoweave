# IF Lab Brief / 实验交接说明

## English

### Goal
Protein-validate **two distinct** cryptic niches on DLPFC Visium section **151508** (same tissue block ideal), plus optional cross-donor L3 on **151669**.

| Priority | Niche | Spots | Class | Do not mix with |
|---------:|-------|------:|-------|-----------------|
| 1 | `151508_L3` | 138 | L3_program | other class on same section |
| 1 | `151508_L6` | 154 | L6_myelin | other class on same section |
| 2 | `151669_L3` | 137 | L3_program | — |

### Antibodies (minimal)

| Target | Role |
|--------|------|
| **ENC1** | L3 primary |
| **HOPX** | L3 primary |
| **MBP** | myelin primary |
| **DAPI** | nuclear counterstain |

### Pass criteria (pre-registered — protein)

- **L3:** ENC1 **or** HOPX higher in ROI vs **same-layer L3 non-ROI** (padj≤0.05); MBP **not** significantly higher in ROI.
- **L6:** MBP higher in ROI vs **rest of section** (padj≤0.05).
- **Cross-donor (optional):** 151669 L3 meets L3 criteria.

### Files for the core

- `niches/<id>/roi_barcodes.csv` — Visium barcodes in ROI
- `niches/<id>/background_same_layer.csv` — same-layer non-ROI controls
- `niches/<id>/background_rest.csv` — all non-ROI
- `niches/<id>/roi.geojson` — spot centroids for overlay
- `briefing_*.png` — RNA spatial maps with ROI outline (pathologist briefing)

### Return format
CSV with columns: `barcode, ENC1, HOPX, MBP [, PLP1]` (background-subtracted mean intensity).
Drop under `results/if_return/` and run:

```bash
python research/discovery_uncertainty_niches/analyze_if_return.py
```

That command alone upgrades the claim ladder when protein gates pass.

## 中文

### 目标
在 **151508** 同一张（或同供体）切片上验证 **两个不同** 生态位：**L3 型 n=138** 与 **L6 型 n=154**；可选第三位点 **151669 L3** 做跨供体。

**禁止** 把 L3 ROI 与 L6 ROI 合并成同一种“cryptic 状态”。

### 抗体
ENC1、HOPX、MBP（+DAPI）；可选 PLP1。

### 通过标准（蛋白，预注册）
- L3：相对 **同层 L3 非 ROI**，ENC1 或 HOPX 升高（padj≤0.05），且 MBP 不升高。
- L6：相对 **全切片非 ROI**，MBP 升高（padj≤0.05）。

### 回传
按 barcode 的 IF 强度表 → `results/if_return/` → 运行 `analyze_if_return.py` 自动生成「经验证生物学」报告。

## RNA proxy pre-check (not protein)

- `151508_L3`: proxy_pass=False · {'pass_proxy': False, 'enc1_same_layer_up_fdr': False, 'hopx_same_layer_up_fdr': False, 'mbp_not_up_same_layer': True, 'level': 'RNA_proxy'}
- `151508_L6`: proxy_pass=True · {'pass_proxy': True, 'mbp_rest_up_fdr': True, 'level': 'RNA_proxy'}
- `151669_L3`: proxy_pass=False · {'pass_proxy': False, 'enc1_same_layer_up_fdr': False, 'hopx_same_layer_up_fdr': False, 'mbp_not_up_same_layer': True, 'level': 'RNA_proxy'}

_RNA proxy can fail while protein still passes (or vice versa). It only stress-tests ROI/control design._
