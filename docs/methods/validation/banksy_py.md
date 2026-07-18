# Validation report — `banksy_py`

**Decision:** scientific **validated**  
**Category:** domain_detection  
**Protocol:** `histoweave.landscape.dlpfc_real.v1` + variance decomposition

## Datasets

DLPFC Visium slices 151507 / 151669 / 151670 / 151673 / 151674 (Maynard et al. 2021).

## Metric

Adjusted Rand Index (ARI) vs manual cortical-layer labels; multi-seed landscape and
factorial variance decomposition of method choice.

## Outcome

Native BANKSY-style neighbourhood-augmented embedding recovers layered structure across
the five-slice difficulty gradient with documented multi-seed ARI.

## Limitations

- This is the **Python scaffold** (`banksy_py`), not the full Bioconductor `Banksy` R path
  (`banksy` wrap is validated separately via the same family proxy).
- Absolute ARI depends on HVG selection, `n_domains` policy, and spatial weight.

## Evidence pointer

`src/histoweave/plugins/builtin/release_manifest.py` → `VALIDATION_EVIDENCE["banksy_py"]`
