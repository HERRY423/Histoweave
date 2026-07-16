# SOTA spatial-domain comparison

The DLPFC benchmark includes four published spatial-domain methods in addition
to the existing ten sklearn and five spatial-aware baselines:

| benchmark name | official backend | benchmark output |
| --- | --- | --- |
| `spagcn` | `SpaGCN==1.2.7` | native SpaGCN labels + Visium refinement |
| `graphst` | `JinmiaoChenLab/GraphST` | `obsm['emb']` + fixed-q Gaussian mixture |
| `bayesspace` | Bioconductor `BayesSpace` | native `spatial.cluster` labels |
| `stagate` | `QIFEIDKN/STAGATE_pyG` | `obsm['STAGATE']` + fixed-q Gaussian mixture |

The canonical spelling is **BayesSpace** (not “BayeSpace”). All methods receive
the same raw count matrix, spatial coordinates, truth-derived domain count, and
seeds. Labels are used only for the final ARI calculation and never for feature
selection or representation learning.

## Isolated environments

The official repositories pin mutually incompatible Python, NumPy, Scanpy, and
deep-learning versions. Install them in separate environments and point the
benchmark at each interpreter:

```powershell
$env:HISTOWEAVE_SPAGCN_PYTHON = "C:\envs\spagcn\python.exe"
$env:HISTOWEAVE_GRAPHST_PYTHON = "C:\envs\graphst\python.exe"
$env:HISTOWEAVE_STAGATE_PYTHON = "C:\envs\stagate\python.exe"
$env:HISTOWEAVE_BAYESSPACE_PYTHON = "C:\envs\histoweave\python.exe"
$env:HISTOWEAVE_R_LIB = "C:\envs\R-library"
python 5x15_spatial_aware\experiment_5x15_methods.py
```

Install the R backend in the R library selected above:

```r
BiocManager::install(c("BayesSpace", "zellkonverter"))
```

CPU is the reproducible default. Set `HISTOWEAVE_SOTA_DEVICE=cuda` to use a
CUDA device for GraphST and STAGATE. `HISTOWEAVE_SOTA_TIMEOUT` controls the
per-cell timeout in seconds (default: 7200).

Each dataset × method × seed cell writes an independent JSON checkpoint. The
aggregated `benchmark_long.csv` contains `status` and `error` fields, so missing
or incompatible official backends remain explicit rather than being replaced
by an approximate implementation.

Official sources:

- SpaGCN: <https://github.com/jianhuupenn/SpaGCN>
- GraphST: <https://github.com/JinmiaoChenLab/GraphST>
- BayesSpace: <https://bioconductor.org/packages/BayesSpace/>
- STAGATE PyG: <https://github.com/QIFEIDKN/STAGATE_pyG>
