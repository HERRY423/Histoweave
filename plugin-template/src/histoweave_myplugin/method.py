"""An example external domain-detection method.

Replace the body of ``run`` with a call into the library you are wrapping. The point of
HistoWeave is that this class is *all* you write — interoperability, provenance, the CLI,
reporting, and benchmarking come for free once the method is registered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from histoweave.plugins import Method, MethodCategory, MethodSpec, ParamSpec, register


@register
class ExampleLouvainDomains(Method):
    spec = MethodSpec(
        name="example_louvain",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="Example plugin: threshold-on-first-PC pseudo-domains.",
        params=(
            ParamSpec("n_domains", "int|None", None, "Clusters; else uns['n_domains']."),
            ParamSpec("key_added", "str", "domain", "obs column for the result."),
        ),
        wraps="example",       # in a real plugin: "squidpy", "Bioconductor::BANKSY", ...
        language="python",
    )

    def run(self, data):
        data = data.copy()
        k = self.params["n_domains"] or data.uns.get("n_domains") or 3

        # --- your real method goes here -------------------------------------
        # e.g. sq.gr.spatial_neighbors(adata); sc.tl.leiden(adata); ...
        # This placeholder just bins cells along the top principal component.
        Xc = data.X - data.X.mean(axis=0, keepdims=True)
        pc1 = np.linalg.svd(Xc, full_matrices=False)[0][:, 0]
        labels = pd.qcut(pc1, q=int(k), labels=False, duplicates="drop")
        # --------------------------------------------------------------------

        data.obs[self.params["key_added"]] = pd.Categorical(
            [f"domain_{int(v)}" for v in labels]
        )
        return self.finalize(data, step="domain_detection")
