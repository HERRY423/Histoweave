"""HistoWeave quickstart — run the full reference pipeline and print a summary.

    python examples/quickstart.py

Produces `quickstart_report.html` in the current directory.
"""

from __future__ import annotations

import logging

import histoweave as ts

_LOGGER = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _LOGGER.info("HistoWeave v%s\n", ts.__version__)

    # 1. A tiny canonical dataset with known ground-truth spatial domains.
    data = ts.datasets.make_synthetic(n_cells=800, n_genes=50, n_domains=4, seed=0)
    _LOGGER.info("Ingested: %r", data)

    # 2. Run the default Phase-1 pipeline (QC -> normalize -> domains -> annotate).
    _LOGGER.info("\nRunning pipeline:")
    result = ts.run_pipeline(data, verbose=True)

    # 3. How well did domain detection recover the planted structure?
    from histoweave._math import adjusted_rand_index

    ari = adjusted_rand_index(
        result.obs["domain_truth"].to_numpy(),
        result.obs["domain"].to_numpy(),
    )
    _LOGGER.info("\nDomain-detection ARI vs ground truth: %.3f", ari)

    # 4. Write a shareable, self-contained HTML report.
    out = ts.build_report(result, "quickstart_report.html")
    _LOGGER.info("Report: %s", out.resolve())

    # 5. Benchmark all registered domain-detection methods.
    from histoweave.benchmark import domain_detection_task, run_benchmark

    bench = run_benchmark(domain_detection_task(data))
    _LOGGER.info("\nLeaderboard (domain_detection):")
    for row in bench.leaderboard:
        _LOGGER.info("  %s. %-16s score=%s", row["rank"], row["method"], row["score"])


if __name__ == "__main__":
    main()
