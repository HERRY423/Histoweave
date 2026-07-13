"""HistoWeave quickstart — run the full reference pipeline and print a summary.

    python examples/quickstart.py

Produces `quickstart_report.html` in the current directory.
"""

from __future__ import annotations

import histoweave as ts


def main() -> None:
    print(f"HistoWeave v{ts.__version__}\n")

    # 1. A tiny canonical dataset with known ground-truth spatial domains.
    data = ts.datasets.make_synthetic(n_cells=800, n_genes=50, n_domains=4, seed=0)
    print("Ingested:", repr(data))

    # 2. Run the default Phase-1 pipeline (QC -> normalize -> domains -> annotate).
    print("\nRunning pipeline:")
    result = ts.run_pipeline(data, verbose=True)

    # 3. How well did domain detection recover the planted structure?
    from histoweave._math import adjusted_rand_index

    ari = adjusted_rand_index(
        result.obs["domain_truth"].to_numpy(),
        result.obs["domain"].to_numpy(),
    )
    print(f"\nDomain-detection ARI vs ground truth: {ari:.3f}")

    # 4. Write a shareable, self-contained HTML report.
    out = ts.build_report(result, "quickstart_report.html")
    print(f"Report: {out.resolve()}")

    # 5. Benchmark all registered domain-detection methods.
    from histoweave.benchmark import domain_detection_task, run_benchmark

    bench = run_benchmark(domain_detection_task(data))
    print("\nLeaderboard (domain_detection):")
    for row in bench.leaderboard:
        print(f"  {row['rank']}. {row['method']:<16} score={row['score']}")


if __name__ == "__main__":
    main()
