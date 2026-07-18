"""Unit tests for maturity policy ranks."""

from histoweave.plugins.interfaces import METHOD_MATURITY_POLICIES, MethodMaturity


def test_maturity_ranks_are_ordered() -> None:
    ranks = {level: METHOD_MATURITY_POLICIES[level].rank for level in MethodMaturity}
    assert ranks[MethodMaturity.EXPERIMENTAL] < ranks[MethodMaturity.BETA]
    assert ranks[MethodMaturity.BETA] < ranks[MethodMaturity.PRODUCTION]
    assert ranks[MethodMaturity.PRODUCTION] < ranks[MethodMaturity.VALIDATED]
