# Delaware Systematic V6 Research Note

This note preserves case-specific reproduction details outside the normative
FVCOM grid-generation workflow. The reusable V6 engine contains no Delaware
paths, component IDs, source-node IDs, or passage-removal authority.

The reproducibility driver is
`scripts/research/delaware/run_systematic_v6_overnight.py`. It injects the
reviewed passage-node set `95`, `106911`, and `106926` only for the frozen
Delaware experiment. Node `95` is the causal node on one bank; `106911` and
`106926` are internal nodes of the opposing over-resolved run bracketed by
retained nodes `110` and `111`. The driver also preserves the case-specific
`thin-16-7abd9f8f29` support probe and frozen input hashes.

The experiment remains research-only. The learned passage operation may alter
wet connectivity and is not enabled by `auto`, by generic
`SystematicV6LoopConfig`, or by the full-grid CLI unless
`--systematic-v6-passage-removal` is explicitly supplied. Its accepted
transaction must still preserve positive manifold geometry, protected and
restricted edges, hard anchors, OBC order, deterministic replay, and the
reviewed topology delta.

The historical r5 controlled proof additionally requires source-lineage edge
`94-109` to remain absent. This is a Delaware evidence check, not a universal
FVCOM acceptance rule.
