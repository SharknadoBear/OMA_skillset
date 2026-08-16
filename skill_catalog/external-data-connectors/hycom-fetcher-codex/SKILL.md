---
name: hycom-fetcher-codex
description: Interactive Codex workflow for inventorying, estimating, downloading, resuming, and validating arbitrary model-neutral HYCOM subsets. Use when Bear wants Codex to inspect live HYCOM schema, review a bounded request and timing/storage estimate, and run the shared hycom-fetcher implementation with an automatic long-run waitbar.
---

# HYCOM Fetcher for Codex

Use the implementation and request contract in the sibling `../hycom-fetcher/` package. Do not copy scripts into this variant.

1. Read `../hycom-fetcher/SKILL.md` and its request contract.
2. Inventory the source and show Bear the variables, coverage, estimated bytes, timing interval, chunk count, storage routing, and monitor decision before a material live download.
3. Use `../hycom-fetcher/scripts/hycom_fetcher.py` for `inventory`, `estimate`, `fetch`, and `health`.
4. Let a valid plan proceed without an additional confirmation. If its conservative estimate is at least 600 seconds, let the shared implementation open the localhost waitbar.
5. Report the final output, health JSON, status JSON, important masking/coverage caveats, and whether checkpoints were retained.

Keep every product native-grid or explicitly point-sampled and model-neutral. Never add FVCOM OBC generation, sigma remapping, credentials, query secrets, or personal paths.
