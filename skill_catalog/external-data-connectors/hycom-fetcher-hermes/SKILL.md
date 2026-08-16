---
name: hycom-fetcher-hermes
description: Noninteractive manifest-driven workflow for inventorying, estimating, downloading, resuming, and validating arbitrary model-neutral HYCOM subsets with shared hycom-fetcher scripts. Use for unattended Hermes runs that require deterministic request hashes, timing/storage gates, persistent JSON status, and automatic long-run monitoring.
---

# HYCOM Fetcher for Hermes

Use the implementation and request contract in the sibling `../hycom-fetcher/` package. Do not copy scripts into this variant.

1. Require a complete request JSON. Do not prompt for missing source, variables, time, bbox, depth, output, or coordinate overrides; fail with a precise manifest error.
2. Run `../hycom-fetcher/scripts/hycom_fetcher.py inventory`, then `estimate`, then `fetch` using the emitted hash-bound plan.
3. Treat unknown timing, failed probes, stale hashes, expired plans, and inadequate storage as terminal blocked outcomes.
4. Let valid short and long plans proceed automatically. Preserve `download_status.json`, checkpoint isolation, retry evidence, health JSON, and the waitbar for long runs.
5. Return machine-readable artifact paths and terminal state. Keep monitor-visible information free of credentials, URL queries, personal paths, and data values.

Do not perform FVCOM OBC generation, spatial model mapping, sigma remapping, or time reconstruction.
