---
name: fvcom-region-bbox
description: Deprecated compatibility shim for the old FVCOM rectangular bounding-box selector. Use fvcom-region-bpoly instead for all new regional FVCOM preprocessing domain selection because the active workflow now creates four-sided deformable polygon boxes, domain-type notes, mission-scope gates, and open-boundary QA.
---

# fvcom-region-bbox Compatibility Shim

This skill name is deprecated. For all new Stage 1 FVCOM regional domain work, use:

`C:/Users/huan111/.codex/skills/fvcom-region-bpoly/SKILL.md`

The old rectangular `RegionBox` approach had a pass rate a little over 50 percent in clean-agent tests. The active replacement is `fvcom-region-bpoly`, which uses a four-sided deformable polygon box as the controlling geometry and keeps the axis-aligned bbox only as a derived data-fetch envelope.

Do not start new tests with this skill unless the user explicitly asks for legacy rectangular behavior.

