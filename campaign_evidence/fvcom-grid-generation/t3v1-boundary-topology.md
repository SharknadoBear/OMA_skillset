# `t3v1` pre-mesh island-topology evidence

- Implementation commit: `3092a6d4712d1b0b25cee97d9238309c9631697b`
- Campaign: `Workspace/Preprocessing/fvcom-grid-generation/run/t3v1`
- Independent agents/cases: Tampa Bay, Galveston Bay, Hawaiian Islands
- Topology acceptance: 3/3
- Benchmark-grid baseline ready: 1/3
- Submission eligible: 0/3; tidal forcing was outside scope

| Case | Action | Count | Terminal outcome | Case-status SHA-256 | Topology evidence SHA-256 |
|---|---|---:|---|---|---|
| Tampa Bay | merge source chains 14/15 | 106 → 105 | `blocked_grid_boundary_readiness` | `ba1c7a7e8957d2b0069b3fda571649dcc6539a1bc1399639e66573244f85fe56` | `8679e41e94768353f77a1c23dbeb6f82e203922e9b73a5726908333ec2249cf6` |
| Galveston Bay | remove exterior-conflicting chain 29 | 29 → 28 | `completed_benchmark_grid_baseline` | `196ef26d5a377419b41ae0b8e7fffcd84b7dd5c7b108dbbd4c27b90a4b2c95e2` | `938f4d819c07f4c061a1075e8fe472edc46b8709c5c7bcbb3b9b451104e0fcdb` |
| Hawaiian Islands | merge source chains 10/16 | 25 → 24 | `blocked_strict_boundary_revalidation` | `28bedba2b293f699fda0b05f13b0dcf59c8d79af4d867d0fc62f1d508595342d` | `f772d9966033399c88d9fb49b15a22c124fc749be31e156d6be28f82372d642f` |

All three compensated wet domains were valid, one-component, mutually disjoint-hole reconstructions with exact chain/hole agreement. Exterior coordinates and OBC coordinates, order, IDs, segmentation, and hard anchors were unchanged.

The Tampa and Hawaiʻi terminal blockers were not topology failures. Their strict v2 open-exterior contract audits passed, while a separate Grid adaptive-geometry revalidation reported `open_boundary_exterior_overlap_below_0_98`.

The campaign-frozen catalog and installed content hashes were identical and unchanged:

- RegionBPoly: `f8bee77206a3c06138724f712929462cb40244390f96d64daa262dd5f448caa5`
- Boundary Arc: `0179a3acbcefa32b7c72c996ace20095e22d70730cd0eb0a6f5b1a1ad0ac9dee`
- CUDEM Bathymetry: `0cc2168d29c5651cbe44029288685c8a2038be96b773e68ea799ba8f2cd4a1c1`
- Grid Generation: `b4bd699a22acd5ecf68eda860d2de128528ba09656dccedb96c2bc934654fa51`

Aggregate artifact hashes:

- `campaign-manifest.json`: `234063cb936588d79dfbdfff69c2d1effceab6bea5ba97324f0dea5863c503a5`
- `campaign-end-hashes.json`: `6b945f4aa551ce169ea01ad966b9a6003381b35cc208bc47eedcfdb940c624fe`
- `results.csv`: `b9a951e4c3820c351aaeb98e01887d879b9ef104507b838479378612a9a6bc60`
- `summary.md`: `1a0d9db4dfaa843ba68e656a84efbdc8df012fd5c737662aa44bf2faa4f1fd3f`
- `topology-qa-contact-sheet.png`: `3912dfdafb854fa2eae27735af786c61e24561f66dfe7252f89348f410d932a8`
