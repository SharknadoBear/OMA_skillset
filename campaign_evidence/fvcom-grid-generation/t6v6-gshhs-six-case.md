# `t6v6` GSHHS-only six-region benchmark evidence

- Tested implementation commit: `31c4a1fa5ffbe33bd96fb1dcda7233f2b1dae69d`
- Campaign: `Workspace/Preprocessing/fvcom-grid-generation/run/t6v6`
- Scientific endpoint: benchmark-grid baseline ready `6/6`
- FVCOM-grid ready: `6/6`
- Submission eligible: `0/6`; tidal forcing and model submission were outside scope
- Coastline policy: GSHHS only; no CUSP dataset, command, or skill invocation

| Case | Nodes | Triangles | OBC nodes | qL3 sigma | Final 2DM SHA-256 | Case-status SHA-256 |
|---|---:|---:|---:|---:|---|---|
| Tampa Bay | 74,893 | 133,667 | 166 | 0.848729 | `42305cc13b37852ec1835b6e10f919d0427cfa9b139da1286fa55622deec2189` | `848d67c72911f3745279667a9eac073dea66cc5ffe35c62207515892f2bd9b01` |
| San Francisco Bay | 88,675 | 165,913 | 311 | 0.884602 | `d2c7a8cdf1df557b64a1cc833f2b9603f31e00d9f1c5a47141adc2e6d89a528d` | `8648ae0233321c147dffb1388e34fbc9b3635213fe4896f33c7f418b41dfcd7c` |
| Galveston Bay | 56,856 | 104,010 | 153 | 0.854395 | `54f003030ce5110670d7dd5a4a6967cf788010a1e484db0f131da78d4fe746ee` | `51705d5e1a19fda866e41bc9652468acbd4dcb7accf39db8181ee0a9cf97c1bc` |
| Mobile Bay | 108,731 | 202,098 | 250 | 0.880456 | `0de4b10a62200acd96d8ab43fadbff69c59b0ddb9c85b1819d9cff60f35490aa` | `c1d8418daecef928d324cbbd295041854b07564893c95cb1466b1cc3ee8c4d66` |
| Hawaiian Islands | 100,355 | 190,602 | 853 | 0.874875 | `d4469565eb88bda3c50fc0e47ed371754a4c0f89d7dcf8017288e3cc57d3399c` | `555f5439f2b0f9074c374b03d36f15dc66b10dc862c3f867c4c229e5f28e69eb` |
| Columbia River estuary | 53,679 | 98,138 | 140 | 0.857381 | `381dd5170a4874bea19afeadaf4e2e373cf0a60faec7fe4a9783bc7798460312` | `3a9a41a1d3bff678d02cc9804a2a7368999bb2fd71ff7e6934aea0ee8437222f` |

Every final mesh has finite positive depths, positive signed element areas, one intended connected manifold component, exact preserved constraints and OBC order, zero unused nodes, maximum valence eight, zero superthin elements, fewer than five million nodes, and a passing exact 2DM roundtrip.

Tampa Bay is the fresh full-workflow proof for the sub-resolution boundary-fan repair. Columbia River retained the finite-coverage gate after an NCEI OPeNDAP failure and recovered the same official NOAA CRM/ETOPO datasets over HTTPS; no bathymetry was fabricated and the domain, resolution, and source priority were unchanged.

The source and installed skill trees remained unchanged and equal throughout the campaign:

- RegionBPoly: `7785d2919b207314b667af2f2c37a4b52cb61e174cc99c356979fb30c1ca8495`
- Boundary Arc: `2b8022077f8bbec76a6493f796b96da171284c4e6ce9dbd9b72a85a1cd4c8872`
- CUDEM Bathymetry: `ea204577ec2da2982c57307c03cee4a734a1f1bbb0965326f5652a14c82beab0`
- Grid Generation: `a88fb868ad415893243503cbadabd0d2ad2875a87b77dcaef3208e812c2eacc6`

Aggregate artifact hashes:

- `campaign-manifest.json`: `83726a83111a866b6d80f57c624331e6cc270d91202acddb575e5666025be79c`
- `campaign-end-hashes.json`: `cdcdf855cc218c557cb6ab97a92268995bc7b49b6b241d27cf7bef5ac483fb36`
- `results.csv`: `86ae46b8d85ac6ddd7bfabd3136e8f1ac7c342dfc3c1f5d96899187449301037`
- `summary.md`: `84d1a038223fd1e544841cd366c8b8abf94cb3bc227eef95980f56dc4f53d08f`
- `qa-contact-sheet.png`: `7f98db7173374a8322006391019773a5330b5422b333cdcbe9f6e8acdb788c71`

Orchestration note: the session scheduler retained a completed Round 1 child slot and rejected creation of a sixth distinct thread. Five distinct subagent threads produced the six isolated case outcomes; the completed Galveston worker was reactivated for a separate fresh Columbia run and reused no prior scientific case artifact.
