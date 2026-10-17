# Barotropic agent workspace contract

Applies to new `barotropic_tide` projects. The caller supplies `case_id` and a project root; preserve that root, including any experiment matrix folders above it. Run Control creates the local directories during initialization and records `agent_layout.schema=fvcom_barotropic_agent_layout_v1` and `agent_layout.role_roots` in `project_manifest.json`. When a role needs remote working files, use the same relative paths beneath the recorded Kestrel project root through `$kestrel-hpc`.

```text
<project>/agents/
  00_lead/
  01_grid/
  02_tpxo/
  03_configuration_build/
```

These are simulation output folders, distinct from this skill package's `agents/openai.yaml` metadata. Their names are fixed across regions, models, reasoning efforts, and repetitions. `benchmark` mode does not require this preparation-role layout.

## Ownership and handoff

| Role | Private working/report root | Canonical final-product ownership |
|---|---|---|
| Lead | `agents/00_lead/` | Root status, provenance ledger and report; assembled `input/<grid_case>/`; run, output and validation stages through the named specialist skills |
| Grid | `agents/01_grid/` | `gridding/<grid_case>/`; stage preconfiguration in its private root for lead-controlled input assembly |
| TPXO | `agents/02_tpxo/` | `forcing/shared_source/` and `forcing/<grid_case>/`; generate FVCOM forcing only after the exact OBC contract is returned |
| Configuration/build | `agents/03_configuration_build/` | `run/build/`; stage provisional/final configuration in its private root for lead-controlled input assembly |

`<grid_case>` retains the request contract's accepted/fresh identities. These folders change neither the selected grid policy nor scientific gates, retries, or execution authorization.

Before delegation, give each worker the case ID, project root, assigned role root, required input contracts, and the canonical paths it may publish. The same assignments apply if the lead performs configuration/build because child-agent slots are limited. All three preparation manifests remain mandatory; four folders do not require four simultaneous agents.

Each preparation role returns `role_manifest.json` in its own role root using the [common worker envelope](state_contract.md). Its `artifact_root` is the project-relative role root; provenance and artifact records also identify any canonical products outside that private root. Do not relocate or duplicate final scientific products merely to place them under `agents/`. The lead passes those three manifest paths to Run Control's `join` command and stores the preparation join record under `agents/00_lead/`.

Only the lead updates shared project status and the command ledger, including by invoking Run Control for a worker's returned evidence. After the join, the lead assigns one writer at a time to assemble or revise `input/<grid_case>/`; independent workers must not concurrently modify the same namelist or input bundle. Preserve immutable attempt and hash rules. Further children use numbered subfolders beneath their owning role root and return through that role; they do not create additional top-level roles.

Keep lead-only working notes and final narrative/resource-use records in `agents/00_lead/`, linking the canonical root report and validation products. Preserve the existing scientific paths under `gridding/`, `forcing/`, `input/`, `run/`, `output/`, and `analysis/`.

## Existing projects

Preserve existing archives and manifest locations when resuming a project without `agent_layout`; absence of this field identifies a legacy layout. Do not move, overwrite, or recertify old artifacts to retrofit the folder convention. Run Control's report continues to discover legacy `worker_manifest.json` files. Any later adoption for new work must retain the original handoffs and record the new paths explicitly in the provenance ledger; reinitialization alone does not migrate the stored project manifest.
