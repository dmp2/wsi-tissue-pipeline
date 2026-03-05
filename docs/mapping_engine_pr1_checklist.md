# PR1 Checklist: Dual Mapping Engine Foundation

This document defines the first pull request for introducing a generic mapping workflow with two engine paths:

- `emlddmm` (current production path)
- `xiv-lddmm-particle` (future spatial-omics path)

The goal of PR1 is architecture and naming cleanup with strict backward compatibility.

## Scope

1. Introduce canonical mapping command names in `scripts/run_pipeline.py`:
   - `prepare-mapping` (keep `step4` and `emlddmm-prep` as aliases)
   - `run-mapping` (keep `step5` and `reconstruct` as aliases)
2. Add `--engine` selector to mapping execution:
   - `--engine emlddmm` (default)
   - `--engine xiv-lddmm-particle` (placeholder for now)
3. Add generic workflow config flag:
   - `--workflow-config` (canonical)
   - `--emlddmm-config` (deprecated alias)
4. Add mapping workflow dispatcher API that routes by engine while preserving existing EM-LDDMM behavior.
5. Update docs from step-number-first framing to mapping-engine-first framing.
6. Preserve compatibility outputs and references where possible (legacy aliases, artifact names, test behavior).

## Non-Goals for PR1

1. Implement xIV compute execution.
2. Add spatial omics data adapters or transformations.
3. Change EM-LDDMM algorithm behavior or output semantics.

## Proposed File Changes

1. `scripts/run_pipeline.py`
   - Add canonical command names and new flags.
   - Keep old commands/flags as aliases with deprecation messaging.
2. `src/wsi_pipeline/mapping/__init__.py` (new)
   - Public mapping entrypoints.
3. `src/wsi_pipeline/mapping/engines.py` (new)
   - Engine enum/literals and validation helpers.
4. `src/wsi_pipeline/mapping/workflow.py` (new)
   - `plan_mapping_workflow` and `run_mapping_workflow` dispatchers.
5. `src/wsi_pipeline/mapping/xiv_particle.py` (new)
   - Placeholder adapter that raises a clear "not implemented yet" error.
6. `src/wsi_pipeline/registration/__init__.py`
   - Re-export canonical mapping entrypoints while preserving `run_emlddmm_workflow`.
7. `src/wsi_pipeline/registration/provenance.py`
   - Normalize canonical replay command to `run-mapping --engine emlddmm`.
   - Continue writing compatibility replay artifact for legacy tooling.
8. `tests/registration/test_step5_cli.py`
   - Extend tests for `run-mapping` and legacy aliases.
9. `tests/registration/test_workflow.py`
   - Adjust assertions for canonical replay behavior while retaining backward-compatible artifacts.
10. `tests/registration/test_mapping_dispatch.py` (new)
    - Validate engine dispatch and xIV placeholder behavior.
11. `docs/mapping_workflow.md` (new)
    - Explain generic mapping lifecycle and engine selection.
12. `docs/emlddmm_registration.md`
    - Reframe as EM-LDDMM engine-specific documentation.
13. `docs/configuration.md`
    - Document `--engine` and `--workflow-config`.
14. `README.md`
    - Surface two-engine strategy and mark xIV support as planned/experimental.
15. `CHANGELOG.md`
    - Add migration notes for canonical names and retained aliases.

## Backward Compatibility Rules

1. Keep existing `step4`/`step5` commands functional.
2. Keep `--emlddmm-config` accepted until a later major cleanup.
3. Keep existing EM-LDDMM outputs stable unless explicitly documented.
4. Keep old replay artifact path available:
   - `reproduce_step5_command.txt`
5. Add canonical replay artifact path:
   - `reproduce_mapping_command.txt`

## Suggested Commit Plan (PR1)

1. `feat(cli): add canonical prepare-mapping/run-mapping commands with aliases`
2. `feat(mapping): add engine dispatcher scaffolding and xiv placeholder`
3. `test(mapping): add dispatch and canonical cli coverage`
4. `docs(mapping): add mapping workflow docs and migration notes`

## Suggested PR Title

`Introduce Canonical Mapping Workflow and Engine Dispatcher (EM-LDDMM + xIV placeholder)`

## Suggested PR Description Seed

This PR introduces a generic mapping workflow surface so users can choose an engine based on task:

- EM-LDDMM for image-only registration workflows (histopathology, CT, MRI, etc.)
- xIV-LDDMM-Particle path scaffolded for future spatial omics mapping support

It also introduces canonical command names (`prepare-mapping`, `run-mapping`) while preserving `step4`/`step5` aliases for backward compatibility.

