# OUR_SEM Prior-Aware Causal Workflow

This repository provides a reviewer-ready, runnable pipeline for:
- synthetic causal graph validation,
- prior-knowledge extraction from completed scenario outputs,
- prior-aware causal graph learning on new/external datasets,
- mediation pathway analysis (TE/NDE/NIE) on external data.

## Legal Notice
Read first: `final_task/COPYRIGHT_NOTICE.md`

## What The Model Does
`OUR_SEM` estimates directed causal structure under domain constraints.

Two model variants are supported:
1. `domain_latent`: latent-aware SEM (more expressive, can be less stable).
2. `domain_structured`: structured observed-domain SEM (more robust fallback).

Core block logic follows domain roles:
- `G` (genetic/reference),
- `Zfix` / `Znoise` (covariates/confounders),
- `Lt` / `Lm` (treatment/mediator latent blocks),
- `R` (retinal/biomarker mediator block),
- `V` (outcome block).

## Data Flow (End-to-End)
1. Generate or load data.
2. Fit standalone OUR_SEM per scenario.
3. Validate causality on synthetic data with truth adjacency.
4. Build reusable priors from previous runs.
5. Apply priors to new/external data with prior-aware standalone runner.
6. For pathway effects, run mediation (`TE`, `NDE`, `NIE`) and visualize.

## Install
```powershell
pip install -r final_task/requirements_github.txt
```

## Quickstart
### 1) Create Small Demo Synthetic Dataset
```powershell
python final_task/scripts/create_github_demo_dataset.py `
  --generator-dir "C:\Users\i1n23\OneDrive - University of Southampton\Documents\codex_folder\Revised_models\final_sem_release_v1\scripts" `
  --out-dir "final_task/data/github_demo" `
  --scenario-name "Demo-Small" `
  --n 300 --p 24 --rho 0.2 --k 3 --ell 3 --seed 20260223 --rmiss 0.1 --missing-mechanism mixed
```

Outputs:
- `final_task/data/github_demo/Demo-Small/Demo-Small_data.csv`
- `final_task/data/github_demo/Demo-Small/Demo-Small_adjacency.csv`
- `final_task/data/github_demo/Demo-Small/Demo-Small_weights.csv`
- `final_task/data/github_demo/Demo-Small/metadata.json`

### 2) Build Reusable Priors From Completed Scenario Runs
```powershell
python final_task/scripts/build_external_prior_knowledge.py `
  --base-dir "c:\Users\i1n23\OneDrive - University of Southampton\Desktop\New folder\NAHES_Dataset"
```

Outputs (timestamped bundle):
- `scenario_run_index.csv`
- `scenario_edge_priors.csv`
- `scenario_edge_priors_high_confidence.csv`
- `scenario_block_priors.csv`
- `global_block_priors.csv`
- `prior_summary.json`

### 3) Run Prior-Aware Standalone OUR_SEM On New Data
```powershell
python final_task/scripts/run_our_sem_standalone_prioraware.py `
  --data-root "final_task/data/github_demo" `
  --scenario "Demo-Small" `
  --out "final_task/results/prioraware_demo_run" `
  --model-root "C:\Users\i1n23\OneDrive - University of Southampton\Documents\codex_folder\Revised_models\final_sem_release_v1\scripts" `
  --our-sem-variant domain_structured `
  --impute median `
  --prior-bundle-dir "final_task/results/external_prior_knowledge_20260223_113339"
```

Outputs:
- `final_task/results/prioraware_demo_run/metrics.csv`
- `final_task/results/prioraware_demo_run/run_diagnostics.json`
- `final_task/results/prioraware_demo_run/adjacency_pred.csv`
- `final_task/results/prioraware_demo_run/weights_hat.csv`
- `final_task/results/prioraware_demo_run/edge_list.csv`

## How To Check Causality On Synthetic Data
Use standalone benchmark outputs (`combined_metrics.csv`, `metrics.csv`) and inspect:
- `Adjacency_F1`,
- `directed_F1`,
- `SHD`,
- `causal_accuracy`,
- convergence/fit diagnostics in `run_diagnostics.json`.

These compare predicted graph vs truth adjacency in synthetic scenarios.

## How To Check Causal Pathways On External Data
For external datasets, run mediation workflow and inspect:
- `mediation_table_all_combos.csv`,
- `summary.csv`,
- `sem_paper_forest_te_nde_nie.png`,
- `sem_paper_summary_te_nde_nie.png`.

Interpretation:
- `TE` = total effect,
- `NDE` = direct effect,
- `NIE` = mediated/indirect effect.

Use `NIE_Significant` (CI excludes 0) as the strict pathway significance criterion.

## Additional Utilities
- Cross-scenario anti-leak + domain summary bundle:
  - `final_task/scripts/build_sem_crosscheck_bundle.py`
- Data policy:
  - `final_task/DATA.md`
