# Data Policy

Use code-based reproducibility for public release.

## Included
- Synthetic generator wrapper: `final_task/scripts/create_github_demo_dataset.py`
- Small demo synthetic dataset: `final_task/data/github_demo/Demo-Small`

## Not Included
- Large/private real datasets.

## Regenerate Demo
```powershell
python final_task/scripts/create_github_demo_dataset.py `
  --generator-dir "C:\Users\i1n23\OneDrive - University of Southampton\Documents\codex_folder\Revised_models\final_sem_release_v1\scripts" `
  --out-dir "final_task/data/github_demo" `
  --scenario-name "Demo-Small" `
  --n 300 --p 24 --rho 0.2 --k 3 --ell 3 --seed 20260223 --rmiss 0.1 --missing-mechanism mixed
```
