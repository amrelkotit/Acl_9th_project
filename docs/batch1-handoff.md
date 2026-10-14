# Milestone 1: first batch handoff

Scope: Notion work packages 1–10. Existing team owners stay as recorded in the shared plan.
Implementation assistance does not replace each member's responsibility to review and explain their work.

## Artifacts

- `notebooks/milestone1.ipynb`: self-contained notebook with ordered source and executed results.
- `src/batch1.py`: auditable implementation used to build the notebook.
- `src/batch1_report.py`: measured report interpretation and model comparison.
- `reports/batch1-report.md`: first-batch report draft; final PDF is a later work package.
- `reports/batch1-evidence/`: small audit tables, definitions, figures, and validation metrics.
- `outputs/batch1/`: full local generated data, lineage, fitted pipelines and trial records (ignored by Git).
- `docs/data-inventory.md`: preserved inventory from the existing pending inventory PR.

## Reproduce locally

Use Python 3.12 with `pip install -r requirements.txt`, or use the already installed local `.deps` directory.
Place the unchanged 14 source CSVs under `data/raw/` (or the downloaded archive at `data/f1-source.zip`).
From repository root, run `python build_notebook.py`, then `python run_batch1.py`, then `python verify_batch1.py`.
The local runner provisions a repository-local kernelspec and executes in a fresh kernel.
Do not run untrusted pickle/joblib files; generated pipelines here are local outputs of this code.

## Kaggle

Existing shared notebook: https://www.kaggle.com/code/amrelkotit/acl-9th-project
Attach the assignment-cited dataset and import `notebooks/milestone1.ipynb` after the existing draft has been preserved.
The notebook locates exactly one complete dataset directory and writes generated files under `/kaggle/working/acl_m1_batch1`.
No credentials, remote module imports, internet access, or raw-data edits are required during notebook execution.
Keep the notebook private. The repository workflow calls for accepted work to be synchronized after review.

## Acceptance state

| Work package | Evidence | Remaining review |
| --- | --- | --- |
| 1 Setup/workflow | Existing private GitHub and Kaggle, local checkout, focused branch, contribution instructions | Confirm all four teammates' access; review PR and synchronize accepted notebook |
| 2 Inventory | 14-table audit, per-column schema/missingness, PK/FK checks, SHA-256 manifest | Team review of audit |
| 3 Cleaning/EDA | Typed working copies, conversion-failure audit, before/after and coverage plots | Review semantic choices |
| 4 Cohort | Explicit exclusions, deterministic duplicate resolution, unique-key assertion | Historical shared-drive/start identity and ambiguous DNS limitations need team verification |
| 5 Features/split | Training-only evidence, prior-history lineage, race-disjoint splits, future-perturbation check | Ablation pending later batch |
| 6 Front row | 2014-boundary analysis, actual-grid definition, counts, joins and figure | Review small-sample interpretation |
| 7 Home race | Country mapping, grid strata and common standardization weights, figure | Review mapping and residual confounding |
| 8 Mechanical | 2014–2021 vs 2022–2024, status mapping and start denominators, figure | Review ambiguous causes; conservative counts may undercount |
| 9 Logistic regression | Training/validation trials, metrics, threshold selection, plots | Final test evaluation deferred until remaining modeling choices are frozen |
| 10 Random forest/comparison | Same-row/feature contract, training/validation metrics and report section | NN comparison and final test evaluation pending later batch |

No whole milestone, teammate access, final submission, or perfect accuracy is claimed complete.
The original Kaggle draft and the inventory PR must not be silently deleted or merged.
