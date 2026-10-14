# ACL 9th Project

CSEN 903 Advanced Computer Lab: Formula 1 data engineering and race prediction project.

## Repository layout

- `notebooks/` — notebook work and instructions for the shared Kaggle notebook.
- `src/` — reusable data preparation, feature engineering, modeling, and evaluation code.
- `reports/` — report outline, figures, and final PDF.
- `data/` — local data instructions only; keep raw CSVs out of Git.
- `docs/` — requirements, decisions, and team workflow.

## Milestone 1 outputs

- One notebook that runs from top to bottom and is shared through Kaggle.
- An analytical report in PDF format.
- An accessible GitHub repository and Kaggle notebook link submitted through the official course form.

## Reproducibility and data rules

- Keep raw CSV files unchanged; clean working copies.
- Use a chronological split: train through 2019, validation on 2020–2021, and test on 2022–2024.
- Use only information available before each race starts as model features. Do not use current-race or post-race information.
- Document the eligible driver-race cohort, duplicate handling, missingness, and limitations.

## Team workflow

1. Discuss the task and name an owner before starting.
2. Create a focused branch: `feat/<short-topic>` or `docs/<short-topic>`.
3. Keep changes scoped; do not commit raw data, credentials, or generated large files.
4. Open a pull request with a short summary, evidence, and any report/notebook impact.
5. Ask one teammate to review; merge only after addressing comments.
6. Keep the shared Kaggle notebook synchronized with the accepted repository work, and record contributors in commit history.

See `docs/CONTRIBUTING.md` for the contribution checklist.

## Current review package

Work packages 1–20 are implemented in `notebooks/milestone1.ipynb`. See `docs/batch2-handoff.md` for reproduction commands and the evidence/sign-off distinction. The integrated report is `reports/milestone1-report.md`; its PDF is `output/pdf/ACL_Milestone_1_Review_Report.pdf`. Compact figures/tables are in `reports/milestone1-evidence/`.

The local run does not confirm Kaggle execution, teammate access/review, historical cohort ground truth, or official submission. Those remain pending. Model choices are frozen before test evaluation; revealed test results must not guide additional tuning.
