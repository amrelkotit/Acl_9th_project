# Milestone 1: work packages 11–20

This extends the same draft PR and preserves the shared plan's owners. The batch-1 report is a historical checkpoint; use `reports/milestone1-report.md` and `output/pdf/ACL_Milestone_1_Review_Report.pdf` for the integrated review.

## Reproduce

Use Python 3.12 and install `requirements.txt` in an isolated environment. Keep the original 14 CSVs under `data/raw/`. Run from the repository root:

```powershell
python build_notebook.py --batch2
python run_batch1.py
python verify_batch2.py
python render_report.py
python package_batch2.py
```

The runner name is retained for compatibility; it executes the current integrated notebook in a fresh kernel. The notebook embeds implementation source and does not import repository modules on Kaggle. Local `.deps` is an optional installed dependency directory, excluded from Git.

The integrated notebook writes to `outputs/batch2/` locally or `/kaggle/working/acl_m1` on Kaggle. Its batch-1 report section is a checkpoint; the final integrated report supersedes earlier pending-work statements. Attach the assignment-cited dataset and confirm SHAP/LIME dependencies in the target runtime before Run All. Do not infer Kaggle reproducibility from the local run.

## Scope and evidence

| Plan rows | Implementation | Remaining sign-off |
| --- | --- | --- |
| 11 | One- and two-hidden-layer NN trials; explicit chronological validation checkpoint selection; same rows/encoded features as LR/RF | Team interpretation and Kaggle runtime |
| 12 | Six remove-one-group retraining ablations; two controlled order comparisons; distribution/range/value audit and downstream metrics | Team justification and review |
| 13 | Validation permutation importance, global and local SHAP, local LIME; complete/missing inference without refitting | Review assumptions and surrogate fidelity |
| 14–15 | Integrated Markdown/PDF, all three historical questions, all generated analysis figures and measured interpretations | All-member review and source ambiguity checks |
| 16 | Review-ready artifacts and submission handoff | **Pending:** team access/review, accepted-work synchronization to Kaggle, Kaggle Run All, official submission and confirmation |
| 17 | Target and unique driver-race checks, explicit cohort/duplicate rules and retained exclusion evidence | Historical shared-drive and start-status uncertainty remains |
| 18 | Pre-race feature register, strictly previous standings/form, no sprint predictors; independent future-perturbation check | Timestamped source availability and historical cases need review |
| 19 | Locked three-model comparison: six named metrics plus separately defined AP on train/validation/test | Team review; no further test-driven tuning |
| 20 | LR baseline, RF, NN, ablation and two order experiments with separate notebook sections | Team explanation of findings |

`reports/milestone1-evidence/` contains compact measured tables and figures. Full generated datasets, fitted models and test predictions remain local in ignored `outputs/batch2/`. `model_choices_frozen.json` records fitted pipeline hashes and choices before test prediction. A changed choice after test scoring raises an error; preserve this lock and disclose any justified later change rather than treating the revealed test set as unseen.

## Pending external actions

Kaggle notebook: https://www.kaggle.com/code/amrelkotit/acl-9th-project . Its original draft was preserved as private version 5. The earlier upload was denied; the new notebook has not replaced it. Do not bypass that denial.

Draft PR: https://github.com/amrelkotit/Acl_9th_project/pull/2 . No merge or teammate approval is claimed. The pre-existing inventory PR remains open.

Official submission deadline: 18 October 2026, 11:59 PM, as stated in the supplied milestone brief. Submit accessible links and the reviewed PDF through the official form only when the final package is ready, and retain the confirmation. No official submission has been made.

Rows 21–40 remain for the following ten-row batches. Some evidence is already produced, but checks and member sign-off must still be reconciled. Do not mark a whole milestone or owner Done from code execution alone. The requirements do not guarantee or require perfect prediction accuracy.
