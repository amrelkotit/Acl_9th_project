# Milestone 1 — final compliance and handoff

All remaining local implementation and acceptance work is prepared and verified. This is not an official submission confirmation or human review sign-off. Original owners/internal targets/Done cells remain unchanged.

## Verification

Fresh local kernel: 37 executed code cells, zero errors. All 26 analysis figures match notebook/report/PDF coverage. All three historical question recounts match. Future-outcome/standings invariance, training-only preprocessing, saved predictions/metrics and frozen choices remain checked. The user’s ≥80% ordinary test-accuracy goal is met; selected RF accuracy is 90.10% versus 84.93% majority baseline.

## Final findings

Hybrid-era home-race estimates change sign under exact-grid versus broad-bin control. Treat the comparison as sensitive and observational, not a proven home advantage.

| control | common_support_rows | home_minus_away_percentage_points |
| --- | --- | --- |
| coarse_grid_strata | 4538 | -0.5665 |
| exact_grid_position | 4504 | 1.9095 |

The alternate smallest-resultId duplicate rule agrees with the baseline in all 82 duplicate groups; that agreement does not prove historical starting-car identity. Constructor changes/history resets and conservative mechanical ambiguity are documented. No post-test model tuning was introduced.

## Official deliverable coverage

| Brief deliverable | Requirement | State | Evidence |
| --- | --- | --- | --- |
| 1 | One sectioned run-all notebook | Local Run All passes; final Kaggle run pending | notebooks/milestone1.ipynb |
| 2 | EDA with figures | Local evidence verified | 26 matched analysis figures; report pages |
| 3 | Unique driver-race table and safe features | Local checks verified; historical assumptions disclosed | starter_cohort.csv; modeling_table.csv; final feature register |
| 4 | Three-model comparison and metrics | Local evidence verified; wording ambiguity disclosed | three_model_metrics.csv; frozen choice manifest |
| 5 | Ablation and two order experiments | Local evidence verified | feature_group_ablation.csv; preprocessing_order_results.csv |
| 6 | Global/local explanations, plots and interpretation | Local evidence verified; limitations disclosed | XAI sections; SHAP reconstruction; LIME fidelity |
| 7 | Inference complete/missing examples | Local evidence verified | inference_examples.csv; 25 inference contract cases |
| 8 | Analytical PDF report | Generated and structurally verified; visual QA before handoff | output/pdf/ACL_Milestone_1_Review_Report.pdf |

## All 40 plan rows

| Plan row | Owner (unchanged) | State | Evidence | PDF pages | Remaining | Done |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Karim Ahmed Mahmoud Abdelmageed | Partial/external verification pending | docs/CONTRIBUTING.md; docs/submission-readiness.md | See linked evidence | All-member private-link access unverified | Unverified |
| 2 | Amr Hazem Mohamed Mohamed Hassan Elkotit | Local evidence verified; member review pending | Source and audit | 2 | Member review; retain documented source/method limitations | Unverified |
| 3 | Amr Hazem Mohamed Mohamed Hassan Elkotit | Local evidence verified; member review pending | Cleaning, cohort and duplicate rule | 2 | Member review; retain documented source/method limitations | Unverified |
| 4 | Mark Bassem Mourad Samoel Emira | Local evidence verified; member review pending | Cleaning, cohort and duplicate rule | 2 | Member review; retain documented source/method limitations | Unverified |
| 5 | Mark Bassem Mourad Samoel Emira | Local evidence verified; member review pending | Features and chronological evaluation design | 6 | Member review; retain documented source/method limitations | Unverified |
| 6 | Loay Mohamed Sabry Farag Ismail | Local evidence verified; member review pending | Question 1 | 4, 8 | Member review; retain documented source/method limitations | Unverified |
| 7 | Loay Mohamed Sabry Farag Ismail | Local evidence verified; member review pending | Question 2 | 4 | Member review; retain documented source/method limitations | Unverified |
| 8 | Loay Mohamed Sabry Farag Ismail | Local evidence verified; member review pending | Question 3 | 5 | Member review; retain documented source/method limitations | Unverified |
| 9 | Mark Bassem Mourad Samoel Emira | Local evidence verified; member review pending | Three-model training | 9 | Member review; retain documented source/method limitations | Unverified |
| 10 | Mark Bassem Mourad Samoel Emira | Local evidence verified; member review pending | Three-model training | 9 | Member review; retain documented source/method limitations | Unverified |
| 11 | Karim Ahmed Mahmoud Abdelmageed | Local evidence verified; member review pending | Three-model training | 9 | Member review; retain documented source/method limitations | Unverified |
| 12 | Amr Hazem Mohamed Mohamed Hassan Elkotit | Local evidence verified; member review pending | Controlled experiments | 1 | Member review; retain documented source/method limitations | Unverified |
| 13 | Karim Ahmed Mahmoud Abdelmageed | Local evidence verified; member review pending | Model explainability | 17 | Member review; retain documented source/method limitations | Unverified |
| 14 | Amr Hazem Mohamed Mohamed Hassan Elkotit | Local evidence verified; member review pending | Source and audit; Controlled experiments | 1, 2 | Member review; retain documented source/method limitations | Unverified |
| 15 | Loay Mohamed Sabry Farag Ismail | Local evidence verified; member review pending | Question 1; Question 2; Question 3 | 4, 5, 8 | Member review; retain documented source/method limitations | Unverified |
| 16 | All four; Karim Ahmed Mahmoud Abdelmageed coordinates | Partial/external verification pending | docs/submission-readiness.md | See linked evidence | Team review, Kaggle execution, official submission and confirmation pending | Unverified |
| 17 | All four | Local evidence verified; member review pending | Cleaning, cohort and duplicate rule | 2 | Member review; retain documented source/method limitations | Unverified |
| 18 | All four | Local evidence verified; member review pending | Features and chronological evaluation design | 6 | Member review; retain documented source/method limitations | Unverified |
| 19 | All four | Local evidence verified; member review pending | Three-model training | 9 | Member review; retain documented source/method limitations | Unverified |
| 20 | All four | Local evidence verified; member review pending | Controlled experiments | 1 | Member review; retain documented source/method limitations | Unverified |
| 21 | All four | Local evidence verified; member review pending | reports/final-evidence/final_report_pages.csv | See linked evidence | Member review; retain documented source/method limitations | Unverified |
| 22 | All four | Local evidence verified; member review pending | Model explainability; Inference function | 17, 22 | Member review; retain documented source/method limitations | Unverified |
| 23 | All four | Partial/external verification pending | docs/submission-readiness.md | See linked evidence | Final external execution/submission pending | Unverified |
| 24 | All four | Local evidence verified; member review pending | docs/plan-rows-snapshot.json | See linked evidence | Member review; retain documented source/method limitations | Unverified |
| 25 | Existing feature owner; all four verify | Local evidence verified; member review pending | reports/batch3-evidence/feature_register_final.csv | See linked evidence | Member review; retain documented source/method limitations | Unverified |
| 26 | Existing experiment owner; all four verify | Local evidence verified; member review pending | Controlled experiments | 1 | Member review; retain documented source/method limitations | Unverified |
| 27 | All four | Local evidence verified; member review pending | reports/final-evidence/final_figure_coverage.csv | See linked evidence | Member review; retain documented source/method limitations | Unverified |
| 28 | Existing model owners; all four verify | Local evidence verified; member review pending | Three-model training | 9 | Member review; retain documented source/method limitations | Unverified |
| 29 | Existing inference owner; all four verify | Local evidence verified; member review pending | Inference function | 22 | Member review; retain documented source/method limitations | Unverified |
| 30 | Existing cleaning/cohort owners; all four verify | Local evidence verified; member review pending | reports/batch3-evidence/row_count_audit.csv | See linked evidence | Member review; retain documented source/method limitations | Unverified |
| 31 | All four | Partial/external verification pending | docs/evaluation-walkthrough.md | See linked evidence | Walkthrough guide prepared; actual four-member mock evaluation not held/verified | Unverified |
| 32 | Existing data-engineering owner; all four verify | Local evidence verified; member review pending | Question 1; Final acceptance addendum | 4, 8, 27 | Member review; retain documented source/method limitations | Unverified |
| 33 | Existing data-engineering owner; all four verify | Local evidence verified; member review pending | Question 2; Final acceptance addendum | 4, 27 | Member review; retain documented source/method limitations | Unverified |
| 34 | Existing data-engineering owner; all four verify | Local evidence verified; member review pending | Question 3; Final acceptance addendum | 5, 27 | Member review; retain documented source/method limitations | Unverified |
| 35 | Existing EDA/feature owners; all four verify | Local evidence verified; member review pending | Final acceptance addendum | 27 | Member review; retain documented source/method limitations | Unverified |
| 36 | Existing model owners; all four verify | Partial/external verification pending | Three-model training; docs/evaluation-walkthrough.md | 9 | Named metrics covered; intended five still unclarified | Unverified |
| 37 | Existing XAI owner; all four verify | Local evidence verified; member review pending | Model explainability | 17 | Member review; retain documented source/method limitations | Unverified |
| 38 | Existing feature/model owners; all four verify | Local evidence verified; member review pending | Features and chronological evaluation design; model_choices_frozen.json | 6 | Member review; retain documented source/method limitations | Unverified |
| 39 | All four; existing submission coordinator | Partial/external verification pending | docs/submission-readiness.md | See linked evidence | Kaggle Run All/output agreement and submission confirmation pending | Unverified |
| 40 | All four | Local evidence verified; member review pending | reports/final-compliance.md; docs/plan-rows-snapshot.json | See linked evidence | Member review; retain documented source/method limitations | Unverified |

## External completion gates

Actual all-member access/review and mock evaluation, final Kaggle restart/Run All and output agreement, accepted repository work, official form submission and saved confirmation require actual evidence. Browser permissions and extension file access may block those operations; they cannot be bypassed. See docs/submission-readiness.md and the current external verification record.

## Reproduce

Run python build_notebook.py --final, python run_batch1.py, python verify_batch2.py, python render_report.py, python package_batch2.py, and python verify_final.py. requirements-tested.txt records the tested local versions, not a guarantee across operating systems. Preserve the existing model-choice lock and do not tune on the revealed test set.

## Review links

[Draft PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2). [Private Kaggle notebook](https://www.kaggle.com/code/amrelkotit/acl-9th-project). Official form: https://forms.gle/esDxHsvfzffYHVAZ9 . The brief says evaluation dates/slots are announced separately; office-hours booking rules are not submission deadlines.
