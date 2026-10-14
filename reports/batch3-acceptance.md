# Milestone 1 — acceptance batch 3 (plan rows 21–30)

Local acceptance evidence is verified. This is a supplement to the integrated analytical report, not a completed milestone or substitute for member review. No model choice, threshold or feature set was changed after test evaluation.

## User accuracy target

Target: at least 80% ordinary binary accuracy, as requested on 14 October 2026. All three frozen models exceed this on the historical 2022–2024 held-out test. This does not guarantee future-race performance. The task predicts podium versus non-podium, not exact finishing position.

| model | accuracy_percent | majority_baseline_percent | accuracy_lift_percentage_points | precision | recall | F1 | meets_user_accuracy_target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | 88.6263 | 84.9335 | 3.6928 | 0.5947 | 0.7696 | 0.6709 | True |
| Random forest | 90.1034 | 84.9335 | 5.1699 | 0.6944 | 0.6127 | 0.6510 | True |
| Shallow neural network | 89.2910 | 84.9335 | 4.3575 | 0.6468 | 0.6373 | 0.6420 | True |

The validation-selected Random forest has test accuracy 90.10%, compared with 84.93% for always predicting no podium: a lift of 5.17 percentage points. Its podium precision is 69.44% and recall 61.27%. The majority baseline itself exceeds the requested 80%, so meeting that number alone does not establish useful podium prediction. AP remains the validation selection metric; test results do not select a new model.

## Row-by-row reconciliation

| Plan row | State | Evidence interpretation | Evidence location |
| --- | --- | --- | --- |
| 21 | Local coverage verified; member review pending | All three questions plus feature validity, model comparison, metrics, experiments and limitations are present in notebook/report/PDF. | report_section_pages.csv; figure_coverage.csv |
| 22 | Local evidence verified; member review pending | Global permutation/SHAP, local SHAP/LIME, noncausal interpretation and complete/missing inference. LIME approximation limitations remain explicit. | xai_summary.json; inference_contract_checks.csv; PDF pp. 17–22 |
| 23 | Partial — external checks pending | Fresh local kernel: 31 code cells, no errors; PDF verified. Kaggle Run All, team access, official submission and confirmation are not verified. | submission_manifest.json; docs/submission-readiness.md |
| 24 | Review policy preserved | Automated checks provide evidence; they do not supply member approval. Zero Done cells may be changed by this audit. | technical_acceptance.json; Notion original table preserved |
| 25 | Local register verified; cutoff assumptions remain | 14 features with sources, calculation, cutoff assumptions, training-only evidence, detail plots and group ablation. | feature_register_final.csv |
| 26 | Local experiment evidence verified | Six remove-one-group retrainings; two order comparisons/four variants; 48 numeric distribution audits and consistent validation rows. | feature_group_ablation.csv; preprocessing_order_results.csv; preprocessing_distribution_effects.csv |
| 27 | Local figure/report coverage verified | 26 inline notebook figures, 26 report references and 26 PDF images; every caption mapped to its PDF page. | figure_coverage.csv; report_section_pages.csv |
| 28 | Local model documentation verified | LR baseline, RF, one/two-hidden-layer NN trials; model settings, limitations and comparison documented. Final contract supersedes initial test-reserved checkpoint. | model_feature_contract_final.json; nn_model_trials.csv; PDF pp. 9–14 |
| 29 | Local inference checks verified | 25 cases: field reordering, missing inputs, unseen categories, invalid values/keys and unknown model; fitted models unchanged. | inference_contract_checks.csv; src/batch2.py: podium_inference |
| 30 | Local audit verified; historical semantics remain | 14 tables, documented key graph, sentinels, date/duration conversion and row-conservation checks. Retrospective starting-car/shared-drive uncertainty remains unresolved. | primary_key_recheck.csv; foreign_key_recheck.csv; row_count_audit.csv |

## Report and figure coverage

The current review PDF has 26 pages and all 26 saved analysis figures. All figure captions and report section headings were found in extracted PDF text; every figure is also present as an inline notebook output. The PDF was visually checked in batch 2 and is unchanged in this batch. This structural check does not replace those visual checks.

| section | PDF_page |
| --- | --- |
| Objective and scope | 2 |
| Source and audit | 2 |
| Cleaning, cohort and duplicate rule | 2 |
| Question 1 — front-row starts by circuit and the 2014 boundary | 4 |
| Question 2 — home-race podiums controlling starting grid | 4 |
| Question 3 — constructor mechanical retirements | 5 |
| Features and chronological evaluation design | 6 |
| Team and contribution verification | 7 |
| Additional detail for Question 1 - unfiltered versus supported leaders | 8 |
| Three-model training, selection and comparison | 9 |
| Controlled experiments | 1 |
| Model explainability | 17 |
| Inference function and missing-data handling | 22 |
| Limitations and review items | 22 |
| Reproducibility and acceptance state | 22 |
| Appendix - detailed feature evidence | 23 |
| Sources and methods references | 26 |

## Feature evidence and inference

The final feature register adds explicit calculations, cutoff assumptions, training-only statistics, detail-plot references and measured validation group-ablation drops. The original batch-1 register/contract are preserved as checkpoints; use feature_register_final.csv and model_feature_contract_final.json for final review. Correlation and group ablation are model evidence, not causal or isolated-feature guarantees.

Twenty-five inference contract cases pass without changing fitted-model hashes. Complete/partial demonstrations remain in the analytical report. The callable still expects engineered pre-race history supplied by the caller; it does not construct history from raw entrant identity or establish calibrated real-world probabilities.

## Remaining external verification

Row 23 remains partial. Confirm each teammate can access the private repository and notebook, review the complete work, synchronize accepted changes to Kaggle after the earlier upload denial is resolved, restart and Run All there, and verify regenerated outputs against the submission version. Submit the reviewed PDF and accessible links through the official form only when ready, then retain confirmation. No submission has been made. Original Notion owners and Done cells remain unchanged.

Historical shared-drive/starting-car identities, ambiguous start status, retrospective grid timing, home-race confounding and vague mechanical statuses remain disclosed review items. The notebook and local verification cannot independently resolve missing source ground truth. Metric-count/local-XAI wording and evaluation preparation are addressed in the remaining plan rows 31–40.

## Reproduce

Run python build_notebook.py --batch3, python run_batch1.py, python verify_batch2.py, then python verify_batch3.py. The model-choice lock is preserved and must match the existing frozen digest. The PDF is not regenerated for this supplement. See docs/submission-readiness.md for external checks.

## Artifact binding

submission_manifest.json records SHA-256 and sizes of the executed notebook, measured analytical report, unchanged PDF and acceptance implementation. These hashes bind this review package; they do not prove external submission or teammate review.

Private review: [PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2). Shared plan: [ACL Project Milestone 1 — Shared Team Plan](https://www.notion.so/3f7d34a5d89d8109be93e6df20766856).

Next batch: plan rows 31–40.
