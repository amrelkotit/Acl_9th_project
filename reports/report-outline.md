# ACL Project — Milestone 1 Report

## Title page
- Project title and course
- Team members and contributions
- Submission date

## 1. Executive summary
State the objective, data, methods, main findings, and limitations.

## 2. Data inventory and preparation
- Describe source tables, columns, keys, relationships, missing values, and sentinels.
- Explain cleaning on working copies and confirm raw files were preserved.
- Document duplicate handling and the one-row-per-eligible-driver-race modeling grain.
- Define the cohort, including starters, pit-lane starters, non-starters, and non-qualifiers.

## 3. Exploratory and data-engineering analyses
- Front-row to podium outcome by circuit and era.
- Home-race performance, controlling for starting position.
- Mechanical retirements across the specified periods.
- For each: question, joins, definitions, figures, interpretation, and limitations.

## 4. Modeling
- Target definition and leakage-safe pre-race features.
- Chronological split: train through 2019, validation 2020–2021, test 2022–2024.
- Logistic regression, random forest, and shallow neural network.
- Report accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
- Compare models on the same cohort and feature set.

## 5. Experiments and explainability
- Feature-group ablation.
- Two preprocessing-order experiments.
- Global permutation importance and SHAP; one local SHAP or LIME example.
- Inference examples for complete and partially missing records.
- Describe explanations as model reliance, not causal effects.

## 6. Discussion and limitations
Summarize findings, data coverage limits, assumptions, and next steps.

## 7. Reproducibility and submission
Link the run-all Kaggle notebook and GitHub repository. Confirm the notebook runs top-to-bottom and submit both links plus this PDF through the official course form.

## Appendix
Add supporting figures, metric tables, and implementation notes as needed.
