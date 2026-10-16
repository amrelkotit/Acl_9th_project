# Formula 1 Podium Prediction Companion — Milestone 1

CSEN 903 Advanced Computer Lab · ACL Team 61

Predict whether a driver finishes in the top three of a Formula 1 main race. Each observation is one driver–race entry, uniquely identified by `(raceId, driverId)`. The prediction cutoff is **after qualifying and before the main race begins**.

## Final submission artifacts

- [Executed notebook](F1_Podium_Prediction_MS1.ipynb): 151 cells, comprising 125 executed Python cells and 26 Markdown cells; execution counts 1–125, 42 embedded figures and no saved execution errors.
- [Analytical report](F1_MS1_Analytical_Report.pdf): 38 pages, including all 42 figures.
- [Final Kaggle Version 13](https://www.kaggle.com/code/amrelkotit/acl-team-61?scriptVersionId=359134370): the exact saved run.

Kaggle records a successful **127.6-second** Version 13 run on 16 October 2026, with GPU T4 x2. The released notebook corrects one stale opening runtime statement from 100.8 to 127.6 seconds; the report's final provenance note records that correction. All Python source, execution counts, outputs, model settings, numerical results and remaining Markdown analyses are preserved. No model was rerun to package this release. Runtime depends on the execution environment.

## Data and engineering

Source: [Formula 1 World Championship (1950–2024), Rohan Rao](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). The dataset slug retains `1950-2020`; the saved analysis uses the 1950–2024 snapshot. Raw data are excluded from Git. CSV SHA-256 fingerprints in the notebook identify the source snapshot.

The 14 relational tables are `circuits`, `constructors`, `constructor_results`, `constructor_standings`, `drivers`, `driver_standings`, `races`, `results`, `qualifying`, `lap_times`, `pit_stops`, `sprint_results`, `seasons` and `status`.

The workflow audits schema, missingness, qualifying coverage and historical eras before cleaning. It converts missing sentinels, dates and qualifying durations, checks 14 primary keys and 23 foreign-key relationships, and validates cleaned exports. Deterministic historical shared-drive deduplication reduces 26,759 raw result rows to 26,668 unique driver–race entries. Raw tables remain unchanged.

Three multi-table analyses investigate circuit front-row conversion and historical-outlier sensitivity, home-country podium associations adjusted for starting-grid bands with uncertainty, and constructor mechanical-retirement rates across 2014–2021 and 2022–2024 with status-classification sensitivity.

Feature engineering produces 16 numerical predictors spanning grid, qualifying, driver form, constructor form, pre-race standings proxies and race context. Historical form and cumulative points use earlier races only. Current-race finishing position, points, laps, status, pit stops and post-race standings are excluded as predictors for that same race.

## Modeling and evaluation

Whole-season chronological partitions preserve temporal order:

| Partition | Seasons | Driver–race rows |
| --- | --- | ---: |
| Train | 2006–2019 | 5,775 |
| Validation | 2020–2021 | 780 |
| Test | 2022–2024 | 1,359 |

Three models use the same observations and 16 features: Logistic Regression, a 200-tree Random Forest, and a shallow TensorFlow neural network with 32/16 hidden units, dropout and validation-loss early stopping. Imputation and scaling fit training observations only; probability thresholds are selected using validation F1.

The primary metric is **Average Precision (AP)**, labeled `PR-AUC` in notebook tables and computed with scikit-learn `average_precision_score`. It differs from trapezoidal PR-curve area and Keras's training-history PR-AUC approximation. ROC-AUC, F1, precision and recall supplement AP. Accuracy is supplementary because podiums are rare. All three models have train, validation and test reports.

| Model | Validation AP | Validation threshold | Test AP | Test F1 | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7363 | 0.29 | 0.7063 | 0.6808 | 0.8999 |
| Random Forest | 0.7315 | 0.35 | 0.6655 | 0.6555 | 0.8793 |
| Neural Network | **0.7496** | 0.28 | 0.6847 | 0.6381 | 0.8882 |

The **Neural Network remains the validation-selected model**. It trained for 35 epochs and restored epoch 25. Logistic Regression performs better on the retrospective test comparison; that result does not revise validation-based selection.

## Experiments, explanations and inference

Two controlled preprocessing-order experiments reverse median imputation versus standard scaling, and training-quantile capping versus median imputation. A robust-scaling reference supplements the first comparison. Fixed Logistic Regression models, identical chronological rows and training-fitted statistics isolate each comparison. Similar rounded metrics do not establish identical transformations or guaranteed improvement.

Feature-group ablation retrains six models, each omitting grid, qualifying, driver form, constructor form, standings or race context, and compares them with the full-feature baseline on validation AP.

Global permutation importance and SHAP, plus local SHAP and LIME, explain the **Logistic Regression baseline**. SHAP represents log-odds attribution; LIME uses 1,500 perturbations for the same validation record and reports surrogate fidelity. Correlation, background choice and unrealistic perturbations limit interpretation. These explanations describe model reliance, not causal effects or the selected NN.

`predict_podium(record)` demonstrates **NN-only feature-level inference** with the existing training-fitted preprocessing and NN threshold. It accepts engineered pre-race features and returns probability, predicted class, threshold and missing-feature names. Complete and partially missing examples are saved. It does not fetch race-weekend data or construct histories from driver IDs, and does not automatically dispatch to another model if a future rerun changes the validation winner.

## Methodological limitations

- Historical shared drives are simplified by deduplication; an entry is not proof of an actual race start. Qualifying coverage, sporting formats and constructor identities change over time.
- Reconstructed points are proxies: sprint points, penalties and official as-of standings differences are not fully resolved. Retrospective source corrections limit historical information-availability claims.
- Absolute Q1 duration is not circuit-normalized pace. An anomalous historical Q1 value is flagged without inventing a correction. Mechanical-status mappings cannot establish every retirement's cause.
- Validation is reused for selection, thresholds, early stopping and exploratory experiments. Earlier test inspection and full-history EDA prevent claiming a pristine untouched-test study.
- Calibration is not established. Associations and explanations are not causal effects or guaranteed race outcomes. Accuracy above 80% alone does not establish useful podium detection.

## View or run on Kaggle

To inspect submitted results, open [Version 13](https://www.kaggle.com/code/amrelkotit/acl-team-61?scriptVersionId=359134370). Viewing saved outputs requires no training. On GitHub, use the notebook's **Preview** tab; download it for a local viewer if GitHub's renderer is unavailable.

For a separate reproduction:

1. Import `F1_Podium_Prediction_MS1.ipynb` into a new Kaggle notebook or use Kaggle's copy workflow. Keep the final saved Version 13 intact.
2. Attach the source dataset above. Ensure `/kaggle/input` contains exactly one CSV for each required name; duplicate filenames from extra datasets fail the loader check. Compare CSV fingerprints with the saved source manifest for numerical reproduction.
3. Use Kaggle's Python environment. The recorded run used GPU T4 x2, pandas 2.3.3 and NumPy 2.1.3. Enable Internet if optional XAI dependencies need installation. `requirements.txt` lists direct imports, not a frozen Kaggle environment.
4. Run all cells in order. Cleaned tables, fitted models, report tables and XAI exports are written under `/kaggle/working`; these generated files and raw datasets are excluded from this release.
5. Check the final audit and outputs. Package/runtime changes, dataset updates and floating-point behavior can change results; the submitted evidence is the saved Version 13 execution.

Local execution requires equivalent Kaggle input/output paths and dependencies. Installing `requirements.txt` alone does not attach data or reproduce Kaggle's GPU stack.

## Repository and submission

The two root artifacts above are the final Milestone 1 submission files. Existing `docs/`, `notebooks/`, `reports/`, `src/` and `data/` documentation remains as project history and workflow guidance. The root notebook is the authoritative executed submission.

Do not commit raw datasets, environments, credentials, caches, downloads or generated models. See [contribution workflow](docs/CONTRIBUTING.md) and [local data policy](data/README.md).

Before university submission, confirm evaluators can access GitHub and the exact Kaggle version without your account, and submit the required links and report through the official course process. This commit does not submit the university form or change repository/notebook visibility.
