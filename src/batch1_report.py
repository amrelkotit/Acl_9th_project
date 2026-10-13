"""Generate the report section from measured results, never placeholder findings."""
from pathlib import Path
import json
import pandas as pd

def write_batch1_report(out, destination):
    out, destination = Path(out), Path(destination)
    def read(name): return pd.read_csv(out/name)
    def table(frame):
        def fmt(v):
            if pd.isna(v): return '—'
            if isinstance(v,float): return f'{v:.4f}'
            return str(v).replace('|','/')
        return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
            ['| '+' | '.join(map(fmt,r))+' |' for r in frame.itertuples(index=False,name=None)])
    def figure(name,caption):
        # Relative links resolve from reports/ to the local run outputs.
        return f'![{caption}](../outputs/batch1/figures/{name}.png)\n\n{caption}'
    inventory = read('table_inventory.csv')
    cohort = json.loads((out/'cohort_summary.json').read_text())
    q1 = read('q1_front_row_rates.csv')
    q1_supported = q1.loc[q1.starts.ge(10)]
    top1 = q1_supported.groupby('era').head(3)
    ranks = q1.pivot(index='circuit_name',columns='era',values='rank_within_era').dropna()
    changed = int(ranks.iloc[:,0].ne(ranks.iloc[:,1]).sum())
    q2 = read('q2_home_summary.csv')
    q2def = json.loads((out/'q2_definitions.json').read_text())
    q3 = read('q3_mechanical_rates.csv')
    metrics = read('statistical_model_metrics.csv')
    registry = read('feature_register.csv')
    primary = metrics.loc[metrics.split.eq('validation')].sort_values('average_precision',ascending=False).iloc[0]
    home = q2.set_index('group').loc['Home','grid_standardized_rate']
    away = q2.set_index('group').loc['Away','grid_standardized_rate']
    lines = [
      '# ACL Milestone 1 — batch 1 report draft (work packages 1–10)',
      '',
      '**Status:** reproducible first batch, not the final milestone submission. Shallow NN, ablation, preprocessing-order experiments, XAI, inference demonstrations, final test scores, team review, and final PDF remain for later batches.',
      '',
      '## Objective and scope',
      'Predict whether an eligible race starter finishes on the podium using information available after qualifying and before the main race. The binary label is positionOrder ≤ 3; exact finishing-position prediction is outside scope. Descriptive questions use historical outcomes, but those current-race outcomes never enter the predictor whitelist.',
      '',
      '## Source and audit',
      'Source: [the assignment-cited Kaggle/Ergast dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). The downloaded snapshot contains all 14 tables and results for all 24 races in 2024. Source_manifest.json records per-file SHA-256 and runtime versions. Every run checks that raw hashes are unchanged.',
      '',table(inventory),
      '',
      'Implemented candidate-key checks found no null/duplicate primary keys; the documented foreign-key checks found no orphan values. Per-column schemas and missingness are in column_inventory.csv and each tested relationship is in foreign_key_checks.csv. These checks establish properties of this snapshot, not externally enforced database constraints.',
      '',
      '## Cleaning, cohort and duplicate rule',
      'Cleaning replaces literal backslash-N and blank tokens with actual missing values on working copies. Numeric columns, dates, and qualifying/lap/pit-stop durations receive semantic types; duration values are also represented in seconds. Conversion failures are counted and must be zero. Missing Q2/Q3 can be structural non-advancement; the baseline does not impute these as if every driver participated.',
      '',
      f"Raw results: {cohort['raw_result_rows']:,} rows. Non-qualifier exclusions: {cohort['excluded_non_qualifiers']:,}; documented non-start exclusions: {cohort['excluded_documented_non_starts']:,}; uncertain zero-grid/zero-lap entries withheld: {cohort['ambiguous_starts_withheld']:,}. Duplicate rows removed after eligibility filtering: {cohort['duplicate_rows_removed']:,}. Final historical cohort: {cohort['eligible_driver_race_rows']:,} unique driver–race rows.",
      '',
      'Within duplicate driver–race groups, prefer a positive-grid source row over a zero-grid handover and break ties using the smallest resultId. Finishing position and target are never used to select the feature row. The target belongs to that retained source row; historical shared-drive outcomes may differ under another justified rule. All affected source rows and selections are retained in duplicate_resolution.csv. Historical starting-car identity is not always recoverable; do not interpret the rule as verified ground truth for every shared drive.',
      '',
      'A zero grid is kept when there is evidence of race participation (positive laps), including pit-lane starters; zero-grid/zero-lap uncertain entries are separately withheld. Zero is not ranked ahead of pole. Status and laps are used only for retrospective cohort auditing/labels, never predictors. Results alone cannot independently establish every DNS or pit-lane start, especially positive-grid zero-lap engine/accident cases. This is conditional prediction for reconstructed starters, not a model of which entrants will start.',
      '',figure('01_before_after_cleaning','Before/after cleaning and cohort construction; unchanged missing-cell totals reflect representation changes, not unexplained imputation.'),
      '',figure('02_qualifying_coverage','Qualifying coverage by season; the training portion supports beginning the modeling window in 2003.'),
      '',figure('03_cohort_by_season','Changing field sizes and podium prevalence make historical and modern eras non-exchangeable.'),
      '',
      '## Question 1 — front-row starts by circuit and the 2014 boundary',
      'Join path: results → races → circuits, with qualifying joined on (raceId, driverId) to distinguish qualifying rank from starting grid. Grain: one retained eligible driver–race. Numerator: podium finishes among actual grid 1/2 starts; denominator: those front-row starts. Compare 1950–2013 against 2014–2024. Indianapolis 500 is excluded as an explicitly different competition format. Grid missing/zero cases are outside front-row denominators; missing qualifying rows do not remove known-grid starts.',
      '',
      'The highest supported rates (at least 10 front-row starts for this display; the unfiltered table is also saved) are:',
      '',table(top1[['era','circuit_name','starts','podiums','rate']]),
      '',f'Among circuits observed in both periods, {changed} have different within-era ranks. Rankings also depend on changing circuit participation and small denominators; they are descriptive, not controlled causal comparisons. The full q1_circuit_changes.csv records common-circuit rate changes.',
      '',figure('05_front_row_by_era','Front-row-to-podium conversion, with numerator/denominator labels; qualifying position is not substituted for grid.'),
      '',
      '## Question 2 — home-race podiums controlling starting grid',
      'Join path: results → races → circuits, and results → drivers. Map nationality to country explicitly (British → UK; Monegasque → Monaco; USA/United States normalized). Ambiguous dual-nationality entries are withheld rather than guessed. Indianapolis is excluded. Only mapped starters with positive grid positions enter this comparison.',
      '',
      'Control method: stratify grid into 1–2, 3–5, 6–10, 11–15, and 16+, then apply the same pooled grid-stratum weights to home and away rates on common support. The notebook saves each stratum’s counts and weights. This does not eliminate within-bin differences or driver/team/era confounding.',
      '',table(q2),
      '',f'Grid-standardized home-minus-away difference: {(home-away)*100:.2f} percentage points. Rows excluded for unmapped nationality: {q2def["unmapped_driver_rows"]:,}; mapped rows lacking positive grid: {q2def["mapped_rows_without_positive_grid"]:,}. This association does not establish a causal home advantage.',
      '',figure('06_home_grid_control','Within-grid-stratum rates and standardized home/away rates.'),
      '',
      '## Question 3 — constructor mechanical retirements',
      'Join path: results → status, results → races, and results → constructors. Periods are 2014–2021 and 2022–2024. Numerator: verified component/system failure statuses in the documented conservative mapping. Denominator: eligible constructor driver-starts, not race weekends or all entrants. Lapped finishers are finished, not retired. Accidents, driver causes, and regulatory decisions are excluded from mechanical failures.',
      '',table(q3.groupby('period').head(5)),
      '',
      'Each constructorId is analyzed separately; renames are not silently combined. Generic Retired, damage, tyre/puncture, wing and other ambiguous causes are not assumed mechanical: their counts are saved in q3_ambiguous_status_counts.csv. Reported mechanical rates may therefore undercount failures with vague cause labels. Denominator size and constructor lifetimes differ.',
      '',figure('07_mechanical_retirements','Mechanical retirements per constructor driver-start, with count labels.'),
      '',
      '## Features and chronological evaluation design',
      'Model window: 2003–2024, chosen because training-only coverage is consistently near-complete from 2003, while 1996–2002 coverage is patchy. Training is 2003–2019; validation is 2020–2021; test is reserved at 2022–2024. Earlier rows are retained for historical questions and prior-race history. All rows from one race stay in the same split.',
      '',
      'Driver form shifts each history before rolling over up to five earlier starts. Constructor form first aggregates each race then shifts before rolling, so the teammate’s current result cannot leak. Standings use the most recent strictly earlier main race within the same season, with source-event assertions and season-start handling. Earlier validation/test races may supply observed history for later races, corresponding to sequential pre-race deployment. Current-weekend sprint results are deliberately excluded; no claim is made that every sprint format shares the same cutoff.',
      '',table(registry[['feature','group','source','training_missing_fraction','training_spearman_with_podium']]),
      '',
      'Feature hypotheses: grid/qualifying represent starting opportunity and qualifying pace; driver and constructor form represent recent competitiveness; prior standings represent season strength; constructor and circuit identities represent team/circuit context; round represents championship progress. The training-only plots show associations and missingness; these motivate candidate inclusion but do not establish incremental predictive value. Batch 2 ablation will assess group contribution. IDs are one-hot categorical values, not continuous ranks. Model selection and preprocessing never fit on validation/test rows.',
      '',figure('04_feature_evidence','Training-only evidence for the candidate features; associations are not causal effects.'),
      '',
      '## Statistical-model comparison (work packages 9–10)',
      'Baseline: logistic regression, a regularized linear model of log-odds; it is interpretable but cannot represent arbitrary interactions without engineered terms. Compare C = 0.1, 1, 10 on training/validation. Random forest averages 250 trees and can learn nonlinearities/interactions, but can overfit, struggle with distribution shifts/unseen categories, and yields less direct interpretation. Compare max_depth = 6 or 12 with min_samples_leaf = 3.',
      '',
      'Both models receive the same rows, feature whitelist and encoded feature names. Each pipeline learns median imputation with missing indicators, scaling, and one-hot categories on training data only. Scaled values do not change the ordering used by tree splits. Unknown categories are ignored by the fitted encoder and remain a limitation.',
      '',
      'Primary selection metric: average precision (AP), because podiums are the minority class and precision–recall ranking is informative. AP is reported separately from trapezoidal PR-AUC; they are not silently equated. Secondary metrics include F1 at the chosen validation threshold and ROC-AUC, with accuracy, precision and recall for context. Thresholds are selected from 0.05–0.95 using validation F1 only. Hyperparameters are ranked by validation AP. The brief names three metrics but elsewhere says five; all six plan metrics plus AP are reported.',
      '',table(metrics),
      '',f'The provisional validation AP leader is {primary["model"]} (AP {primary["average_precision"]:.4f}). This is not the final three-model comparison and not evidence of unseen test performance. Training metrics can be optimistic; validation has also been used for choices.',
      '',figure('08_statistical_model_curves','Validation ROC and precision–recall curves; the prevalence line is the PR baseline.'),
      '',figure('logistic_regression_confusion','Logistic-regression validation confusion matrix at its validation-selected threshold.'),
      '',figure('random_forest_confusion','Random-forest validation confusion matrix at its validation-selected threshold.'),
      '',
      '## Remaining work and interpretation limits',
      'No test scores or test predictions are computed in this batch. After NN, ablation, preprocessing experiments and all choices are frozen, evaluate every locked model once on the test split and report all metrics. Add XAI (global permutation importance/SHAP and local LIME/SHAP), inference demonstrations, and the complete PDF with every figure. Every team member must review and defend the whole pipeline. Confirm teammate access before marking setup complete.',
      '',
      'Independent per-driver classifiers need not predict exactly three podiums per race; the brief requests binary classification, not constrained race rankings. Dataset grid values reflect recorded final grids rather than timestamped snapshots, so cutoff validity relies on their interpretation as the grid fixed before the start. Retrospective starter selection and historical shared drives remain material limitations. No numerical grade or perfect prediction accuracy is guaranteed.',
      '',
      '## Sources',
      '- MS1 Description (Project).pdf, pages 1–8 (local official course brief).',
      '- Introduction for Formula 1 Beginners (Project).pdf, pages 1–7 (local course companion).',
      '- [scikit-learn leakage and preprocessing guidance](https://scikit-learn.org/stable/common_pitfalls.html).',
      '- [scikit-learn average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).',
    ]
    destination.parent.mkdir(parents=True,exist_ok=True)
    text = '\n\n'.join(lines)
    destination.write_text(text,encoding='utf-8')
    return text
