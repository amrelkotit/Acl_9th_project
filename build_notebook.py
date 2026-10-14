"""Build a self-contained notebook; no GitHub modules are needed on Kaggle."""
from pathlib import Path
import ast
import sys
sys.path.insert(0,str(Path(__file__).parent/'.deps'))
import nbformat as nbf

root = Path(__file__).parent
batch3_enabled = '--batch3' in sys.argv
batch2_enabled = '--batch2' in sys.argv or batch3_enabled
source = (root/'src'/'batch1.py').read_text(encoding='utf-8')
report_source = (root/'src'/'batch1_report.py').read_text(encoding='utf-8')
lines = source.splitlines(keepends=True)
tree = ast.parse(source)
functions = {node.name:''.join(lines[node.lineno-1:node.end_lineno]) for node in tree.body if isinstance(node,ast.FunctionDef)}
first = next(n.lineno for n in tree.body if isinstance(n,ast.FunctionDef))
definitions = ''.join(lines[:first-1])
# Constant mappings are module-level and needed before the respective functions.
for node in tree.body:
    if isinstance(node,ast.Assign) and node.lineno>first:
        definitions += ''.join(lines[node.lineno-1:node.end_lineno])+'\n'
cells = [nbf.v4.new_markdown_cell('''# ACL Milestone 1 — Formula 1 Podium Prediction
## Batch 1: work packages 1–10
This notebook implements audit/cleaning, an explicit starter cohort, historical analyses,
leakage-safe features, logistic regression, and random forest. It is a first-batch draft,
not the final milestone. NN, experiments, XAI, inference, final test evaluation, and final
PDF remain. Raw CSVs are immutable. Test years 2022–2024 are reserved for final evaluation.

Source: [assignment-cited F1 dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020).
Attach that dataset to Kaggle, then Run All. Local execution runs from repository root.
The historical questions use outcomes descriptively; model selection uses training/validation only.'''),
 nbf.v4.new_code_cell('''from pathlib import Path
import os, sys
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists() and (PROJECT_ROOT.parent / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if (PROJECT_ROOT / '.deps').exists():
    sys.path.insert(0, str(PROJECT_ROOT / '.deps'))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.mplconfig'))
OUT = Path('/kaggle/working/acl_m1_batch1') if Path('/kaggle/input').exists() else PROJECT_ROOT/'outputs'/'batch1'
OUT.mkdir(parents=True, exist_ok=True)
from IPython.display import display, Markdown, Image
'''+definitions)]

def stage(title,explanation,names,call):
    cells.append(nbf.v4.new_markdown_cell('## '+title+'\n\n'+explanation))
    if names:
        cells.append(nbf.v4.new_code_cell('\n\n'.join(functions[n] for n in names)))
    cells.append(nbf.v4.new_code_cell(call))

stage('1–2. Source inventory and integrity audit',
 'Verify all 14 source tables, candidate keys, columns, missingness and foreign keys. Fail on key/orphan problems; record source hashes.',
 ['markdown_table','write_json','file_hash','locate_data','audit_data'],
 'DATA_DIR = locate_data(PROJECT_ROOT)\nraw, raw_hashes, inventory, columns, foreign_keys = audit_data(DATA_DIR, OUT)\ndisplay(foreign_keys)')
stage('3–4. Typed cleaning and eligible driver–race cohort',
 'Normalize sentinels on copies; parse dates/durations. Prefer a positive-grid source row in historical duplicate groups, then smallest resultId. Never choose using the target. Withhold uncertain zero-grid/zero-lap entries and retain evidence of exclusions.',
 ['duration_seconds','clean_data','save_figure','build_cohort','exploratory_analysis'],
 'clean = clean_data(raw, OUT)\ncohort, ordered_races, cohort_summary = build_cohort(clean, OUT)\ncoverage = exploratory_analysis(raw, clean, cohort, OUT)')
stage('5. Feature engineering and chronological split',
 'Training-only coverage motivates 2003 as the model-window start. Split 2003–2019 / 2020–2021 / reserved 2022–2024. Lag driver and constructor form and as-of join strictly earlier within-season standings. Encode IDs as categories; never use current-race outcomes. Feature evidence is training-only.',
 ['add_prior_standings','engineer_features'],
 'model, numeric_features, categorical_features, feature_register = engineer_features(cohort, clean, ordered_races, OUT)\ndisplay(feature_register)')
stage('6. Front-row conversion before and from 2014',
 'Join results → races → circuits and qualifying. Actual grid 1/2 defines front row; denominator is front-row starts. Use 1950–2013 vs 2014–2024, excluding Indianapolis. Save all circuits; the plot displays circuits with at least 10 starts to expose small-sample issues.',
 ['question_front_row'],
 'front_row_rates, circuit_changes, front_row_definitions = question_front_row(cohort, clean, OUT)\ndisplay(circuit_changes.sort_values("change",ascending=False).head(10))')
stage('7. Home-race effect controlling starting grid',
 'Join drivers and circuits through results/races. Map nationality to country explicitly; withhold ambiguous dual nationality. Standardize home/away outcomes to the same pooled grid-stratum distribution on common support. This controls grid coarsely and is not a causal analysis.',
 ['question_home'],
 'home_summary, home_strata, home_definitions = question_home(cohort, clean, OUT)\ndisplay(home_strata)')
stage('8. Mechanical retirements by constructor',
 'Join results → races/status/constructors. Compare 2014–2021 vs 2022–2024 per constructorId. Divide documented mechanical retirements by driver-starts. Exclude accidents, driver and regulatory causes; show ambiguous statuses separately rather than labeling all DNF as mechanical.',
 ['question_mechanical'],
 'mechanical_rates, status_mapping = question_mechanical(cohort, clean, OUT)\ndisplay(status_mapping)')
stage('9–10. Logistic regression and random forest',
 'Use identical rows and features with train-only fitted preprocessing. Logistic regression is the baseline. Select hyperparameters by validation average precision (AP); select thresholds by validation F1. Report trapezoidal PR-AUC and AP separately. Do not evaluate the reserved test yet: freeze choices after the remaining models/experiments.',
 ['make_preprocessor','metrics','train_statistical_models'],
 'statistical_models, model_metrics, model_trials = train_statistical_models(model, numeric_features, categorical_features, OUT)\ndisplay(model_trials)\ndisplay(model_metrics)')
stage('Batch 1 interpretation and report section',
 'The following report is generated from this run’s evidence. It is a draft for the first batch, not a completed submission.',
 ['verify_raw_unchanged'],
 report_source+'\n\nREPORT_PATH = OUT/"batch1_report.md"\nreport_text = write_batch1_report(OUT, REPORT_PATH)\nverify_raw_unchanged(DATA_DIR, raw_hashes)\n# Figures have already been displayed in their matching notebook sections.\nimport re\ndisplay(Markdown(re.sub(r"!\\[[^]]*\\]\\([^)]*\\)", "[Figure shown in the matching notebook section above]", report_text)))')
if batch2_enabled:
    cells[0].source = cells[0].source.replace('## Batch 1: work packages 1–10','## Integrated implementation: work packages 1–20').replace(
        'It is a first-batch draft,\nnot the final milestone. NN, experiments, XAI, inference, final test evaluation, and final\nPDF remain.',
        'The later sections add the shallow NN, controlled experiments, XAI/inference, and a frozen three-model test evaluation. Team review, Kaggle synchronization and official submission remain pending.')
    cells[1].source = cells[1].source.replace('/kaggle/working/acl_m1_batch1','/kaggle/working/acl_m1').replace("PROJECT_ROOT/'outputs'/'batch1'","PROJECT_ROOT/'outputs'/'batch2'")
    source2 = (root/'src'/'batch2.py').read_text(encoding='utf-8')
    lines2 = source2.splitlines(keepends=True)
    tree2 = ast.parse(source2)
    nodes2 = {node.name:''.join(lines2[node.lineno-1:node.end_lineno]) for node in tree2.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    first2 = min(node.lineno for node in tree2.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)))
    header2 = ''.join(lines2[:first2-1])
    cells.append(nbf.v4.new_markdown_cell('## Batch 2: work packages 11–20\n\nThe preceding batch-1 report is a checkpoint. The integrated report at the end supersedes its pending-work statements. All experiments below use training/validation only until the explicit model-choice freeze.'))
    cells.append(nbf.v4.new_code_cell(header2+'\n'+nodes2['QuantileClipper']))
    def stage2(title,explanation,names,call):
        cells.append(nbf.v4.new_markdown_cell('## '+title+'\n\n'+explanation))
        cells.append(nbf.v4.new_code_cell('\n\n'.join(nodes2[n] for n in names)))
        cells.append(nbf.v4.new_code_cell(call))
    stage2('11. Shallow feed-forward neural network',
        'Compare one hidden layer (32 units) and two (32, 16). Train using the same encoded features and training rows as both statistical models. Adam updates use training rows only; explicit 2020–2021 validation AP selects architecture/checkpoint, then validation F1 selects the threshold. Record the full learning history and limitations.',
        ['choose_threshold','train_shallow_network'],
        'all_models, three_model_validation, nn_trials = train_shallow_network(model, numeric_features, categorical_features, statistical_models, OUT)\ndisplay(nn_trials)')
    stage2('12. Feature-group ablation and two preprocessing-order experiments',
        'Retrain a fixed random forest after removing each of the six feature groups separately. Keep rows, RF settings and decision threshold fixed. Experiment 1 compares imputation/scaling orders; experiment 2 compares clipping/scaling orders after imputation. Both order variants use a fixed logistic-regression model and omit numeric missing indicators to isolate the comparison at equal dimension. Bounds/statistics are learned on training only.',
        ['feature_group_ablation','preprocessing_order_experiments','feature_detail_plots'],
        'ablation_results = feature_group_ablation(model, numeric_features, categorical_features, feature_register, all_models, OUT)\norder_results, distribution_effects = preprocessing_order_experiments(model, numeric_features, categorical_features, all_models, OUT)\nfeature_detail_names = feature_detail_plots(model, numeric_features, categorical_features, OUT)')
    stage2('13a. Inference on complete and partially missing pre-race records',
        'The callable consumes engineered pre-race features, uses the fitted pipeline without refitting, handles missing fields, and rejects unknown/outcome columns. Demonstrate both records for every model and verify fitted-model hashes remain unchanged. Scores are not asserted to be calibrated probabilities.',
        ['podium_inference','inference_demonstration'],
        'inference_examples, complete_pre_race_record = inference_demonstration(model, numeric_features, categorical_features, all_models, OUT)\ndisplay(inference_examples)')
    stage2('13b. Global and local explanations of the validation-selected model',
        'Choose the model by validation AP. Compute raw-feature permutation importance, global SHAP on 48 validation records with training-only background, and local SHAP plus LIME for one complete record. Check SHAP additivity to the predicted probability; record LIME local fidelity. Explain model reliance, never causal effects. Sample/background dependence and artificial perturbations are limitations.',
        ['explain_validation_model'],
        'xai_summary, permutation_table, shap_table, lime_table = explain_validation_model(model, numeric_features, categorical_features, all_models, OUT, complete_pre_race_record)\ndisplay(lime_table)')
    stage2('17–20. Freeze choices, then evaluate the three locked models',
        'After all validation-driven work above, keep the full predeclared feature set and baseline main-model preprocessing. Freeze pipeline hashes, hyperparameters, NN checkpoint and thresholds before computing any test predictions. Report all six plan metrics plus separately defined AP on train/validation/test. A changed choice after test scoring raises an error rather than silently retuning.',
        ['freeze_model_choices','evaluate_frozen_models','acceptance_checks_batch2'],
        'frozen_choices = freeze_model_choices(model, numeric_features, categorical_features, all_models, OUT)\nfinal_model_metrics = evaluate_frozen_models(model, numeric_features, categorical_features, all_models, frozen_choices, OUT)\nbatch2_acceptance = acceptance_checks_batch2(model, numeric_features, categorical_features, all_models, OUT)\nverify_raw_unchanged(DATA_DIR, raw_hashes)')
    report2 = (root/'src'/'batch2_report.py').read_text(encoding='utf-8')
    cells.append(nbf.v4.new_markdown_cell('## 14–16. Integrated analytical report and review handoff\n\nThe report incorporates both batches and all figures. Row 16 remains pending team review, successful Kaggle execution, and actual submission confirmation. The local PDF is generated from this same report after notebook execution.'))
    cells.append(nbf.v4.new_code_cell(report2+'\n\nintegrated_report = write_integrated_report(OUT, OUT/"milestone1_report.md")\ndisplay(Markdown(re.sub(r"!\\[[^]]*\\]\\([^)]*\\)", "[Figure displayed in the matching notebook section]", integrated_report)))'))
    cells.append(nbf.v4.new_markdown_cell('## Remaining: acceptance/enhancement rows 21–40\n\nTeam access/review, historical source ambiguities, Kaggle synchronization/runtime verification, metric/LIME wording clarification, and official submission remain pending. Test results must not guide additional model tuning. No team review or submission is claimed complete.'))
else:
    cells.append(nbf.v4.new_markdown_cell('''## Remaining batches
Work packages 11–20 add the shallow NN, experiments, XAI/inference, report integration and acceptance checks.
Rows 21–40 add the remaining acceptance/enhancement checks. Final test metrics are pending until choices are frozen.
Setup is not fully complete until all four teammates can access the private repository and notebook.
No task is marked Done solely because code exists; inspect the run evidence and limitations.'''))
if batch3_enabled:
    cells[0].source = cells[0].source.replace('work packages 1–20','work packages 1–20; acceptance rows 21–30')
    cells[-1] = nbf.v4.new_markdown_cell('## Batch 3: acceptance and enhancement rows 21–30\n\nThe integrated PDF/report above remains the analytical review draft. The following checks add evidence and a final feature register/contract without changing any frozen model choice. The user’s 80% ordinary binary accuracy target is checked alongside the no-podium baseline and podium metrics. Kaggle execution, team access/review and official submission remain pending.')
    cells.append(nbf.v4.new_code_cell((root/'src'/'batch3.py').read_text(encoding='utf-8')))
    cells.append(nbf.v4.new_code_cell('accuracy_target_results, final_feature_register, inference_contract_checks, batch3_summary = run_batch3_acceptance(model, numeric_features, categorical_features, all_models, OUT, DATA_DIR)\ndisplay(accuracy_target_results.loc[accuracy_target_results.split.eq("test"), ["model","accuracy","always_no_podium_accuracy","accuracy_lift_percentage_points","precision","recall","F1","meets_user_accuracy_target"]])\ndisplay(final_feature_register)\ndisplay(inference_contract_checks)'))
    cells.append(nbf.v4.new_markdown_cell('## Handoff after rows 21–30\n\nRun verify_batch3.py locally to bind notebook/report/PDF hashes and produce the coverage matrix. The original owner assignments and Done cells require member review. Row 23 remains partial until Kaggle Run All, accessible team links, official submission and confirmation are verified. Remaining ten-row batch: 31–40. The test set has been revealed; no further tuning may use its scores.'))
nb = nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
target = root/'notebooks'/'milestone1.ipynb'
nbf.write(nb,target)
print(f'Built {target} with {len(cells)} cells.')
