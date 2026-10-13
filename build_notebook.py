"""Build a self-contained notebook; no GitHub modules are needed on Kaggle."""
from pathlib import Path
import ast
import sys
sys.path.insert(0,str(Path(__file__).parent/'.deps'))
import nbformat as nbf

root = Path(__file__).parent
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
cells.append(nbf.v4.new_markdown_cell('''## Remaining batches
Work packages 11–20 add the shallow NN, experiments, XAI/inference, report integration and acceptance checks.
Rows 21–40 add the remaining acceptance/enhancement checks. Final test metrics are pending until choices are frozen.
Setup is not fully complete until all four teammates can access the private repository and notebook.
No task is marked Done solely because code exists; inspect the run evidence and limitations.'''))
nb = nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
target = root/'notebooks'/'milestone1.ipynb'
nbf.write(nb,target)
print(f'Built {target} with {len(cells)} cells.')
