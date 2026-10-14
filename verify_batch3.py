"""Bind rows 21–30 evidence to the executed notebook and unchanged review PDF."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'.deps'))
import pandas as pd
import nbformat
from pypdf import PdfReader

out=root/'outputs/batch2'
evidence=out/'batch3'
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def normal(text): return re.sub(r'[^a-z0-9]','',text.casefold())
def table(frame):
    def value(x):
        if isinstance(x,float): return f'{x:.4f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])

notebook_path=root/'notebooks/milestone1.ipynb'
report_path=out/'milestone1_report.md'
pdf_path=root/'output/pdf/ACL_Milestone_1_Review_Report.pdf'
notebook=nbformat.read(notebook_path,as_version=4)
nbformat.validate(notebook)
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
assert not [o for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='error']
assert (root/'src/batch3.py').read_text(encoding='utf-8') in '\n'.join(c.source for c in notebook.cells)
inline_figures=[o for c in notebook.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
report=report_path.read_text(encoding='utf-8')
published_report=root/'reports/milestone1-report.md'
assert report==published_report.read_text(encoding='utf-8').replace('(milestone1-evidence/figures/','(figures/')
plots=re.findall(r'!\[([^]]*)\]\(([^)]*)\)',report)
assert len(plots)==len(inline_figures)==26
assert {Path(path).name for _,path in plots}=={p.name for p in (out/'figures').glob('*.png')}
pdf=PdfReader(pdf_path)
texts=[page.extract_text() or '' for page in pdf.pages]
normalized=[normal(t) for t in texts]
assert sum(len(page.images) for page in pdf.pages)==len(plots)
coverage=[]
for ordinal,(caption,relative) in enumerate(plots,1):
    pages=[i+1 for i,text in enumerate(normalized) if normal(caption) in text]
    assert pages,('Missing PDF caption',caption)
    coverage.append({'notebook_figure_ordinal':ordinal,'figure':Path(relative).name,
        'PDF_page':pages[0],'caption':caption,'figure_sha256':digest(out/relative)})
pd.DataFrame(coverage).to_csv(evidence/'figure_coverage.csv',index=False)
headings=[line[3:] for line in report.splitlines() if line.startswith('## ')]
sections=[]
for heading in headings:
    pages=[i+1 for i,text in enumerate(normalized) if normal(heading) in text]
    assert pages,('Missing PDF section',heading)
    sections.append({'section':heading,'PDF_page':pages[0]})
pd.DataFrame(sections).to_csv(evidence/'report_section_pages.csv',index=False)
checks=json.loads((evidence/'technical_acceptance.json').read_text())
frozen=json.loads((out/'model_choices_frozen.json').read_text())
assert frozen['choice_sha256']=='f650c67a131721a39f46243d89dc94b2950b3d970971eb7f2a4ce79c8f4a37d3'
assert checks['inference_cases_passed']==25
assert checks['feature_register_entries']==14
assert checks['model_choice_manifest_unchanged'] and checks['raw_files_unchanged']
accuracy=pd.read_csv(evidence/'accuracy_target_audit.csv')
test=accuracy[accuracy.split.eq('test')].copy()
test['accuracy_percent']=100*test.accuracy
test['majority_baseline_percent']=100*test.always_no_podium_accuracy
assert test.meets_user_accuracy_target.all()
rows=[
    (21,'Local coverage verified; member review pending','All three questions plus feature validity, model comparison, metrics, experiments and limitations are present in notebook/report/PDF.','report_section_pages.csv; figure_coverage.csv'),
    (22,'Local evidence verified; member review pending','Global permutation/SHAP, local SHAP/LIME, noncausal interpretation and complete/missing inference. LIME approximation limitations remain explicit.','xai_summary.json; inference_contract_checks.csv; PDF pp. 17–22'),
    (23,'Partial — external checks pending',f'Fresh local kernel: {sum(c.cell_type=="code" for c in notebook.cells)} code cells, no errors; PDF verified. Kaggle Run All, team access, official submission and confirmation are not verified.','submission_manifest.json; docs/submission-readiness.md'),
    (24,'Review policy preserved','Automated checks provide evidence; they do not supply member approval. Zero Done cells may be changed by this audit.','technical_acceptance.json; Notion original table preserved'),
    (25,'Local register verified; cutoff assumptions remain','14 features with sources, calculation, cutoff assumptions, training-only evidence, detail plots and group ablation.','feature_register_final.csv'),
    (26,'Local experiment evidence verified','Six remove-one-group retrainings; two order comparisons/four variants; 48 numeric distribution audits and consistent validation rows.','feature_group_ablation.csv; preprocessing_order_results.csv; preprocessing_distribution_effects.csv'),
    (27,'Local figure/report coverage verified','26 inline notebook figures, 26 report references and 26 PDF images; every caption mapped to its PDF page.','figure_coverage.csv; report_section_pages.csv'),
    (28,'Local model documentation verified','LR baseline, RF, one/two-hidden-layer NN trials; model settings, limitations and comparison documented. Final contract supersedes initial test-reserved checkpoint.','model_feature_contract_final.json; nn_model_trials.csv; PDF pp. 9–14'),
    (29,'Local inference checks verified','25 cases: field reordering, missing inputs, unseen categories, invalid values/keys and unknown model; fitted models unchanged.','inference_contract_checks.csv; src/batch2.py: podium_inference'),
    (30,'Local audit verified; historical semantics remain','14 tables, documented key graph, sentinels, date/duration conversion and row-conservation checks. Retrospective starting-car/shared-drive uncertainty remains unresolved.','primary_key_recheck.csv; foreign_key_recheck.csv; row_count_audit.csv')]
matrix=pd.DataFrame(rows,columns=['Plan row','State','Evidence interpretation','Evidence location'])
matrix.to_csv(evidence/'plan_rows_21_30.csv',index=False)
manifest={'scope':'Plan rows 21–30; local acceptance evidence, not final submission',
    'artifacts':{str(p.relative_to(root)).replace('\\','/'):{'sha256':digest(p),'bytes':p.stat().st_size}
                 for p in [notebook_path,published_report,pdf_path,root/'src/batch3.py']},
    'frozen_choice_sha256':frozen['choice_sha256'],
    'local_run_all_verified':True,'code_cells':sum(c.cell_type=='code' for c in notebook.cells),
    'analysis_figures':len(plots),'PDF_pages':len(pdf.pages),
    'ordinary_test_accuracy_target_met_all_models':True,
    'Kaggle_uploaded_and_executed':False,'all_member_access_and_review':False,
    'official_submission_and_confirmation':False,'historical_start_ground_truth_fully_resolved':False,
    'Done_cell_updates_authorized_by_this_audit':0,
    'Kaggle_note':'Original private version 5 remains preserved. Earlier upload denial is not bypassed.',
    'official_submission_deadline':'2026-10-18 23:59 Africa/Cairo (supplied brief; unchanged)'}
(evidence/'submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
selected=checks['accuracy_target']['selected_model']
result=test.set_index('model').loc[selected]
text='\n\n'.join([
    '# Milestone 1 — acceptance batch 3 (plan rows 21–30)',
    'Local acceptance evidence is verified. This is a supplement to the integrated analytical report, not a completed milestone or substitute for member review. No model choice, threshold or feature set was changed after test evaluation.',
    '## User accuracy target',
    'Target: at least 80% ordinary binary accuracy, as requested on 14 October 2026. All three frozen models exceed this on the historical 2022–2024 held-out test. This does not guarantee future-race performance. The task predicts podium versus non-podium, not exact finishing position.',
    table(test[['model','accuracy_percent','majority_baseline_percent','accuracy_lift_percentage_points','precision','recall','F1','meets_user_accuracy_target']]),
    f'The validation-selected {selected} has test accuracy {100*result.accuracy:.2f}%, compared with {100*result.always_no_podium_accuracy:.2f}% for always predicting no podium: a lift of {result.accuracy_lift_percentage_points:.2f} percentage points. Its podium precision is {100*result.precision:.2f}% and recall {100*result.recall:.2f}%. The majority baseline itself exceeds the requested 80%, so meeting that number alone does not establish useful podium prediction. AP remains the validation selection metric; test results do not select a new model.',
    '## Row-by-row reconciliation',table(matrix),
    '## Report and figure coverage',
    f'The current review PDF has {len(pdf.pages)} pages and all {len(plots)} saved analysis figures. All figure captions and report section headings were found in extracted PDF text; every figure is also present as an inline notebook output. The PDF was visually checked in batch 2 and is unchanged in this batch. This structural check does not replace those visual checks.',
    table(pd.DataFrame(sections)),
    '## Feature evidence and inference',
    'The final feature register adds explicit calculations, cutoff assumptions, training-only statistics, detail-plot references and measured validation group-ablation drops. The original batch-1 register/contract are preserved as checkpoints; use feature_register_final.csv and model_feature_contract_final.json for final review. Correlation and group ablation are model evidence, not causal or isolated-feature guarantees.',
    'Twenty-five inference contract cases pass without changing fitted-model hashes. Complete/partial demonstrations remain in the analytical report. The callable still expects engineered pre-race history supplied by the caller; it does not construct history from raw entrant identity or establish calibrated real-world probabilities.',
    '## Remaining external verification',
    'Row 23 remains partial. Confirm each teammate can access the private repository and notebook, review the complete work, synchronize accepted changes to Kaggle after the earlier upload denial is resolved, restart and Run All there, and verify regenerated outputs against the submission version. Submit the reviewed PDF and accessible links through the official form only when ready, then retain confirmation. No submission has been made. Original Notion owners and Done cells remain unchanged.',
    'Historical shared-drive/starting-car identities, ambiguous start status, retrospective grid timing, home-race confounding and vague mechanical statuses remain disclosed review items. The notebook and local verification cannot independently resolve missing source ground truth. Metric-count/local-XAI wording and evaluation preparation are addressed in the remaining plan rows 31–40.',
    '## Reproduce',
    'Run python build_notebook.py --batch3, python run_batch1.py, python verify_batch2.py, then python verify_batch3.py. The model-choice lock is preserved and must match the existing frozen digest. The PDF is not regenerated for this supplement. See docs/submission-readiness.md for external checks.',
    '## Artifact binding',
    'submission_manifest.json records SHA-256 and sizes of the executed notebook, measured analytical report, unchanged PDF and acceptance implementation. These hashes bind this review package; they do not prove external submission or teammate review.',
    'Private review: [PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2). Shared plan: [ACL Project Milestone 1 — Shared Team Plan](https://www.notion.so/3f7d34a5d89d8109be93e6df20766856).',
    'Next batch: plan rows 31–40.'])+'\n'
(evidence/'batch3-acceptance.md').write_text(text,encoding='utf-8')
target=root/'reports/batch3-evidence'
target.mkdir(parents=True,exist_ok=True)
for path in evidence.iterdir():
    if path.suffix in {'.csv','.json'}: shutil.copy2(path,target/path.name)
(root/'reports/batch3-acceptance.md').write_text(text,encoding='utf-8')
print(f'PASS: rows 21–30 reconciled; {len(inline_figures)} notebook/report/PDF figures; {checks["inference_cases_passed"]} inference cases.')
print(test[['model','accuracy_percent','majority_baseline_percent','accuracy_lift_percentage_points']].to_string(index=False))
print('Pending: Kaggle execution, team access/review, official submission and historical ground-truth verification.')
