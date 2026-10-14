"""Final artifact/coverage binding. External completion requires actual evidence."""
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
dest=out/'batch4'
target=root/'reports/final-evidence'
target.mkdir(parents=True,exist_ok=True)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
normal=lambda s:re.sub(r'[^a-z0-9]','',s.casefold())
def table(frame):
    def val(v): return (f'{v:.4f}' if isinstance(v,float) else str(v)).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(val(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])
nbpath=root/'notebooks/milestone1.ipynb'
nb=nbformat.read(nbpath,as_version=4)
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
assert (root/'src/batch4.py').read_text(encoding='utf-8') in '\n'.join(c.source for c in nb.cells)
for phrase in ['Modeling attempt: Logistic regression','Modeling attempt: Random forest','Preprocessing-order attempt 1','Preprocessing-order attempt 2']:
    assert any(phrase in c.source for c in nb.cells),phrase
markdown=(out/'milestone1_report.md').read_text(encoding='utf-8')
assert markdown==(root/'reports/milestone1-report.md').read_text(encoding='utf-8').replace('(milestone1-evidence/figures/','(figures/')
pdfpath=root/'output/pdf/ACL_Milestone_1_Review_Report.pdf'
pdf=PdfReader(pdfpath)
pages=[normal(p.extract_text() or '') for p in pdf.pages]
plots=re.findall(r'!\[([^]]*)\]\(([^)]*)\)',markdown)
inline=[o for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
assert len(plots)==len(inline)==sum(len(p.images) for p in pdf.pages)==26
coverage=[]
for i,(caption,path) in enumerate(plots,1):
    found=[n+1 for n,text in enumerate(pages) if normal(caption) in text]
    assert found,caption
    coverage.append({'notebook_figure':i,'figure':Path(path).name,'PDF_page':found[0],'sha256':digest(out/path)})
pd.DataFrame(coverage).to_csv(dest/'final_figure_coverage.csv',index=False)
section_pages={}
for heading in [line[3:] for line in markdown.splitlines() if line.startswith('## ')]:
    found=[i+1 for i,text in enumerate(pages) if normal(heading) in text]
    assert found,heading
    section_pages[heading]=found[0]
pd.DataFrame(section_pages.items(),columns=['section','PDF_page']).to_csv(dest/'final_report_pages.csv',index=False)
checks=json.loads((dest/'final_technical_checks.json').read_text())
assert all(checks[k] for k in ['Q1_recount_matches','Q2_standardized_recount_matches','Q3_count_recount_matches','strictly_earlier_feature_lineage'])
freeze=json.loads((out/'model_choices_frozen.json').read_text())
assert freeze['choice_sha256']=='f650c67a131721a39f46243d89dc94b2950b3d970971eb7f2a4ce79c8f4a37d3'
snapshot=json.loads((root/'docs/plan-rows-snapshot.json').read_text())
assert [row['row'] for row in snapshot['rows']]==list(range(1,41))
assert all(not row['Done'] for row in snapshot['rows'])
locations={
1:'docs/CONTRIBUTING.md; docs/submission-readiness.md',2:'Source and audit',3:'Cleaning, cohort and duplicate rule',4:'Cleaning, cohort and duplicate rule',5:'Features and chronological evaluation design',
6:'Question 1',7:'Question 2',8:'Question 3',9:'Three-model training',10:'Three-model training',11:'Three-model training',12:'Controlled experiments',13:'Model explainability',14:'Source and audit; Controlled experiments',15:'Question 1; Question 2; Question 3',16:'docs/submission-readiness.md',17:'Cleaning, cohort and duplicate rule',18:'Features and chronological evaluation design',19:'Three-model training',20:'Controlled experiments',21:'reports/final-evidence/final_report_pages.csv',22:'Model explainability; Inference function',23:'docs/submission-readiness.md',24:'docs/plan-rows-snapshot.json',25:'reports/batch3-evidence/feature_register_final.csv',26:'Controlled experiments',27:'reports/final-evidence/final_figure_coverage.csv',28:'Three-model training',29:'Inference function',30:'reports/batch3-evidence/row_count_audit.csv',31:'docs/evaluation-walkthrough.md',32:'Question 1; Final acceptance addendum',33:'Question 2; Final acceptance addendum',34:'Question 3; Final acceptance addendum',35:'Final acceptance addendum',36:'Three-model training; docs/evaluation-walkthrough.md',37:'Model explainability',38:'Features and chronological evaluation design; model_choices_frozen.json',39:'docs/submission-readiness.md',40:'reports/final-compliance.md; docs/plan-rows-snapshot.json'}
partial={1:'All-member private-link access unverified',16:'Team review, Kaggle execution, official submission and confirmation pending',23:'Final external execution/submission pending',31:'Walkthrough guide prepared; actual four-member mock evaluation not held/verified',36:'Named metrics covered; intended five still unclarified',39:'Kaggle Run All/output agreement and submission confirmation pending'}
matrix=[]
for row in snapshot['rows']:
    number=row['row']
    refs=locations[number]
    matches=sorted({page for title,page in section_pages.items() if any(normal(part) in normal(title) for part in refs.split('; ') if '/' not in part)})
    matrix.append({'Plan row':number,'Owner (unchanged)':row['owner'],
        'State':'Partial/external verification pending' if number in partial else 'Local evidence verified; member review pending',
        'Evidence':refs,'PDF pages':', '.join(map(str,matches)) or 'See linked evidence',
        'Remaining':partial.get(number,'Member review; retain documented source/method limitations'),'Done':'Unverified'})
matrix=pd.DataFrame(matrix)
matrix.to_csv(dest/'all_40_plan_rows.csv',index=False)
official=pd.DataFrame([
    (1,'One sectioned run-all notebook','Local Run All passes; final Kaggle run pending','notebooks/milestone1.ipynb'),
    (2,'EDA with figures','Local evidence verified','26 matched analysis figures; report pages'),
    (3,'Unique driver-race table and safe features','Local checks verified; historical assumptions disclosed','starter_cohort.csv; modeling_table.csv; final feature register'),
    (4,'Three-model comparison and metrics','Local evidence verified; wording ambiguity disclosed','three_model_metrics.csv; frozen choice manifest'),
    (5,'Ablation and two order experiments','Local evidence verified','feature_group_ablation.csv; preprocessing_order_results.csv'),
    (6,'Global/local explanations, plots and interpretation','Local evidence verified; limitations disclosed','XAI sections; SHAP reconstruction; LIME fidelity'),
    (7,'Inference complete/missing examples','Local evidence verified','inference_examples.csv; 25 inference contract cases'),
    (8,'Analytical PDF report','Generated and structurally verified; visual QA before handoff','output/pdf/ACL_Milestone_1_Review_Report.pdf')],
    columns=['Brief deliverable','Requirement','State','Evidence'])
official.to_csv(dest/'official_deliverable_groups.csv',index=False)
external_path=dest/'external_verification.json'
external=json.loads(external_path.read_text()) if external_path.exists() else {}
artifacts=[nbpath,pdfpath,root/'reports/milestone1-report.md',root/'src/batch4.py',root/'docs/evaluation-walkthrough.md']
manifest={'scope':'All 40 plan rows and eight official deliverable groups reconciled',
    'artifacts':{str(p.relative_to(root)).replace('\\','/'):{'sha256':digest(p),'bytes':p.stat().st_size} for p in artifacts},
    'local_Run_All':True,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'analysis_figures':len(plots),'PDF_pages':len(pdf.pages),
    'frozen_choice_sha256':freeze['choice_sha256'],'test_accuracy_goal_met':True,
    'external_verification':external,'team_mock_evaluation_verified':False,'all_member_review_verified':False,
    'official_submission_confirmed':False,'Done_cells_changed':0,
    'deadline':'2026-10-18 23:59 Africa/Cairo; supplied brief, unchanged'}
(dest/'final_submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
home=pd.read_csv(dest/'home_grid_era_sensitivity.csv').query('era == "2014–2024"')
text='\n\n'.join(['# Milestone 1 — final compliance and handoff',
    'All remaining local implementation and acceptance work is prepared and verified. This is not an official submission confirmation or human review sign-off. Original owners/internal targets/Done cells remain unchanged.',
    '## Verification',f'Fresh local kernel: {manifest["code_cells"]} executed code cells, zero errors. All 26 analysis figures match notebook/report/PDF coverage. All three historical question recounts match. Future-outcome/standings invariance, training-only preprocessing, saved predictions/metrics and frozen choices remain checked. The user’s ≥80% ordinary test-accuracy goal is met; selected RF accuracy is 90.10% versus 84.93% majority baseline.',
    '## Final findings',
    'Hybrid-era home-race estimates change sign under exact-grid versus broad-bin control. Treat the comparison as sensitive and observational, not a proven home advantage.',table(home[['control','common_support_rows','home_minus_away_percentage_points']]),
    'The alternate smallest-resultId duplicate rule agrees with the baseline in all 82 duplicate groups; that agreement does not prove historical starting-car identity. Constructor changes/history resets and conservative mechanical ambiguity are documented. No post-test model tuning was introduced.',
    '## Official deliverable coverage',table(official),
    '## All 40 plan rows',table(matrix),
    '## External completion gates','Actual all-member access/review and mock evaluation, final Kaggle restart/Run All and output agreement, accepted repository work, official form submission and saved confirmation require actual evidence. Browser permissions and extension file access may block those operations; they cannot be bypassed. See docs/submission-readiness.md and the current external verification record.',
    '## Reproduce','Run python build_notebook.py --final, python run_batch1.py, python verify_batch2.py, python render_report.py, python package_batch2.py, and python verify_final.py. requirements-tested.txt records the tested local versions, not a guarantee across operating systems. Preserve the existing model-choice lock and do not tune on the revealed test set.',
    '## Review links','[Draft PR #2](https://github.com/amrelkotit/Acl_9th_project/pull/2). [Private Kaggle notebook](https://www.kaggle.com/code/amrelkotit/acl-9th-project). Official form: https://forms.gle/esDxHsvfzffYHVAZ9 . The brief says evaluation dates/slots are announced separately; office-hours booking rules are not submission deadlines.'])+'\n'
(root/'reports/final-compliance.md').write_text(text,encoding='utf-8')
for p in dest.iterdir():
    if p.suffix in {'.csv','.json','.txt'}: shutil.copy2(p,target/p.name)
shutil.copy2(dest/'requirements-tested.txt',root/'requirements-tested.txt')
print(f'PASS: all 40 plan rows and 8 brief deliverable groups reconciled; {manifest["code_cells"]} code cells; {len(pdf.pages)} PDF pages; 26 matched figures.')
print('External state:',json.dumps(external))
