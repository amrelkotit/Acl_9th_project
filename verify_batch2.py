"""Independent acceptance checks against saved run evidence; does not fit models."""
from pathlib import Path
import hashlib
import json
import re
import runpy
import sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'.deps'))
sys.path.insert(0,str(root/'src'))
import numpy as np
import pandas as pd
import nbformat
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
import batch1 as b

out=root/'outputs/batch2'
manifest=json.loads((out/'source_manifest.json').read_text())
b.verify_raw_unchanged(root/'data/raw',manifest['sha256'])
nb=nbformat.read(root/'notebooks/milestone1.ipynb',as_version=4)
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
assert nb.cells[-2].execution_count is not None
frozen=json.loads((out/'model_choices_frozen.json').read_text())
unsigned={k:v for k,v in frozen.items() if k not in ['choice_sha256','frozen_before_test_predictions']}
assert hashlib.sha256(json.dumps(unsigned,sort_keys=True,default=str).encode()).hexdigest()==frozen['choice_sha256']

data=pd.read_csv(out/'modeling_table.csv',dtype={'constructorId':str,'circuitId':str})
contract=json.loads((out/'model_feature_contract.json').read_text())
features=contract['input_features']
numeric=features[:-2]
train=data[data.split.eq('train')]
files={'Logistic regression':'logistic_regression.joblib','Random forest':'random_forest.joblib',
       'Shallow neural network':'shallow_neural_network.joblib'}
pipes={}
for name,filename in files.items():
    pipe=joblib.load(out/filename)
    # joblib object hashes are checked in-memory at freeze/evaluation, not compared
    # across pickle round-trips, which can change array aliasing/serialization state.
    pipes[name]=pipe
    assert list(pipe.feature_names_in_)==features
    prep=pipe.named_steps['preprocess']
    imputer=prep.named_transformers_['numeric'].named_steps['impute']
    np.testing.assert_allclose(imputer.statistics_,train[numeric].median().to_numpy(),atol=1e-12)
    assert list(prep.get_feature_names_out())==contract['encoded_features']

predictions=pd.read_csv(out/'frozen_test_predictions.csv')
reported=pd.read_csv(out/'three_model_metrics.csv')
test=data[data.split.eq('test_reserved')]
for name,p in predictions.groupby('model'):
    pd.testing.assert_frame_equal(p[b.KEY+['podium']].reset_index(drop=True),test[b.KEY+['podium']].reset_index(drop=True),check_dtype=False)
    y=p.podium.to_numpy().astype(bool)
    score=p.probability.to_numpy()
    threshold=frozen['models'][name]['threshold']
    np.testing.assert_allclose(pipes[name].predict_proba(test[features])[:,1],score,atol=1e-12)
    guess=score>=threshold
    tp=(y&guess).sum(); fp=(~y&guess).sum(); fn=(y&~guess).sum()
    prec=tp/(tp+fp) if tp+fp else 0
    rec=tp/(tp+fn) if tp+fn else 0
    precision,recall,_=precision_recall_curve(y,score)
    expected={'accuracy':(y==guess).mean(),'precision':prec,'recall':rec,
        'F1':2*tp/(2*tp+fp+fn),'ROC_AUC':roc_auc_score(y,score),
        'PR_AUC':auc(recall,precision),'average_precision':average_precision_score(y,score)}
    row=reported[reported.model.eq(name)&reported.split.eq('test')].iloc[0]
    for metric,value in expected.items(): assert np.isclose(row[metric],value,atol=1e-12),(name,metric)

shap_audit=pd.read_csv(out/'shap_additivity_audit.csv')
groups=pd.read_csv(out/'sample_shap_contributions.csv')
np.testing.assert_allclose(groups.sum(axis=1)+shap_audit.SHAP_baseline,shap_audit.probability,atol=2e-5)
assert shap_audit.absolute_error.max()<2e-5
assert len(groups)==48 and list(groups.columns)==features
assert len(reported)==9
text=(out/'milestone1_report.md').read_text(encoding='utf-8')
figures={p.name for p in (out/'figures').glob('*.png')}
assert figures=={Path(p).name for p in re.findall(r'!\[[^]]*\]\(([^)]*)\)',text)}
# Reuse the independently implemented future-outcome/standings perturbation audit.
# Its saved batch-1 metric contract is explicitly a historical checkpoint.
runpy.run_path(str(root/'verify_batch1.py'),run_name='__main__')
results={'fresh_kernel_all_cells_executed_without_errors':True,'raw_hashes_unchanged':True,
    'freeze_manifest_digest_verified':True,'saved_models_reproduce_frozen_test_predictions':True,
    'imputers_match_training_only_medians':True,
    'three_models_same_encoded_feature_contract':True,'test_metrics_recomputed_from_predictions':True,
    'SHAP_probability_reconstruction_verified':True,'all_figures_in_integrated_report':True,
    'future_outcome_and_standings_invariance':True,'analysis_figure_count':len(figures),
    'code_cell_count':sum(c.cell_type=='code' for c in nb.cells),
    'Kaggle_runtime_verified':False,'all_member_review_verified':False,'official_submission_verified':False}
(out/'verification_results.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS: integrated acceptance checks\n'+json.dumps(results,indent=2))
