"""Independent checks, including future-outcome invariance of pre-race features."""
from pathlib import Path
import json
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/'.deps'))
sys.path.insert(0,str(root/'src'))
import numpy as np
import pandas as pd
import nbformat
import batch1 as b

out = root/'outputs'/'batch1'
manifest = json.loads((out/'source_manifest.json').read_text())
b.verify_raw_unchanged(root/'data'/'raw',manifest['sha256'])
data = {n:pd.read_csv(root/'data'/'raw'/f'{n}.csv',keep_default_na=False) for n in b.PK}
check = root/'outputs'/'verification'
check.mkdir(parents=True,exist_ok=True)
clean = b.clean_data(data,check)
cohort,races,_ = b.build_cohort(clean,check)
b.save_figure = lambda *args,**kwargs: None
# Suppress pyplot work for the metamorphic checks; final notebook figures are already saved.
b.plt.show = lambda: None
base,numeric,categorical,_ = b.engineer_features(cohort,clean,races,check)
cutoff = int(races.loc[(races.year==2020)&(races['round']==5),'event_idx'].iloc[0])
changed = cohort.copy()
mask = changed.event_idx.ge(cutoff)
changed.loc[mask,'podium'] = 1-changed.loc[mask,'podium']
changed.loc[mask,'positionOrder'] = 100-changed.loc[mask,'positionOrder']
altered_clean = {n:d.copy(deep=True) for n,d in clean.items()}
future_races = set(races.loc[races.event_idx.ge(cutoff),'raceId'])
for name in ['driver_standings','constructor_standings']:
    future = altered_clean[name].raceId.isin(future_races)
    altered_clean[name].loc[future,'points'] = 999999
    altered_clean[name].loc[future,'position'] = 999
perturbed,*_ = b.engineer_features(changed,altered_clean,races,check)
cols = b.KEY+numeric+categorical
before = base.loc[base.event_idx.le(cutoff),cols].sort_values(b.KEY).reset_index(drop=True)
after = perturbed.loc[perturbed.event_idx.le(cutoff),cols].sort_values(b.KEY).reset_index(drop=True)
pd.testing.assert_frame_equal(before,after)
print('PASS: changing current/future outcomes and standings does not change current/earlier pre-race features.')
metric = pd.read_csv(out/'statistical_model_metrics.csv')
assert set(metric.split)=={'train','validation'}
assert len(metric)==4
assert np.isfinite(metric.select_dtypes(include='number')).all().all()
assert metric[['accuracy','precision','recall','F1','ROC_AUC','PR_AUC','average_precision']].ge(0).all().all()
assert metric[['accuracy','precision','recall','F1','ROC_AUC','PR_AUC','average_precision']].le(1).all().all()
contract = json.loads((out/'model_feature_contract.json').read_text())
assert contract['test_evaluated'] is False
nb = nbformat.read(root/'notebooks'/'milestone1.ipynb',as_version=4)
nbformat.validate(nb)
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
assert not [o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
print('PASS: valid executed notebook, finite metrics and reserved test contract.')
result = {'raw_hashes_unchanged':True,'future_outcome_and_standings_invariance':True,
    'executed_notebook_no_errors':True,'test_model_scores_computed':False}
b.write_json(out/'verification_results.json',result)
