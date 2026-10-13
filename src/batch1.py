"""Milestone 1 work packages 1–10. Functions are embedded in the Kaggle notebook."""
from pathlib import Path
import hashlib
import json
import platform
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve, auc,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay)
import sklearn
import joblib

SEED = 42
KEY = ['raceId', 'driverId']
DATASET = 'https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020'
PK = {
 'circuits':['circuitId'], 'constructors':['constructorId'],
 'constructor_results':['constructorResultsId'], 'constructor_standings':['constructorStandingsId'],
 'drivers':['driverId'], 'driver_standings':['driverStandingsId'], 'races':['raceId'],
 'results':['resultId'], 'qualifying':['qualifyId'], 'lap_times':['raceId','driverId','lap'],
 'pit_stops':['raceId','driverId','stop'], 'sprint_results':['resultId'],
 'seasons':['year'], 'status':['statusId']}
FK = {'races':[('circuitId','circuits','circuitId'),('year','seasons','year')]}
for _name in ['constructor_results','constructor_standings','driver_standings',
              'qualifying','results','sprint_results','lap_times','pit_stops']:
    FK[_name] = [('raceId','races','raceId')]
    if _name not in ['constructor_results','constructor_standings']:
        FK[_name].append(('driverId','drivers','driverId'))
    if _name not in ['driver_standings','lap_times','pit_stops']:
        FK[_name].append(('constructorId','constructors','constructorId'))
    if _name in ['results','sprint_results']:
        FK[_name].append(('statusId','status','statusId'))

NUMERIC = {'number','grid','position','positionOrder','points','laps','milliseconds',
 'fastestLap','rank','fastestLapSpeed','lap','stop','year','round','wins','lat','lng','alt'}

def markdown_table(frame, digits=4):
    """No optional tabulate dependency needed on Kaggle."""
    def cell(x):
        if pd.isna(x): return '—'
        if isinstance(x, (float, np.floating)): return f'{x:.{digits}f}'
        return str(x).replace('|', '/')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(cell(v) for v in r)+' |' for r in frame.itertuples(index=False,name=None)])

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')

def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def locate_data(project_root):
    """Use a single complete dataset directory; never mix unrelated input CSVs."""
    project_root = Path(project_root)
    if Path('/kaggle/input').exists():
        candidates = [p.parent for p in Path('/kaggle/input').rglob('results.csv')
                      if set(PK).issubset({x.stem for x in p.parent.glob('*.csv')})]
        if len(candidates) != 1:
            raise ValueError(f'Attach exactly one complete F1 dataset; found {candidates}')
        return candidates[0]
    raw = project_root/'data'/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    archive = project_root/'data'/'f1-source.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for name in PK:
                dest = raw/f'{name}.csv'
                contents = z.read(f'{name}.csv')
                if dest.exists():
                    assert dest.read_bytes() == contents, f'Refusing to overwrite changed raw file: {dest}'
                else:
                    dest.write_bytes(contents)
    assert set(PK).issubset({x.stem for x in raw.glob('*.csv')}), 'Download the assignment dataset first.'
    return raw

def audit_data(data_dir, out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    paths = {n:Path(data_dir)/f'{n}.csv' for n in PK}
    hashes = {n:file_hash(p) for n,p in paths.items()}
    raw = {n:pd.read_csv(p,keep_default_na=False) for n,p in paths.items()}
    audit, columns, relationships = [], [], []
    for name, df in raw.items():
        missing = df.apply(lambda s:s.astype(str).str.strip().isin(['',r'\N']))
        pk_null = int(missing[PK[name]].any(axis=1).sum())
        dup = int(df.duplicated(PK[name]).sum())
        audit.append({'table':name,'rows':len(df),'columns':len(df.columns),
                      'primary_key':', '.join(PK[name]),'null_PK_rows':pk_null,
                      'duplicate_PK_rows':dup,'missing_cells':int(missing.sum().sum())})
        for c in df:
            columns.append({'table':name,'column':c,'raw_dtype':str(df[c].dtype),
                            'missing_count':int(missing[c].sum()),'missing_fraction':float(missing[c].mean())})
        for c,parent,pc in FK.get(name,[]):
            source = df.loc[~missing[c],c].astype(str)
            orphan = int((~source.isin(raw[parent][pc].astype(str))).sum())
            relationships.append({'child_table':name,'foreign_key':c,'parent_table':parent,
                                  'parent_key':pc,'orphan_rows':orphan})
    audit, columns, relationships = map(pd.DataFrame,[audit,columns,relationships])
    assert audit.null_PK_rows.sum()==0 and audit.duplicate_PK_rows.sum()==0
    assert relationships.orphan_rows.sum()==0
    audit.to_csv(out/'table_inventory.csv',index=False)
    columns.to_csv(out/'column_inventory.csv',index=False)
    relationships.to_csv(out/'foreign_key_checks.csv',index=False)
    manifest = {'source':DATASET,'download_date':'2026-10-14','sha256':hashes,
                'python':platform.python_version(),'pandas':pd.__version__,
                'numpy':np.__version__,'sklearn':sklearn.__version__}
    write_json(out/'source_manifest.json',manifest)
    print(audit.to_string(index=False))
    return raw, hashes, audit, columns, relationships

def duration_seconds(value):
    if pd.isna(value): return np.nan
    parts = str(value).lstrip('+').split(':')
    try:
        total = 0.0
        for part in parts: total = total*60+float(part)
        return total
    except ValueError:
        return np.nan

def clean_data(raw, out):
    clean, conversions = {}, []
    dest = Path(out)/'cleaned'
    dest.mkdir(exist_ok=True)
    durations = {'qualifying':['q1','q2','q3'],'lap_times':['time'],
                 'pit_stops':['duration'],'results':['time','fastestLapTime'],
                 'sprint_results':['time','fastestLapTime']}
    for name, source in raw.items():
        df = source.copy(deep=True)
        for c in df:
            if df[c].dtype == 'object' or pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].astype('string').str.strip().replace({'':pd.NA,r'\N':pd.NA})
            if c.endswith('Id') or c in NUMERIC:
                before = df[c].notna()
                df[c] = pd.to_numeric(df[c],errors='coerce')
                conversions.append({'table':name,'column':c,'type':'numeric',
                                    'new_missing':int((before & df[c].isna()).sum())})
            elif c=='dob' or c=='date' or c.endswith('_date'):
                before = df[c].notna()
                df[c] = pd.to_datetime(df[c],format='%Y-%m-%d',errors='coerce')
                conversions.append({'table':name,'column':c,'type':'date',
                                    'new_missing':int((before & df[c].isna()).sum())})
        for c in durations.get(name,[]):
            df[c+'_seconds'] = df[c].map(duration_seconds)
            conversions.append({'table':name,'column':c,'type':'duration_seconds',
                'new_missing':int((df[c].notna() & df[c+'_seconds'].isna()).sum())})
        clean[name] = df
        df.to_csv(dest/f'{name}.csv',index=False)
    conversions = pd.DataFrame(conversions)
    conversions.to_csv(Path(out)/'type_conversion_audit.csv',index=False)
    assert conversions.new_missing.sum()==0, 'Investigate parsing failures before proceeding.'
    return clean

def save_figure(fig, out, name):
    fig.tight_layout()
    path = Path(out)/'figures'/f'{name}.png'
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path,dpi=150,bbox_inches='tight')
    plt.show()
    plt.close(fig)
    return path

def build_cohort(clean, out):
    races = clean['races'].sort_values(['date','round','raceId']).copy()
    races['event_idx'] = np.arange(len(races))
    r = clean['results'].merge(races[['raceId','year','round','date','name','circuitId','event_idx']],
        on='raceId',validate='many_to_one').merge(clean['status'].rename(columns={'status':'status_text'}),
        on='statusId',validate='many_to_one')
    assert len(r)==len(clean['results'])
    no_qual = r.status_text.isin(['Did not qualify','Did not prequalify','107% Rule'])
    nonstart = (r.laps.eq(0) & r.status_text.isin(['Withdrew','Excluded','Injured','Injury',
                           'Driver unwell','Illness','Safety concerns','Safety']))
    ambiguous = r.grid.eq(0) & r.laps.eq(0) & ~no_qual & ~nonstart
    r['cohort_exclusion'] = np.select([no_qual,nonstart,ambiguous],
        ['non_qualifier','documented_non_start','ambiguous_zero_grid_zero_laps'],default='included')
    r.loc[r.cohort_exclusion.ne('included')].to_csv(Path(out)/'excluded_entries.csv',index=False)
    r.loc[ambiguous].to_csv(Path(out)/'ambiguous_start_entries.csv',index=False)
    starters = r.loc[r.cohort_exclusion.eq('included')].copy()
    duplicate_rows = starters.loc[starters.duplicated(KEY,keep=False)].copy()
    # Prefer an identifiable starting car over a zero-grid handover; then source ID.
    # Neither label nor finishing position is used to pick the retained feature row.
    starters['_has_grid'] = starters.grid.gt(0).astype(int)
    starters = starters.sort_values(KEY+['_has_grid','resultId'],ascending=[True,True,False,True])
    chosen = starters.drop_duplicates(KEY,keep='first').copy()
    duplicate_rows['retained'] = duplicate_rows.resultId.isin(chosen.resultId)
    duplicate_rows.to_csv(Path(out)/'duplicate_resolution.csv',index=False)
    chosen = chosen.drop(columns='_has_grid').sort_values(['event_idx','driverId']).reset_index(drop=True)
    chosen['podium'] = chosen.positionOrder.le(3).astype(int)
    assert not chosen.duplicated(KEY).any()
    assert chosen.positionOrder.notna().all()
    # Historical zero-grid rows can mean missing grid or a shared drive; do not call all pit-lane starters.
    chosen['grid_zero_flag'] = chosen.grid.eq(0).astype(int)
    chosen['grid_position'] = chosen.grid.where(chosen.grid.gt(0))
    chosen.to_csv(Path(out)/'starter_cohort.csv',index=False)
    summary = {'raw_result_rows':len(r),'excluded_non_qualifiers':int(no_qual.sum()),
        'excluded_documented_non_starts':int(nonstart.sum()),'ambiguous_starts_withheld':int(ambiguous.sum()),
        'duplicate_rows_removed':len(starters)-len(chosen),'eligible_driver_race_rows':len(chosen),
        'retained_grid_zero_rows':int(chosen.grid_zero_flag.sum()),
        'rule':'Prefer positive-grid source row; break ties by smallest resultId, never by finishing position.',
        'limitation':'Results do not independently verify every DNS or pit-lane start. Positive-grid zero-lap engine/accident cases may include start-line failures. Cohort is conditional on reconstructed race starters.'}
    write_json(Path(out)/'cohort_summary.json',summary)
    print(json.dumps(summary,indent=2))
    return chosen, races, summary

def exploratory_analysis(raw, clean, cohort, out):
    missing_raw = raw['results'].apply(lambda s:s.astype(str).isin(['',r'\N'])).sum().sum()
    missing_clean = clean['results'][list(raw['results'].columns)].isna().sum().sum()
    fig, axes = plt.subplots(1,3,figsize=(12,3.8))
    axes[0].bar(['Raw sentinels','Typed missing'],[missing_raw,missing_clean],color=['#64748b','#0f766e'])
    axes[0].set_title('Missing cells represented consistently')
    axes[1].bar(['Raw results','Starter cohort'],[len(raw['results']),len(cohort)])
    axes[1].set_title('Cohort/exclusion effect on row count')
    axes[2].bar(['Before','After'],[clean['results'].duplicated(KEY,keep=False).sum(),cohort.duplicated(KEY,keep=False).sum()])
    axes[2].set_title('Rows in duplicate driver–race keys')
    save_figure(fig,out,'01_before_after_cleaning')
    q = clean['qualifying']
    assert not q.duplicated(KEY).any(), 'Resolve qualifying duplicates explicitly.'
    coverage = cohort[KEY+['year']].merge(q[KEY+['position']],on=KEY,how='left',validate='one_to_one')
    coverage['observed'] = coverage.position.notna()
    coverage = coverage.groupby('year').agg(starters=('observed','size'),qualifying_coverage=('observed','mean')).reset_index()
    coverage.to_csv(Path(out)/'qualifying_coverage_by_season.csv',index=False)
    fig, ax = plt.subplots(figsize=(11,3.8))
    ax.plot(coverage.year,coverage.qualifying_coverage,marker='.',label='Observed qualifying position')
    ax.axvline(2003,color='#a16207',linestyle='--',label='Model window begins (training EDA)')
    ax.set(xlabel='Season',ylabel='Fraction of eligible starters',ylim=(0,1.05),title='Historical qualifying coverage')
    ax.legend()
    save_figure(fig,out,'02_qualifying_coverage')
    base = cohort.groupby('year').agg(starts=('podium','size'),podium_rate=('podium','mean'))
    fig, axes = plt.subplots(1,2,figsize=(11,3.8))
    axes[0].plot(base.index,base.starts); axes[0].set(title='Eligible driver starts by season',xlabel='Season',ylabel='Starts')
    axes[1].plot(base.index,base.podium_rate); axes[1].set(title='Podium class prevalence by season',xlabel='Season',ylabel='Podium fraction')
    save_figure(fig,out,'03_cohort_by_season')
    return coverage

def add_prior_standings(frame, standings, races, entity, prefix):
    """As-of join strictly earlier main-race standings, reset within each season."""
    history = standings.merge(races[['raceId','year','event_idx']],on='raceId',validate='many_to_one')
    history = history.rename(columns={'event_idx':prefix+'_source_event','points':prefix+'_points_before',
                                     'position':prefix+'_rank_before'})
    cols = [entity,'year',prefix+'_source_event',prefix+'_points_before',prefix+'_rank_before']
    history = history[cols].sort_values(prefix+'_source_event')
    result = pd.merge_asof(frame.sort_values('event_idx'),history,left_on='event_idx',
        right_on=prefix+'_source_event',by=[entity,'year'],direction='backward',allow_exact_matches=False)
    ok = result[prefix+'_source_event'].notna()
    assert (result.loc[ok,prefix+'_source_event'] < result.loc[ok,'event_idx']).all()
    # Zero points is safe only when no earlier record exists for that entrant this season.
    result.loc[~ok,prefix+'_points_before'] = 0.0
    return result.sort_values(['event_idx','driverId']).reset_index(drop=True)

def engineer_features(cohort, clean, races, out):
    f = cohort.copy()
    q = clean['qualifying'].copy()
    q['q1_gap_to_best'] = q.q1_seconds-q.groupby('raceId').q1_seconds.transform('min')
    q = q[KEY+['position','q1_gap_to_best']].rename(columns={'position':'qualifying_position'})
    f = f.merge(q,on=KEY,how='left',validate='one_to_one').sort_values(['event_idx','driverId'])
    f['driver_form_podium_last5'] = f.groupby('driverId').podium.transform(lambda s:s.shift().rolling(5,min_periods=1).mean())
    f['driver_form_finish_last5'] = f.groupby('driverId').positionOrder.transform(lambda s:s.shift().rolling(5,min_periods=1).mean())
    f['driver_form_source_event'] = f.groupby('driverId').event_idx.shift()
    history = f.groupby(['constructorId','raceId','event_idx'],as_index=False).agg(team_race_podium_rate=('podium','mean'))
    history = history.sort_values('event_idx')
    history['constructor_form_podium_last5'] = history.groupby('constructorId').team_race_podium_rate.transform(lambda s:s.shift().rolling(5,min_periods=1).mean())
    history['constructor_form_source_event'] = history.groupby('constructorId').event_idx.shift()
    f = f.merge(history[['constructorId','raceId','constructor_form_podium_last5','constructor_form_source_event']],
                on=['constructorId','raceId'],validate='many_to_one')
    for source in ['driver_form_source_event','constructor_form_source_event']:
        valid = f[source].notna()
        assert (f.loc[valid,source] < f.loc[valid,'event_idx']).all()
    f = add_prior_standings(f,clean['driver_standings'],races,'driverId','driver_standings')
    f = add_prior_standings(f,clean['constructor_standings'],races,'constructorId','constructor_standings')
    # Modeling window chosen from training-only qualifying coverage; history still includes earlier races.
    model = f.loc[f.year.between(2003,2024)].copy().reset_index(drop=True)
    numeric = ['grid_position','grid_zero_flag','qualifying_position','q1_gap_to_best',
        'driver_form_podium_last5','driver_form_finish_last5','constructor_form_podium_last5',
        'driver_standings_points_before','driver_standings_rank_before',
        'constructor_standings_points_before','constructor_standings_rank_before','round']
    categorical = ['constructorId','circuitId']
    features = numeric+categorical
    for c in numeric: model[c] = pd.to_numeric(model[c],errors='coerce').astype(float)
    for c in categorical: model[c] = model[c].astype(int).astype(str)
    model['split'] = np.select([model.year.le(2019),model.year.between(2020,2021)],['train','validation'],default='test_reserved')
    assert not model.duplicated(KEY).any()
    groups = model.groupby('split').agg(rows=('podium','size'),races=('raceId','nunique'),start_year=('year','min'),end_year=('year','max'))
    groups.to_csv(Path(out)/'split_summary.csv')
    race_sets = [set(model.loc[model.split.eq(s),'raceId']) for s in ['train','validation','test_reserved']]
    assert all(not race_sets[i]&race_sets[j] for i in range(3) for j in range(i+1,3))
    assert not set(features)&{'podium','positionOrder','statusId','points','laps','milliseconds','rank','fastestLapTime'}
    model[KEY+['year','split','podium']+features].to_csv(Path(out)/'modeling_table.csv',index=False)
    model[KEY+['event_idx','driver_form_source_event','constructor_form_source_event',
        'driver_standings_source_event','constructor_standings_source_event']].to_csv(Path(out)/'feature_lineage.csv',index=False)
    specs = {
      'grid_position':('grid','results.grid','Actual positive starting position after penalties; zero represented separately.'),
      'grid_zero_flag':('grid','results.grid','Zero-grid indicator; never interpret zero as better than pole.'),
      'qualifying_position':('qualifying','qualifying.position','Qualifying rank available after session; not equivalent to grid.'),
      'q1_gap_to_best':('qualifying','qualifying.q1','Seconds behind fastest recorded Q1 at same race; session-normalized.'),
      'driver_form_podium_last5':('form','earlier results','Mean podium label over up to five earlier starts, shifted first.'),
      'driver_form_finish_last5':('form','earlier results','Mean positionOrder over up to five earlier starts, shifted first.'),
      'constructor_form_podium_last5':('form','earlier results','Mean of team per-race podium rates over up to five earlier races; constructorId history.'),
      'driver_standings_points_before':('standings','driver_standings + races','Most recent strictly earlier race within same season; zero before first record.'),
      'driver_standings_rank_before':('standings','driver_standings + races','Most recent strictly earlier rank this season; missing before first record.'),
      'constructor_standings_points_before':('standings','constructor_standings + races','Most recent strictly earlier constructor points in season; zero before first record.'),
      'constructor_standings_rank_before':('standings','constructor_standings + races','Most recent strictly earlier constructor rank in season; missing before first record.'),
      'round':('context','races.round','Known scheduled championship round; context for season progress.'),
      'constructorId':('identity','results.constructorId','Pre-race entrant team identity; categorical, not an ordinal number.'),
      'circuitId':('context','races.circuitId','Known circuit identity; categorical, not an ordinal number.')}
    train = model.loc[model.split.eq('train')]
    evidence = []
    for c in features:
        group, source, meaning = specs[c]
        correlation = train[c].corr(train.podium,method='spearman') if c in numeric and train[c].nunique()>1 else np.nan
        evidence.append({'feature':c,'group':group,'source':source,'pre_race_validity':meaning,
                         'training_missing_fraction':train[c].isna().mean(),'training_spearman_with_podium':correlation,
                         'evidence':'04_feature_evidence.png (training only); group ablation pending batch 2'})
    registry = pd.DataFrame(evidence)
    registry.to_csv(Path(out)/'feature_register.csv',index=False)
    fig, axes = plt.subplots(4,4,figsize=(15,13))
    for ax,c in zip(axes.ravel(),features):
        if c in numeric and train[c].nunique()>8:
            bins = pd.qcut(train[c],5,duplicates='drop')
            rates = train.groupby(bins,observed=True).podium.agg(['mean','size'])
        else:
            rates = train.groupby(c,observed=True).podium.agg(['mean','size']).sort_values('size',ascending=False).head(10)
        ax.bar(np.arange(len(rates)),rates['mean'])
        ax.set_xticks(np.arange(len(rates)),[str(v) for v in rates.index],rotation=65,ha='right',fontsize=7)
        ax.set(title=c,ylabel='Training podium rate',ylim=(0,1))
    for ax in axes.ravel()[len(features):]: ax.axis('off')
    save_figure(fig,out,'04_feature_evidence')
    print(groups.to_string())
    return model, numeric, categorical, registry

COUNTRY = {'British':'UK','German':'Germany','Spanish':'Spain','Finnish':'Finland',
 'Japanese':'Japan','French':'France','Polish':'Poland','Brazilian':'Brazil','Italian':'Italy',
 'Australian':'Australia','Austrian':'Austria','American':'USA','Dutch':'Netherlands',
 'Colombian':'Colombia','Portuguese':'Portugal','Canadian':'Canada','Indian':'India',
 'Hungarian':'Hungary','Irish':'Ireland','Danish':'Denmark','Argentine':'Argentina',
 'Argentinian':'Argentina','Czech':'Czech Republic','Malaysian':'Malaysia','Swiss':'Switzerland',
 'Belgian':'Belgium','Monegasque':'Monaco','Swedish':'Sweden','Venezuelan':'Venezuela',
 'New Zealander':'New Zealand','Chilean':'Chile','Mexican':'Mexico','South African':'South Africa',
 'Liechtensteiner':'Liechtenstein','Rhodesian':'Zimbabwe','Uruguayan':'Uruguay','Thai':'Thailand',
 'East German':'Germany','Russian':'Russia','Indonesian':'Indonesia','Chinese':'China'}

def question_front_row(cohort, clean, out):
    a = cohort.loc[~cohort.name.str.contains('Indianapolis',case=False)].merge(
        clean['circuits'][['circuitId','name']].rename(columns={'name':'circuit_name'}),
        on='circuitId',validate='many_to_one')
    a = a.merge(clean['qualifying'][KEY+['position']].rename(columns={'position':'qualifying_position'}),
                on=KEY,how='left',validate='one_to_one')
    a['era'] = np.where(a.year.lt(2014),'1950–2013','2014–2024')
    result = a.loc[a.grid.isin([1,2])].groupby(['circuitId','circuit_name','era']).podium.agg(starts='size',podiums='sum',rate='mean').reset_index()
    result['rank_within_era'] = result.groupby('era').rate.rank(method='min',ascending=False)
    result = result.sort_values(['era','rate','starts'],ascending=[True,False,False])
    result.to_csv(Path(out)/'q1_front_row_rates.csv',index=False)
    paired = result.pivot(index='circuit_name',columns='era',values='rate').dropna()
    paired['change'] = paired['2014–2024']-paired['1950–2013']
    paired.to_csv(Path(out)/'q1_circuit_changes.csv')
    compar = a.loc[a.qualifying_position.notna()]
    distinction = {'qualifying_rows':len(compar),
        'qualifying_grid_disagreements':int(compar.qualifying_position.ne(compar.grid).sum()),
        'front_row_qualifying_not_grid':int((compar.qualifying_position.isin([1,2])&~compar.grid.isin([1,2])).sum()),
        'Indianapolis_excluded':True}
    write_json(Path(out)/'q1_definitions.json',distinction)
    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    for ax,era in zip(axes,['1950–2013','2014–2024']):
        shown = result.loc[(result.era==era)&(result.starts>=10)].head(10).sort_values('rate')
        ax.barh(shown.circuit_name,shown.rate,color='#0f766e')
        ax.set(xlim=(0,1.07),xlabel='Podiums / front-row starts',title=era+' (at least 10 starts)')
        for i,(_,r) in enumerate(shown.iterrows()): ax.text(r.rate+.01,i,f'{r.podiums}/{r.starts}',va='center',fontsize=8)
    save_figure(fig,out,'05_front_row_by_era')
    print(result.groupby('era').head(5).to_string(index=False))
    return result, paired, distinction

def question_home(cohort, clean, out):
    a = cohort.loc[~cohort.name.str.contains('Indianapolis',case=False)].merge(
        clean['drivers'][['driverId','nationality']],on='driverId',validate='many_to_one').merge(
        clean['circuits'][['circuitId','country']],on='circuitId',validate='many_to_one')
    a['home_country'] = a.nationality.str.strip().map(COUNTRY)
    a['country'] = a.country.replace({'United States':'USA'})
    a['mapping_known'] = a.home_country.notna()
    a.loc[~a.mapping_known,['driverId','nationality']].drop_duplicates().to_csv(Path(out)/'q2_unmapped_nationalities.csv',index=False)
    mapping = a[['nationality','home_country']].drop_duplicates().sort_values('nationality')
    mapping.to_csv(Path(out)/'q2_country_mapping.csv',index=False)
    eligible = a.loc[a.mapping_known & a.grid.gt(0)].copy()
    eligible['home'] = eligible.home_country.eq(eligible.country)
    eligible['grid_stratum'] = pd.cut(eligible.grid,bins=[0,2,5,10,15,np.inf],labels=['1–2','3–5','6–10','11–15','16+'])
    grouped = eligible.groupby(['grid_stratum','home'],observed=True).podium.agg(starts='size',podiums='sum',rate='mean').reset_index()
    grouped.to_csv(Path(out)/'q2_grid_strata.csv',index=False)
    rate = grouped.pivot(index='grid_stratum',columns='home',values='rate').dropna()
    counts = grouped.pivot(index='grid_stratum',columns='home',values='starts').loc[rate.index]
    weights = counts.sum(axis=1)/counts.sum().sum()
    adjusted = (rate.mul(weights,axis=0)).sum()
    raw_rates = eligible.groupby('home').podium.mean()
    summary = pd.DataFrame({'group':['Away','Home'],
        'raw_podium_rate':[raw_rates[False],raw_rates[True]],
        'grid_standardized_rate':[adjusted[False],adjusted[True]]})
    summary.to_csv(Path(out)/'q2_home_summary.csv',index=False)
    weights.rename('common_grid_weight').to_csv(Path(out)/'q2_standardization_weights.csv')
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    rate.rename(columns={False:'Away',True:'Home'}).plot.bar(ax=axes[0],rot=0)
    axes[0].set(title='Podium rates within starting-grid strata',ylabel='Podium fraction',xlabel='Starting grid')
    axes[1].bar(summary.group,summary.grid_standardized_rate,color=['#64748b','#0f766e'])
    axes[1].set(title='Same grid distribution for home and away',ylabel='Standardized podium fraction')
    save_figure(fig,out,'06_home_grid_control')
    exclusions = {'unmapped_driver_rows':int((~a.mapping_known).sum()),
        'mapped_rows_without_positive_grid':int((a.mapping_known & ~a.grid.gt(0)).sum()),
        'eligible_rows':len(eligible),'strata_with_common_support':list(map(str,rate.index)),
        'limitation':'Grid strata retain within-bin differences; driver, constructor, circuit and era confounding remain. Not a causal effect. Dual nationalities excluded rather than guessed.'}
    write_json(Path(out)/'q2_definitions.json',exclusions)
    print(summary.to_string(index=False))
    return summary, grouped, exclusions

MECHANICAL = {'Engine','Gearbox','Transmission','Clutch','Hydraulics','Electrical',
 'Radiator','Suspension','Brakes','Differential','Overheating','Mechanical','Driveshaft',
 'Fuel pressure','Water pressure','Throttle','Steering','Electronics','Heat shield fire',
 'Exhaust','Oil leak','Water leak','Fuel pump','Track rod','Oil pressure','Engine fire',
 'Engine misfire','Pneumatics','Wheel bearing','Fuel system','Oil line','Launch control',
 'Power loss','Drivetrain','Ignition','Battery','Halfshaft','Crankshaft','Alternator',
 'Oil pump','Fuel leak','Injection','Distributor','Turbo','CV joint','Water pump',
 'Spark plugs','Fuel pipe','Oil pipe','Axle','Water pipe','Magneto','Supercharger',
 'Power Unit','ERS','Cooling system'}
ACCIDENT_DRIVER_REG = {'Accident','Collision','Spun off','Disqualified','Underweight',
 'Collision damage','Fatal accident','Driver unwell','Injured','Injury','Illness','Physical','Eye injury'}

def question_mechanical(cohort, clean, out):
    def classify(status):
        if status in MECHANICAL: return 'mechanical'
        if status=='Finished' or status.startswith('+'): return 'finished'
        if status in ACCIDENT_DRIVER_REG: return 'accident_driver_regulation'
        if status in ['Did not qualify','Did not prequalify','107% Rule','Withdrew','Excluded']: return 'non_start_or_exclusion'
        return 'ambiguous_or_other'
    mapping = clean['status'].copy()
    mapping['group'] = mapping.status.map(classify)
    mapping['rule'] = np.where(mapping.group.eq('mechanical'),'Explicit component failure / mechanical system; conservative mapping',
                              'Not counted as a proven mechanical failure')
    mapping.to_csv(Path(out)/'q3_status_mapping.csv',index=False)
    a = cohort.loc[cohort.year.between(2014,2024)].merge(
        clean['constructors'][['constructorId','name']].rename(columns={'name':'constructor_name'}),
        on='constructorId',validate='many_to_one')
    a['mechanical'] = a.status_text.isin(MECHANICAL).astype(int)
    a['ambiguous'] = a.status_text.map(classify).eq('ambiguous_or_other').astype(int)
    a['period'] = np.where(a.year.le(2021),'2014–2021','2022–2024')
    summary = a.groupby(['constructorId','constructor_name','period']).agg(
        starts=('mechanical','size'),mechanical_retirements=('mechanical','sum'),ambiguous_statuses=('ambiguous','sum')).reset_index()
    summary['mechanical_rate'] = summary.mechanical_retirements/summary.starts
    summary = summary.sort_values(['period','mechanical_rate','starts'],ascending=[True,False,False])
    summary.to_csv(Path(out)/'q3_mechanical_rates.csv',index=False)
    a.loc[a.ambiguous.eq(1)].groupby(['period','status_text']).size().rename('count').to_csv(Path(out)/'q3_ambiguous_status_counts.csv')
    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    for ax,period in zip(axes,['2014–2021','2022–2024']):
        shown = summary.loc[summary.period.eq(period)].sort_values('mechanical_rate')
        ax.barh(shown.constructor_name,shown.mechanical_rate,color='#0f766e')
        ax.set(xlabel='Mechanical retirements / constructor starts',title=period)
        for i,(_,r) in enumerate(shown.iterrows()): ax.text(r.mechanical_rate+.002,i,f'{r.mechanical_retirements}/{r.starts}',va='center',fontsize=8)
        ax.set_xlim(0,max(shown.mechanical_rate)*1.3+.01)
    save_figure(fig,out,'07_mechanical_retirements')
    print(summary.groupby('period').head(5).to_string(index=False))
    return summary, mapping

def make_preprocessor(numeric, categorical):
    return ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                             ('scale',StandardScaler())]),numeric),
        ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)],
        sparse_threshold=0)

def metrics(y, probabilities, threshold):
    pred = probabilities>=threshold
    precision, recall, _ = precision_recall_curve(y,probabilities)
    return {'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
        'recall':recall_score(y,pred,zero_division=0),'F1':f1_score(y,pred,zero_division=0),
        'ROC_AUC':roc_auc_score(y,probabilities),'PR_AUC':auc(recall,precision),
        'average_precision':average_precision_score(y,probabilities)}

def train_statistical_models(model, numeric, categorical, out):
    """No test labels, predictions, scores or plots are used in this batch."""
    features = numeric+categorical
    train = model.loc[model.split.eq('train')]
    val = model.loc[model.split.eq('validation')]
    Xtr, Xval = train[features],val[features]
    ytr, yval = train.podium,val.podium
    candidates = [('Logistic regression',{'C':c},LogisticRegression(C=c,max_iter=3000,random_state=SEED)) for c in [0.1,1.0,10.0]]
    candidates += [('Random forest',{'max_depth':d,'min_samples_leaf':3,'n_estimators':250},
        RandomForestClassifier(n_estimators=250,max_depth=d,min_samples_leaf=3,random_state=SEED,n_jobs=2)) for d in [6,12]]
    trials, fitted = [], {}
    for name,params,estimator in candidates:
        pipe = Pipeline([('preprocess',make_preprocessor(numeric,categorical)),('model',estimator)])
        pipe.fit(Xtr,ytr)
        p = pipe.predict_proba(Xval)[:,1]
        score = average_precision_score(yval,p)
        trials.append({'model':name,'parameters':json.dumps(params),'validation_AP':score})
        if name not in fitted or score > fitted[name]['validation_AP']:
            fitted[name] = {'pipeline':pipe,'parameters':params,'validation_AP':score}
    trials = pd.DataFrame(trials)
    trials.to_csv(Path(out)/'statistical_model_trials.csv',index=False)
    results, thresholds, predictions = [], {}, []
    fig, axes = plt.subplots(1,2,figsize=(11,4.2))
    for name,info in fitted.items():
        pipe = info['pipeline']
        p = pipe.predict_proba(Xval)[:,1]
        search = pd.DataFrame({'threshold':np.linspace(.05,.95,91)})
        search['F1'] = [f1_score(yval,p>=t,zero_division=0) for t in search.threshold]
        # Stable tie break: prefer threshold closest to 0.5, then lower threshold.
        search['distance_to_half'] = abs(search.threshold-.5)
        best = search.sort_values(['F1','distance_to_half','threshold'],ascending=[False,True,True]).iloc[0]
        threshold = float(best.threshold)
        info['threshold'] = threshold
        thresholds[name] = {'threshold':threshold,'parameters':info['parameters']}
        search.to_csv(Path(out)/(name.lower().replace(' ','_')+'_threshold_search.csv'),index=False)
        for split,frame in [('train',train),('validation',val)]:
            probability = pipe.predict_proba(frame[features])[:,1]
            results.append({'model':name,'split':split,'threshold':threshold,**metrics(frame.podium,probability,threshold)})
        predictions.append(val[KEY+['podium']].assign(model=name,probability=p))
        RocCurveDisplay.from_predictions(yval,p,name=name,ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yval,p,name=name,ax=axes[1])
        joblib.dump(pipe,Path(out)/(name.lower().replace(' ','_')+'.joblib'))
        fig_cm,ax_cm = plt.subplots(figsize=(4,3.5))
        ConfusionMatrixDisplay.from_predictions(yval,p>=threshold,ax=ax_cm,display_labels=['Not podium','Podium'],colorbar=False)
        ax_cm.set_title(name+' — validation')
        save_figure(fig_cm,out,name.lower().replace(' ','_')+'_confusion')
    axes[0].set_title('Validation ROC curves')
    axes[1].axhline(yval.mean(),color='#64748b',linestyle='--',label='Class prevalence')
    axes[1].set_title('Validation precision–recall curves')
    axes[1].legend()
    save_figure(fig,out,'08_statistical_model_curves')
    results = pd.DataFrame(results)
    results.to_csv(Path(out)/'statistical_model_metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(Path(out)/'validation_predictions.csv',index=False)
    write_json(Path(out)/'selected_statistical_models.json',thresholds)
    names = [tuple(i['pipeline'].named_steps['preprocess'].get_feature_names_out()) for i in fitted.values()]
    assert names[0]==names[1], 'Both models must see exactly the same encoded features.'
    write_json(Path(out)/'model_feature_contract.json',{'input_features':features,'encoded_features':names[0],
        'train_keys_sha256':hashlib.sha256(train[KEY].to_csv(index=False).encode()).hexdigest(),
        'validation_keys_sha256':hashlib.sha256(val[KEY].to_csv(index=False).encode()).hexdigest(),
        'primary_metric':'Average precision (AP), explicitly distinct from trapezoidal PR-AUC',
        'secondary_metrics':['PR_AUC','F1','ROC_AUC'],'test_evaluated':False})
    print(results.to_string(index=False))
    return fitted, results, trials

def verify_raw_unchanged(data_dir, hashes):
    assert all(file_hash(Path(data_dir)/f'{name}.csv')==digest for name,digest in hashes.items())
    print('PASS: all 14 raw CSV hashes unchanged.')
