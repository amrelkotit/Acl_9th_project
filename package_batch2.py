"""Copy only compact review evidence; keep data, model binaries and predictions local."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
source=root/'outputs/batch2'
target=root/'reports/milestone1-evidence'
target.mkdir(parents=True,exist_ok=True)
excluded={'modeling_table.csv','starter_cohort.csv','feature_lineage.csv','excluded_entries.csv',
          'duplicate_resolution.csv','validation_predictions.csv','frozen_test_predictions.csv'}
copied=[]
for p in source.iterdir():
    if p.suffix in {'.csv','.json'} and p.name not in excluded:
        shutil.copy2(p,target/p.name)
        copied.append(p.name)
shutil.copytree(source/'figures',target/'figures',dirs_exist_ok=True)
text=(source/'milestone1_report.md').read_text(encoding='utf-8').replace('(figures/','(milestone1-evidence/figures/')
(root/'reports/milestone1-report.md').write_text(text,encoding='utf-8')
print(f'Packaged {len(copied)} compact tables/manifests and {len(list((target/"figures").glob("*.png")))} figures.')
