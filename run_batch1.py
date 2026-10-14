"""Execute the generated notebook in a fresh local kernel and preserve outputs."""
from pathlib import Path
import json
import os
import sys
root = Path(__file__).resolve().parent
deps = root/'.deps'
if deps.exists(): sys.path.insert(0,str(deps))
import nbformat
from nbclient import NotebookClient

os.environ['PYTHONPATH'] = str(deps)+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['JUPYTER_PATH'] = str(root/'.jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(root/'.jupyter'/'runtime')
os.environ['MPLCONFIGDIR'] = str(root/'.mplconfig')
kernel = root/'.jupyter'/'kernels'/'acl-ms1'
kernel.mkdir(parents=True,exist_ok=True)
(root/'.jupyter'/'runtime').mkdir(exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'ACL local runtime','language':'python','env':{'PYTHONPATH':str(deps)}}))
path = root/'notebooks'/'milestone1.ipynb'
nb = nbformat.read(path,as_version=4)
print('Starting a fresh notebook kernel...',flush=True)
client = NotebookClient(nb,timeout=900,startup_timeout=40,kernel_name='acl-ms1',resources={'metadata':{'path':str(root)}},
    on_cell_start=lambda cell,cell_index: print(f'Cell {cell_index+1}/{len(nb.cells)}: {cell.cell_type}',flush=True))
try:
    client.execute()
finally:
    nbformat.write(nb,path)
errors = [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
assert not errors,errors
assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
print(f'PASS: fresh-kernel Run All completed; {sum(c.cell_type=="code" for c in nb.cells)} code cells, zero errors.')
