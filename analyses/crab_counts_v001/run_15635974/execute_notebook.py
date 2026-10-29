"""Execute the count notebook under Slurm and preserve a job-scoped receipt."""
from pathlib import Path
import os, sys, json, hashlib, shutil

assert os.environ.get('SLURM_JOB_ID'), 'Notebook execution requires Slurm'
BASE = Path(__file__).resolve().parent
job = os.environ['SLURM_JOB_ID']
run = BASE / f'run_{job}'
run.mkdir(exist_ok=False)
figures = run / 'figures'
figures.mkdir()
os.environ['CRAB_COUNT_OUTPUT'] = str(figures)
os.environ['JUPYTER_RUNTIME_DIR'] = str(run / 'jupyter_runtime')
share = run / 'jupyter_share'
spec = share / 'kernels' / 'crab-counts'
spec.mkdir(parents=True)
(spec / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Project Python (Slurm)', 'language': 'python'
}))
os.environ['JUPYTER_PATH'] = str(share)

import nbformat
from nbclient import NotebookClient
from PIL import Image
from jupyter_client import KernelManager


class LocalIPCKernelManager(KernelManager):
    """Keep this same-node, finite notebook kernel off TCP."""
    def __init__(self, *args, **kwargs):
        kwargs['transport'] = 'ipc'
        kwargs['ip'] = str(run / 'ipc')
        super().__init__(*args, **kwargs)


source = BASE / 'crab_counts_over_time.ipynb'
shutil.copyfile(source, run / 'notebook_input.ipynb')
for name in ['execute_notebook.py', 'run.sh']:
    shutil.copyfile(BASE / name, run / name)
nb = nbformat.read(source, as_version=4)
nbformat.validate(nb)
client = NotebookClient(nb, timeout=120, kernel_name='crab-counts',
                        kernel_manager_class=LocalIPCKernelManager,
                        resources={'metadata': {'path': str(BASE)}}, allow_errors=False)
client.execute()
for cell in nb.cells:
    if cell.cell_type == 'code':
        assert cell.execution_count is not None
        assert not any(o.output_type == 'error' for o in cell.outputs)
images = [o for c in nb.cells if c.cell_type == 'code' for o in c.outputs
          if o.output_type in ('display_data', 'execute_result') and 'image/png' in o.get('data', {})]
assert len(images) == 2, len(images)
for path in figures.glob('*.png'):
    with Image.open(path) as im:
        im.verify()
# Also read back the provenance JSON produced by the notebook itself.
provenance = json.loads((figures / 'count_provenance.json').read_text())
assert provenance['tests_and_data_validations_passed']
# Leave a normal portable kernel label for future Slurm-backed interactive use.
nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbformat.validate(nb)
nbformat.write(nb, run / 'crab_counts_over_time.executed.ipynb')
nbformat.write(nb, source)
artifacts = []
for p in sorted(run.rglob('*')):
    if p.is_file() and 'jupyter_' not in str(p.relative_to(run)):
        artifacts.append({'path': str(p.relative_to(run)), 'bytes': p.stat().st_size,
                          'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
receipt = {'status': 'complete', 'job_id': job, 'code_cells_executed': sum(c.cell_type == 'code' for c in nb.cells),
           'embedded_plots': len(images), 'tests_and_data_validations_passed': True,
           'notebook_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'python': sys.version, 'artifacts': artifacts}
(run / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('COMPLETE', source, flush=True)
print('OUTPUTS', run, flush=True)
