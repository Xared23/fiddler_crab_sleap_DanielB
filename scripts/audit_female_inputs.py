#!/usr/bin/env python3
"""Read-only inventory of newly uploaded labels; run through Slurm only."""
import json
import os
from audit_labels import inventory
from pipeline_common import ROOT, atomic_json, versions

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Slurm required')
report = {
    'sources': [inventory(ROOT / name) for name in (
        'data/labels/Female_data/female_labels.v002.slp',
        'data/labels/burrows/BurrowLabels3.slp')],
    'versions': versions(), 'job_id': os.environ['SLURM_JOB_ID'],
}
atomic_json(ROOT / 'checks/female_v002/INPUT_AUDIT.json', report)
print(json.dumps(report, indent=2, default=str), flush=True)
