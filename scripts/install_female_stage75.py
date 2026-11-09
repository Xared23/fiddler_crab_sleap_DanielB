#!/usr/bin/env python3
"""Finite CPU test/install for the authorized female centroid 75-epoch stage.

No job submission/cancellation/requeue, signals, training, polling, or watcher.
Installation uses atomic inode replacement: a currently sourced shell keeps its
old open file, and the tested new launcher is read at the next natural requeue.
Original preflight, manifests, model configs and male code remain unchanged.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from pipeline_common import ROOT, atomic_json, load_run, read_json, sha256
import female_workflow as workflow
import female_stage75 as stage

JOB = '16998958'
CHECKS = stage.CHECKS


def baseline():
    run = load_run(stage.RUN_CONFIG)
    preflight = read_json(ROOT / 'checks/female_v002/PREFLIGHT_COMPLETE.json')
    actual = workflow.input_signature(run)
    assert preflight['passed'] and all(preflight[k] == v for k, v in actual.items()), 'Baseline changed'
    return run, preflight, actual


def test():
    run, preflight, actual = baseline()
    patches = read_json(CHECKS / 'patches.json')
    assert set(patches) == stage.ALLOWED_CHANGES
    candidates = CHECKS / 'candidates'
    originals = CHECKS / 'originals'
    candidates.mkdir(exist_ok=True)
    originals.mkdir(exist_ok=True)
    for name, replacements in patches.items():
        path = ROOT / name
        original = path.read_text()
        text = original
        for replacement in replacements:
            assert text.count(replacement['old']) == 1, f'Patch context ambiguous: {name}'
            text = text.replace(replacement['old'], replacement['new'], 1)
        target = candidates / path.name
        target.write_text(text)
        if path.suffix == '.py':
            compile(text, str(target), 'exec')
        else:
            subprocess.run(['bash', '-n', str(target)], check=True)
        backup = originals / path.name
        if not backup.exists():
            shutil.copy2(path, backup)
        assert sha256(backup) == sha256(path)
    expected = dict(actual, female_implementation=dict(actual['female_implementation']))
    for name in stage.ALLOWED_CHANGES:
        expected['female_implementation'][name] = sha256(candidates / Path(name).name)
    atomic_json(CHECKS / 'CANDIDATE_SIGNATURE.json', expected)
    additions = {p: sha256(ROOT / p) for p in stage.ADDITIONS}
    results = []
    for pattern in ('test_female_workflow.py', 'test_female_stage75.py'):
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
            '-p', pattern, '-v'], cwd=ROOT, text=True, capture_output=True)
        (CHECKS / f'{pattern}.stdout.txt').write_text(result.stdout)
        (CHECKS / f'{pattern}.stderr.txt').write_text(result.stderr)
        print(result.stdout, result.stderr, flush=True)
        results.append({'pattern': pattern, 'exit_code': result.returncode})
    assert additions == {p: sha256(ROOT / p) for p in stage.ADDITIONS}, 'Code changed during tests'
    assert actual == workflow.input_signature(run), 'Baseline changed during tests'
    report = {'passed': all(r['exit_code'] == 0 for r in results), 'suites': results,
        'test_job_id': os.environ['SLURM_JOB_ID'], 'active_signature': expected,
        'additions': additions,
        'checks': ['exact epoch75 full-state save', 'real Lightning 75-epoch CPU training',
            'early-stop deferral to exact target', 'real Lightning continuation to epoch76',
            'identical weights/Adam moments/LR scheduler after resume vs uninterrupted',
            'atomic immutable stage publication', 'fatal save failures block completion',
            'exact epoch75 downstream selection instead of earlier best',
            'unrelated drift rejection', 'existing female regression tests']}
    atomic_json(CHECKS / 'TESTS.json', report)
    if not report['passed']:
        raise RuntimeError('Stage tests failed; live worker and launcher remain unchanged')
    receipt = {'schema_version': 1, 'run_id': 'female_v002', 'tests_passed': True,
        'baseline_preflight': preflight, 'active_signature': expected,
        'policy': stage.load_policy(), 'additions': additions,
        'test_report_sha256': sha256(CHECKS / 'TESTS.json')}
    stage.validate_extension(receipt, preflight, expected)
    if (CHECKS / 'APPROVED.json').exists():
        raise FileExistsError('Approval already exists; no overwrite')
    atomic_json(CHECKS / 'APPROVED.json', receipt)
    print('FEMALE_STAGE75_TESTED_READY; live files unchanged', flush=True)


def import_live():
    spec = importlib.util.spec_from_file_location('female_workflow_installed', ROOT / 'scripts/female_workflow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install():
    run, preflight, actual = baseline()
    receipt = read_json(CHECKS / 'APPROVED.json')
    expected = read_json(CHECKS / 'CANDIDATE_SIGNATURE.json')
    stage.verify_extension(preflight, expected)
    # A single scheduler query, not a polling loop. Require ample time in an
    # already-running allocation so no startup can see a half-installed pair.
    result = subprocess.run(['squeue', '-h', '-j', JOB, '-o', '%T|%L'],
                            capture_output=True, text=True, check=True)
    fields = result.stdout.strip().split('|')
    assert len(fields) == 2 and fields[0] == 'RUNNING', 'Install only during an active allocation'
    remaining = fields[1]
    assert '-' not in remaining
    parts = [int(p) for p in remaining.split(':')]
    seconds = sum(value * 60**i for i, value in enumerate(reversed(parts)))
    assert seconds >= 300, f'Too close to next-hop startup: {remaining}'
    directory = ROOT / 'models/female_v002/centroid'
    assert read_json(directory / 'checkpoint_progress.json')['epoch'] < 74
    with (directory / '.worker.lock').open('a') as lock:
        held = False
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        except BlockingIOError:
            held = True
        assert held, 'Training worker must already hold its run lock'
    # Replace executable files atomically; NEVER truncate an active sourced file.
    for name in sorted(stage.ALLOWED_CHANGES):
        target = ROOT / name
        candidate = CHECKS / 'candidates' / target.name
        assert sha256(candidate) == expected['female_implementation'][name]
        fd, temp_name = tempfile.mkstemp(prefix='.stage75.', dir=target.parent)
        os.close(fd)
        try:
            shutil.copy2(candidate, temp_name)
            assert sha256(Path(temp_name)) == sha256(candidate)
            os.replace(temp_name, target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
    live = import_live()
    assert live.verify_preflight(run) == expected
    assert live.input_signature(run) == expected
    atomic_json(CHECKS / 'INSTALLED.json', {'job_id': os.environ['SLURM_JOB_ID'],
        'centroid_job_id': JOB, 'activation': 'next_natural_requeue_no_restart_or_cancellation',
        'approval_sha256': sha256(CHECKS / 'APPROVED.json'),
        'observed_progress': read_json(directory / 'checkpoint_progress.json'),
        'observed_hop_state': read_json(directory / 'hop_state.json'),
        'active_signature': expected,
        'downstream_job_ids_unchanged': ['16998986', '16999003', '16999004', '16999056', '16999177']})
    print('FEMALE_STAGE75_INSTALLED_AND_VERIFIED; activates at next natural hop', flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['test', 'install'])
    args = parser.parse_args()
    test() if args.action == 'test' else install()
