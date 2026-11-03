"""Finite authorized 300 -> 2000 demo replacement and narrow non-force Git push.

All execution is Slurm-only. Never train/infer, alter original sources, discard
old artifacts, include unrelated edits, rewrite history, or modify Git config.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import prepare_crab_demo as demo
from pipeline_common import atomic_json, sha256

ROOT = demo.ROOT
OLD = ROOT/'demo/crab_cam2_2_middle_300'
ARCHIVE = ROOT/'archive/demo/crab_cam2_2_middle_300'
NEW = ROOT/'demo/crab_cam2_2_middle_2000'
REMOTE = 'git@github.com:Xared23/fiddler_crab_sleap_DanielB.git'
BRANCH = 'my-code'
CODE_PATHS = ['.gitignore', 'README.md', 'docs/README.md', 'docs/PROJECT.md',
              'docs/demo/PREPARATION.md', 'docs/demo/REPLACEMENT_2000.md',
              'docs/demo/README_2000.template.md', 'scripts/prepare_crab_demo.py',
              'scripts/replace_crab_demo_2000.py', 'tests/test_crab_demo.py',
              'tests/test_crab_demo_2000.py', 'jobs/replace_crab_demo_2000.sh']


def allowed_path(path):
    return (path in CODE_PATHS or path.startswith('demo/crab_cam2_2_middle_300/')
            or path.startswith('demo/crab_cam2_2_middle_2000/'))


def git(*args, timeout=60, check=True):
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0',
               GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
    result = subprocess.run(['git', *args], cwd=ROOT, env=env, text=True,
                            capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'git {args[0]} failed ({result.returncode}): {result.stderr}')
    return result


def check_local_state(expected_head=None):
    assert git('symbolic-ref', '--short', 'HEAD').stdout.strip() == BRANCH, 'Unexpected branch; not changing it'
    assert git('remote', 'get-url', 'origin').stdout.strip() == REMOTE, 'Unexpected origin'
    assert not git('diff', '--cached', '--name-only').stdout.strip(), 'Existing staged changes; refusing to mix them'
    head = git('rev-parse', 'HEAD').stdout.strip()
    if expected_head is not None:
        assert head == expected_head, 'HEAD changed during build; review before committing'
    return head


def run_tests(logdir):
    # Import after selecting the profile: existing controls bind its actual bounds.
    sys.path.insert(0, str(ROOT/'tests'))
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
                               for name in ['test_crab_demo', 'test_crab_demo_2000'])
    with (logdir/'tests.txt').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    atomic_json(logdir/'TESTS.json', dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors)))
    assert result.wasSuccessful(), 'Test gate failed; see tests.txt; no video encoded'
    print('TESTS_PASSED', result.testsRun, flush=True)


def verify_bundle():
    receipt = json.loads((NEW/'DEMO_COMPLETE.json').read_text())
    assert receipt['status'] == 'validated_complete'
    assert (receipt['frames'], receipt['source_frame_start'], receipt['source_frame_stop_exclusive']) == (2000,13400,15400)
    expected = {f'{demo.NAME}{suffix}' for suffix in ['.mp4','.predictions.slp','.tracked.slp']}
    expected |= {'README.md','observations.parquet','frames.parquet','frame_map.csv','encoding.log'}
    assert set(receipt['artifacts']) == expected
    assert {p.name for p in NEW.iterdir()} == expected | {'DEMO_COMPLETE.json'}
    for name, pin in receipt['artifacts'].items():
        path = NEW/name
        assert path.stat().st_size == pin['bytes'] < 100*1024**2
        assert sha256(path) == pin['sha256']
    return receipt


def publish(logdir, head, source_pins):
    """A network failure leaves validated local files intact and does not retry."""
    check_local_state(head)
    for name, digest in source_pins.items():
        assert sha256(ROOT/name) == digest, f'File changed during build: {name}'
    remote = git('ls-remote', '--exit-code', 'origin', f'refs/heads/{BRANCH}', timeout=90, check=False)
    (logdir/'remote_check.txt').write_text(remote.stdout + remote.stderr)
    if remote.returncode:
        raise RuntimeError('Cannot inspect origin/my-code; no commit/push attempted. See remote_check.txt.')
    lines = remote.stdout.splitlines()
    assert len(lines) == 1 and lines[0].split() == [head, f'refs/heads/{BRANCH}'], (
        'Remote and starting local HEAD differ; refusing to upload unrelated history or force-push')
    paths = CODE_PATHS + [str(OLD.relative_to(ROOT)), str(NEW.relative_to(ROOT))]
    git('add', '-A', '--', *paths)
    staged = git('diff', '--cached', '--name-only', '-z').stdout.split('\0')
    staged = [p for p in staged if p]
    assert staged and all(allowed_path(p) for p in staged), 'Unexpected staged paths'
    (logdir/'staged_paths.json').write_text(json.dumps(staged, indent=2)+'\n')
    # Explicit path-only commit also excludes any unrelated concurrently staged files.
    result = git('commit', '--only', '-m', 'Replace 300-frame crab demo with aligned 2000-frame bundle',
                 '--', *paths, timeout=120)
    (logdir/'commit.txt').write_text(result.stdout + result.stderr)
    commit = git('rev-parse', 'HEAD').stdout.strip()
    assert git('rev-parse', 'HEAD^').stdout.strip() == head
    committed = git('diff-tree', '--no-commit-id', '--name-only', '-r', '-z', commit).stdout.split('\0')
    assert all(allowed_path(p) for p in committed if p)
    push = git('push', '--porcelain', 'origin', f'{commit}:refs/heads/{BRANCH}', timeout=180, check=False)
    (logdir/'push.txt').write_text(push.stdout + push.stderr)
    atomic_json(logdir/'PUSH_RESULT.json', dict(commit=commit, parent=head, remote=REMOTE,
        branch=BRANCH, returncode=push.returncode, status='pushed' if push.returncode == 0 else 'push_failed_or_unconfirmed',
        forced=False))
    if push.returncode:
        raise RuntimeError('Push failed or unconfirmed; no retry. Local commit is preserved; see push.txt.')
    print('PUSHED', commit, 'origin/my-code', flush=True)
    return commit


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    logdir = ROOT/'logs/demo_preparation'/f'replace_{os.environ["SLURM_JOB_ID"]}'
    logdir.mkdir(parents=True, exist_ok=False)
    demo.configure_demo_2000()
    head = check_local_state()
    assert OLD.is_dir() and (OLD/'DEMO_COMPLETE.json').is_file()
    assert not NEW.exists() and not ARCHIVE.exists(), 'No overwrite/repeated build allowed'
    tracked_old = git('ls-files', '--', str(OLD.relative_to(ROOT))).stdout.splitlines()
    assert any(p.endswith('.mp4') for p in tracked_old), 'Current branch does not track the old video; inspect before replacement'
    source_pins = {p:sha256(ROOT/p) for p in CODE_PATHS}
    old_pins = {p.name:sha256(p) for p in OLD.iterdir() if p.is_file()}
    assert len(old_pins) == len(list(OLD.iterdir())), 'Unexpected subdirectory in old demo'
    atomic_json(logdir/'PLAN.json', dict(starting_head=head, source_sha256=source_pins, old_demo_sha256=old_pins,
        source_start=13400, frames=2000, target_branch=BRANCH, archive=str(ARCHIVE.relative_to(ROOT))))
    run_tests(logdir)
    demo.build()
    receipt = verify_bundle()
    # Archive only after the entire replacement passes source/SLP/video validations.
    for name, digest in old_pins.items():
        assert sha256(OLD/name) == digest, 'Old demo changed concurrently; refusing to move it'
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    OLD.rename(ARCHIVE)
    for name, digest in old_pins.items():
        assert sha256(ARCHIVE/name) == digest
    atomic_json(logdir/'LOCAL_REPLACEMENT_COMPLETE.json', dict(status='validated_local_replacement',
        frames=2000, observations=receipt['observations'], old_demo_sha256=old_pins,
        archive=str(ARCHIVE.relative_to(ROOT)), new_demo=str(NEW.relative_to(ROOT))))
    print('LOCAL_REPLACEMENT_COMPLETE', NEW, flush=True)
    try:
        commit = publish(logdir, head, source_pins)
    except Exception as exc:
        atomic_json(logdir/'PUBLICATION_BLOCKED.json', dict(local_replacement_complete=True,
            error=str(exc), note='No automatic retry, force push, credential/config changes or history rewrite'))
        raise
    atomic_json(logdir/'COMPLETE.json', dict(status='validated_and_pushed', commit=commit,
        frames=2000, branch=BRANCH, job_id=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    main()
