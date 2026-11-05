"""Build/commit the authorized lossless chunked demo, then try one normal push.

Network access is not required for the local commit. No video encoder, inference,
training, force push, branch/config changes, or automatic retries.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import chunk_crab_demo as demo
from pipeline_common import ROOT, atomic_json, sha256

OLD = ROOT/'demo/crab_cam2_2_middle_300'
ARCHIVE = ROOT/'archive/demo/crab_cam2_2_middle_300'
NEW = demo.FINAL
REMOTE = 'git@github.com:Xared23/fiddler_crab_sleap_DanielB.git'
BRANCH = 'my-code'
CODE_PATHS = ['.gitignore','README.md','docs/README.md','docs/PROJECT.md',
    'docs/demo/PREPARATION.md','docs/demo/REPLACEMENT_2000.md','docs/demo/README_2000.template.md',
    'scripts/prepare_crab_demo.py','scripts/chunk_crab_demo.py','scripts/replace_crab_demo_2000.py',
    'tests/test_crab_demo.py','tests/test_crab_demo_chunks.py','jobs/replace_crab_demo_2000.sh']


def allowed_path(path):
    return path in CODE_PATHS or path.startswith('demo/crab_cam2_2_middle_300/') or path.startswith('demo/crab_cam2_2_middle_2000/')


def git(*args, timeout=120, check=True):
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15')
    result=subprocess.run(['git',*args],cwd=ROOT,env=env,text=True,capture_output=True,timeout=timeout)
    if check and result.returncode: raise RuntimeError(f'git {args[0]} failed: {result.stderr}')
    return result


def check_local_state(expected_head=None):
    assert git('symbolic-ref','--short','HEAD').stdout.strip()==BRANCH
    assert git('remote','get-url','origin').stdout.strip()==REMOTE
    assert not git('diff','--cached','--name-only').stdout.strip(), 'Existing staged work; refusing to mix changes'
    head=git('rev-parse','HEAD').stdout.strip()
    if expected_head is not None: assert head==expected_head, 'HEAD changed during build'
    return head


def run_tests(logdir):
    sys.path.insert(0,str(ROOT/'tests'))
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
        for name in ['test_crab_demo','test_crab_demo_chunks'])
    with (logdir/'tests.txt').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_json(logdir/'TESTS.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors)))
    assert result.wasSuccessful(), 'Test gate failed; see tests.txt'
    print('TESTS_PASSED',result.testsRun,flush=True)


def commit_bundle(logdir, head, pins, receipt):
    check_local_state(head)
    for name,digest in pins.items(): assert sha256(ROOT/name)==digest, f'Source changed during build: {name}'
    paths=CODE_PATHS+[str(OLD.relative_to(ROOT)),str(NEW.relative_to(ROOT))]
    git('add','-A','--',*paths)
    staged=[p for p in git('diff','--cached','--name-only','-z').stdout.split('\0') if p]
    assert staged and all(allowed_path(p) for p in staged)
    expected={str(NEW.relative_to(ROOT)/p) for p in receipt['artifacts']} | {str(NEW.relative_to(ROOT)/'DEMO_COMPLETE.json')}
    indexed=set(git('ls-files','--',str(NEW.relative_to(ROOT))).stdout.splitlines())
    assert indexed==expected, ('Missing/extra demo files in Git index',expected-indexed,indexed-expected)
    assert not git('ls-files','--',str(OLD.relative_to(ROOT))).stdout.strip()
    atomic_json(logdir/'staged_paths.json',staged)
    result=git('commit','--only','-m','Replace 300-frame demo with lossless aligned 2000-frame chunks','--',*paths,timeout=180)
    (logdir/'commit.txt').write_text(result.stdout+result.stderr)
    commit=git('rev-parse','HEAD').stdout.strip()
    assert git('rev-parse','HEAD^').stdout.strip()==head
    changed=[p for p in git('diff-tree','--no-commit-id','--name-only','-r','-z',commit).stdout.split('\0') if p]
    assert all(allowed_path(p) for p in changed)
    # Verify payload bytes really are in this commit, not merely in the worktree.
    for path in sorted(expected):
        actual=git('hash-object','--',path).stdout.strip()
        saved=git('rev-parse',f'{commit}:{path}').stdout.strip()
        assert saved==actual, path
    assert not git('ls-tree','-r','--name-only',commit,'--',str(OLD.relative_to(ROOT))).stdout.strip()
    atomic_json(logdir/'LOCAL_COMMIT_COMPLETE.json',dict(status='validated_and_committed',
        commit=commit,parent=head,frames=2000,files_verified_in_commit=len(expected),branch=BRANCH))
    print('LOCAL_COMMIT_COMPLETE',commit,'demo_files',len(expected),flush=True)
    return commit


def try_push(logdir,commit):
    try:
        push=git('push','--porcelain','origin',f'{commit}:refs/heads/{BRANCH}',timeout=240,check=False)
        (logdir/'push.txt').write_text(push.stdout+push.stderr)
        code=push.returncode
    except subprocess.TimeoutExpired:
        code=None
        (logdir/'push.txt').write_text('Push timed out; remote outcome unconfirmed. No automatic retry.\n')
    result=dict(status='pushed' if code==0 else 'local_commit_ready_push_not_confirmed',
        commit=commit,branch=BRANCH,returncode=code,forced=False,
        manual_command=f'git push origin {BRANCH}')
    atomic_json(logdir/'PUSH_RESULT.json',result)
    if code==0:
        # Read-only verification of the branch we just pushed.
        remote=git('ls-remote','--exit-code','origin',f'refs/heads/{BRANCH}',timeout=60,check=False)
        (logdir/'remote_verification.txt').write_text(remote.stdout+remote.stderr)
        result['remote_commit_verified']=(remote.returncode==0 and remote.stdout.split()[0]==commit)
        atomic_json(logdir/'PUSH_RESULT.json',result)
    print('PUSH_RESULT',json.dumps(result),flush=True)
    return result


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    logdir=ROOT/'logs/demo_preparation'/f'replace_{os.environ["SLURM_JOB_ID"]}'
    logdir.mkdir(parents=True,exist_ok=False)
    head=check_local_state()
    # Compute nodes may not inherit the user's terminal Git identity. Preserve
    # the identity already used by this branch, in this process only; never
    # invent an identity or change local/global Git configuration.
    identity_source='existing_git_environment'
    if git('var','GIT_AUTHOR_IDENT',check=False).returncode:
        identity=git('show','-s','--format=%an%x00%ae','HEAD').stdout.rstrip('\n').split('\0')
        assert len(identity)==2 and identity[0] and '@' in identity[1], 'No usable existing commit identity'
        os.environ.update(GIT_AUTHOR_NAME=identity[0],GIT_AUTHOR_EMAIL=identity[1],
                          GIT_COMMITTER_NAME=identity[0],GIT_COMMITTER_EMAIL=identity[1])
        identity_source='preserved_from_starting_HEAD_for_this_process_only'
    git('var','GIT_AUTHOR_IDENT')
    assert OLD.is_dir() and (OLD/'DEMO_COMPLETE.json').is_file()
    assert not NEW.exists() and not ARCHIVE.exists(), 'No overwrite permitted'
    assert any(p.endswith('.mp4') for p in git('ls-files','--',str(OLD.relative_to(ROOT))).stdout.splitlines())
    pins={p:sha256(ROOT/p) for p in CODE_PATHS}
    old_pins={p.name:sha256(p) for p in OLD.iterdir() if p.is_file()}
    assert len(old_pins)==len(list(OLD.iterdir()))
    atomic_json(logdir/'PLAN.json',dict(starting_head=head,source_sha256=pins,old_demo_sha256=old_pins,
        requested_frames=2000,mode='stream_copy_only',branch=BRANCH,commit_identity_source=identity_source))
    run_tests(logdir)
    receipt=demo.build(logdir)
    for path,pin in receipt['artifacts'].items():
        assert (NEW/path).stat().st_size==pin['bytes']<demo.LIMIT
        assert sha256(NEW/path)==pin['sha256']
    for name,digest in old_pins.items(): assert sha256(OLD/name)==digest
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True); OLD.rename(ARCHIVE)
    for name,digest in old_pins.items(): assert sha256(ARCHIVE/name)==digest
    atomic_json(logdir/'LOCAL_REPLACEMENT_COMPLETE.json',dict(status='validated_local_replacement',
        frames=2000,chunks=receipt['chunk_count'],observations=receipt['observations'],
        old_demo_sha256=old_pins,archive=str(ARCHIVE.relative_to(ROOT)),new_demo=str(NEW.relative_to(ROOT))))
    commit=commit_bundle(logdir,head,pins,receipt)
    result=try_push(logdir,commit)
    if result['status']=='pushed':
        atomic_json(logdir/'COMPLETE.json',dict(status='validated_committed_and_pushed',commit=commit,frames=2000))
    else:
        print('ACTION_REQUIRED: local demo is complete AND committed; push from the normal network-enabled terminal.',flush=True)
        sys.exit(3)


if __name__=='__main__':
    main()
