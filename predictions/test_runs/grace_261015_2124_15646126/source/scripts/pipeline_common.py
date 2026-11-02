"""Shared, versioned workflow invariants. File operations only; no submissions."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXIT_REQUEUE = 75


def project_path(value) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    path.relative_to(ROOT)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def exclusive_lock(directory: Path, name: str = ".worker.lock"):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another worker holds {directory / name}; refusing concurrent writes") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def versions() -> dict:
    return {name: importlib.metadata.version(name) for name in
            ("sleap-io", "sleap-nn", "torch", "lightning", "numpy")}


def load_run(config) -> dict:
    path = project_path(config)
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("schema_version") != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("run_id", "")):
        raise ValueError("Invalid schema_version/run_id")
    if not 0 < cfg["dataset"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if cfg["prediction"]["chunk_frames"] <= 0:
        raise ValueError("chunk_frames must be positive")
    for key, value in cfg["videos"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ValueError(f"Invalid video key: {key}")
        project_path(value)
    return cfg


def dataset_dir(cfg: dict) -> Path:
    return ROOT / "data" / "labels" / cfg["run_id"]


def model_dir(cfg: dict, role: str) -> Path:
    if role not in ("centroid", "centered"):
        raise ValueError(role)
    return ROOT / "models" / cfg["run_id"] / role


def prediction_dir(cfg: dict, video_key: str) -> Path:
    if video_key not in cfg["videos"]:
        raise ValueError(f"Unknown video key {video_key}")
    return ROOT / "predictions" / cfg["run_id"] / video_key


def pin_manifest(directory: Path, value: dict) -> str:
    """Call under the run lock. A run ID cannot silently change its inputs."""
    path = directory / "manifest.json"
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Run inputs/configuration changed: {path}. Use a NEW run_id.")
    else:
        atomic_json(path, value)
    return fingerprint(value)


def dataset_manifest(cfg: dict) -> dict:
    directory = dataset_dir(cfg)
    marker = read_json(directory / "DATASET_COMPLETE.json")
    manifest = read_json(directory / "manifest.json")
    if marker["manifest_sha256"] != fingerprint(manifest):
        raise RuntimeError("Dataset manifest changed")
    if manifest["dataset_config"] != cfg["dataset"]:
        raise RuntimeError("Dataset configuration changed; prepare a new run")
    for source in manifest["sources"]:
        if sha256(project_path(source["path"])) != source["sha256"]:
            raise RuntimeError(f"Source labels changed: {source['path']}")
    for item in marker["outputs"]:
        if sha256(project_path(item["path"])) != item["sha256"]:
            raise RuntimeError(f"Prepared split changed: {item['path']}")
    return {"manifest": manifest, "complete": marker}


def chunk_ranges(n_frames: int, chunk_frames: int):
    if n_frames <= 0 or chunk_frames <= 0:
        raise ValueError("Video length and chunk size must be positive")
    return [(start, min(start + chunk_frames, n_frames)) for start in range(0, n_frames, chunk_frames)]


def validate_labels(labels, video: Path, start: int, stop: int, nodes: list[str]) -> None:
    """Empty-detection frames count as covered; missing frames never do."""
    if len(labels.videos) != 1 or Path(labels.videos[0].filename).resolve() != video.resolve():
        raise RuntimeError("Unexpected prediction video")
    indices = [int(lf.frame_idx) for lf in labels]
    if indices != list(range(start, stop)):
        raise RuntimeError(f"Missing, duplicated, or unordered prediction frames in [{start}, {stop})")
    for lf in labels:
        if lf.video is not labels.videos[0] or lf.user_instances:
            raise RuntimeError("Unexpected video/user instance in predictions")
        for inst in lf.instances:
            if list(inst.skeleton.node_names) != nodes:
                raise RuntimeError("Unexpected prediction skeleton/node order")


def save_slp_atomic(labels, destination: Path, validator) -> None:
    """Keep the old destination until a replacement has passed read-back validation."""
    import sleap_io as sio
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.slp")
    try:
        labels.save(str(temporary), format="slp", embed=False)
        check = sio.load_slp(temporary, open_videos=False)
        validator(check)
        del check
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def hop_start(directory: Path, maximum: int) -> int:
    path = directory / "hop_state.json"
    state = read_json(path) if path.exists() else {"hops": 0, "no_progress_hops": 0, "progress": -1}
    if state["hops"] >= maximum:
        raise RuntimeError(f"Reached safety cap of {maximum} hops")
    state["hops"] += 1
    # Record an unfinished attempt first, so SIGKILL also counts against the guard.
    state["no_progress_hops"] += 1
    atomic_json(path, state)
    return state["hops"]


def hop_progress(directory: Path, progress: int, no_progress_limit: int) -> None:
    path = directory / "hop_state.json"
    state = read_json(path)
    if progress > state["progress"]:
        state["progress"] = int(progress)
        state["no_progress_hops"] = 0
    atomic_json(path, state)
    if state["no_progress_hops"] >= no_progress_limit:
        raise RuntimeError(f"No durable progress for {state['no_progress_hops']} hops; stopping for review")
