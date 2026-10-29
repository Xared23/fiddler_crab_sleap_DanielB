#!/usr/bin/env python3
"""Predict one gputest chunk of a video with the native 8K top-down models.

Writes a per-chunk .slp, then either exits 75 (more frames remain; requeue)
or merges chunks into <stem>.predictions.slp and writes PREDICT_COMPLETE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

from pipeline_common import ROOT as PROJECT_ROOT

EXIT_REQUEUE = 75

CENTROID = PROJECT_ROOT / "models" / "dan_centroid_native"
CENTERED = PROJECT_ROOT / "models" / "dan_centered_native"


def _resolve_project_video(raw: Path) -> Path:
    """Resolve a video path against fiddler_crab_sleap; reject anything outside."""
    root = PROJECT_ROOT.resolve()
    text = raw.as_posix()
    if text.startswith("./"):
        text = text[2:]
    if text.startswith(f"{root.name}/"):
        text = text[len(root.name) + 1 :]
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    video = candidate.resolve()
    try:
        video.relative_to(root)
    except ValueError:
        logger.error(f"Video must be under {root} (got {video})")
        raise SystemExit(1) from None
    if not video.is_file():
        logger.error(f"Video not found: {video}")
        raise SystemExit(1)
    return video


def _n_frames(video: Path) -> int:
    import sleap_io as sio

    vid = sio.load_video(video.as_posix())
    return int(len(vid))


def _read_int(path: Path, default: int = 0) -> int:
    if not path.is_file():
        return default
    return int(path.read_text().strip())


def _chunk_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import sleap_io as sio

        sio.load_slp(path.as_posix())
    except Exception:
        return False
    return True


def predict_chunk(video: Path, output: Path, start: int, stop: int,
                  model_paths, batch_size=1, peak_threshold=0.2,
                  max_instances=None):
    """Shared native-resolution inference; no tracking or regional filtering.

    Versioned callers pass a temporary .slp path and publish only after read-back.
    Empty frames are retained so processing coverage is explicit.
    """
    from sleap_nn.inference.run import predict
    return predict(video.as_posix(), model_paths=[str(p) for p in model_paths],
                   frames=list(range(start, stop)), output_path=str(output),
                   output_format="slp", device="cuda", batch_size=batch_size,
                   peak_threshold=peak_threshold, max_instances=max_instances,
                   tracker_config=None, filter_config=None,
                   clean_empty_frames=False, embed="false")


def _merge_chunks(chunks: List[Path], out_path: Path) -> None:
    import sleap_io as sio

    if not chunks:
        raise RuntimeError("No chunk files to merge")
    tmp_path = out_path.with_suffix(".tmp.slp")
    if tmp_path.exists():
        tmp_path.unlink()
    if out_path.exists():
        logger.warning(f"Replacing incomplete merge file {out_path}")
        out_path.unlink()

    merged = sio.load_slp(chunks[0].as_posix())
    video = merged.videos[0]
    n_chunks = len(chunks)
    for i, chunk in enumerate(chunks[1:], start=2):
        part = sio.load_slp(chunk.as_posix())
        for lf in part.labeled_frames:
            lf.video = video
            merged.append(lf)
        if i % 50 == 0 or i == n_chunks:
            logger.info(f"Merged {i}/{n_chunks} chunks")
    merged.save(tmp_path.as_posix(), embed=False, format="slp")
    tmp_path.replace(out_path)
    logger.info(f"Wrote {out_path}")


def _finish(run_dir: Path, video: Path, n_frames: int) -> int:
    chunks = sorted(run_dir.glob("*.frames_*.slp"))
    merged = run_dir / f"{video.stem}.predictions.slp"
    _merge_chunks(chunks, merged)
    complete = run_dir / "PREDICT_COMPLETE"
    complete.write_text(f"frames={n_frames}\nmerged={merged}\n")
    logger.info(f"Wrote {complete}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-config", type=Path,
                        help="Pinned versioned prediction config; overrides legacy model/output defaults.")
    parser.add_argument("--chunk-frames", type=int, default=40)
    parser.add_argument("--max-hops", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--peak-threshold", type=float, default=0.2)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip inference and merge existing chunk .slp files.",
    )
    args = parser.parse_args(argv)

    video = _resolve_project_video(args.video)
    if args.run_config is not None:
        from prediction_run import run_hop, finalize
        if args.merge_only:
            return finalize(args.run_config, video)
        return run_hop(args.run_config, video)
    if args.run_dir is None:
        parser.error("--run-dir is required without --run-config")
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".video").write_text(f"{video}\n")

    complete = run_dir / "PREDICT_COMPLETE"
    if complete.exists():
        logger.info(f"Already complete: {complete}")
        return 0

    if args.merge_only:
        n_frames = _n_frames(video)
        logger.info(f"Merge-only {video.name}: {n_frames} frames")
        return _finish(run_dir, video, n_frames)

    for model in (CENTROID, CENTERED):
        if not (model / "best.ckpt").is_file():
            logger.error(f"Missing trained weights: {model / 'best.ckpt'}")
            return 1

    hops_file = run_dir / ".gputest_hops"
    hop = _read_int(hops_file, 0)
    if hop > args.max_hops:
        logger.error(f"Reached max predict hops ({args.max_hops})")
        return 1

    n_frames = _n_frames(video)
    start = _read_int(run_dir / ".next_frame", 0)
    logger.info(f"Video {video.name}: {n_frames} frames; next={start}")

    if start >= n_frames:
        return _finish(run_dir, video, n_frames)

    end = min(start + args.chunk_frames, n_frames) - 1
    chunk_path = run_dir / f"{video.stem}.frames_{start:06d}_{end:06d}.slp"
    logger.info(f"Predicting frames {start}-{end} -> {chunk_path}")

    if _chunk_ok(chunk_path):
        logger.info(f"Chunk already on disk; skipping inference: {chunk_path}")
    else:
        if chunk_path.exists():
            logger.warning(f"Removing incomplete chunk {chunk_path}")
            chunk_path.unlink()

        try:
            predict_chunk(video, chunk_path, start, end + 1, [CENTROID, CENTERED],
                          args.batch_size, args.peak_threshold, args.max_instances)
        except Exception:
            logger.exception("Predict raised")
            if chunk_path.exists() and not _chunk_ok(chunk_path):
                chunk_path.unlink()
            return 1

        if not _chunk_ok(chunk_path):
            logger.error(f"Predict returned but chunk is unreadable: {chunk_path}")
            return 1

    next_frame = end + 1
    (run_dir / ".next_frame").write_text(f"{next_frame}\n")
    logger.info(f"Chunk done; next_frame={next_frame}/{n_frames}")

    if next_frame < n_frames:
        logger.info("More frames remain; requeue")
        return EXIT_REQUEUE

    return _finish(run_dir, video, n_frames)


if __name__ == "__main__":
    sys.exit(main())
