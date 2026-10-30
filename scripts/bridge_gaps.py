#!/usr/bin/env python3

"""
Bridge short gaps between body-first tracklets (command-line entry point).

Subcommands (``run`` is the default, so the v001 invocation still works)::

    # bridge once
    .venv/bin/python scripts/bridge_gaps.py \\
        --observations demo/crab_cam2_2_middle_300/observations.parquet \\
        --out outputs/gap_bridging/demo_v001 \\
        --config configs/tracking/gap_bridging_v001.yaml

    # fit the sqrt gate radius r0 + k sqrt(dt) to measured displacements
    .venv/bin/python scripts/bridge_gaps.py fit \\
        --observations demo/crab_cam2_2_middle_300/observations.parquet

    # sweep the horizon max_gap, with frame-deletion hold-out (plus decoys and
    # flicker singletons) and figures
    .venv/bin/python scripts/bridge_gaps.py sweep \\
        --observations demo/crab_cam2_2_middle_300/observations.parquet \\
        --out logs/gap_sweep --max-gaps 5,10,15,30,45,60 --holdout --decoys 4 --flicker 2

    # export interpolated frames as weak labels (fine-tuning candidates)
    .venv/bin/python scripts/bridge_gaps.py export \\
        --bridged outputs/gap_bridging/demo_v001 --out outputs/weak_labels/demo

See docs/GAP_BRIDGING.md for the method and output schema.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ----- numerics -----
import numpy as np  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# ----- local modules -----
from gap_bridging import export, fit, identity, pipeline, tables  # noqa: E402
from gap_bridging.config import BridgeConfig, load_config  # noqa: E402
from gap_bridging.types import DECISIONS  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

COMMANDS = ('run', 'fit', 'sweep', 'export')
DEFAULT_MAX_GAPS = (5, 10, 15, 30, 45, 60)
DEFAULT_HOLDOUT_GAPS = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60)
SUMMARY_NAME = 'sweep_summary.csv'
BY_GAP_NAME = 'holdout_by_gap.csv'
FIGURES_DIR = 'figures'
CONFIDENCES = ('high', 'medium', 'low')
SUMMARY_COLUMNS = (('max_gap', 'links', 'trajectories', 'mean_length', 'median_length',
                    'p90_length', 'units_len1', 'frac_frames_in_units_ge_30', 'recovered',
                    'interpolated')
                   + tuple(f'interpolated_{c}' for c in CONFIDENCES)
                   + tuple(f'decision_{d}' for d in DECISIONS)
                   + ('n_recovered_tail', 'claw_side_refusals', 'spurious', 'identity_starts',
                      'identity_starts_filtered', 'units_len1_filtered', 'mean_length_filtered',
                      'median_length_filtered')
                   + ('holdout_planted', 'holdout_precision', 'holdout_recall',
                      'holdout_unplanted_links', 'holdout_decoy_planted',
                      'holdout_decoy_false_link', 'holdout_flicker_planted',
                      'holdout_flicker_spurious'))
BY_GAP_COLUMNS = ('max_gap', 'gap', 'planted', 'recovered', 'false_link', 'missed', 'recall',
                  'precision', 'decoy_planted', 'decoy_recovered', 'decoy_false_link',
                  'flicker_planted', 'flicker_recovered', 'flicker_spurious')

# ============================================================
# HELPERS
# ============================================================


def int_list(text: str) -> list[int]:
    """Parse ``"5,10,15"`` into ``[5, 10, 15]`` (argparse type)."""
    try:
        values = [int(v) for v in text.split(',') if v.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from error
    if not values or any(v < 0 for v in values):
        raise argparse.ArgumentTypeError(f"expected non-negative integers, got {text!r}")
    return values


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    """Write ``rows`` (dicts) with a header; floats NaN as ``nan``."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def _nan(value) -> float:
    """None (JSON null) as NaN, else float."""
    return math.nan if value is None else float(value)


# ============================================================
# FIT
# ============================================================


def fit_command(observations: Path, max_dt: int = 60, q: float = 0.99, min_samples: int = 200,
                write_yaml: Path | None = None) -> dict:
    """Fit ``r0 + k sqrt(dt)`` to the displacement quantiles; print and optionally write YAML.

    Returns
    -------
    dict
        The fit (:func:`gap_bridging.fit.fit_radius`) plus ``quantiles``
        (list of rows) and ``yaml`` (path or None).
    """
    obs = pipeline.read_observations(pipeline.input_files(observations))
    quantiles = fit.displacement_quantiles(obs, max_dt=max_dt, q=q, min_samples=min_samples)
    result = fit.fit_radius(quantiles)
    print(f"{'dt':>4} {'q' + format(q, 'g'):>10} {'n':>8} {'fit':>8}")
    for row in quantiles.to_pylist():
        fitted = result['radius_r0_px'] + result['radius_k_px'] * math.sqrt(row['dt'])
        print(f"{row['dt']:4d} {row['quantile']:10.3f} {row['n']:8d} {fitted:8.3f}")
    print(f"fit: r0 = {result['radius_r0_px']:.2f} px, k = {result['radius_k_px']:.2f} px, "
          f"rmse = {result['rmse']:.2f} px over {result['n_dt']} dt values", flush=True)
    if write_yaml is not None:
        write_yaml = Path(write_yaml)
        write_yaml.parent.mkdir(parents=True, exist_ok=True)
        write_yaml.write_text(
            f"# Gate radius fit (bridge_gaps.py fit): nnls of r0 + k*sqrt(dt) to the q={q:g}\n"
            f"# quantile of net shell displacement, dt 1..{max_dt}, input {observations}.\n"
            f"# rmse {result['rmse']:.2f} px over {result['n_dt']} dt values.\n"
            f"radius_r0_px: {result['radius_r0_px']:.2f}\n"
            f"radius_k_px: {result['radius_k_px']:.2f}\n")
        print(f"wrote {write_yaml}", flush=True)
    return dict(result, quantiles=quantiles.to_pylist(), yaml=None if write_yaml is None
                else str(write_yaml))


# ============================================================
# SWEEP
# ============================================================


def run_summary(run_dir: Path, marker: dict) -> dict:
    """Continuity statistics of one bridging run (one ``sweep_summary.csv`` row, no hold-out).

    Unit length is ``end_frame - start_frame + 1`` of each trajectory
    (interpolated frames included), as in the v001 report. The ``*_filtered``
    columns repeat the statistics over identity trajectories only
    (``trajectory_status == 'identity'``; spurious units are flagged, never
    removed); ``identity_starts`` counts trajectories starting after the
    video's first frame; ``claw_side_refusals`` counts ``refused_gate`` audit
    rows with reason ``claw_side``.
    """
    traj = pq.read_table(run_dir / 'trajectories.parquet',
                         columns=['trajectory_id', 'start_frame', 'end_frame',
                                  tables.STATUS_FIELD.name])
    status = dict(zip(traj['trajectory_id'].to_pylist(),
                      traj[tables.STATUS_FIELD.name].to_pylist()))
    filtered = identity.filtered_lengths(traj, status)
    links = pq.read_table(run_dir / 'links.parquet', columns=['decision', 'reason'])
    claw_side = pc.sum(pc.and_(pc.equal(links['decision'], 'refused_gate'),
                               pc.equal(links['reason'], 'claw_side'))).as_py() or 0
    length = (np.asarray(traj['end_frame'].to_numpy(), np.int64)
              - np.asarray(traj['start_frame'].to_numpy(), np.int64) + 1)
    obs = pq.read_table(run_dir / 'observations.parquet', columns=['source', 'fill_confidence'])
    interp = obs.filter(pc.equal(obs['source'], 'interpolated'))['fill_confidence'].to_pylist()
    counts = marker['counts']
    total = int(length.sum())
    row = {
        'max_gap': marker['config']['max_gap'], 'links': counts['links'],
        'trajectories': counts['trajectories'],
        'mean_length': float(length.mean()) if length.size else math.nan,
        'median_length': float(np.median(length)) if length.size else math.nan,
        'p90_length': float(np.percentile(length, 90)) if length.size else math.nan,
        'units_len1': int((length == 1).sum()),
        'frac_frames_in_units_ge_30': float(length[length >= 30].sum() / total) if total else math.nan,
        'recovered': counts['n_recovered'], 'interpolated': counts['n_interpolated'],
    }
    row.update({f'interpolated_{c}': interp.count(c) for c in CONFIDENCES})
    row.update({f'decision_{d}': counts['decisions'][d] for d in DECISIONS})
    row.update({
        'n_recovered_tail': counts['n_recovered_tail'], 'claw_side_refusals': int(claw_side),
        'spurious': counts['spurious_trajectories'], 'identity_starts': counts['identity_starts'],
        'identity_starts_filtered': counts['identity_starts_identity_only'],
        'units_len1_filtered': filtered['units_len1'],
        'mean_length_filtered': _nan(filtered['mean']),
        'median_length_filtered': _nan(filtered['median']),
    })
    return row


def sweep(observations: Path, out: Path, max_gaps: list[int], cfg: BridgeConfig,
          holdout_run: bool = False, holdout_gaps: list[int] = list(DEFAULT_HOLDOUT_GAPS),
          per_gap: int = 8, seed: int = 0, force: bool = False, video: Path | None = None,
          bridges_from: int | None = None, figures_on: bool = True, decoys: int = 0,
          flicker: int = 0) -> list[dict]:
    """Bridge at every horizon, optionally score a hold-out, write the CSVs and figures.

    Parameters
    ----------
    observations : Path
        Observation parquet file or directory of parts.
    out : Path
        Receives ``max_gap_<N>/`` (pipeline outputs), ``holdout_<N>/``,
        ``sweep_summary.csv``, ``holdout_by_gap.csv`` and ``figures/``.
    max_gaps : list of int
        Horizons; ``cfg.max_gap`` is replaced by each in turn.
    cfg : BridgeConfig
        Base configuration.
    holdout_run : bool
        Run :func:`gap_bridging.holdout.run_holdout` per horizon, with the
        planted gap lengths ``holdout_gaps`` capped at the horizon.
    holdout_gaps, per_gap, seed, decoys, flicker
        Hold-out planting parameters (see :func:`gap_bridging.holdout.plant_gaps`).
    force : bool
        Overwrite existing outputs.
    video : Path, optional
        Video for the rendered bridges; default the single ``*.mp4`` next to
        a file input, else the bridge figure is skipped.
    bridges_from : int, optional
        Horizon whose run is rendered (default the largest).
    figures_on : bool
        Write the figures.

    Returns
    -------
    list of dict
        The ``sweep_summary.csv`` rows.
    """
    from gap_bridging import holdout
    out = Path(out)
    max_gaps = sorted(set(int(g) for g in max_gaps))
    if not max_gaps:
        raise ValueError("no horizons given")
    for name in (SUMMARY_NAME, BY_GAP_NAME):
        if (out / name).exists() and not force:
            raise FileExistsError(f"outputs already exist in {out}: {name}; use --force")
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, by_gap_rows = [], []
    for horizon in max_gaps:
        run_cfg = replace(cfg, max_gap=horizon)
        run_dir = out / f'max_gap_{horizon}'
        print(f"[sweep] max_gap {horizon}: bridging into {run_dir}", flush=True)
        marker = pipeline.run(observations, run_dir, run_cfg, force=force)
        row = run_summary(run_dir, marker)
        row.update({'holdout_planted': 0, 'holdout_precision': math.nan,
                    'holdout_recall': math.nan, 'holdout_unplanted_links': -1,
                    'holdout_decoy_planted': 0, 'holdout_decoy_false_link': 0,
                    'holdout_flicker_planted': 0, 'holdout_flicker_spurious': 0})
        gaps = [g for g in holdout_gaps if 1 <= g <= horizon]
        if holdout_run and gaps:
            print(f"[sweep] max_gap {horizon}: hold-out gaps {gaps} x {per_gap}", flush=True)
            result = holdout.run_holdout(observations, out / f'holdout_{horizon}', run_cfg, gaps,
                                         per_gap, seed=seed, force=force, decoys=decoys,
                                         flicker=flicker)
            overall = result['scores'][-1]
            row.update({'holdout_planted': overall['planted'],
                        'holdout_precision': _nan(overall['precision']),
                        'holdout_recall': _nan(overall['recall']),
                        'holdout_unplanted_links': result['n_unplanted_links'],
                        'holdout_decoy_planted': overall['decoy_planted'],
                        'holdout_decoy_false_link': overall['decoy_false_link'],
                        'holdout_flicker_planted': overall['flicker_planted'],
                        'holdout_flicker_spurious': overall['flicker_spurious']})
            counted = ('planted', 'recovered', 'false_link', 'missed', 'decoy_planted',
                       'decoy_recovered', 'decoy_false_link', 'flicker_planted',
                       'flicker_recovered', 'flicker_spurious')
            for score in result['scores'][:-1]:
                by_gap_rows.append({'max_gap': horizon, 'gap': score['gap_frames'],
                                    **{k: score[k] for k in counted},
                                    'recall': _nan(score['recall']),
                                    'precision': _nan(score['precision'])})
        summary_rows.append(row)
    write_csv(out / SUMMARY_NAME, SUMMARY_COLUMNS, summary_rows)
    if holdout_run:
        write_csv(out / BY_GAP_NAME, BY_GAP_COLUMNS, by_gap_rows)
    if figures_on:
        make_figures(observations, out, cfg, holdout_run and bool(by_gap_rows), video,
                     bridges_from if bridges_from is not None else max_gaps[-1])
    return summary_rows


def make_figures(observations: Path, out: Path, cfg: BridgeConfig, with_holdout: bool,
                 video: Path | None, bridges_from: int) -> list[Path]:
    """Write the sweep, hold-out, radius-fit and bridge figures under ``out/figures``."""
    from gap_bridging import figures
    fig_dir = out / FIGURES_DIR
    written = figures.fig_sweep(out / SUMMARY_NAME, fig_dir / 'sweep.pdf')
    runs = sorted((p for p in out.glob('max_gap_*') if (p / tables.COMPLETE_NAME).is_file()),
                  key=lambda p: int(p.name.rsplit('_', 1)[1]))
    if runs:
        written += figures.fig_start_curve(runs, fig_dir / 'start_curve.pdf')
    if with_holdout:
        written += figures.fig_holdout(out / BY_GAP_NAME, fig_dir / 'holdout.pdf')
    obs = pipeline.read_observations(pipeline.input_files(observations))
    quantiles = fit.displacement_quantiles(obs)
    if quantiles.num_rows >= 2:
        written += figures.fig_radius_fit(quantiles, fit.fit_radius(quantiles), cfg,
                                          fig_dir / 'radius_fit.pdf')
    if video is None and Path(observations).is_file():
        candidates = sorted(Path(observations).parent.glob('*.mp4'))
        video = candidates[0] if len(candidates) == 1 else None
    run_obs = out / f'max_gap_{bridges_from}' / 'observations.parquet'
    if video is not None and run_obs.exists():
        pages = figures.fig_bridges(run_obs, video, fig_dir / f'bridges_max_gap_{bridges_from}')
        print(f"[sweep] rendered {len(pages)} bridge page(s) from max_gap {bridges_from}",
              flush=True)
        written += pages
    else:
        print("[sweep] no video or run for the rendered bridges; skipped", flush=True)
    print(f"[sweep] figures: {[str(p) for p in written]}", flush=True)
    return written


# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the ``run``, ``fit``, ``sweep`` and ``export`` subcommands."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     epilog=__doc__.split('\n\n', 1)[1].rsplit('\n\nSam', 1)[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='bridge gaps once (default subcommand)')
    run.add_argument('--observations', type=Path, required=True,
                     help='observation parquet file, or a directory of parquet parts')
    run.add_argument('--out', type=Path, required=True, help='output directory')
    run.add_argument('--config', type=Path, default=None,
                     help='YAML config (default: built-in defaults = gap_bridging_v001.yaml)')
    run.add_argument('--max-gap', type=int, default=None,
                     help='maximum number of missing frames to bridge (0 disables linking)')
    run.add_argument('--margin', type=float, default=None,
                     help='minimum cost margin to the runner-up, on the active gate model\'s '
                          'cost scale: minimum_cost_margin (px/frame) under linear, '
                          'minimum_cost_margin_sqrt (fraction of R) under sqrt')
    run.add_argument('--recovery-radius', type=float, default=None,
                     help='radius (px) for adopting a rejected detection')
    run.add_argument('--force', action='store_true', help='overwrite existing outputs')

    fit_p = sub.add_parser('fit', help='fit the sqrt gate radius to displacement quantiles')
    fit_p.add_argument('--observations', type=Path, required=True)
    fit_p.add_argument('--max-dt', type=int, default=60, help='largest frame separation')
    fit_p.add_argument('--quantile', type=float, default=0.99, help='displacement quantile')
    fit_p.add_argument('--min-samples', type=int, default=200,
                       help='drop dt values with fewer samples')
    fit_p.add_argument('--write-yaml', type=Path, default=None,
                       help='write radius_r0_px / radius_k_px as a YAML config overlay')

    sw = sub.add_parser('sweep', help='bridge at several horizons, hold-out and figures')
    sw.add_argument('--observations', type=Path, required=True)
    sw.add_argument('--out', type=Path, required=True, help='sweep output directory')
    sw.add_argument('--max-gaps', type=int_list, default=list(DEFAULT_MAX_GAPS),
                    help='comma-separated horizons (default 5,10,15,30,45,60)')
    sw.add_argument('--config', type=Path, default=None, help='base YAML config')
    sw.add_argument('--holdout', action='store_true', help='run the frame-deletion hold-out')
    sw.add_argument('--holdout-gaps', type=int_list, default=list(DEFAULT_HOLDOUT_GAPS),
                    help='planted gap lengths, capped at each horizon')
    sw.add_argument('--per-gap', type=int, default=8, help='planted tracklets per gap length')
    sw.add_argument('--seed', type=int, default=0, help='hold-out seed')
    sw.add_argument('--decoys', type=int, default=0,
                    help='decoy pairs per planted gap length: a donor right half with '
                         'different appearance cues placed next to a target\'s right half')
    sw.add_argument('--flicker', type=int, default=0,
                    help='flicker singletons per planted gap length: g frames deleted on '
                         'each side of one kept frame')
    sw.add_argument('--video', type=Path, default=None,
                    help='video for the rendered bridges (default: single *.mp4 next to the input)')
    sw.add_argument('--bridges-from', type=int, default=None,
                    help='horizon whose bridges are rendered (default the largest)')
    sw.add_argument('--no-figures', action='store_true', help='skip the figures')
    sw.add_argument('--force', action='store_true', help='overwrite existing outputs')

    ex = sub.add_parser('export', help='export interpolated frames as weak labels '
                                       '(fine-tuning candidates, not labels)')
    export.add_arguments(ex)
    return parser


def with_margin(cfg: BridgeConfig, margin: float | None) -> BridgeConfig:
    """Set the active gate model's runner-up margin (``--margin``); None leaves cfg unchanged.

    Linear: ``minimum_cost_margin`` (px/frame); sqrt: ``minimum_cost_margin_sqrt``
    (dimensionless). Raises ValueError for a negative margin.
    """
    if margin is None:
        return cfg
    field = 'minimum_cost_margin' if cfg.gate_model == 'linear' else 'minimum_cost_margin_sqrt'
    updated = replace(cfg, **{field: float(margin)})
    assert updated.cost_thresholds()['minimum_cost_margin'] == float(margin)
    return updated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments; without a subcommand the v001 ``run`` flags are assumed."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ('-h', '--help'):
        argv = ['run'] + argv
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch the subcommand.

    Returns
    -------
    int
        0 on success, 1 on a configuration, input or overwrite error.
    """
    args = parse_args(argv)
    try:
        if args.command == 'run':
            cfg = with_margin(load_config(args.config, max_gap=args.max_gap,
                                          recovery_radius_px=args.recovery_radius), args.margin)
            marker = pipeline.run(args.observations, args.out, cfg, force=args.force)
            print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
        elif args.command == 'fit':
            fit_command(args.observations, args.max_dt, args.quantile, args.min_samples,
                        args.write_yaml)
        elif args.command == 'export':
            marker = export.run(args)
            print(f"[export] {marker['counts']['rows']} weak-label rows -> {args.out}",
                  flush=True)
        else:
            rows = sweep(args.observations, args.out, args.max_gaps, load_config(args.config),
                         holdout_run=args.holdout, holdout_gaps=args.holdout_gaps,
                         per_gap=args.per_gap, seed=args.seed, force=args.force,
                         video=args.video, bridges_from=args.bridges_from,
                         figures_on=not args.no_figures, decoys=args.decoys,
                         flicker=args.flicker)
            print(f"[sweep] wrote {args.out / SUMMARY_NAME} ({len(rows)} horizons)", flush=True)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"bridge_gaps: error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
