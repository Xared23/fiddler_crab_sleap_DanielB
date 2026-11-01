#!/usr/bin/env python3

"""
Compare linker variants on the hold-out of their horizon sweeps, by fold (v004).

A variant is a list of sweep directories (one per seed) written by
``bridge_gaps.py sweep --holdout``. ``fold_scores`` re-scores one
``holdout_<h>/`` directory on the planted tracklets of one fold
(:func:`gap_bridging.labels.truth_folds`, keyed on the ORIGINAL tracklet id),
so a learned scorer trained on the ``train`` fold is judged on ``test``
plants only; ``fold='all'`` reproduces the sweep's ``holdout_by_gap.csv``.
Each truth row is kept by its own fold; decoy donors need no special rule,
since :func:`gap_bridging.holdout.score_links` skips them and reads the donor
from its target's ``partner_id``. ``compare_variants`` first checks that all
variants were scored on the same plants (same sweep count, same horizons,
identical ``holdout_truth.parquet`` per sweep index and horizon), then pools
every variant's seeds
(:func:`gap_bridging.holdout.pool_sweeps`), applies the horizon rule
(:func:`gap_bridging.holdout.apply_rule`) and checks each variant against a
baseline (``promotion_check``: precision 1 and no recall loss at every planted
gap up to ``through_gap``). ``write_comparison`` writes the CSVs, the JSON
verdicts and a Markdown report.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Mapping, Sequence

# ----- numerics -----
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----- local modules -----
from gap_bridging import holdout, labels
from gap_bridging.labels import DEFAULT_SALT, DEFAULT_TEST_FRACTION

# ============================================================
# CONSTANTS
# ============================================================

FOLD_CHOICES = ('test', 'train', 'all')
DEFAULT_THROUGH_GAP = 30
FALLBACK_HORIZON = 30
HOLDOUT_DIR = re.compile(r'^holdout_(\d+)$')
# Same columns as bridge_gaps.BY_GAP_COLUMNS / POOLED_*_COLUMNS (bridge_gaps imports
# this package, so the tuples are repeated here; a test checks they agree).
BY_GAP_COLUMNS = ('max_gap', 'gap', 'planted', 'recovered', 'false_link', 'missed', 'recall',
                  'precision', 'decoy_planted', 'decoy_recovered', 'decoy_false_link',
                  'flicker_planted', 'flicker_recovered', 'flicker_spurious')
POOLED_GAP_COLUMNS = (('max_gap', 'gap') + holdout.POOL_COUNTS
                      + ('links', 'recall', 'precision', 'precision_lower'))
POOLED_HORIZON_COLUMNS = (('max_gap',) + holdout.POOL_COUNTS
                          + ('links', 'recall', 'precision', 'precision_lower', 'n_gaps',
                             'min_gap_precision', 'rule_pass', 'failing_gaps'))
COMPARISON_GAP_NAME = 'comparison_by_gap.csv'
COMPARISON_HORIZON_NAME = 'comparison_by_horizon.csv'
PROMOTION_NAME = 'promotion.json'
COMPARISON_GAP_COLUMNS = ('variant',) + POOLED_GAP_COLUMNS
COMPARISON_HORIZON_COLUMNS = ('variant',) + POOLED_HORIZON_COLUMNS

# ============================================================
# FOLD SCORES
# ============================================================


def _horizon(holdout_dir: Path) -> int:
    """``max_gap`` of a ``holdout_<h>/`` directory (name and JSON must agree when both exist)."""
    match = HOLDOUT_DIR.match(Path(holdout_dir).name)
    jsons = sorted(Path(holdout_dir).glob('holdout_*.json'))
    from_json = None
    if len(jsons) == 1:
        from_json = int(json.loads(jsons[0].read_text())['config']['max_gap'])
    if match is None and from_json is None:
        raise ValueError(f"{holdout_dir}: cannot tell max_gap (no holdout_<h> name or JSON)")
    if match is not None and from_json is not None and int(match.group(1)) != from_json:
        raise ValueError(f"{holdout_dir}: directory says max_gap {match.group(1)}, "
                         f"JSON says {from_json}")
    return int(match.group(1)) if match is not None else from_json


def _keep_fold(truth: pa.Table, fold: str) -> pa.Table:
    """Truth rows whose own ``fold`` (of their ``original_id``) is ``fold``.

    Decoy donors get no special treatment: :func:`gap_bridging.holdout.score_links`
    skips ``decoy_donor`` rows and takes the donor from the target row's
    ``partner_id``, so whether a donor row is kept never changes a score.
    """
    return truth.filter(pc.equal(truth['fold'], fold))


def holdout_dirs(sweep_dir: Path) -> list[tuple[int, Path]]:
    """``(horizon, dir)`` of the completed ``holdout_<h>/`` runs of a sweep, by horizon.

    A directory counts when it holds ``holdout_<h>.json`` (written last by
    :func:`gap_bridging.holdout.run_holdout`; the criterion of
    :func:`gap_bridging.train.holdout_dirs`), the truth table and the bridged
    links and observations.
    """
    sweep_dir = Path(sweep_dir)
    found = []
    for child in sweep_dir.iterdir() if sweep_dir.is_dir() else ():
        match = HOLDOUT_DIR.match(child.name)
        if (match and child.is_dir() and (child / f'{child.name}.json').is_file()
                and (child / holdout.TRUTH_NAME).is_file()
                and (child / holdout.BRIDGED_DIR / 'links.parquet').is_file()
                and (child / holdout.BRIDGED_DIR / 'observations.parquet').is_file()):
            found.append((int(match.group(1)), child))
    return sorted(found)


def _by_gap_row(max_gap: int, score: dict) -> dict:
    """One ``holdout_by_gap.csv`` row (as ``bridge_gaps.sweep`` writes it) from a scores row."""
    counted = [c for c in BY_GAP_COLUMNS if c not in ('max_gap', 'gap', 'recall', 'precision')]
    row = {'max_gap': int(max_gap), 'gap': int(score['gap_frames']),
           **{k: int(score[k]) for k in counted}}
    row['recall'] = math.nan if score['recall'] is None else float(score['recall'])
    row['precision'] = math.nan if score['precision'] is None else float(score['precision'])
    return {name: row[name] for name in BY_GAP_COLUMNS}


def fold_scores(holdout_dir: Path, fold: str, salt: str = DEFAULT_SALT,
                test_fraction: float = DEFAULT_TEST_FRACTION) -> list[dict]:
    """Hold-out scores of one ``holdout_<h>/`` directory on the plants of one fold.

    Parameters
    ----------
    holdout_dir : Path
        Output of :func:`gap_bridging.holdout.run_holdout` (``holdout_<h>.json``,
        ``holdout_truth.parquet``, ``bridged/links.parquet`` and
        ``bridged/observations.parquet``).
    fold : str
        ``'test'``, ``'train'`` or ``'all'`` (every truth row; no fold lookup).
    salt, test_fraction
        Passed to :func:`gap_bridging.labels.truth_folds`.

    Returns
    -------
    list of dict
        One row per planted gap length of the whole truth table, ascending,
        with the ``holdout_by_gap.csv`` columns (:data:`BY_GAP_COLUMNS`); the
        all-gaps row is excluded, as in that CSV. A gap without a plant of
        this fold has zero counts and NaN ratios, so every fold of every seed
        holds the same ``(max_gap, gap)`` keys and the train and test counts
        add up to the ``'all'`` counts.
    """
    if fold not in FOLD_CHOICES:
        raise ValueError(f"fold must be one of {FOLD_CHOICES}, got {fold!r}")
    holdout_dir = Path(holdout_dir)
    max_gap = _horizon(holdout_dir)
    truth = pq.read_table(holdout_dir / holdout.TRUTH_NAME)
    links = pq.read_table(holdout_dir / holdout.BRIDGED_DIR / 'links.parquet',
                          columns=['tracklet_a', 'tracklet_b', 'decision'])
    bridged = pq.read_table(holdout_dir / holdout.BRIDGED_DIR / 'observations.parquet',
                            columns=['tracklet_id', 'trajectory_status'])
    bridged = bridged.filter(pc.is_valid(bridged['tracklet_id']))
    status = dict(zip(bridged['tracklet_id'].to_pylist(),
                      bridged['trajectory_status'].to_pylist()))
    gaps = sorted(set(int(g) for g in truth['gap_frames'].to_pylist()))
    if fold != 'all':
        with_folds = labels.truth_folds(truth, salt, test_fraction)
        if 'fold' not in with_folds.schema.names or with_folds.num_rows != truth.num_rows:
            raise RuntimeError("truth_folds must return the truth rows plus a 'fold' column")
        truth = _keep_fold(with_folds, fold).select(holdout.TRUTH_SCHEMA.names)
    scores = {r['gap_frames']: r for r in holdout.score_links(links, truth, status).to_pylist()
              if r['gap_frames'] != holdout.ALL_GAPS}
    empty = holdout._score_row(0, [], [], [])
    rows = [_by_gap_row(max_gap, scores.get(g, {**empty, 'gap_frames': g})) for g in gaps]
    assert all(r['recovered'] + r['false_link'] + r['missed'] == r['planted'] for r in rows)
    return rows


def sweep_fold_rows(sweep_dir: Path, fold: str, salt: str = DEFAULT_SALT,
                    test_fraction: float = DEFAULT_TEST_FRACTION) -> list[dict]:
    """:func:`fold_scores` of every ``holdout_<h>/`` directory of a sweep, by ascending ``h``.

    Only completed runs count (:func:`holdout_dirs`).

    Raises
    ------
    FileNotFoundError
        If the sweep has no completed ``holdout_<h>/`` directory.
    """
    sweep_dir = Path(sweep_dir)
    dirs = [d for _, d in holdout_dirs(sweep_dir)]
    if not dirs:
        raise FileNotFoundError(f"no completed holdout_<max_gap>/ directories under {sweep_dir}")
    return [row for d in dirs for row in fold_scores(d, fold, salt, test_fraction)]


# ============================================================
# COMPARISON
# ============================================================


def _sorted_truth(path: Path) -> pa.Table:
    """A truth table without schema metadata, rows sorted on every column."""
    table = pq.read_table(path)
    table = table.replace_schema_metadata(None)
    return table.sort_by([(name, 'ascending') for name in table.schema.names])


def check_same_plants(variants: Mapping[str, Sequence[Path]]) -> list[int]:
    """Check that every variant was scored on the same planted gaps.

    ``run_holdout`` draws decoys with the variant's own config, and seed lists
    are free-form, so per-gap recall is comparable across variants only when,
    for every sweep index and horizon, the variants hold the same truth table.

    Parameters
    ----------
    variants : mapping
        ``name -> sweep directories``; the first variant is the reference.

    Returns
    -------
    list of int
        The common horizons, ascending.

    Raises
    ------
    ValueError
        If the variants have different numbers of sweep directories, a sweep
        holds a different set of completed horizons than the reference's
        first sweep, or a truth table (rows sorted, schema metadata ignored)
        differs from the reference variant's at the same sweep index and
        horizon. The message names the variant, the sweep index and the
        horizon.
    """
    names = list(variants)
    reference = names[0]
    counts = {name: len(variants[name]) for name in names}
    if len(set(counts.values())) != 1:
        raise ValueError(f"variants have different numbers of sweep directories: {counts}")
    runs = {name: [dict(holdout_dirs(Path(d))) for d in variants[name]] for name in names}
    horizons = sorted(runs[reference][0])
    for name in names:
        for index, found in enumerate(runs[name]):
            if sorted(found) != horizons:
                raise ValueError(f"variant {name!r} sweep {index} ({variants[name][index]}) "
                                 f"holds horizons {sorted(found)}, expected {horizons} "
                                 f"(variant {reference!r} sweep 0)")
    for index in range(counts[reference]):
        for horizon in horizons:
            expected = _sorted_truth(runs[reference][index][horizon] / holdout.TRUTH_NAME)
            for name in names[1:]:
                got = _sorted_truth(runs[name][index][horizon] / holdout.TRUTH_NAME)
                if not got.equals(expected):
                    raise ValueError(
                        f"variant {name!r} sweep {index} ({variants[name][index]}) horizon "
                        f"{horizon}: holdout truth differs from variant {reference!r} "
                        f"({variants[reference][index]}); the variants were not scored on "
                        f"the same plants")
    return horizons


def promotion_check(current_by_gap: Sequence[dict], candidate_by_gap: Sequence[dict],
                    horizon: int, through_gap: int = DEFAULT_THROUGH_GAP) -> dict:
    """Does a candidate keep precision 1 and lose no recall at one horizon?

    Parameters
    ----------
    current_by_gap, candidate_by_gap : sequence of dict
        Pooled per-gap rows (:func:`gap_bridging.holdout.pool_sweeps`) of the
        current linker and the candidate: ``max_gap``, ``gap``, ``precision``,
        ``recall``.
    horizon : int
        ``max_gap`` whose rows are compared.
    through_gap : int
        Only planted gaps ``<= through_gap`` are checked.

    Returns
    -------
    dict
        ``horizon``, ``through_gap``, ``gaps`` (the checked gaps),
        ``precision_ok`` (candidate precision 1.0 or NaN, i.e. no link, at
        every checked gap), ``recall_ok`` (candidate recall ``>=`` current
        recall at every checked gap; a NaN recall on either side passes),
        ``failing`` (``"gap:precision"``, ``"gap:recall"`` or
        ``"gap:missing"`` strings, gap order) and ``ok`` (both checks pass
        and at least one gap was checked).
    """
    def at(rows: Sequence[dict]) -> dict[int, dict]:
        return {int(r['gap']): r for r in rows
                if int(r['max_gap']) == int(horizon) and int(r['gap']) <= through_gap}

    current, candidate = at(current_by_gap), at(candidate_by_gap)
    gaps = sorted(set(current) | set(candidate))
    failing: list[str] = []
    precision_ok = recall_ok = True
    for gap in gaps:
        if gap not in current or gap not in candidate:
            failing.append(f"{gap}:missing")
            precision_ok = recall_ok = False
            continue
        p = float(candidate[gap]['precision'])
        if not math.isnan(p) and p < 1.0:
            failing.append(f"{gap}:precision")
            precision_ok = False
        r_new, r_old = float(candidate[gap]['recall']), float(current[gap]['recall'])
        if not (math.isnan(r_new) or math.isnan(r_old)) and r_new < r_old - 1e-12:
            failing.append(f"{gap}:recall")
            recall_ok = False
    if not gaps:
        failing.append(f"{horizon}:no_rows")
    return {'horizon': int(horizon), 'through_gap': int(through_gap), 'gaps': gaps,
            'precision_ok': precision_ok and bool(gaps), 'recall_ok': recall_ok and bool(gaps),
            'failing': failing, 'ok': precision_ok and recall_ok and bool(gaps)}


def compare_variants(variants: Mapping[str, Sequence[Path]], fold: str = 'test',
                     through_gap: int = DEFAULT_THROUGH_GAP, rule_precision: float = 1.0,
                     salt: str = DEFAULT_SALT, test_fraction: float = DEFAULT_TEST_FRACTION,
                     baseline: str | None = None) -> dict:
    """Pool every variant's seeds on one fold, apply the horizon rule and check promotion.

    Parameters
    ----------
    variants : mapping
        ``name -> sweep directories`` (one per seed); insertion order is the
        report order.
    fold, salt, test_fraction
        See :func:`fold_scores`.
    through_gap : int
        Largest planted gap of the promotion check.
    rule_precision : float
        Precision floor of :func:`gap_bridging.holdout.apply_rule`.
    baseline : str, optional
        Variant every other is checked against (default the first). The
        check runs at the baseline's chosen horizon, or at
        :data:`FALLBACK_HORIZON` when no horizon passes the rule.

    Returns
    -------
    dict
        ``fold``, ``through_gap``, ``rule_precision``, ``baseline``,
        ``horizon`` (of the promotion check) and ``variants``: ``name ->
        {'sweeps', 'by_gap', 'by_horizon', 'chosen', 'promotion'}`` with the
        pooled rows, the rule table (``by_horizon`` plus ``rule_pass`` etc.)
        and the chosen horizon. The baseline is checked against itself, which
        passes exactly when its own precision is 1 at every checked gap.

    Raises
    ------
    ValueError
        If there is no variant, a variant has no sweep, ``baseline`` is not a
        variant, or the variants were not scored on the same plants
        (:func:`check_same_plants`).
    """
    if not variants:
        raise ValueError("no variants to compare")
    if any(not dirs for dirs in variants.values()):
        raise ValueError("every variant needs at least one sweep directory")
    names = list(variants)
    baseline = names[0] if baseline is None else baseline
    if baseline not in variants:
        raise ValueError(f"baseline {baseline!r} is not a variant ({names})")
    check_same_plants(variants)
    out: dict = {}
    for name in names:
        sweeps = [sweep_fold_rows(Path(d), fold, salt, test_fraction) for d in variants[name]]
        by_gap, by_horizon = holdout.pool_sweeps(sweeps)
        chosen, table = holdout.apply_rule(by_gap, by_horizon, rule_precision)
        out[name] = {'sweeps': [str(d) for d in variants[name]], 'by_gap': by_gap,
                     'by_horizon': table, 'chosen': chosen}
    horizon = out[baseline]['chosen']
    horizon = FALLBACK_HORIZON if horizon is None else horizon
    for name in names:
        out[name]['promotion'] = promotion_check(out[baseline]['by_gap'], out[name]['by_gap'],
                                                 horizon, through_gap)
    return {'fold': fold, 'through_gap': int(through_gap), 'rule_precision': float(rule_precision),
            'baseline': baseline, 'horizon': int(horizon), 'variants': out}


# ============================================================
# REPORT
# ============================================================


def _fmt(value: float, digits: int = 3) -> str:
    """Fixed-point float, ``–`` for NaN."""
    return '–' if value is None or math.isnan(float(value)) else f"{float(value):.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], right: int = 1) -> list[str]:
    """Markdown table lines; columns from ``right`` on are right-aligned."""
    align = ['---' if i < right else '---:' for i in range(len(header))]
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    return lines


def markdown_tables(result: dict) -> str:
    """Markdown report of :func:`compare_variants`.

    One table per horizon (rows = planted gap; per variant a precision cell
    ``p (recovered/links)`` and a recall cell ``r (recovered/planted)``), a
    per-horizon summary table (per variant pooled precision, its 95% lower
    bound, recall and whether the horizon passes the rule), the chosen
    horizons and the promotion verdicts against the baseline.
    """
    names = list(result['variants'])
    variants = result['variants']
    horizons = sorted({int(r['max_gap']) for v in variants.values() for r in v['by_gap']})
    lines = [f"Fold `{result['fold']}`; baseline `{result['baseline']}`; rule precision "
             f"{result['rule_precision']:g}; promotion at `max_gap` {result['horizon']} "
             f"through planted gap {result['through_gap']}."]
    if result.get('command'):
        lines += ['', 'Generated by:', '', '```bash', result['command'], '```']
    for horizon in horizons:
        lines += ['', f"#### `max_gap` {horizon}", '']
        index = {name: {int(r['gap']): r for r in variants[name]['by_gap']
                        if int(r['max_gap']) == horizon} for name in names}
        gaps = sorted({g for rows in index.values() for g in rows})
        header = ['planted gap'] + [f"{name} {kind}" for name in names
                                    for kind in ('precision', 'recall')]
        rows = []
        for gap in gaps:
            row = [str(gap)]
            for name in names:
                r = index[name].get(gap)
                if r is None:
                    row += ['n/a', 'n/a']
                    continue
                row += [f"{_fmt(r['precision'])} ({r['recovered']}/{r['links']})",
                        f"{_fmt(r['recall'])} ({r['recovered']}/{r['planted']})"]
            rows.append(row)
        lines += _table(header, rows)
    lines += ['', '#### Per horizon', '']
    header = ['`max_gap`'] + [f"{name} {kind}" for name in names
                              for kind in ('precision', 'lower 95%', 'recall', 'rule')]
    rows = []
    for horizon in horizons:
        row = [str(horizon)]
        for name in names:
            r = next((r for r in variants[name]['by_horizon'] if int(r['max_gap']) == horizon),
                     None)
            if r is None:
                row += ['n/a'] * 4
                continue
            row += [f"{_fmt(r['precision'], 4)} ({r['recovered']}/{r['links']})",
                    _fmt(r['precision_lower'], 4),
                    f"{_fmt(r['recall'], 4)} ({r['recovered']}/{r['planted']})",
                    'pass' if r['rule_pass'] else f"fail {r['failing_gaps']}"]
        rows.append(row)
    lines += _table(header, rows)
    lines += ['', '#### Verdicts', '']
    rows = []
    for name in names:
        v, p = variants[name], variants[name]['promotion']
        rows.append([name, str(v['chosen']), 'yes' if p['precision_ok'] else 'no',
                     'yes' if p['recall_ok'] else 'no',
                     '**promote**' if p['ok'] and name != result['baseline']
                     else ('baseline' if name == result['baseline'] else 'no'),
                     ' '.join(p['failing']) or '–'])
    lines += _table(['variant', 'chosen `max_gap`', 'precision 1', 'recall ≥ baseline',
                     'verdict', 'failing'], rows, right=99)
    return '\n'.join(lines) + '\n'


def _json_safe(value):
    """NaN floats as None, recursively."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: list[dict]) -> None:
    """CSV with a header (NaN as ``nan``)."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def write_comparison(result: dict, out_dir: Path, docs_path: Path | None = None,
                     force: bool = False) -> None:
    """Write ``comparison_by_gap.csv``, ``comparison_by_horizon.csv``, ``promotion.json``
    under ``out_dir`` and the Markdown report to ``docs_path``.

    Every target is checked before anything is written: an existing output
    (CSV, JSON or the Markdown file) raises FileExistsError unless ``force``.
    """
    out_dir = Path(out_dir)
    targets = [out_dir / n for n in (COMPARISON_GAP_NAME, COMPARISON_HORIZON_NAME,
                                     PROMOTION_NAME)]
    if docs_path is not None:
        targets.append(Path(docs_path))
    existing = [str(p) for p in targets if p.exists()]
    if existing and not force:
        raise FileExistsError(f"outputs already exist: {existing}; use --force")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(result['variants'])
    gap_rows = [{'variant': n, **{c: r[c] for c in POOLED_GAP_COLUMNS}}
                for n in names for r in result['variants'][n]['by_gap']]
    horizon_rows = [{'variant': n, **{c: r[c] for c in POOLED_HORIZON_COLUMNS}}
                    for n in names for r in result['variants'][n]['by_horizon']]
    _write_csv(out_dir / COMPARISON_GAP_NAME, COMPARISON_GAP_COLUMNS, gap_rows)
    _write_csv(out_dir / COMPARISON_HORIZON_NAME, COMPARISON_HORIZON_COLUMNS, horizon_rows)
    verdicts = {k: result[k] for k in ('fold', 'through_gap', 'rule_precision', 'baseline',
                                       'horizon')}
    verdicts.update({'command': result.get('command'),
                     'variants': {n: {'sweeps': result['variants'][n]['sweeps'],
                                      'chosen': result['variants'][n]['chosen'],
                                      'promotion': result['variants'][n]['promotion']}
                                  for n in names}})
    (out_dir / PROMOTION_NAME).write_text(json.dumps(_json_safe(verdicts), indent=2,
                                                     sort_keys=True) + '\n')
    if docs_path is not None:
        Path(docs_path).parent.mkdir(parents=True, exist_ok=True)
        Path(docs_path).write_text(markdown_tables(result))


# ============================================================
# CLI HELPERS
# ============================================================


def parse_variant(text: str) -> tuple[str, list[Path]]:
    """Parse ``name=DIR[,DIR...]`` (argparse type)."""
    name, sep, dirs = text.partition('=')
    paths = [Path(d) for d in dirs.split(',') if d.strip()]
    if not sep or not name.strip() or not paths:
        raise argparse.ArgumentTypeError(f"expected name=DIR[,DIR...], got {text!r}")
    return name.strip(), paths


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``compare`` options to ``parser``."""
    parser.add_argument('--variant', type=parse_variant, action='append', required=True,
                        help='name=SWEEP_DIR[,SWEEP_DIR...] (one directory per seed); '
                             'repeat per variant; the first is the baseline')
    parser.add_argument('--out', type=Path, required=True,
                        help='receives comparison_by_gap.csv, comparison_by_horizon.csv and '
                             'promotion.json')
    parser.add_argument('--fold', choices=FOLD_CHOICES, default='test',
                        help='plants scored: test or train fold of the original tracklet id, '
                             'or all (default test)')
    parser.add_argument('--through-gap', type=int, default=DEFAULT_THROUGH_GAP,
                        help='largest planted gap of the promotion check (default 30)')
    parser.add_argument('--rule-precision', type=float, default=1.0,
                        help='precision floor of the horizon rule (default 1.0)')
    parser.add_argument('--docs', type=Path, default=None,
                        help='write the Markdown tables to this file')
    parser.add_argument('--salt', default=DEFAULT_SALT, help='fold hash salt')
    parser.add_argument('--test-fraction', type=float, default=DEFAULT_TEST_FRACTION,
                        help='fraction of original tracklets in the test fold')
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs')


def main(args: argparse.Namespace) -> dict:
    """Run :func:`compare_variants` and :func:`write_comparison` from parsed arguments.

    Returns
    -------
    dict
        The comparison (with ``command``, the generating command line).
    """
    variants: dict[str, list[Path]] = {}
    for name, dirs in args.variant:
        if name in variants:
            raise ValueError(f"variant {name!r} given twice")
        variants[name] = dirs
    result = compare_variants(variants, fold=args.fold, through_gap=args.through_gap,
                              rule_precision=args.rule_precision, salt=args.salt,
                              test_fraction=args.test_fraction)
    result['command'] = getattr(args, 'command_line', None) or shlex.join(
        [Path(sys.executable).name] + sys.argv)
    write_comparison(result, args.out, args.docs, args.force)
    for name, v in result['variants'].items():
        p = v['promotion']
        print(f"[compare] {name}: chosen max_gap {v['chosen']}; at {p['horizon']} through "
              f"gap {p['through_gap']}: precision_ok {p['precision_ok']}, recall_ok "
              f"{p['recall_ok']} {' '.join(p['failing'])}", flush=True)
    print(f"[compare] wrote {Path(args.out) / COMPARISON_GAP_NAME}, {COMPARISON_HORIZON_NAME}, "
          f"{PROMOTION_NAME}" + (f" and {args.docs}" if args.docs else ''), flush=True)
    return result
