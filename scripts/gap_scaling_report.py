#!/usr/bin/env python3

"""
Tables and figures of the gap-bridging scaling study (v007).

Called by ``gap_scaling.py report``; reads the study root written by
``jobs/gap_scaling_v007.slurm`` (windows, compare, natural, measure, models)
and writes one generated Markdown file of tables plus three figures. The
narrative and the verdict live in docs/GAP_BRIDGING.md (v007 section).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

# ----- numerics and plotting -----
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----- local modules -----
import gap_scaling as gs  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

STYLE = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 14,
         "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
         "xtick.top": False, "ytick.right": False}
SCHEME_COLOR = {'current': 'black', 's1': '#1f4e9c', 's1p50': '#c0392b'}
LABEL = dict(gs.VARIANTS)
FIG_PREFIX = 'gap_bridging_v007_scaling'

# ============================================================
# LOADING
# ============================================================


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Rows of a CSV file as dicts."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _style(variant: str) -> Dict:
    """Colour by scheme, solid for frozen or hand cost, dashed for retrained."""
    scheme = variant.split('_')[0]
    if scheme not in SCHEME_COLOR:
        raise ValueError(f"unknown variant {variant!r}")
    return {'color': SCHEME_COLOR[scheme], 'lw': 2.0, 'marker': 'o', 'ms': 5,
            'ls': '--' if variant.endswith('_rt') else '-'}


def load_study(root: Path, sizes: Sequence[int]) -> Dict:
    """Every table the report needs, keyed by name."""
    if not sizes:
        raise ValueError("no window sizes")
    root = Path(root)
    windows = [json.loads((root / 'windows' / f"w{s}" / 'window.json').read_text()) for s in sizes]
    measure = []
    for s in sizes:
        for scheme in gs.SCHEMES:
            measure += _read_csv(root / 'measure' / f"w{s}_{scheme}.csv")
    study = {'windows': windows, 'holdout': gs.holdout_rows(root, sizes),
             'models': gs.model_rows(root, sizes),
             'natural': _read_csv(root / 'natural' / 'summary.csv'),
             'agreement': _read_csv(root / 'natural' / 'agreement.csv'),
             'tracklets': _read_csv(root / 'natural' / 'tracklets.csv'),
             'plants_only': _read_csv(root / 'natural' / 'plants_only.csv'),
             'measure': measure, 'sizes': [int(s) for s in sizes], 'root': root}
    assert len(study['natural']) >= len(sizes)
    return study


def _cell(rows: List[Dict], **match) -> Dict:
    """The single row of ``rows`` matching every ``key=value`` (compared as strings)."""
    hits = [r for r in rows if all(str(r[k]) == str(v) for k, v in match.items())]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} rows match {match}")
    return hits[0]


def runtime_rows(study: Dict) -> List[Dict]:
    """Median wall and stage seconds and peak RSS per window and scheme."""
    rows = []
    for size in study['sizes']:
        for scheme in gs.SCHEMES:
            reps = [r for r in study['measure'] if r['label'] == f"w{size}:{scheme}"]
            if not reps:
                raise KeyError(f"no measurements for w{size}:{scheme}")
            med = {k: float(np.median([float(r[k]) for r in reps]))
                   for k in ('wall_s', 'stage_total', 'stage_link', 'stage_assemble')}
            rows.append({'size': size, 'scheme': scheme, 'repeats': len(reps), **med,
                         'max_rss_mib': max(float(r['max_rss_kb']) for r in reps) / 1024.0,
                         'links': int(reps[0]['links'])})
    return rows


# ============================================================
# TABLES
# ============================================================


def tables_markdown(study: Dict) -> str:
    """All report tables as Markdown sections."""
    sizes, out = study['sizes'], []
    win = [[w['size'], f"{w['start']}-{w['last']}", w['rows'], w['tracklets'],
            f"{w['per_gap']} ({w['density_target']})", w['decoys'], w['flicker'],
            _cell(study['holdout'], size=w['size'], fold='all', variant='current',
                  max_gap=30)['planted'],
            _cell(study['holdout'], size=w['size'], fold='test', variant='current',
                  max_gap=30)['planted']] for w in study['windows']]
    out.append("## Windows\n\n" + gs.markdown_table(
        ['frames', 'original frames', 'rows', 'tracklets', 'plants per gap (density target)',
         'decoys', 'flicker', 'plants incl. decoy targets (all, 5 seeds)', 'of which test fold'], win))
    mod = [[m['size'], f"{m['train_pairs']} ({m['train_positive']})",
            f"{m['test_pairs']} ({m['test_positive']})", m['threshold'], m['pair_threshold'],
            f"{m['curve_recovered']} / {m['curve_false_link']} of {m['curve_planted']}",
            m['test_precision'], m['test_recall']] for m in study['models']]
    out.append("## Retrained S1 per window (pipeline rule)\n\n" + gs.markdown_table(
        ['frames', 'train pairs (positive)', 'test pairs (positive)', 'rule threshold',
         'pair threshold', 'train-fold plants through gap 30 (recovered / false)',
         'pairwise test precision', 'pairwise test recall'], mod))
    out.append(_promotion_table(study))
    out.append(_plants_only_table(study))
    for fold, gap in (('test', 30), ('all', 30), ('all', 60)):
        out.append(_holdout_table(study, fold, gap))
    out.append(_natural_tables(study))
    rt = runtime_rows(study)
    out.append("## Runtime and memory (`bridge_gaps.py run`, max_gap 30; times are medians "
               "and peak RSS the maximum over repeats)\n\n"
               + gs.markdown_table(['frames', 'scheme', 'wall s', 'pipeline s', 'link stage s',
                                    'assemble s', 'peak RSS MiB', 'links'],
                                   [[r['size'], r['scheme'], r['wall_s'], r['stage_total'],
                                     r['stage_link'], r['stage_assemble'], r['max_rss_mib'],
                                     r['links']] for r in rt])
               + "\n\n" + _exponent_table(rt, sizes))
    return '\n\n'.join(out) + '\n'


def _promotion_table(study: Dict) -> str:
    """Promotion verdict per window and variant on the test fold (horizon 30)."""
    rows = []
    for size in study['sizes']:
        for variant, _ in gs.VARIANTS:
            r = _cell(study['holdout'], size=size, fold='test', variant=variant, max_gap=30)
            rows.append([size, LABEL[variant], 'pass' if r['promotion_ok'] else 'fail',
                         r['precision_ok'], r['recall_ok'],
                         f"{r['demo_links']} (floor {r['demo_floor']})", r['failing'] or '-',
                         r['chosen_horizon'] if r['chosen_horizon'] is not None else 'none',
                         f"{'pass' if r['compare_ok'] else 'fail'} at {r['compare_horizon']}"])
    return ("## Promotion rule per window (test fold, horizon 30, through gap 30; floor = "
            "hand-cost natural links on that window)\n\n"
            "The rule is applied at horizon 30 on every window. `compare` itself checks at the "
            "hand cost's chosen horizon (largest horizon whose pooled precision is 1.0 at "
            "every planted gap, else the fallback 30), which moves between windows; its "
            "chosen horizons and verdicts are in the last two columns.\n\n"
            + gs.markdown_table(
                ['frames', 'variant', 'verdict', 'precision 1.0', 'recall >= hand',
                 'natural links', 'failing', 'own chosen horizon', "compare's verdict"], rows))


def _plants_only_table(study: Dict) -> str:
    """Plants-only totals through gap 30 at horizon 30 and the rule without decoy targets."""
    rows = []
    for fold in ('test', 'all'):
        for size in study['sizes']:
            for variant, _ in gs.VARIANTS:
                r = _cell(study['plants_only'], size=size, fold=fold, variant=variant)
                rows.append([fold, size, LABEL[variant],
                             f"{r['recovered']} / {r['false_link']} of {r['planted']}",
                             r['gaps_with_false'], r['precision_ok'] == 'True',
                             r['recall_ok'] == 'True'])
    return ("## Plants only (decoy targets excluded), horizon 30, gaps 1-30\n\n"
            "Decoy targets are planted pairs with a donor right half placed 3 px from the "
            "true right half; with 1-frame halves no linker can tell them apart. "
            "Recall >= hand is per gap against the hand cost on the same plants.\n\n"
            + gs.markdown_table(['fold', 'frames', 'variant', 'recovered / false of planted',
                                 'gaps with a false link', 'precision 1.0',
                                 'recall >= hand'], rows))


def _holdout_table(study: Dict, fold: str, gap: int) -> str:
    """Recovered / false of planted per window (rows) and variant (columns)."""
    rows = []
    for size in study['sizes']:
        row: List = [size]
        for variant, _ in gs.VARIANTS:
            r = _cell(study['holdout'], size=size, fold=fold, variant=variant, max_gap=gap)
            row.append(f"{r['recovered']} / {r['false_link']} of {r['planted']} "
                       f"({100 * r['recall']:.1f}%)")
        rows.append(row)
    return (f"## Hold-out at horizon {gap}, {fold} {'fold' if fold == 'test' else 'plants'} "
            "(recovered / false links of planted, recall)\n\n"
            + gs.markdown_table(['frames'] + [LABEL[v] for v, _ in gs.VARIANTS], rows))


def _natural_tables(study: Dict) -> str:
    """Natural-link outcomes and agreement with the hand cost at horizon 30."""
    nat = [[r['size'], LABEL[r['variant']], r['links'], float(r['links_per_kframe']),
            r['traj_n'], float(r['traj_median']), float(r['traj_mean']),
            float(r['traj_frac_frames_ge_150']), float(r['traj_frac_full']),
            r['identity_starts']] for r in study['natural']]
    agr = [[r['size'], LABEL[r['variant']], r['both'], r['only_a'], r['only_b'],
            float(r['jaccard'])] for r in study['agreement']]
    trk = [[r['size'], r['tracklet_n'], float(r['tracklet_median']), float(r['tracklet_mean']),
            float(r['tracklet_frac_frames_ge_150']), float(r['tracklet_frac_full'])]
           for r in study['tracklets']]
    return ("## Input tracklets per window\n\n" + gs.markdown_table(
        ['frames', 'tracklets', 'median span', 'mean span', 'frame share in spans >= 150',
         'share spanning window'], trk)
        + "\n\n## Natural links at horizon 30 (unplanted window)\n\n" + gs.markdown_table(
        ['frames', 'variant', 'links', 'links per 1000 frames', 'trajectories',
         'median length', 'mean length', 'frame share in trajectories >= 150',
         'share spanning window', 'identity starts'], nat)
        + "\n\n## Agreement with the hand cost (natural links, horizon 30)\n\n"
        + gs.markdown_table(['frames', 'variant', 'shared', 'gained', 'lost', 'Jaccard'], agr))


def _exponent_table(rt: List[Dict], sizes: Sequence[int]) -> str:
    """Power-law exponents of time and memory against window length."""
    rows = []
    for scheme in gs.SCHEMES:
        mine = [r for r in rt if r['scheme'] == scheme]
        rows.append([scheme] + [round(gs.scaling_exponent([r['size'] for r in mine],
                                                          [r[k] for r in mine]), 3) + 0.0
                                for k in ('wall_s', 'stage_total', 'stage_link', 'max_rss_mib')])
    return ("Fitted exponent b of value ~ frames^b over the windows "
            f"{', '.join(map(str, sizes))}:\n\n"
            + gs.markdown_table(['scheme', 'wall', 'pipeline', 'link stage', 'peak RSS'], rows))


# ============================================================
# FIGURES
# ============================================================


def fig_accuracy(study: Dict, path: Path, dpi: int) -> None:
    """Test recall, false links per 100 plants and natural links per 1000 frames vs frames."""
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), sharex=True)
    sizes = study['sizes']
    for variant, _ in gs.VARIANTS:
        test = [_cell(study['holdout'], size=s, fold='test', variant=variant, max_gap=30)
                for s in sizes]
        allp = [_cell(study['holdout'], size=s, fold='all', variant=variant, max_gap=30)
                for s in sizes]
        nat = [float(_cell(study['natural'], size=s, variant=variant)['links_per_kframe'])
               for s in sizes]
        st = _style(variant)
        axes[0].plot(sizes, [r['recall'] for r in test], label=LABEL[variant], **st)
        axes[1].plot(sizes, [100.0 * r['false_link'] / r['planted'] for r in allp], **st)
        axes[2].plot(sizes, nat, **st)
    axes[0].set_ylabel(r"Test recall, horizon 30")
    axes[1].set_ylabel(r"False links per 100 plants (all)")
    axes[2].set_ylabel(r"Natural links per $10^3$ frames")
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel(r"Window length, frames")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_runtime(study: Dict, path: Path, dpi: int) -> None:
    """Wall time and peak RSS of one bridging run against window length (log-log)."""
    plt.rcParams.update(STYLE)
    rt = runtime_rows(study)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    names = {'current': LABEL['current'], 's1': 'v006 S1', 's1p50': 'v005 S1 p>=0.5'}
    for scheme in gs.SCHEMES:
        mine = [r for r in rt if r['scheme'] == scheme]
        st = _style(scheme + '_frozen' if scheme != 'current' else scheme)
        x = [r['size'] for r in mine]
        axes[0].plot(x, [r['wall_s'] for r in mine], label=names[scheme], **st)
        axes[1].plot(x, [r['max_rss_mib'] for r in mine], **st)
    axes[0].set_ylabel(r"Wall time, $t$ (s)")
    axes[1].set_ylabel(r"Peak memory (MiB)")
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(study['sizes'], [str(s) for s in study['sizes']])
        ax.minorticks_off()
    axes[0].set_yticks([3, 5, 10, 15], ['3', '5', '10', '15'])
    axes[1].set_yticks([300, 500, 1000], ['300', '500', '1000'])
    for ax in axes:
        ax.minorticks_off()
        ax.set_xlabel(r"Window length, frames")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0),
               frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def fig_lengths(study: Dict, path: Path, dpi: int) -> None:
    """Complementary CDF of trajectory lengths on the largest window, input tracklets in grey."""
    plt.rcParams.update(STYLE)
    size = max(study['sizes'])
    arrays = np.load(Path(study['root']) / 'natural' / f"lengths_w{size}.npz")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    order = ['tracklets'] + [v for v, _ in gs.VARIANTS if v in arrays.files]
    for name in order:
        lengths = np.sort(arrays[name])
        ccdf = 1.0 - np.arange(lengths.size) / lengths.size
        st = ({'color': '0.6', 'lw': 2.0} if name == 'tracklets'
              else {k: v for k, v in _style(name).items() if k not in ('marker', 'ms')})
        ax.step(lengths, ccdf, where='pre',
                label='input tracklets' if name == 'tracklets' else LABEL[name], **st)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"Length, $\ell$ (frames)")
    ax.set_ylabel(r"Fraction longer, $P(L \geq \ell)$")
    ax.set_title(f"{size}-frame window, horizon 30", fontsize=14)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ============================================================
# ENTRY
# ============================================================


def write_report(root: Path, sizes: Sequence[int], docs: Path, figures: Path, dpi: int) -> None:
    """Write the generated tables document and the three figures."""
    if dpi < 72:
        raise ValueError(f"dpi too low: {dpi}")
    study = load_study(root, sizes)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {k: figures / f"{FIG_PREFIX}_{k}.png" for k in ('accuracy', 'runtime', 'lengths')}
    fig_accuracy(study, paths['accuracy'], dpi)
    fig_runtime(study, paths['runtime'], dpi)
    fig_lengths(study, paths['lengths'], dpi)
    rel = {k: Path('figures') / p.name for k, p in paths.items()}
    head = ("# Gap bridging v007: scaling to the 2,000-frame demo (generated tables)\n\n"
            "Generated by `scripts/gap_scaling.py report` from `logs/gap_scaling_v007/` "
            "(Slurm chain `jobs/gap_scaling_v007_submit.sh`); the method, caveats and the "
            "verdict are in [GAP_BRIDGING.md](GAP_BRIDGING.md#v007-scaling-to-the-2000-frame-"
            "demo). Variants: the v003 hand cost; v006 S1 at its pipeline threshold and v005 "
            "S1 at p >= 0.5 (dummy tied), each as committed (frozen, trained on the 300-frame "
            "demo) and retrained on the window's own hand-cost sweeps (5 seeds, short "
            "profile).\n\n"
            f"![accuracy]({rel['accuracy']})\n\n![runtime]({rel['runtime']})\n\n"
            f"![lengths]({rel['lengths']})\n\n")
    docs.write_text(head + tables_markdown(study))
    print(f"[report] wrote {docs} and {', '.join(str(p) for p in paths.values())}", flush=True)
