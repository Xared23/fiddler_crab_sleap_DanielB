#!/usr/bin/env python3

"""
Tests for scripts/gap_scaling.py (gap-bridging scaling study, v007).

Example tests pin the study's real windows (centred on the 300-frame demo,
nested, inside 13410-15409) and the v005 planting recipe; Hypothesis property
tests cover the window geometry, planting ratios, link agreement, length
summaries, the GNU time parser and the power-law exponent.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
import sys
from pathlib import Path

# ----- numerics and tables -----
import numpy as np
import pyarrow as pa
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

# ----- local modules -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import gap_scaling as gs  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

LO, HI = 13410, 15409
GNU_TIME = """\tCommand being timed: "python x"
\tUser time (seconds): 1.94
\tSystem time (seconds): 0.52
\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:03.46
\tMaximum resident set size (kbytes): 330668
"""

# ============================================================
# WINDOWS
# ============================================================


def test_study_windows_are_nested_and_centred():
    """The four study windows: 300 is the demo itself, 2000 the whole bundle."""
    bounds = {s: gs.window_bounds(s, LO, HI) for s in (300, 600, 1000, 2000)}
    assert bounds == {300: (14250, 14549), 600: (14100, 14699),
                      1000: (13900, 14899), 2000: (LO, HI)}
    sizes = sorted(bounds)
    for small, large in zip(sizes, sizes[1:]):
        assert bounds[large][0] <= bounds[small][0] and bounds[small][1] <= bounds[large][1]


@given(lo=st.integers(0, 10_000), extra=st.integers(0, 3000),
       anchor_off=st.integers(0, 3000), anchor_len=st.integers(1, 500),
       size_off=st.integers(0, 3000))
@example(lo=LO, extra=1700, anchor_off=840, anchor_len=300, size_off=1700)
def test_window_bounds_properties(lo, extra, anchor_off, anchor_len, size_off):
    """Length, containment of the anchor and the source range, for any feasible input."""
    hi = lo + anchor_len + extra - 1 + anchor_off
    anchor = lo + anchor_off
    assume(anchor + anchor_len - 1 <= hi)
    size = min(anchor_len + size_off, hi - lo + 1)
    start, last = gs.window_bounds(size, lo, hi, anchor, anchor_len)
    assert last - start + 1 == size
    assert lo <= start and last <= hi
    assert start <= anchor and anchor + anchor_len - 1 <= last


@given(size=st.integers(-5, 5000))
def test_window_bounds_rejects_bad_sizes(size):
    """Sizes below the anchor or above the source raise ValueError."""
    assume(not 300 <= size <= HI - LO + 1)
    with pytest.raises(ValueError):
        gs.window_bounds(size, LO, HI)


def test_cut_window_keeps_order_and_range():
    """Rows inside [start, last] only, in their original order."""
    table = pa.table({'frame': [5, 1, 3, 9, 3], 'v': [0, 1, 2, 3, 4]})
    out = gs.cut_window(table, 2, 5)
    assert out['v'].to_pylist() == [0, 2, 4]
    with pytest.raises(ValueError):
        gs.cut_window(table, 5, 2)


# ============================================================
# PLANTING
# ============================================================


def test_planting_reproduces_v005_recipe():
    """planting(8) is --per-gap 8 --decoys 4 --flicker 2; targets keep 8 per 300 frames."""
    assert gs.planting(8) == {'per_gap': 8, 'decoys': 4, 'flicker': 2}
    assert [gs.density_target(s) for s in (300, 600, 1000, 2000)] == [8, 16, 27, 53]


@given(per_gap=st.integers(1, 10_000))
def test_planting_ratios(per_gap):
    """Decoys and flicker never exceed the recipe ratios and stay at least 1."""
    plan = gs.planting(per_gap)
    assert plan['per_gap'] == per_gap
    assert 1 <= plan['decoys'] <= max(1, per_gap / 2)
    assert 1 <= plan['flicker'] <= max(1, per_gap / 4)


@given(per_gap=st.integers(-100, 0))
def test_planting_rejects_non_positive(per_gap):
    """Zero or negative plants raise ValueError."""
    with pytest.raises(ValueError):
        gs.planting(per_gap)


# ============================================================
# NATURAL LINKS
# ============================================================


def test_linked_pairs_only_linked_rows():
    """Refused rows are not links."""
    links = pa.table({'tracklet_a': ['a', 'b', 'c'], 'tracklet_b': ['x', 'y', 'z'],
                      'decision': ['linked', 'refused_gate', 'linked']})
    assert gs.linked_pairs(links) == {('a', 'x'), ('c', 'z')}
    with pytest.raises(ValueError):
        gs.linked_pairs(links.drop(['decision']))


pairs = st.sets(st.tuples(st.integers(0, 20), st.integers(0, 20)), max_size=40)


@given(a=pairs, b=pairs)
def test_agreement_properties(a, b):
    """Counts partition both sets; Jaccard is symmetric, in [0, 1], 1 iff equal."""
    ab, ba = gs.agreement(a, b), gs.agreement(b, a)
    assert ab['both'] + ab['only_a'] == len(a) and ab['both'] + ab['only_b'] == len(b)
    assert (ab['only_a'], ab['only_b']) == (ba['only_b'], ba['only_a'])
    if a or b:
        assert 0.0 <= ab['jaccard'] <= 1.0 and ab['jaccard'] == ba['jaccard']
        assert (ab['jaccard'] == 1.0) == (a == b)
    else:
        assert math.isnan(ab['jaccard'])


@given(lengths=st.lists(st.integers(1, 400), min_size=1, max_size=200),
       window=st.integers(400, 2000))
def test_length_summary_bounds(lengths, window):
    """Quantiles ordered inside [min, max]; fractions in [0, 1]; n is the count."""
    out = gs.length_summary(np.array(lengths), window)
    arr = np.array(lengths, float)
    assert out['n'] == len(lengths)
    assert arr.min() <= out['p10'] <= out['median'] <= out['p90'] <= arr.max()
    assert out['mean'] == pytest.approx(arr.mean())
    for key in ('frac_frames_ge_30', 'frac_frames_ge_150', 'frac_full'):
        assert 0.0 <= out[key] <= 1.0
    assert out['frac_frames_ge_150'] <= out['frac_frames_ge_30']


def test_length_summary_edges():
    """Empty input gives NaNs; a length beyond the window raises."""
    assert math.isnan(gs.length_summary(np.zeros(0), 10)['median'])
    assert gs.length_summary(np.array([10.0, 10.0]), 10)['frac_full'] == 1.0
    with pytest.raises(ValueError):
        gs.length_summary(np.array([11.0]), 10)


def test_tracklet_and_trajectory_lengths():
    """Spans are last - first + 1 over accepted rows; null tracklets ignored."""
    obs = pa.table({'tracklet_id': ['a', 'a', 'a', 'b', None, 'c'],
                    'frame': [10, 12, 30, 5, 7, 8],
                    'accepted_for_tracking': [True, True, False, True, True, False]})
    assert sorted(gs.tracklet_lengths(obs).tolist()) == [1.0, 3.0]
    traj = pa.table({'start_frame': [0, 4], 'end_frame': [0, 9]})
    assert gs.trajectory_lengths(traj).tolist() == [1.0, 6.0]


# ============================================================
# RUNTIME AND MEMORY
# ============================================================


def test_parse_gnu_time_example():
    """The fields of a real ``/usr/bin/time -v`` block."""
    out = gs.parse_gnu_time(GNU_TIME)
    assert out == {'wall_s': pytest.approx(3.46), 'max_rss_kb': 330668.0,
                   'user_s': 1.94, 'system_s': 0.52}
    with pytest.raises(ValueError):
        gs.parse_gnu_time('nothing here')


@given(h=st.integers(0, 30), m=st.integers(0, 59), s=st.floats(0, 59.99))
def test_clock_seconds_round_trip(h, m, s):
    """h:mm:ss.ss and m:ss.ss both decode to the same seconds."""
    total = h * 3600 + m * 60 + round(s, 2)
    text = f"{h}:{m:02d}:{round(s, 2):05.2f}"
    assert gs._clock_seconds(text) == pytest.approx(total, abs=1e-6)
    assert gs._clock_seconds(f"{h * 60 + m}:{round(s, 2):05.2f}") == pytest.approx(total, abs=1e-6)


@given(exponent=st.floats(-3, 3), scale=st.floats(1e-3, 1e3),
       sizes=st.lists(st.integers(1, 10_000), min_size=2, max_size=8, unique=True))
@example(exponent=1.0, scale=1.0, sizes=[300, 600, 1000, 2000])
def test_scaling_exponent_recovers_power_law(exponent, scale, sizes):
    """An exact power law returns its exponent; rescaling values leaves it unchanged."""
    x = np.array(sizes, float)
    assume(np.ptp(np.log(x)) > 1e-3)
    y = scale * x ** exponent
    assert gs.scaling_exponent(x, y) == pytest.approx(exponent, abs=1e-6)
    assert gs.scaling_exponent(x, 7.0 * y) == pytest.approx(exponent, abs=1e-6)


def test_scaling_exponent_errors():
    """One point, unequal lengths or non-positive values raise ValueError."""
    for sizes, values in (([1], [1]), ([1, 2], [1]), ([1, 2], [0, 1]), ([2, 2], [1, 3])):
        with pytest.raises(ValueError):
            gs.scaling_exponent(sizes, values)


# ============================================================
# OVERLAYS AND TABLES
# ============================================================


def test_overlay_text(tmp_path):
    """Threshold and dummy are equal; bare scorer path when it sits beside the overlay."""
    scorer = tmp_path / 'm.joblib'
    scorer.write_bytes(b'x')
    text = gs.overlay_text(scorer, 0.5, tmp_path)
    assert 'scorer_path: m.joblib\n' in text
    assert 'learned_min_probability: 0.5\n' in text
    assert 'learned_unmatched_probability: 0.5\n' in text
    assert str(scorer.resolve()) in gs.overlay_text(scorer, 0.5, tmp_path / 'other')
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            gs.overlay_text(scorer, bad)


def test_markdown_table_alignment():
    """Numeric columns right-aligned, text columns left; ragged rows raise."""
    text = gs.markdown_table(['name', 'n'], [['a', 1], ['b', 2.5]])
    assert text.splitlines()[1] == '|---|---:|'
    with pytest.raises(ValueError):
        gs.markdown_table(['a', 'b'], [[1]])


# ============================================================
# REPORT HELPERS
# ============================================================


def test_report_helpers():
    """Row lookup, line style per variant and the runtime medians."""
    report = pytest.importorskip('gap_scaling_report')
    rows = [{'size': 300, 'variant': 'current', 'v': 1}, {'size': 600, 'variant': 'current', 'v': 2}]
    assert report._cell(rows, size=600, variant='current')['v'] == 2
    with pytest.raises(KeyError):
        report._cell(rows, variant='current')
    assert report._style('s1_rt')['ls'] == '--' and report._style('s1_frozen')['ls'] == '-'
    assert report._style('current')['color'] == 'black'
    with pytest.raises(ValueError):
        report._style('s9_rt')
    measure = [{'label': f"w{s}:{k}", 'wall_s': str(w), 'stage_total': '1', 'stage_link': '0.1',
                'stage_assemble': '0.5', 'max_rss_kb': str(1024 * s), 'links': '10'}
               for s in (300, 600) for k in gs.SCHEMES for w in (1.0, 2.0, 9.0)]
    rt = report.runtime_rows({'sizes': [300, 600], 'measure': measure})
    assert len(rt) == 6 and all(r['wall_s'] == 2.0 and r['repeats'] == 3 for r in rt)
    assert [r['max_rss_mib'] for r in rt if r['scheme'] == 'current'] == [300.0, 600.0]
    text = report._exponent_table(rt, [300, 600])
    assert '| current | 0 | 0 | 0 | 1 |' in text


def test_role_scores_matches_compare_on_committed_sweeps():
    """Plants plus decoy targets reproduce compare's test-fold counts (v005/v006 logs)."""
    sweeps = ROOT / 'logs/gap_sweep_v005/current_short'
    by_gap = ROOT / 'logs/gap_sweep_v006/compare_test/comparison_by_gap.csv'
    if not (sweeps / 'seed_4/holdout_30').is_dir() or not by_gap.is_file():
        pytest.skip('v005 sweeps or the v006 compare output are absent (git-ignored logs)')
    import csv
    total = {}
    for seed in range(5):
        for gap, c in gs.role_scores(sweeps / f"seed_{seed}/holdout_30", 'test',
                                     ('plant', 'decoy_target')).items():
            t = total.setdefault(gap, [0, 0, 0])
            for i, k in enumerate(('planted', 'recovered', 'false_link')):
                t[i] += c[k]
    with open(by_gap, newline='') as handle:
        ref = {int(r['gap']): [int(r['planted']), int(r['recovered']), int(r['false_link'])]
               for r in csv.DictReader(handle)
               if r['variant'] == 'current' and r['max_gap'] == '30'}
    assert total == ref
    with pytest.raises(ValueError):
        gs.role_scores(sweeps / 'seed_0/holdout_30', 'nope', ('plant',))


def test_regression_D015_seed_dirs_skip_logs(tmp_path):
    """D-015: seed_<k>.log files beside the seed folders are not seeds; order is numeric."""
    variant = tmp_path / 'sweeps' / 'w300' / 'current'
    for k in (0, 1, 10, 2):
        (variant / f"seed_{k}").mkdir(parents=True)
        (variant / f"seed_{k}.log").write_text('log')
    (variant / 'seed_x').mkdir()
    assert [p.name for p in gs._seed_dirs(tmp_path, 300, 'current')] == \
        ['seed_0', 'seed_1', 'seed_2', 'seed_10']
    assert gs._seed_dirs(tmp_path, 600, 'current') == []
