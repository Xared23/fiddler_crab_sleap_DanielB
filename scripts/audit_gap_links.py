#!/usr/bin/env python3

"""
Render the links a candidate gap-bridging run gained and lost (visual audit).

Example (v004 horizon 5 as current, horizon 30 as candidate)::

    .venv/bin/python scripts/audit_gap_links.py \\
        --current logs/gap_sweep_v004/current/seed_0/max_gap_5 \\
        --candidate logs/gap_sweep_v004/current/seed_0/max_gap_30 \\
        --observations demo/crab_cam2_2_middle_300/observations.parquet \\
        --out logs/gap_audit/gap5_vs_gap30

writes ``<out>_gained_p<k>.png``, ``<out>_lost_p<k>.png`` and ``<out>.csv``.
See scripts/gap_bridging/audit.py.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

# ----- local modules -----
from gap_bridging import audit  # noqa: E402

# ============================================================
# CLI
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the audit options."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[1].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    audit.add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the audit and print its counts.

    Returns
    -------
    int
        0 on success, 1 on a missing input, an existing output or a bad
        argument.
    """
    args = build_parser().parse_args(argv)
    try:
        result = audit.run_from_args(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"audit_gap_links: error: {error}", file=sys.stderr, flush=True)
        return 1
    print(audit.summary_line(result), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
