#!/usr/bin/env python3

"""
Bridge short gaps between body-first tracklets (command-line entry point).

Example::

    .venv/bin/python scripts/bridge_gaps.py \\
        --observations demo/crab_cam2_2_middle_300/observations.parquet \\
        --out outputs/gap_bridging/demo_v001 \\
        --config configs/tracking/gap_bridging_v001.yaml

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
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ----- local modules -----
from gap_bridging import pipeline  # noqa: E402
from gap_bridging.config import load_config  # noqa: E402

# ============================================================
# CLI
# ============================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--observations', type=Path, required=True,
                        help='observation parquet file, or a directory of parquet parts')
    parser.add_argument('--out', type=Path, required=True, help='output directory')
    parser.add_argument('--config', type=Path, default=None,
                        help='YAML config (default: built-in defaults = gap_bridging_v001.yaml)')
    parser.add_argument('--max-gap', type=int, default=None,
                        help='maximum number of missing frames to bridge (0 disables linking)')
    parser.add_argument('--margin', type=float, default=None,
                        help='minimum cost margin to the runner-up (px-equivalent)')
    parser.add_argument('--recovery-radius', type=float, default=None,
                        help='radius (px) for adopting a rejected detection')
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline; print the completion marker as JSON.

    Returns
    -------
    int
        0 on success, 1 on a configuration, input or overwrite error.
    """
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, max_gap=args.max_gap, minimum_cost_margin=args.margin,
                          recovery_radius_px=args.recovery_radius)
        marker = pipeline.run(args.observations, args.out, cfg, force=args.force)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"bridge_gaps: error: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(marker, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
