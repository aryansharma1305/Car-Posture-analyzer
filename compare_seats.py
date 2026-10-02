#!/usr/bin/env python3
"""Compare driver comfort across the seats under test.

Groups every session in a log directory by the seat_id in its metadata
sidecar, scores each one against the posture model it was recorded under,
and reports the comparison. Frames the model could not observe are
excluded, not averaged in.

    python3 compare_seats.py [log_dir]
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from driver_comfort_analyzer import DriverComfortAnalyzer
from driver_model import get_reference
from session_schema import SessionMetadata


def load_angles(csv_path):
    """Per-row angle dicts, passing the signed columns through when recorded."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    samples = []
    for _, row in df.iterrows():
        angles = {}
        for column, key in (("trunk_deg", "trunk_from_vertical"),
                            ("neck_deg", "neck_from_vertical"),
                            ("trunk_signed", "trunk_signed"),
                            ("neck_signed", "neck_signed"),
                            ("hip_L", "left_hip_angle"),
                            ("hip_R", "right_hip_angle"),
                            ("knee_L", "left_knee_angle"),
                            ("knee_R", "right_knee_angle")):
            if column in df.columns and pd.notna(row[column]):
                angles[key] = float(row[column])
        samples.append({"angles": angles})
    return samples


def main(log_dir="posture_logs"):
    by_seat = defaultdict(list)
    references = {}
    unattributed = []

    for csv_path in sorted(Path(log_dir).glob("*.csv")):
        meta = SessionMetadata.load(csv_path)
        if meta is None or not meta.seat_id:
            unattributed.append(csv_path.name)
            continue
        by_seat[meta.seat_id].extend(load_angles(csv_path))
        references[meta.seat_id] = meta.posture_reference

    if unattributed:
        print(f"Skipped {len(unattributed)} session(s) with no seat_id "
              f"(record with --seat-id to include them):")
        for name in unattributed[:5]:
            print(f"  - {name}")
        if len(unattributed) > 5:
            print(f"  ... and {len(unattributed) - 5} more")
        print()

    if not by_seat:
        print("No attributed sessions found. Nothing to compare.")
        return 1

    # One analyzer per reference, since the seats must be scored under the
    # model they were recorded against.
    results = {}
    for seat_id, samples in by_seat.items():
        analyzer = DriverComfortAnalyzer(get_reference(references[seat_id]))
        results.update(analyzer.analyze_seat_comparison({seat_id: samples}))

    print(f"{'seat':20} {'mean':>7} {'sd':>6} {'min':>6} {'max':>6} "
          f"{'frames':>7} {'excluded':>9}")
    print("-" * 68)
    for seat_id in sorted(results):
        r = results[seat_id]
        if "error" in r:
            print(f"{seat_id:20} {'-':>7} {'-':>6} {'-':>6} {'-':>6} "
                  f"{r['sample_size']:>7} {r['excluded_samples']:>9}   "
                  f"<- {r['error']}")
            continue
        print(f"{seat_id:20} {r['average_comfort']:7.1f} {r['comfort_std']:6.1f} "
              f"{r['min_comfort']:6.1f} {r['max_comfort']:6.1f} "
              f"{r['sample_size']:>7} {r['excluded_samples']:>9}")

    scored = {s: r for s, r in results.items() if "average_comfort" in r}
    if len(scored) >= 2:
        best = max(scored, key=lambda s: scored[s]["average_comfort"])
        worst = min(scored, key=lambda s: scored[s]["average_comfort"])
        gap = scored[best]["average_comfort"] - scored[worst]["average_comfort"]
        print(f"\n{best} scores {gap:.1f} points above {worst}.")
        print("Note: this is a mean difference, not a significance test. "
              "Treat it as a direction to investigate.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "posture_logs"))
