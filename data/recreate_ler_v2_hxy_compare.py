#!/usr/bin/env python3

import argparse
import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


@dataclass(frozen=True)
class Record:
    shots: int
    errors: int
    discards: int
    seconds: float
    decoder: str
    strong_id: str
    metadata: dict[str, Any]
    custom_counts: dict[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a ler_v2-style comparison plot with the historical H_XY trajectory series and a new movement-aware H_XY trajectory series."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "all_paper_data_v2.csv",
        help="Input aggregate CSV file.",
    )
    parser.add_argument(
        "--legacy-traj-dir",
        type=Path,
        action="append",
        required=True,
        help="Shard directory for the historical H_XY trajectory series. Repeat to combine directories.",
    )
    parser.add_argument(
        "--movement-traj-dir",
        type=Path,
        action="append",
        required=True,
        help="Shard directory for the new movement-aware H_XY trajectory series. Repeat to combine directories.",
    )
    parser.add_argument(
        "--dephase-only-traj-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional shard directory for a movement dephasing-only H_XY trajectory series. Repeat to combine directories.",
    )
    parser.add_argument(
        "--legacy-label",
        default="|H_XY> (Simulated SR/LR CCZs)",
        help="Legend label for the historical H_XY trajectory series.",
    )
    parser.add_argument(
        "--movement-label",
        default="|H_XY> (+ movement)",
        help="Legend label for the movement-aware H_XY trajectory series.",
    )
    parser.add_argument(
        "--dephase-only-label",
        default="|H_XY> (+ movement dephasing only)",
        help="Legend label for the movement dephasing-only H_XY trajectory series.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "ler_v2_hxy_movement_compare.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--x-scale",
        choices=["log", "linear"],
        default="log",
        help="X-axis scale for expected attempts.",
    )
    parser.add_argument(
        "--auto-focus-x",
        action="store_true",
        help="Fit the x-axis tightly around the plotted curves instead of using the wide paper-style range.",
    )
    parser.add_argument(
        "--hide-exact-simulation",
        action="store_true",
        help="Omit the historical exact-simulation curve from the plot and legend.",
    )
    return parser.parse_args()


def load_records(csv_path: Path) -> list[Record]:
    records: list[Record] = []
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        for row in reader:
            shots, errors, discards, seconds, decoder, strong_id, metadata, custom_counts = row
            records.append(
                Record(
                    shots=int(shots),
                    errors=int(errors),
                    discards=int(discards),
                    seconds=float(seconds),
                    decoder=decoder,
                    strong_id=strong_id,
                    metadata=json.loads(metadata),
                    custom_counts=json.loads(custom_counts),
                )
            )
    return records


def pick_record(records: list[Record], filters: dict[str, Any]) -> Record:
    matches = []
    for record in records:
        if all(record.metadata.get(key) == value for key, value in filters.items()):
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"Expected 1 record for {filters}, found {len(matches)}")
    return matches[0]


def aggregate_shard_dir(shard_dir: Path) -> Record:
    files = sorted(shard_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No shard files found in {shard_dir}")

    shots = 0
    errors = 0
    discards = 0
    seconds = 0.0
    decoder = None
    strong_id = f"aggregate:{shard_dir.name}"
    metadata = None
    custom_counts: Counter[str] = Counter()

    for file in files:
        row = next(csv.reader([file.read_text().strip()]))
        file_shots, file_errors, file_discards, file_seconds, file_decoder, _, file_metadata, file_custom_counts = row
        shots += int(file_shots)
        errors += int(file_errors)
        discards += int(file_discards)
        seconds += float(file_seconds)
        decoder = file_decoder
        if metadata is None:
            metadata = json.loads(file_metadata)
        custom_counts.update(json.loads(file_custom_counts))

    assert metadata is not None
    assert decoder is not None
    return Record(
        shots=shots,
        errors=errors,
        discards=discards,
        seconds=seconds,
        decoder=decoder,
        strong_id=strong_id,
        metadata=metadata,
        custom_counts=dict(custom_counts),
    )


def aggregate_shard_dirs(shard_dirs: list[Path]) -> Record:
    aggregates = [aggregate_shard_dir(shard_dir) for shard_dir in shard_dirs]
    metadata = dict(aggregates[0].metadata)
    metadata["c"] = " + ".join(record.metadata.get("c", record.strong_id) for record in aggregates)
    combined_counts: Counter[str] = Counter()
    for record in aggregates:
        combined_counts.update(record.custom_counts)
    return Record(
        shots=sum(record.shots for record in aggregates),
        errors=sum(record.errors for record in aggregates),
        discards=sum(record.discards for record in aggregates),
        seconds=sum(record.seconds for record in aggregates),
        decoder=aggregates[0].decoder,
        strong_id="aggregate:" + "+".join(path.name for path in shard_dirs),
        metadata=metadata,
        custom_counts=dict(combined_counts),
    )


def curve_points(record: Record) -> tuple[list[float], list[float]]:
    gaps = sorted({int(key[1:]) for key in record.custom_counts})
    attempts: list[float] = []
    logical_error_rates: list[float] = []
    error_prefixes = {"X", "Y", "Z", "E"}
    for threshold in gaps:
        kept = sum(
            count
            for key, count in record.custom_counts.items()
            if int(key[1:]) >= threshold
        )
        failures = sum(
            count
            for key, count in record.custom_counts.items()
            if key[0] in error_prefixes and int(key[1:]) >= threshold
        )
        if failures == 0:
            break
        attempts.append(record.shots / kept)
        logical_error_rates.append(failures / kept)
    return attempts, logical_error_rates


def focused_x_limits(
    all_x_values: list[float],
    x_scale: str,
) -> tuple[float, float]:
    xmin = min(all_x_values)
    xmax = max(all_x_values)
    if x_scale == "linear":
        span = xmax - xmin
        pad = max(0.08 * span, 0.05)
        return xmin - pad, xmax + pad
    return xmin / 1.08, xmax * 1.08


def plot(
    records: list[Record],
    legacy_hxy: Record,
    movement_hxy: Record,
    dephase_only_hxy: Record | None,
    legacy_label: str,
    movement_label: str,
    dephase_only_label: str,
    output_path: Path,
    x_scale: str,
    auto_focus_x: bool,
    hide_exact_simulation: bool,
) -> None:
    y_work = pick_record(
        records,
        {
            "c": "e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform",
            "p": 0.001,
            "ourbase": 1,
        },
    )
    gjs24 = pick_record(
        records,
        {
            "c": "GJS24",
            "p": 0.001,
            "f": 3,
        },
    )
    cclp25 = pick_record(
        records,
        {
            "c": "CCLP25",
            "p": 0.001,
            "f": 3,
        },
    )

    fig, ax = plt.subplots(figsize=(5.29, 4.08), dpi=100)
    series = [
        (y_work, "|Y>", "#de5d83", "#f8dee6", "o"),
        (legacy_hxy, legacy_label, "#2ebd85", "#ddf6ea", "s"),
        (movement_hxy, movement_label, "#f28e2b", "#fbe2c4", "D"),
        (gjs24, "Ref. [9]", "#4997d0", "#daeaf5", "o"),
        (cclp25, "Ref. [10]", "#9966cc", "#eae0f4", "o"),
    ]
    if dephase_only_hxy is not None:
        series.insert(3, (dephase_only_hxy, dephase_only_label, "#8c564b", "#ead9d2", "v"))
    if not hide_exact_simulation:
        ideal = pick_record(
            records,
            {
                "c": "Tcirq-handoff-1002",
                "p": 0.001,
                "f": 3,
            },
        )
        series.insert(1, (ideal, "Exact simulation", "#008080", "#cfe9e7", "^"))

    all_x_values: list[float] = []
    for record, label, color, fill, marker in series:
        x_values, y_values = curve_points(record)
        all_x_values.extend(x_values)
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.5,
            marker=marker,
            markersize=4.6,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markerfacecolor=fill,
            label=label,
        )

    ax.axhline(1e-8, color="#d4af37", linewidth=1.2)
    ax.text(
        min(all_x_values) if auto_focus_x else 1.06,
        1.08e-8,
        "2048-bit factoring",
        color="#d4af37",
        fontsize=7.5,
        va="bottom",
    )

    ax.set_xscale(x_scale)
    ax.set_yscale("log")
    if auto_focus_x:
        ax.set_xlim(*focused_x_limits(all_x_values, x_scale))
    else:
        ax.set_xlim(1, 200)
    ax.set_ylim(1e-10, 1e-2)
    ax.set_xlabel("Expected attempts")
    ax.set_ylabel("Logical error rate")
    if x_scale == "linear":
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(True, which="major", color="#c8c8c8", linewidth=0.8)
    ax.grid(True, which="minor", color="#f1f1f1", linewidth=0.4)

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.6,
            marker="o" if marker == "o" else marker,
            markersize=5,
            markeredgecolor=color,
            markerfacecolor=fill,
        )
        for _, _, color, fill, marker in series
    ]
    labels = [label for _, label, _, _, _ in series]
    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=7.2,
        frameon=False,
        handlelength=2.2,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = load_records(args.csv)
    legacy_hxy = aggregate_shard_dirs(args.legacy_traj_dir)
    movement_hxy = aggregate_shard_dirs(args.movement_traj_dir)
    dephase_only_hxy = (
        aggregate_shard_dirs(args.dephase_only_traj_dir)
        if args.dephase_only_traj_dir
        else None
    )
    plot(
        records=records,
        legacy_hxy=legacy_hxy,
        movement_hxy=movement_hxy,
        dephase_only_hxy=dephase_only_hxy,
        legacy_label=args.legacy_label,
        movement_label=args.movement_label,
        dephase_only_label=args.dephase_only_label,
        output_path=args.output,
        x_scale=args.x_scale,
        auto_focus_x=args.auto_focus_x,
        hide_exact_simulation=args.hide_exact_simulation,
    )
    print(f"Wrote {args.output}")
    if movement_hxy.metadata.get("movement_cycle_us") is not None:
        print(
            "movement_cycle_us=",
            movement_hxy.metadata["movement_cycle_us"],
            "previous_cycle_us=",
            movement_hxy.metadata.get("previous_cycle_us"),
        )


if __name__ == "__main__":
    main()
