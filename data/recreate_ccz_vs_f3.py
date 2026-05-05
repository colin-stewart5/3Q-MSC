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
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


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
        description="Recreate the CCZ-vs-f3 comparison plot from the paper CSV plus one or more trajectory-run shard directories."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "all_paper_data_v2.csv",
        help="Input aggregate CSV file containing the baseline rows.",
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory containing trajectory shard files emitted by handoff_trajectory_run.py. Repeat to combine multiple shard directories into one plotted trajectory series.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "ccz_vs_f3.recreated.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--traj-label",
        default="|H_XY> (Trajectory CCZ)",
        help="Legend label for the aggregated trajectory series used with --traj-dir.",
    )
    parser.add_argument(
        "--traj-series",
        action="append",
        default=None,
        help="Explicit plotted trajectory series in the form 'LABEL=dir1[,dir2,...]'. Repeat to plot multiple aggregated trajectory curves.",
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
        "--x-min",
        type=float,
        default=None,
        help="Optional explicit x-axis minimum.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional explicit x-axis maximum.",
    )
    parser.add_argument(
        "--zoom-inset",
        action="store_true",
        help="Add a zoomed inset to highlight small separations between nearby curves.",
    )
    parser.add_argument(
        "--zoom-x-min",
        type=float,
        default=1.56,
        help="Inset x-axis minimum when --zoom-inset is used.",
    )
    parser.add_argument(
        "--zoom-x-max",
        type=float,
        default=1.62,
        help="Inset x-axis maximum when --zoom-inset is used.",
    )
    parser.add_argument(
        "--zoom-y-min",
        type=float,
        default=3.7e-3,
        help="Inset y-axis minimum when --zoom-inset is used.",
    )
    parser.add_argument(
        "--zoom-y-max",
        type=float,
        default=4.3e-3,
        help="Inset y-axis maximum when --zoom-inset is used.",
    )
    parser.add_argument(
        "--hide-ideal",
        action="store_true",
        help="Omit the historical ideal-CCZ curve from the plot and legend.",
    )
    parser.add_argument(
        "--hide-ref9",
        action="store_true",
        help="Omit the Ref. [9] curve from the plot and legend.",
    )
    parser.add_argument(
        "--hide-ref10",
        action="store_true",
        help="Omit the Ref. [10] curve from the plot and legend.",
    )
    return parser.parse_args()


def parse_trajectory_series_specs(
    traj_series_specs: list[str] | None,
    traj_dirs: list[Path] | None,
    traj_label: str,
) -> list[tuple[str, list[Path]]]:
    if traj_series_specs:
        parsed: list[tuple[str, list[Path]]] = []
        for spec in traj_series_specs:
            if "=" not in spec:
                raise ValueError(
                    f"Invalid --traj-series value {spec!r}. Expected 'LABEL=dir1[,dir2,...]'."
                )
            label, raw_dirs = spec.split("=", 1)
            dirs = [Path(part) for part in raw_dirs.split(",") if part]
            if not label or not dirs:
                raise ValueError(
                    f"Invalid --traj-series value {spec!r}. Expected 'LABEL=dir1[,dir2,...]'."
                )
            parsed.append((label, dirs))
        return parsed

    if traj_dirs:
        return [(traj_label, traj_dirs)]

    raise ValueError("Provide either --traj-series or at least one --traj-dir.")


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
    labels = [record.metadata.get("c", record.strong_id) for record in aggregates]
    metadata["c"] = " + ".join(labels)
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
        # Once the finite-shot aggregate has zero observed failures, the
        # resulting "LER=0" tail is not a meaningful plotted point.
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

    # Multiplicative padding preserves spacing on a log axis.
    return xmin / 1.08, xmax * 1.08


def plot(
    records: list[Record],
    trajectory_series: list[tuple[Record, str]],
    output_path: Path,
    x_scale: str,
    auto_focus_x: bool,
    x_min: float | None,
    x_max: float | None,
    zoom_inset: bool,
    zoom_x_min: float,
    zoom_x_max: float,
    zoom_y_min: float,
    zoom_y_max: float,
    hide_ideal: bool,
    hide_ref9: bool,
    hide_ref10: bool,
) -> None:
    this_work = pick_record(
        records,
        {
            "c": "e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform",
            "p": 0.001,
            "ourbase": 1,
        },
    )
    fig, ax = plt.subplots(figsize=(5.29, 4.08), dpi=100, constrained_layout=True)
    all_x_values: list[float] = []

    series = [
        (this_work, "|Y>", "#e85d7f", "o"),
    ]
    trajectory_styles = [
        ("#2ebd85", "s"),
        ("#ff8c42", "D"),
        ("#00a6a6", "^"),
        ("#7a6ff0", "v"),
    ]
    for i, (record, label) in enumerate(trajectory_series):
        color, marker = trajectory_styles[i % len(trajectory_styles)]
        series.append((record, label, color, marker))
    if not hide_ideal:
        ideal = pick_record(
            records,
            {
                "c": "Tcirq-handoff-1002",
                "p": 0.001,
                "f": 3,
            },
        )
        series.insert(1, (ideal, "|H_XY> (ideal CCZ)", "#00a6a6", "^"))
    if not hide_ref9:
        gjs24 = pick_record(
            records,
            {
                "c": "GJS24",
                "p": 0.001,
                "f": 3,
            },
        )
        series.append((gjs24, "Ref. [9]", "#68a8ea", "o"))
    if not hide_ref10:
        cclp25 = pick_record(
            records,
            {
                "c": "CCLP25",
                "p": 0.001,
                "f": 3,
            },
        )
        series.append((cclp25, "Ref. [10]", "#b76be3", "o"))

    inset = None
    if zoom_inset:
        inset = inset_axes(ax, width="42%", height="42%", loc="lower left", borderpad=1.2)

    for record, label, color, marker in series:
        x_values, y_values = curve_points(record)
        all_x_values.extend(x_values)
        ax.plot(
            x_values,
            y_values,
            color=color,
            linewidth=1.5,
            marker=marker,
            markersize=4.2,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markerfacecolor="white",
            label=label,
        )
        if inset is not None:
            inset.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.2,
                marker=marker,
                markersize=3.2,
                markeredgecolor=color,
                markeredgewidth=0.7,
                markerfacecolor="white",
            )

    ax.axhline(1e-8, color="#d4af37", linewidth=1.2)
    ax.text(
        min(all_x_values) if auto_focus_x else 1.05,
        1.08e-8,
        "2048-bit factoring",
        color="#d4af37",
        fontsize=9,
        va="bottom",
    )

    ax.set_xscale(x_scale)
    ax.set_yscale("log")
    if x_min is not None or x_max is not None:
        left = x_min if x_min is not None else min(all_x_values)
        right = x_max if x_max is not None else max(all_x_values)
        ax.set_xlim(left, right)
    elif auto_focus_x:
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
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)
    if inset is not None:
        inset.set_xlim(zoom_x_min, zoom_x_max)
        inset.set_ylim(zoom_y_min, zoom_y_max)
        inset.set_xscale("linear")
        inset.set_yscale("linear")
        inset.grid(True, which="major", color="#dddddd", linewidth=0.6)
        inset.tick_params(axis="both", labelsize=7)
        mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#666666", lw=0.8)
    fig.savefig(output_path)
    print(f"Wrote {output_path}")


def main() -> None:
    args = parse_args()
    records = load_records(args.csv)
    trajectory_series = [
        (aggregate_shard_dirs(series_dirs), label)
        for label, series_dirs in parse_trajectory_series_specs(
            args.traj_series,
            args.traj_dir,
            args.traj_label,
        )
    ]
    plot(
        records,
        trajectory_series,
        args.output,
        x_scale=args.x_scale,
        auto_focus_x=args.auto_focus_x,
        x_min=args.x_min,
        x_max=args.x_max,
        zoom_inset=args.zoom_inset,
        zoom_x_min=args.zoom_x_min,
        zoom_x_max=args.zoom_x_max,
        zoom_y_min=args.zoom_y_min,
        zoom_y_max=args.zoom_y_max,
        hide_ideal=args.hide_ideal,
        hide_ref9=args.hide_ref9,
        hide_ref10=args.hide_ref10,
    )


if __name__ == "__main__":
    main()
