#!/usr/bin/env python3

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    color: str
    fill: str
    marker: str
    filters: dict[str, Any]


SERIES_SPECS = [
    SeriesSpec(
        label="This work",
        color="#de5d83",
        fill="#f8dee6",
        marker="^",
        filters={
            "c": "e2eY-unitstab-g3-3ps1_mid7x3-ue13x0-uniform",
            "p": 0.001,
            "ourbase": 1,
        },
    ),
    SeriesSpec(
        label="This work",
        color="#de5d83",
        fill="#f8dee6",
        marker="p",
        filters={
            "c": "e2e-cls-Y-unitstab-hybrid-fd5-df13",
            "p": 0.001,
            "ourbase": 1,
        },
    ),
    SeriesSpec(
        label="Exact simulation",
        color="#008080",
        fill="#cfe9e7",
        marker="^",
        filters={
            "c": "Tcirq-handoff-1002",
            "p": 0.001,
            "f": 3,
        },
    ),
    SeriesSpec(
        label="GJS24",
        color="#4997d0",
        fill="#daeaf5",
        marker="^",
        filters={
            "c": "GJS24",
            "p": 0.001,
            "f": 3,
        },
    ),
    SeriesSpec(
        label="GJS24",
        color="#4997d0",
        fill="#daeaf5",
        marker="p",
        filters={
            "c": "GJS24",
            "p": 0.001,
            "f": 5,
        },
    ),
    SeriesSpec(
        label="CCLP25",
        color="#9966cc",
        fill="#eae0f4",
        marker="^",
        filters={
            "c": "CCLP25",
            "p": 0.001,
            "f": 3,
        },
    ),
    SeriesSpec(
        label="CCLP25",
        color="#9966cc",
        fill="#eae0f4",
        marker="p",
        filters={
            "c": "CCLP25",
            "p": 0.001,
            "f": 5,
        },
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate MSC_foldedH/data/ler_v2.png from all_paper_data_v2.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=SCRIPT_DIR / "all_paper_data_v2.csv",
        help="Input aggregate CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "ler_v2.recreated.png",
        help="Output PNG path.",
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
        attempts.append(record.shots / kept)
        logical_error_rates.append(failures / kept)
    return attempts, logical_error_rates


def plot(records: list[Record], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.29, 4.08), dpi=100)

    for spec in SERIES_SPECS:
        record = pick_record(records, spec.filters)
        x_values, y_values = curve_points(record)
        ax.plot(
            x_values,
            y_values,
            color=spec.color,
            linewidth=1.5,
            marker=spec.marker,
            markersize=4.6,
            markeredgecolor=spec.color,
            markeredgewidth=0.8,
            markerfacecolor=spec.fill,
        )

    ax.axhline(1e-8, color="#d4af37", linewidth=1.2)
    ax.text(
        1.06,
        1.08e-8,
        "2048-bit factoring",
        color="#d4af37",
        fontsize=7.5,
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 200)
    ax.set_ylim(1e-10, 1e-2)
    ax.set_xlabel("Expected attempts")
    ax.set_ylabel("Logical error rate")
    ax.grid(True, which="major", color="#c8c8c8", linewidth=0.8)
    ax.grid(True, which="minor", color="#f1f1f1", linewidth=0.4)

    legend_order = [
        ("This work", "#de5d83", "#f8dee6"),
        ("Exact simulation", "#008080", "#cfe9e7"),
        ("GJS24", "#4997d0", "#daeaf5"),
        ("CCLP25", "#9966cc", "#eae0f4"),
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=5,
            markeredgecolor=color,
            markerfacecolor=fill,
        )
        for _, color, fill in legend_order
    ]
    labels = [label for label, _, _ in legend_order]
    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=7.3,
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
    plot(records, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
