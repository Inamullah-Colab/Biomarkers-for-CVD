#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render Table7/8/9 CSV files as PNG images.")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--run-root", required=True, help="Relative run root, e.g. final_task/results/run_xxx")
    ap.add_argument("--tables-subdir", default="tables_7_8_9")
    return ap.parse_args()


def render_dataframe_png(df: pd.DataFrame, title: str, out_png: Path) -> None:
    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"{title}\n\nNo pathways passed the current validation criteria.",
            ha="center",
            va="center",
            fontsize=13,
        )
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
        return

    show_df = df.copy()
    max_rows = min(30, len(show_df))
    show_df = show_df.head(max_rows)

    fig_h = max(3.0, 0.4 * (max_rows + 2))
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=10)

    table = ax.table(
        cellText=show_df.values,
        colLabels=show_df.columns,
        cellLoc="center",
        loc="upper left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.15)

    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    run_root = base / args.run_root
    table_dir = run_root / args.tables_subdir

    specs = [
        ("Table7_classical_validated_pathways_boot2000.csv", "Table 7: Classical Validated Pathways"),
        ("Table8_SIGMA_novel_pathways_boot2000.csv", "Table 8: SIGMA Novel Pathways"),
        ("Table9_classical_novel_pathways_boot2000.csv", "Table 9: Classical Novel Pathways"),
    ]

    for csv_name, title in specs:
        csv_fp = table_dir / csv_name
        if not csv_fp.exists():
            raise FileNotFoundError(f"Missing table CSV: {csv_fp}")
        df = pd.read_csv(csv_fp)
        out_png = csv_fp.with_suffix(".png")
        render_dataframe_png(df, title, out_png)


if __name__ == "__main__":
    main()

