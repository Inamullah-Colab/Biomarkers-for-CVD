#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Table 7/8/9 style outputs from bootstrapped mediation results.")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--run-root", required=True, help="Relative run root, e.g. final_task/results/run_xxx")
    ap.add_argument("--dataset-subdir", default="NHANES_with_1000G_reference_proxy")
    ap.add_argument("--validated-thr", type=float, default=0.80)
    ap.add_argument("--novel-low", type=float, default=0.65)
    ap.add_argument("--novel-high", type=float, default=0.80)
    return ap.parse_args()


def sigma_transform(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w = out["NIE_SignConsistency"].clip(lower=0.0, upper=1.0)
    out["UncertaintyWeight"] = w
    for c in ["TE_Estimate", "TE_CI_Lower", "TE_CI_Upper", "NDE_Estimate", "NDE_CI_Lower", "NDE_CI_Upper", "NIE_Estimate", "NIE_CI_Lower", "NIE_CI_Upper"]:
        out[c] = out[c] * w
    return out


def to_table_fmt(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Pathway",
        "TE_Estimate",
        "TE_CI_Lower",
        "TE_CI_Upper",
        "NDE_Estimate",
        "NDE_CI_Lower",
        "NDE_CI_Upper",
        "NIE_Estimate",
        "NIE_CI_Lower",
        "NIE_CI_Upper",
    ]
    if "UncertaintyWeight" in df.columns:
        keep.append("UncertaintyWeight")
    t = df[keep].copy()
    t = t.rename(
        columns={
            "Pathway": "Pathways",
            "TE_Estimate": "TEc Estimate",
            "TE_CI_Lower": "TEc CI Lower",
            "TE_CI_Upper": "TEc CI Upper",
            "NDE_Estimate": "NDE Estimate",
            "NDE_CI_Lower": "NDE CI Lower",
            "NDE_CI_Upper": "NDE CI Upper",
            "NIE_Estimate": "NIE_d Estimate",
            "NIE_CI_Lower": "NIE_d CI Lower",
            "NIE_CI_Upper": "NIE_d CI Upper",
        }
    )
    return t.sort_values("NIE_d Estimate", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def write_bundle(out_dir: Path, name: str, df_table: pd.DataFrame) -> None:
    csv_fp = out_dir / f"{name}.csv"
    md_fp = out_dir / f"{name}_preview.md"
    df_table.to_csv(csv_fp, index=False)
    md = [f"# {name}", "", f"Rows: {len(df_table)}", "", df_table.head(30).to_markdown(index=False)]
    md_fp.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    run_root = base / args.run_root
    ds_dir = run_root / "mediation" / args.dataset_subdir
    in_fp = ds_dir / "mediation_table_all_combos.csv"
    if not in_fp.exists():
        raise FileNotFoundError(f"Missing input table: {in_fp}")

    out_dir = run_root / "tables_7_8_9"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_fp)
    df["NIE_SignConsistency"] = pd.to_numeric(df["NIE_SignConsistency"], errors="coerce").fillna(0.0)
    if "NIE_Significant" in df.columns:
        sig = (
            df["NIE_Significant"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        sig = pd.Series(False, index=df.index)

    # Statistical robustness gate:
    # "validated" and "novel" both require NIE significance (CI excludes zero),
    # with sign-consistency used only as a stability ranking split.
    validated = df[sig & (df["NIE_SignConsistency"] >= float(args.validated_thr))].copy()
    novel = df[
        sig
        & (df["NIE_SignConsistency"] >= float(args.novel_low))
        & (df["NIE_SignConsistency"] < float(args.novel_high))
    ].copy()

    # Table 7: classical, validated pathways
    t7 = to_table_fmt(validated)
    write_bundle(out_dir, "Table7_classical_validated_pathways_boot2000", t7)

    # Table 8: SIGMA-style (uncertainty propagated), novel pathways
    t8_sigma = to_table_fmt(sigma_transform(novel))
    write_bundle(out_dir, "Table8_SIGMA_novel_pathways_boot2000", t8_sigma)

    # Table 9: classical, novel pathways
    t9 = to_table_fmt(novel)
    write_bundle(out_dir, "Table9_classical_novel_pathways_boot2000", t9)

    note = []
    note.append("Table 7/8/9 Generation Note")
    note.append("=" * 50)
    note.append(f"Input table: {in_fp}")
    note.append("Statistical gate: NIE_Significant must be True (95% CI excludes 0)")
    note.append(f"Validated threshold: NIE_SignConsistency >= {args.validated_thr}")
    note.append(f"Novel threshold: {args.novel_low} <= NIE_SignConsistency < {args.novel_high}")
    note.append("")
    note.append("Structural uncertainty propagation used in Table 8:")
    note.append("- UncertaintyWeight = NIE_SignConsistency")
    note.append("- TEc/NDE/NIE_d estimates and CI bounds multiplied by UncertaintyWeight")
    note.append("")
    note.append(f"Table7 rows: {len(t7)}")
    note.append(f"Table8 rows: {len(t8_sigma)}")
    note.append(f"Table9 rows: {len(t9)}")
    (out_dir / "TABLE7_8_9_METHOD_NOTE.txt").write_text("\n".join(note), encoding="utf-8")


if __name__ == "__main__":
    main()
