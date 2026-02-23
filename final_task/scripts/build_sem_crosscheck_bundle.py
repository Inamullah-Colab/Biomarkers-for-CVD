#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSIONS = ["LowDim", "MidDim", "HigDim"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a single cross-check bundle: standalone scenario picks, anti-leak checks, domain summaries, external validation links."
    )
    ap.add_argument("--base-dir", required=True)
    ap.add_argument(
        "--standalone-roots",
        nargs="+",
        default=[
            "final_task/results/our_sem_standalone_2026-02-22",
            "final_task/results/our_sem_standalone_2026-02-22_lowdimd_domain_structured",
            "final_task/other_models_run_2026-02-22/results/our_sem_final_release_v1",
        ],
    )
    ap.add_argument(
        "--external-validation-root",
        default="final_task/results/mediation_realdata_validation_2026-02-22_adaptive",
    )
    ap.add_argument("--out-root", default="final_task/results")
    return ap.parse_args()


def block_of(node: str) -> str:
    if node.startswith("G"):
        return "G"
    if node.startswith("Zfix"):
        return "Zfix"
    if node.startswith("Znoise"):
        return "Znoise"
    if node.startswith("Lt"):
        return "Lt"
    if node.startswith("Lm"):
        return "Lm"
    if node.startswith("R"):
        return "R"
    if node.startswith("V"):
        return "V"
    return "Other"


def read_combined_metrics(csv_fp: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_fp)
    df["combined_metrics_path"] = str(csv_fp)
    df["result_root"] = str(csv_fp.parent)
    return df


def choose_best_per_dimension(all_rows: pd.DataFrame) -> pd.DataFrame:
    ok = all_rows[all_rows["status"].astype(str).isin(["ok", "ok_no_truth"])].copy()
    if ok.empty:
        return ok

    for c in ["Adjacency_F1", "directed_F1", "num_pred_edges", "runtime_sec"]:
        if c not in ok.columns:
            ok[c] = np.nan
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    ok["dimension"] = ok["scenario"].astype(str).str.extract(r"^(LowDim|MidDim|HigDim)", expand=False)
    ok = ok[ok["dimension"].isin(DIMENSIONS)].copy()
    if ok.empty:
        return ok

    ok["score_a"] = ok["Adjacency_F1"].fillna(-1.0)
    ok["score_b"] = ok["directed_F1"].fillna(-1.0)
    ok["score_c"] = ok["num_pred_edges"].fillna(-1.0)
    ok = ok.sort_values(["dimension", "score_a", "score_b", "score_c"], ascending=[True, False, False, False])
    best = ok.groupby("dimension", as_index=False).head(1).copy()
    return best


def scenario_dir_from_row(row: pd.Series) -> Path:
    return Path(row["result_root"]) / str(row["scenario"])


def anti_leak_check(row: pd.Series) -> dict:
    scenario_dir = scenario_dir_from_row(row)
    diag_fp = scenario_dir / "run_diagnostics.json"
    out = {
        "dimension": row.get("dimension", ""),
        "scenario": row.get("scenario", ""),
        "status": row.get("status", ""),
        "anti_leak_flag_in_error": "[ANTI-LEAK]" in str(row.get("error", "")),
        "diagnostics_found": diag_fp.exists(),
        "data_truth_same_path": False,
        "data_root": "",
        "truth_root": "",
    }
    if not diag_fp.exists():
        return out
    try:
        d = json.loads(diag_fp.read_text(encoding="utf-8"))
        dr = str(d.get("data_root", "")).strip()
        tr = str(d.get("truth_root", "")).strip()
        out["data_root"] = dr
        out["truth_root"] = tr
        if dr and tr and (Path(dr).resolve() == Path(tr).resolve()):
            out["data_truth_same_path"] = True
    except Exception:
        pass
    return out


def summarize_domain_edges(row: pd.Series) -> pd.DataFrame:
    scenario_dir = scenario_dir_from_row(row)
    adj_fp = scenario_dir / "adjacency_pred.csv"
    if not adj_fp.exists():
        return pd.DataFrame(columns=["dimension", "scenario", "from_block", "to_block", "edge_count"])

    adj = pd.read_csv(adj_fp, index_col=0)
    edges = []
    arr = adj.values
    rows = list(adj.index)
    cols = list(adj.columns)
    for i, src in enumerate(rows):
        for j, dst in enumerate(cols):
            if i == j:
                continue
            try:
                v = int(arr[i, j])
            except Exception:
                continue
            if v == 0:
                continue
            edges.append((block_of(str(src)), block_of(str(dst))))

    if not edges:
        return pd.DataFrame(columns=["dimension", "scenario", "from_block", "to_block", "edge_count"])

    tmp = pd.DataFrame(edges, columns=["from_block", "to_block"])
    out = tmp.groupby(["from_block", "to_block"], as_index=False).size().rename(columns={"size": "edge_count"})
    out["dimension"] = row.get("dimension", "")
    out["scenario"] = row.get("scenario", "")
    out = out[["dimension", "scenario", "from_block", "to_block", "edge_count"]]
    out = out.sort_values("edge_count", ascending=False)
    return out


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    frames = []
    missing_roots = []
    for rel in args.standalone_roots:
        root = base / rel
        fp = root / "combined_metrics.csv"
        if fp.exists():
            frames.append(read_combined_metrics(fp))
        else:
            missing_roots.append(str(root))

    if not frames:
        raise RuntimeError("No combined_metrics.csv found in standalone roots.")

    all_rows = pd.concat(frames, ignore_index=True)
    selected = choose_best_per_dimension(all_rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / args.out_root / f"sem_crosscheck_bundle_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows.to_csv(out_dir / "all_standalone_metrics_merged.csv", index=False)
    selected.to_csv(out_dir / "selected_scenarios_by_dimension.csv", index=False)

    anti = []
    edge_summaries = []
    for _, row in selected.iterrows():
        anti.append(anti_leak_check(row))
        edge_summaries.append(summarize_domain_edges(row))

    anti_df = pd.DataFrame(anti)
    anti_df.to_csv(out_dir / "anti_leak_crosscheck.csv", index=False)

    if edge_summaries:
        edge_df = pd.concat(edge_summaries, ignore_index=True)
    else:
        edge_df = pd.DataFrame(columns=["dimension", "scenario", "from_block", "to_block", "edge_count"])
    edge_df.to_csv(out_dir / "domain_edge_pattern_summary.csv", index=False)

    ext_root = base / args.external_validation_root
    ext_summary_fp = ext_root / "summary_all_datasets.csv"
    if ext_summary_fp.exists():
        ext_summary = pd.read_csv(ext_summary_fp)
        ext_summary.to_csv(out_dir / "external_validation_summary.csv", index=False)
    else:
        ext_summary = pd.DataFrame()

    lines = []
    lines.append("# SEM Crosscheck Bundle")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("## Selected Standalone Scenarios (Best per Dimension)")
    if selected.empty:
        lines.append("- No completed scenario found (`status in {ok, ok_no_truth}`) for LowDim/MidDim/HigDim.")
    else:
        for _, r in selected.iterrows():
            lines.append(
                f"- {r['dimension']}: `{r['scenario']}` from `{r['result_root']}` "
                f"(Adjacency_F1={r.get('Adjacency_F1', np.nan)}, directed_F1={r.get('directed_F1', np.nan)})"
            )
    lines.append("")
    lines.append("## Anti-Leak Crosscheck")
    if anti_df.empty:
        lines.append("- No anti-leak diagnostics available.")
    else:
        any_flag = bool(anti_df["anti_leak_flag_in_error"].any()) or bool(anti_df["data_truth_same_path"].any())
        lines.append(f"- Any anti-leak risk flag: `{any_flag}`")
        for _, r in anti_df.iterrows():
            lines.append(
                f"- {r['dimension']} / {r['scenario']}: anti_leak_error={r['anti_leak_flag_in_error']}, "
                f"data_truth_same_path={r['data_truth_same_path']}"
            )
    lines.append("")
    lines.append("## Domain Knowledge Extraction")
    if edge_df.empty:
        lines.append("- No adjacency files found for selected scenarios.")
    else:
        for d in DIMENSIONS:
            sub = edge_df[edge_df["dimension"] == d].head(5)
            if sub.empty:
                continue
            lines.append(f"- {d} top block patterns:")
            for _, r in sub.iterrows():
                lines.append(f"  - {r['from_block']} -> {r['to_block']}: {int(r['edge_count'])} edges")
    lines.append("")
    lines.append("## External Validation Links")
    lines.append(f"- External validation root: `{ext_root}`")
    if not ext_summary.empty:
        lines.append(f"- Summary file: `{ext_summary_fp}`")
        for ds in ext_summary["dataset"].astype(str).tolist():
            ds_dir = ext_root / ds
            lines.append(f"- {ds}:")
            lines.append(f"  - `{ds_dir / 'mediation_table_all_combos.csv'}`")
            lines.append(f"  - `{ds_dir / 'sem_paper_forest_te_nde_nie.png'}`")
            lines.append(f"  - `{ds_dir / 'sem_paper_summary_te_nde_nie.png'}`")
    else:
        lines.append("- External validation summary not found.")
    lines.append("")
    lines.append("## Missing Roots")
    if missing_roots:
        for m in missing_roots:
            lines.append(f"- `{m}`")
    else:
        lines.append("- None")

    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(str(out_dir))


if __name__ == "__main__":
    main()

