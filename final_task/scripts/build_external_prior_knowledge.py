#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build reusable prior knowledge from existing SEM standalone outputs (all scenarios)."
    )
    ap.add_argument("--base-dir", required=True)
    ap.add_argument(
        "--source-roots",
        nargs="+",
        default=[
            "final_task/results/our_sem_standalone_2026-02-22",
            "final_task/results/our_sem_standalone_2026-02-22_lowdimd_domain_structured",
            "final_task/results/our_sem_standalone_2026-02-23_higdimd_domain_structured",
            "final_task/other_models_run_2026-02-22/results/our_sem_final_release_v1",
        ],
    )
    ap.add_argument("--out-root", default="final_task/results")
    ap.add_argument("--min-edge-frequency", type=float, default=0.5)
    return ap.parse_args()


def block_of(node: str) -> str:
    n = str(node)
    if n.startswith("G"):
        return "G"
    if n.startswith("Zfix"):
        return "Zfix"
    if n.startswith("Znoise"):
        return "Znoise"
    if n.startswith("Lt"):
        return "Lt"
    if n.startswith("Lm"):
        return "Lm"
    if n.startswith("R"):
        return "R"
    if n.startswith("V"):
        return "V"
    return "Other"


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_metrics_for_scenario(scenario_dir: Path) -> dict:
    out = {
        "status": "",
        "method": "",
        "our_sem_variant": "",
        "backend": "",
        "Adjacency_F1": np.nan,
        "directed_F1": np.nan,
        "threshold_q": np.nan,
    }
    metrics_fp = scenario_dir / "metrics.csv"
    if metrics_fp.exists():
        try:
            df = pd.read_csv(metrics_fp)
            if len(df) > 0:
                row = df.iloc[0]
                for k in out.keys():
                    if k in row.index:
                        out[k] = row[k]
        except Exception:
            pass

    diag_fp = scenario_dir / "run_diagnostics.json"
    if diag_fp.exists():
        try:
            d = json.loads(diag_fp.read_text(encoding="utf-8"))
            if not out["our_sem_variant"]:
                out["our_sem_variant"] = str(d.get("our_sem_variant", ""))
            meta = d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {}
            if not out["backend"]:
                out["backend"] = str(meta.get("backend", ""))
        except Exception:
            pass
    return out


def extract_edges(adj_fp: Path, run_id: str, scenario: str) -> pd.DataFrame:
    adj = pd.read_csv(adj_fp, index_col=0)
    cols = list(adj.columns)
    rows = list(adj.index)
    arr = adj.values
    out = []
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
            out.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "source": str(src),
                    "target": str(dst),
                    "from_block": block_of(src),
                    "to_block": block_of(dst),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    source_roots = [base / p for p in args.source_roots]

    run_rows = []
    edge_rows = []
    missing_roots = []

    for root in source_roots:
        if not root.exists():
            missing_roots.append(str(root))
            continue
        run_id = root.name
        adj_files = sorted(root.rglob("adjacency_pred.csv"))
        for adj_fp in adj_files:
            scenario_dir = adj_fp.parent
            scenario = scenario_dir.name
            m = load_metrics_for_scenario(scenario_dir)
            run_row = {
                "run_id": run_id,
                "source_root": str(root),
                "scenario": scenario,
                "scenario_dir": str(scenario_dir),
                "adjacency_pred_path": str(adj_fp),
                "status": str(m.get("status", "")),
                "method": str(m.get("method", "")),
                "our_sem_variant": str(m.get("our_sem_variant", "")),
                "backend": str(m.get("backend", "")),
                "Adjacency_F1": safe_float(m.get("Adjacency_F1", np.nan)),
                "directed_F1": safe_float(m.get("directed_F1", np.nan)),
                "threshold_q": safe_float(m.get("threshold_q", np.nan)),
            }
            run_rows.append(run_row)
            e = extract_edges(adj_fp, run_id=run_id, scenario=scenario)
            if not e.empty:
                edge_rows.append(e)

    if not run_rows:
        raise RuntimeError("No scenario runs found from source roots.")

    run_df = pd.DataFrame(run_rows)
    ok_runs = run_df[run_df["status"].isin(["ok", "ok_no_truth"])].copy()

    if edge_rows:
        edge_df = pd.concat(edge_rows, ignore_index=True)
    else:
        edge_df = pd.DataFrame(columns=["run_id", "scenario", "source", "target", "from_block", "to_block"])

    # keep only runs marked ok when available
    if not ok_runs.empty and not edge_df.empty:
        keep_pairs = set(zip(ok_runs["run_id"], ok_runs["scenario"]))
        edge_df = edge_df[
            edge_df.apply(lambda r: (str(r["run_id"]), str(r["scenario"])) in keep_pairs, axis=1)
        ].copy()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base / args.out_root / f"external_prior_knowledge_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df.to_csv(out_dir / "scenario_run_index.csv", index=False)

    if edge_df.empty:
        scen_edge = pd.DataFrame(columns=["scenario", "source", "target", "support_count", "run_count", "edge_frequency"])
        scen_block = pd.DataFrame(columns=["scenario", "from_block", "to_block", "support_count", "run_count", "edge_frequency"])
        global_block = pd.DataFrame(columns=["from_block", "to_block", "support_count", "run_count", "edge_frequency"])
    else:
        scenario_counts = run_df.groupby("scenario", as_index=False).size().rename(columns={"size": "run_count"})

        scen_edge = (
            edge_df.groupby(["scenario", "source", "target"], as_index=False)
            .size()
            .rename(columns={"size": "support_count"})
            .merge(scenario_counts, on="scenario", how="left")
        )
        scen_edge["edge_frequency"] = scen_edge["support_count"] / scen_edge["run_count"].clip(lower=1)
        scen_edge = scen_edge.sort_values(["scenario", "edge_frequency", "support_count"], ascending=[True, False, False])

        block_edges = (
            edge_df.groupby(["scenario", "from_block", "to_block"], as_index=False)
            .size()
            .rename(columns={"size": "edge_count_total"})
        )
        block_runs = (
            edge_df[["run_id", "scenario", "from_block", "to_block"]]
            .drop_duplicates()
            .groupby(["scenario", "from_block", "to_block"], as_index=False)
            .size()
            .rename(columns={"size": "run_support"})
        )
        scen_block = (
            block_edges.merge(block_runs, on=["scenario", "from_block", "to_block"], how="left")
            .merge(scenario_counts, on="scenario", how="left")
        )
        scen_block["run_frequency"] = scen_block["run_support"] / scen_block["run_count"].clip(lower=1)
        scen_block["mean_edges_per_support_run"] = scen_block["edge_count_total"] / scen_block["run_support"].clip(lower=1)
        scen_block = scen_block.sort_values(
            ["scenario", "run_frequency", "run_support", "edge_count_total"],
            ascending=[True, False, False, False],
        )

        global_run_count = int(len(run_df))
        global_edges = (
            edge_df.groupby(["from_block", "to_block"], as_index=False)
            .size()
            .rename(columns={"size": "edge_count_total"})
        )
        global_runs = (
            edge_df[["run_id", "scenario", "from_block", "to_block"]]
            .drop_duplicates()
            .groupby(["from_block", "to_block"], as_index=False)
            .size()
            .rename(columns={"size": "run_support"})
        )
        global_block = global_edges.merge(global_runs, on=["from_block", "to_block"], how="left")
        global_block["run_count"] = global_run_count
        global_block["run_frequency"] = global_block["run_support"] / max(1, global_run_count)
        global_block["mean_edges_per_support_run"] = global_block["edge_count_total"] / global_block["run_support"].clip(lower=1)
        global_block = global_block.sort_values(
            ["run_frequency", "run_support", "edge_count_total"],
            ascending=[False, False, False],
        )

    scen_edge.to_csv(out_dir / "scenario_edge_priors.csv", index=False)
    scen_block.to_csv(out_dir / "scenario_block_priors.csv", index=False)
    global_block.to_csv(out_dir / "global_block_priors.csv", index=False)

    # High-confidence priors
    hi = scen_edge[scen_edge["edge_frequency"] >= float(args.min_edge_frequency)].copy()
    hi.to_csv(out_dir / "scenario_edge_priors_high_confidence.csv", index=False)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_roots": [str(p) for p in source_roots],
        "missing_source_roots": missing_roots,
        "total_runs_indexed": int(len(run_df)),
        "ok_runs": int((run_df["status"].isin(["ok", "ok_no_truth"])).sum()),
        "unique_scenarios": sorted(run_df["scenario"].astype(str).unique().tolist()),
        "total_edges_indexed": int(len(edge_df)),
        "min_edge_frequency": float(args.min_edge_frequency),
        "files": {
            "scenario_run_index": str(out_dir / "scenario_run_index.csv"),
            "scenario_edge_priors": str(out_dir / "scenario_edge_priors.csv"),
            "scenario_edge_priors_high_confidence": str(out_dir / "scenario_edge_priors_high_confidence.csv"),
            "scenario_block_priors": str(out_dir / "scenario_block_priors.csv"),
            "global_block_priors": str(out_dir / "global_block_priors.csv"),
        },
    }
    (out_dir / "prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = []
    readme.append("# External Prior Knowledge Bundle")
    readme.append("")
    readme.append("This bundle compiles prior knowledge from existing standalone SEM outputs,")
    readme.append("so external datasets can reuse priors without rerunning all scenario-level checks.")
    readme.append("")
    readme.append(f"- Generated: `{summary['generated_at']}`")
    readme.append(f"- Total runs indexed: `{summary['total_runs_indexed']}`")
    readme.append(f"- OK runs: `{summary['ok_runs']}`")
    readme.append(f"- Unique scenarios: `{len(summary['unique_scenarios'])}`")
    readme.append(f"- Total edges indexed: `{summary['total_edges_indexed']}`")
    readme.append("")
    readme.append("## Files")
    for k, v in summary["files"].items():
        readme.append(f"- {k}: `{v}`")
    if missing_roots:
        readme.append("")
        readme.append("## Missing Source Roots")
        for p in missing_roots:
            readme.append(f"- `{p}`")
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(str(out_dir))


if __name__ == "__main__":
    main()
