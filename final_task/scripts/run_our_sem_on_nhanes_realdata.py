#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run OUR_SEM_MODEL on NHANES real datasets with domain mapping.")
    ap.add_argument("--base-dir", required=True, help="NAHES_Dataset directory containing input CSVs.")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "NHANES_cvd_extended_with_macular_full_matched.csv",
            "NHANES_with_1000G_reference_proxy.csv",
        ],
    )
    ap.add_argument("--out-dir", default="final_task/results/our_sem_realdata_validation_2026-02-22")
    ap.add_argument("--threshold-q", type=float, default=0.65)
    ap.add_argument("--imputer", default="median", choices=["auto", "none", "mean", "median", "most_frequent", "knn", "iterative"])
    ap.add_argument("--variant", default="domain_latent", choices=["base", "truth_aligned", "domain_structured", "domain_latent"])
    ap.add_argument("--transform", default="log1p_signed", choices=["none", "log1p_signed"])
    ap.add_argument("--allow-fallback", action="store_true", help="Fallback to domain_structured if domain_latent fails.")
    return ap.parse_args()


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def get_domain_sets(cols: list[str]) -> dict[str, list[str]]:
    # Genetic proxy block: keep numeric proxy composition only.
    g_candidates = ["GREF_AFR", "GREF_AMR", "GREF_EAS", "GREF_EUR", "GREF_SAS", "GREF_entropy"]
    g = [c for c in g_candidates if c in cols]

    z_candidates = ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDFMPIR", "BMXBMI", "WTMEC2YR", "WTSAF2YR"]
    z = [c for c in z_candidates if c in cols]
    # Prevent proxy duplication: GREF_* is derived from RIDRETH1 mapping.
    if g and "RIDRETH1" in z:
        z.remove("RIDRETH1")

    lt_candidates = ["LBXTC", "LBDHDD", "LBDLDL", "LBXTR", "LBXAPB", "LBXGLU", "LBXGH", "LBXCRP", "LBXSCR", "URXUMA", "URXUCR", "LBXIN"]
    lt = [c for c in lt_candidates if c in cols]

    lm_candidates = ["BMXWAIST", "BPXPLS", "MCQ160C", "MCQ160E"]
    lm = [c for c in lm_candidates if c in cols]

    r_candidates = [
        "AVR_Knudtson",
        "CRAE_Knudtson",
        "CRVE_Knudtson",
        "Vessel_density",
        "Distance_tortuosity",
        "Squared_curvature_tortuosity",
        "Artery_Distance_tortuosity",
        "Artery_Squared_curvature_tortuosity",
        "Vein_Distance_tortuosity",
        "Vein_Squared_curvature_tortuosity",
    ]
    r = [c for c in r_candidates if c in cols]

    v_candidates = ["BPXSY1", "BPXDI1", "BPXPLS"]
    v = [c for c in v_candidates if c in cols]
    return {"G": g, "Z": z, "Lt": lt, "Lm": lm, "R": r, "V": v}


def build_model_df(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    keep_order = groups["G"] + groups["Z"] + groups["Lt"] + groups["Lm"] + groups["R"] + groups["V"]
    keep_order = [c for c in keep_order if c in df.columns]
    # Preserve order but avoid duplicate selections that can return DataFrame on sub[col].
    keep_order = list(dict.fromkeys(keep_order))
    sub = df[keep_order].copy()

    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    rename = {}
    for c in groups["G"]:
        if c in sub.columns:
            rename[c] = f"G_{c}"
    for c in groups["Z"]:
        if c in sub.columns:
            rename[c] = f"Zfix_{c}"
    for c in groups["Lt"]:
        if c in sub.columns:
            rename[c] = f"Lt_{c}"
    for c in groups["Lm"]:
        if c in sub.columns:
            rename[c] = f"Lm_{c}"
    for c in groups["R"]:
        if c in sub.columns:
            rename[c] = f"R_{c}"
    for c in groups["V"]:
        if c in sub.columns:
            rename[c] = f"V_{c}"

    sub = sub.rename(columns=rename)
    sub = sub.dropna(axis=1, how="all")
    nunique = sub.nunique(dropna=True)
    sub = sub.loc[:, nunique > 1]
    return sub


def adjacency_to_edges(W_hat: np.ndarray, cols: list[str], A_pred: np.ndarray) -> pd.DataFrame:
    out = []
    n = len(cols)
    for i in range(n):
        for j in range(n):
            if i == j or A_pred[i, j] == 0:
                continue
            w = _safe_float(W_hat[i, j])
            out.append(
                {
                    "source": cols[i],
                    "target": cols[j],
                    "weight": w,
                    "abs_weight": abs(w),
                }
            )
    if not out:
        return pd.DataFrame(columns=["source", "target", "weight", "abs_weight"])
    return pd.DataFrame(out).sort_values("abs_weight", ascending=False).reset_index(drop=True)


def find_paths(A_pred: np.ndarray, cols: list[str], max_len: int = 4) -> pd.DataFrame:
    graph = {c: [] for c in cols}
    n = len(cols)
    for i in range(n):
        for j in range(n):
            if i != j and A_pred[i, j] == 1:
                graph[cols[i]].append(cols[j])

    sources = [c for c in cols if c.startswith(("Lt_", "Lm_", "R_", "G_"))]
    targets = {c for c in cols if c.startswith("V_")}

    rows = []
    for s in sources:
        stack = [(s, [s])]
        while stack:
            cur, path = stack.pop()
            if len(path) - 1 >= max_len:
                continue
            for nxt in graph.get(cur, []):
                if nxt in path:
                    continue
                npth = path + [nxt]
                if nxt in targets and len(npth) > 1:
                    rows.append({"source": s, "target": nxt, "path_len": len(npth) - 1, "path": " -> ".join(npth)})
                stack.append((nxt, npth))

    if not rows:
        return pd.DataFrame(columns=["source", "target", "path_len", "path"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(["path_len", "path"]).reset_index(drop=True)


def run_one(
    dataset_name: str,
    dataset_fp: Path,
    out_root: Path,
    imputer: str,
    variant: str,
    transform: str,
    threshold_q: float,
    allow_fallback: bool,
):
    here = Path(__file__).resolve()
    bench_root = Path(r"C:\Users\i1n23\OneDrive - University of Southampton\Documents\codex_folder\Revised_models")
    if str(bench_root) not in sys.path:
        sys.path.insert(0, str(bench_root))

    from benchmark_missing_sem_model.run_missing_benchmark_sem_model import (  # noqa: WPS433
        binarize_from_weights,
        build_forbid_mask,
        choose_threshold,
        impute_data,
        run_our_sem_model,
    )

    df = pd.read_csv(dataset_fp)
    groups = get_domain_sets(list(df.columns))
    df_model = build_model_df(df, groups)

    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_before = float(df_model.isna().mean().mean()) if df_model.shape[1] > 0 else float("nan")
    df_imp, imp_meta = impute_data(df_model, strategy=imputer, random_state=123)
    missing_after = float(df_imp.isna().mean().mean()) if df_imp.shape[1] > 0 else float("nan")

    if df_imp.shape[1] < 8:
        summary = {
            "dataset": dataset_name,
            "status": "error",
            "error": f"Too few mapped columns after preprocessing: {df_imp.shape[1]}",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    t0 = time.time()
    W_hat, err, meta = run_our_sem_model(
        df_imp,
        alpha=0.01,
        use_domain_priors=True,
        variant=variant,
        x_transform=transform,
    )
    runtime = time.time() - t0
    used_variant = variant

    if W_hat is None and allow_fallback and variant == "domain_latent":
        W_hat, err, meta = run_our_sem_model(
            df_imp,
            alpha=0.01,
            use_domain_priors=True,
            variant="domain_structured",
            x_transform=transform,
        )
        used_variant = "domain_structured"

    if W_hat is None:
        summary = {
            "dataset": dataset_name,
            "status": "error",
            "error": str(err),
            "runtime_sec": runtime,
            "n_rows": int(df_imp.shape[0]),
            "n_cols_model": int(df_imp.shape[1]),
            "missing_before": missing_before,
            "missing_after": missing_after,
            "imputation": imp_meta.get("imputation"),
            "imputer_backend": imp_meta.get("imputer_backend"),
            "group_counts": {k: int(len(v)) for k, v in groups.items()},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    W_hat = np.asarray(W_hat, dtype=float)
    th = choose_threshold(W_hat, strategy="quantile", q=threshold_q)
    A_pred = binarize_from_weights(W_hat, th)
    A_pred[build_forbid_mask(list(df_imp.columns))] = 0

    pd.DataFrame(W_hat, index=df_imp.columns, columns=df_imp.columns).to_csv(out_dir / "adjacency_weights.csv")
    pd.DataFrame(A_pred.astype(int), index=df_imp.columns, columns=df_imp.columns).to_csv(out_dir / "adjacency_pred.csv")

    edge_df = adjacency_to_edges(W_hat, list(df_imp.columns), A_pred)
    edge_df.to_csv(out_dir / "edge_list.csv", index=False)

    path_df = find_paths(A_pred, list(df_imp.columns), max_len=4)
    path_df.to_csv(out_dir / "pathways.csv", index=False)

    summary = {
        "dataset": dataset_name,
        "status": "ok",
        "runtime_sec": runtime,
        "n_rows": int(df_imp.shape[0]),
        "n_cols_model": int(df_imp.shape[1]),
        "n_edges": int(A_pred.sum()),
        "threshold_q": float(threshold_q),
        "threshold_value": float(th),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "imputation": imp_meta.get("imputation"),
        "imputer_backend": imp_meta.get("imputer_backend"),
        "variant_requested": variant,
        "variant_used": used_variant,
        "transform": transform,
        "group_counts": {k: int(len(v)) for k, v in groups.items()},
    }
    if isinstance(meta, dict):
        for k, v in meta.items():
            if k not in summary:
                summary[k] = v

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(report_fp: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    lines = []
    lines.append("# OUR_SEM Real-Data Validation Report (2026-02-22)")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Inputs: {', '.join(args.inputs)}")
    lines.append(f"- Imputer: {args.imputer}")
    lines.append(f"- Variant: {args.variant}")
    lines.append(f"- Transform: {args.transform}")
    lines.append(f"- Threshold q: {args.threshold_q}")
    lines.append("")
    lines.append("## Per-dataset Results")
    for s in summaries:
        lines.append(f"### {s.get('dataset', 'unknown')}")
        lines.append(f"- Status: {s.get('status')}")
        if s.get("status") == "ok":
            lines.append(f"- Rows x model cols: {s.get('n_rows')} x {s.get('n_cols_model')}")
            lines.append(f"- Predicted edges: {s.get('n_edges')}")
            lines.append(f"- Threshold value: {s.get('threshold_value')}")
            lines.append(f"- Runtime sec: {s.get('runtime_sec')}")
            lines.append(f"- Variant used: {s.get('variant_used')}")
            lines.append(f"- Missing before/after impute: {s.get('missing_before')} / {s.get('missing_after')}")
            lines.append(f"- Group counts: {s.get('group_counts')}")
        else:
            lines.append(f"- Error: {s.get('error')}")
        lines.append("")

    report_fp.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    out_root = base_dir / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for in_name in args.inputs:
        fp = base_dir / in_name
        if not fp.exists():
            summaries.append({"dataset": in_name, "status": "error", "error": f"Missing input file: {fp}"})
            continue
        ds_name = Path(in_name).stem
        summary = run_one(
            dataset_name=ds_name,
            dataset_fp=fp,
            out_root=out_root,
            imputer=args.imputer,
            variant=args.variant,
            transform=args.transform,
            threshold_q=args.threshold_q,
            allow_fallback=bool(args.allow_fallback),
        )
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(out_root / "summary_all_datasets.csv", index=False)
    (out_root / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    write_report(base_dir / "final_task" / "reports" / "OUR_SEM_REALDATA_VALIDATION_2026-02-22.md", summaries, args)


if __name__ == "__main__":
    main()
