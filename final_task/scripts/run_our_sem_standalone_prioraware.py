#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _import_core_functions(model_root: Path):
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    try:
        from run_missing_benchmark_sem_model import (  # type: ignore
            binarize_from_weights,
            choose_threshold,
            impute_data,
            run_our_sem_model,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Cannot import core OUR_SEM functions from: {model_root}. "
            f"Set --model-root correctly."
        ) from exc
    def _fallback_build_forbid_mask(cols: list[str]) -> np.ndarray:
        n = len(cols)
        idx = {c: i for i, c in enumerate(cols)}
        M = np.zeros((n, n), dtype=bool)
        for s in cols:
            for t in cols:
                if s == t:
                    continue
                bs = block_of(s)
                bt = block_of(t)
                forbid = False
                # Conservative domain constraints
                if bs == "G" and bt == "V":
                    forbid = True
                if bs in ("Lt", "Lm") and bt == "V":
                    forbid = True
                if bs in ("Zfix", "Znoise") and bt == "G":
                    forbid = True
                if forbid:
                    M[idx[s], idx[t]] = True
        return M

    try:
        from run_missing_benchmark_sem_model import build_forbid_mask as _bfm  # type: ignore
        build_forbid_mask = _bfm
    except Exception:
        build_forbid_mask = _fallback_build_forbid_mask

    return binarize_from_weights, build_forbid_mask, choose_threshold, impute_data, run_our_sem_model


def load_prior_bundle(prior_dir: Path):
    edge_fp = prior_dir / "scenario_edge_priors_high_confidence.csv"
    block_fp = prior_dir / "global_block_priors.csv"
    edge_df = pd.DataFrame(columns=["source", "target", "edge_frequency"])
    block_df = pd.DataFrame(columns=["from_block", "to_block", "run_frequency"])
    if edge_fp.exists():
        edge_df = pd.read_csv(edge_fp)
    if block_fp.exists():
        block_df = pd.read_csv(block_fp)
    return edge_df, block_df


def apply_priors_to_weights(
    W_hat: np.ndarray,
    cols: list[str],
    edge_priors: pd.DataFrame,
    block_priors: pd.DataFrame,
    edge_min_freq: float,
    block_min_freq: float,
    edge_strength: float,
    block_strength: float,
) -> tuple[np.ndarray, dict]:
    W = np.asarray(W_hat, dtype=float).copy()
    n = len(cols)
    if n == 0:
        return W, {"edge_boosts": 0, "block_boosts": 0}

    base = np.abs(W[np.nonzero(W)])
    base_scale = float(np.nanmedian(base)) if base.size else 1.0
    if not np.isfinite(base_scale) or base_scale <= 0:
        base_scale = 1.0

    idx = {c: i for i, c in enumerate(cols)}
    edge_boosts = 0
    block_boosts = 0

    if not edge_priors.empty:
        e = edge_priors.copy()
        if "edge_frequency" not in e.columns:
            e["edge_frequency"] = 1.0
        e = e[pd.to_numeric(e["edge_frequency"], errors="coerce").fillna(0.0) >= float(edge_min_freq)]
        for _, r in e.iterrows():
            s = str(r.get("source", ""))
            t = str(r.get("target", ""))
            if s in idx and t in idx and s != t:
                freq = _safe_float(r.get("edge_frequency", 1.0), 1.0)
                W[idx[s], idx[t]] += float(edge_strength) * float(freq) * base_scale
                edge_boosts += 1

    if not block_priors.empty:
        b = block_priors.copy()
        freq_col = "run_frequency" if "run_frequency" in b.columns else "edge_frequency"
        if freq_col not in b.columns:
            b[freq_col] = 0.0
        b = b[pd.to_numeric(b[freq_col], errors="coerce").fillna(0.0) >= float(block_min_freq)]
        block_map = {(str(r["from_block"]), str(r["to_block"])): _safe_float(r[freq_col], 0.0) for _, r in b.iterrows()}
        for i in range(n):
            bi = block_of(cols[i])
            for j in range(n):
                if i == j:
                    continue
                bj = block_of(cols[j])
                f = block_map.get((bi, bj), 0.0)
                if f > 0:
                    W[i, j] += float(block_strength) * float(f) * base_scale
                    block_boosts += 1

    return W, {"edge_boosts": int(edge_boosts), "block_boosts": int(block_boosts), "base_scale": float(base_scale)}


def find_scenario_csv(data_root: Path, scenario: str) -> Path:
    p1 = data_root / scenario / f"{scenario}_data.csv"
    if p1.exists():
        return p1
    p2 = data_root / f"{scenario}.csv"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Cannot find data for scenario={scenario} under {data_root}")


def edge_list_from_adjacency(W: np.ndarray, A: np.ndarray, cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(cols)
    for i in range(n):
        for j in range(n):
            if i == j or int(A[i, j]) == 0:
                continue
            rows.append(
                {
                    "source": cols[i],
                    "target": cols[j],
                    "weight": float(W[i, j]),
                    "abs_weight": abs(float(W[i, j])),
                    "from_block": block_of(cols[i]),
                    "to_block": block_of(cols[j]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight", "abs_weight", "from_block", "to_block"])
    return pd.DataFrame(rows).sort_values("abs_weight", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prior-aware standalone OUR_SEM runner (importable + CLI)."
    )
    ap.add_argument("--data-root", required=True, help="Root containing scenario data.")
    ap.add_argument("--scenario", required=True, help="Scenario name (folder name or csv stem).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ap.add_argument(
        "--model-root",
        default=r"C:\Users\i1n23\OneDrive - University of Southampton\Documents\codex_folder\Revised_models\final_sem_release_v1\scripts",
        help="Folder containing run_missing_benchmark_sem_model.py",
    )
    ap.add_argument("--impute", default="median", choices=["auto", "none", "mean", "median", "most_frequent", "knn", "iterative"])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--threshold-strategy", default="quantile", choices=["quantile", "mean", "median"])
    ap.add_argument("--threshold-q", type=float, default=0.35)
    ap.add_argument("--our-sem-variant", default="domain_latent", choices=["base", "truth_aligned", "domain_structured", "domain_latent"])
    ap.add_argument("--our-sem-transform", default="log1p_signed", choices=["none", "log1p_signed"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--prior-bundle-dir", default=None, help="Path to external_prior_knowledge_* bundle.")
    ap.add_argument("--prior-edge-min-freq", type=float, default=0.5)
    ap.add_argument("--prior-block-min-freq", type=float, default=0.6)
    ap.add_argument("--prior-edge-strength", type=float, default=0.35)
    ap.add_argument("--prior-block-strength", type=float, default=0.10)
    return ap.parse_args()


def run_prior_aware_our_sem(args: argparse.Namespace) -> dict:
    (
        binarize_from_weights,
        build_forbid_mask,
        choose_threshold,
        impute_data,
        run_our_sem_model,
    ) = _import_core_functions(Path(args.model_root))

    data_fp = find_scenario_csv(Path(args.data_root), str(args.scenario))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_fp)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    nunique = df.nunique(dropna=True)
    df = df.loc[:, nunique > 1]

    miss_before = float(df.isna().mean().mean()) if df.size else float("nan")
    df_imp, imp_meta = impute_data(df, strategy=str(args.impute), random_state=int(args.seed))
    miss_after = float(df_imp.isna().mean().mean()) if df_imp.size else float("nan")

    t0 = time.time()
    try:
        W_hat, err, meta = run_our_sem_model(
            df_imp,
            alpha=float(args.alpha),
            use_domain_priors=True,
            variant=str(args.our_sem_variant),
            x_transform=str(args.our_sem_transform),
        )
    except TypeError:
        W_hat, err, meta = run_our_sem_model(
            df_imp,
            alpha=float(args.alpha),
            variant=str(args.our_sem_variant),
            x_transform=str(args.our_sem_transform),
        )
    runtime = time.time() - t0

    if W_hat is None:
        out = {
            "scenario": str(args.scenario),
            "status": "error",
            "error": str(err),
            "runtime_sec": float(runtime),
            "missing_before": miss_before,
            "missing_after": miss_after,
            "imputation": imp_meta.get("imputation"),
            "imputer_backend": imp_meta.get("imputer_backend"),
        }
        pd.DataFrame([out]).to_csv(out_dir / "metrics.csv", index=False)
        return out

    W_hat = np.asarray(W_hat, dtype=float)
    prior_meta = {"prior_bundle_used": False}

    if args.prior_bundle_dir:
        prior_dir = Path(args.prior_bundle_dir)
        edge_priors, block_priors = load_prior_bundle(prior_dir)
        W_hat, prior_usage = apply_priors_to_weights(
            W_hat=W_hat,
            cols=list(df_imp.columns),
            edge_priors=edge_priors,
            block_priors=block_priors,
            edge_min_freq=float(args.prior_edge_min_freq),
            block_min_freq=float(args.prior_block_min_freq),
            edge_strength=float(args.prior_edge_strength),
            block_strength=float(args.prior_block_strength),
        )
        prior_meta = {
            "prior_bundle_used": True,
            "prior_bundle_dir": str(prior_dir),
            "prior_edge_min_freq": float(args.prior_edge_min_freq),
            "prior_block_min_freq": float(args.prior_block_min_freq),
            "prior_edge_strength": float(args.prior_edge_strength),
            "prior_block_strength": float(args.prior_block_strength),
            **prior_usage,
        }

    th = choose_threshold(W_hat, strategy=str(args.threshold_strategy), q=float(args.threshold_q))
    A_pred = binarize_from_weights(W_hat, th)
    A_pred[build_forbid_mask(list(df_imp.columns))] = 0

    pd.DataFrame(W_hat, index=df_imp.columns, columns=df_imp.columns).to_csv(out_dir / "weights_hat.csv")
    pd.DataFrame(A_pred.astype(int), index=df_imp.columns, columns=df_imp.columns).to_csv(out_dir / "adjacency_pred.csv")
    edge_list_from_adjacency(W_hat, A_pred, list(df_imp.columns)).to_csv(out_dir / "edge_list.csv", index=False)

    row = {
        "scenario": str(args.scenario),
        "status": "ok_no_truth",
        "error": "none",
        "runtime_sec": float(runtime),
        "threshold_strategy": str(args.threshold_strategy),
        "threshold_q": float(args.threshold_q),
        "threshold_value": float(th),
        "num_pred_edges": int(A_pred.sum()),
        "imputation": imp_meta.get("imputation"),
        "imputer_backend": imp_meta.get("imputer_backend"),
        "missing_before": miss_before,
        "missing_after": miss_after,
        "our_sem_variant": str(args.our_sem_variant),
        "our_sem_transform": str(args.our_sem_transform),
    }
    if isinstance(meta, dict):
        for k, v in meta.items():
            if k not in row:
                row[k] = v
    for k, v in prior_meta.items():
        row[k] = v

    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "run_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": str(args.scenario),
                "data_file": str(data_fp),
                "method": "OUR_SEM_MODEL_PRIOR_AWARE",
                "our_sem_variant": str(args.our_sem_variant),
                "our_sem_transform": str(args.our_sem_transform),
                "threshold_strategy": str(args.threshold_strategy),
                "threshold_q": float(args.threshold_q),
                "impute": str(args.impute),
                "prior_meta": prior_meta,
                "meta": meta if isinstance(meta, dict) else {},
            },
            f,
            indent=2,
        )
    return row


def main() -> None:
    args = parse_args()
    row = run_prior_aware_our_sem(args)
    print(json.dumps({"status": row.get("status"), "scenario": row.get("scenario"), "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
