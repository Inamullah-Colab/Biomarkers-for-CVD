#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


EXPOSURES_DEFAULT = ["LBXTC", "LBDHDD", "LBXTR", "LBXAPB", "LBDLDL"]
OUTCOMES_DEFAULT = ["BPXSY1", "BPXDI1", "BPXPLS"]
COVARS_DEFAULT = ["RIDAGEYR", "RIAGENDR", "BMXBMI", "RIDRETH1"]
MEDIATORS_DEFAULT = [
    "AVR_Knudtson",
    "Vein_Vessel_density",
    "Vein_Fractal_dimension",
    "Vessel_density",
    "Artery_Distance_tortuosity",
    "Artery_Squared_curvature_tortuosity",
    "Fractal_dimension",
    "Distance_tortuosity",
]
WEIGHT_COL_DEFAULT = "WTMEC2YR"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute TE/NDE/NIE tables on NHANES or prefix-structured synthetic datasets.")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "NHANES_cvd_extended_with_macular_full_matched.csv",
            "NHANES_with_1000G_reference_proxy.csv",
        ],
    )
    ap.add_argument("--out-dir", default="final_task/results/mediation_realdata_validation_2026-02-22")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--outcomes", nargs="+", default=None)
    ap.add_argument("--exposures", nargs="+", default=None)
    ap.add_argument("--mediators", nargs="+", default=None)
    ap.add_argument("--covars", nargs="+", default=None)
    ap.add_argument("--min-complete-rows", type=int, default=300)
    return ap.parse_args()


def fit_model(formula: str, data: pd.DataFrame, weight_col: str | None):
    if weight_col and weight_col in data.columns:
        return smf.wls(formula, data=data, weights=data[weight_col]).fit()
    return smf.ols(formula, data=data).fit()


def ci_excludes_zero(lo: float, hi: float) -> bool:
    if np.isnan(lo) or np.isnan(hi):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def infer_schema_lists(
    df: pd.DataFrame,
    exposures_cli: list[str] | None,
    outcomes_cli: list[str] | None,
    mediators_cli: list[str] | None,
    covars_cli: list[str] | None,
) -> tuple[list[str], list[str], list[str], list[str], str | None, str]:
    cols = list(df.columns)
    colset = set(cols)

    has_nhanes = any(c in colset for c in EXPOSURES_DEFAULT) and any(c in colset for c in OUTCOMES_DEFAULT)
    has_prefix = any(c.startswith("V") for c in cols) and any(c.startswith("R") for c in cols)

    if has_nhanes:
        schema = "nhanes_named"
        ex_default = [c for c in EXPOSURES_DEFAULT if c in colset]
        out_default = [c for c in OUTCOMES_DEFAULT if c in colset]
        med_default = [c for c in MEDIATORS_DEFAULT if c in colset]
        cov_default = [c for c in COVARS_DEFAULT if c in colset]
        weight_col = WEIGHT_COL_DEFAULT if WEIGHT_COL_DEFAULT in colset else None
    elif has_prefix:
        schema = "prefix_lettercoded"
        ex_default = [c for c in cols if c.startswith(("Lt", "Lm"))]
        out_default = [c for c in cols if c.startswith("V")]
        med_default = [c for c in cols if c.startswith("R")]
        cov_default = [c for c in cols if c.startswith(("Zfix", "Znoise"))]
        weight_col = None
    else:
        schema = "fallback"
        ex_default = [c for c in cols if c.startswith(("Lt", "Lm"))]
        out_default = [c for c in cols if c.startswith("V")]
        med_default = [c for c in cols if c.startswith("R")]
        cov_default = [c for c in cols if c.startswith(("Zfix", "Znoise"))]
        weight_col = WEIGHT_COL_DEFAULT if WEIGHT_COL_DEFAULT in colset else None

    exposures = [c for c in (exposures_cli or ex_default) if c in colset]
    outcomes = [c for c in (outcomes_cli or out_default) if c in colset]
    mediators = [c for c in (mediators_cli or med_default) if c in colset]
    covars = [c for c in (covars_cli or cov_default) if c in colset]
    return exposures, outcomes, mediators, covars, weight_col, schema


def run_one_combo(
    df: pd.DataFrame,
    exposure: str,
    outcome: str,
    mediators: list[str],
    covars: list[str],
    weight_col: str | None,
    min_complete_rows: int,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    med_list = [m for m in mediators if m in df.columns]
    req = [exposure, outcome] + med_list + [c for c in covars if c in df.columns]
    if weight_col:
        req.append(weight_col)
    req = [c for c in req if c in df.columns]
    d = df[req].dropna().copy()

    if len(d) < min_complete_rows or not med_list:
        return pd.DataFrame()

    covars_used = [c for c in covars if c in d.columns]
    cov_expr = " + ".join(covars_used) if covars_used else ""
    out_expr = f"{outcome} ~ {exposure} + {' + '.join(med_list)}"
    if cov_expr:
        out_expr = f"{out_expr} + {cov_expr}"
    total_expr = f"{outcome} ~ {exposure}"
    if cov_expr:
        total_expr = f"{total_expr} + {cov_expr}"

    rng = np.random.default_rng(seed)
    out_mod = fit_model(out_expr, d, weight_col)
    total_mod = fit_model(total_expr, d, weight_col)
    nde_hat = out_mod.params.get(exposure, np.nan)
    te_hat = total_mod.params.get(exposure, np.nan)

    a_hat = {}
    b_hat = {}
    med_expr_map = {}
    for m in med_list:
        m_expr = f"{m} ~ {exposure}"
        if cov_expr:
            m_expr = f"{m_expr} + {cov_expr}"
        med_expr_map[m] = m_expr
        am = fit_model(m_expr, d, weight_col)
        a_hat[m] = am.params.get(exposure, np.nan)
        b_hat[m] = out_mod.params.get(m, np.nan)

    te_bs = []
    nde_bs = []
    nie_bs = {m: [] for m in med_list}
    n = len(d)
    for _ in range(bootstrap):
        s = d.iloc[rng.integers(0, n, n)]
        try:
            out_bs = fit_model(out_expr, s, weight_col)
            tot_bs = fit_model(total_expr, s, weight_col)
            nde_bs.append(out_bs.params.get(exposure, np.nan))
            te_bs.append(tot_bs.params.get(exposure, np.nan))
            for m in med_list:
                am = fit_model(med_expr_map[m], s, weight_col)
                a = am.params.get(exposure, np.nan)
                b = out_bs.params.get(m, np.nan)
                nie_bs[m].append(a * b)
        except Exception:
            continue

    te_bs = np.asarray(te_bs, dtype=float)
    nde_bs = np.asarray(nde_bs, dtype=float)
    te_est = float(np.nanmean(te_bs)) if te_bs.size else float(te_hat)
    nde_est = float(np.nanmean(nde_bs)) if nde_bs.size else float(nde_hat)
    te_lo, te_hi = (np.nan, np.nan) if te_bs.size == 0 else np.nanpercentile(te_bs, [2.5, 97.5])
    nde_lo, nde_hi = (np.nan, np.nan) if nde_bs.size == 0 else np.nanpercentile(nde_bs, [2.5, 97.5])

    rows = []
    for m in med_list:
        arr = np.asarray(nie_bs[m], dtype=float)
        nie_est = float(np.nanmean(arr)) if arr.size else float(a_hat[m] * b_hat[m])
        nie_lo, nie_hi = (np.nan, np.nan) if arr.size == 0 else np.nanpercentile(arr, [2.5, 97.5])
        rows.append(
            {
                "Pathway": f"{exposure} -> {m} -> {outcome}",
                "TE_Estimate": te_est,
                "TE_CI_Lower": float(te_lo),
                "TE_CI_Upper": float(te_hi),
                "NDE_Estimate": nde_est,
                "NDE_CI_Lower": float(nde_lo),
                "NDE_CI_Upper": float(nde_hi),
                "NIE_Estimate": float(nie_est),
                "NIE_CI_Lower": float(nie_lo),
                "NIE_CI_Upper": float(nie_hi),
                "NIE_Significant": bool(ci_excludes_zero(float(nie_lo), float(nie_hi))),
                "NIE_SignConsistency": float(np.nanmean(np.sign(arr) == np.sign(nie_est))) if arr.size else float("nan"),
                "N_rows": int(len(d)),
                "Bootstrap_Used": int(te_bs.size),
            }
        )
    return pd.DataFrame(rows)


def run_dataset(
    df: pd.DataFrame,
    ds_name: str,
    out_root: Path,
    bootstrap: int,
    seed: int,
    exposures_cli: list[str] | None,
    outcomes_cli: list[str] | None,
    mediators_cli: list[str] | None,
    covars_cli: list[str] | None,
    min_complete_rows: int,
) -> dict:
    out_dir = out_root / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    exposures, outcomes, mediators, covars, weight_col, schema = infer_schema_lists(
        df=df,
        exposures_cli=exposures_cli,
        outcomes_cli=outcomes_cli,
        mediators_cli=mediators_cli,
        covars_cli=covars_cli,
    )

    all_tables = []
    for ex in exposures:
        for out in outcomes:
            t = run_one_combo(
                df=df,
                exposure=ex,
                outcome=out,
                mediators=mediators,
                covars=covars,
                weight_col=weight_col,
                min_complete_rows=min_complete_rows,
                bootstrap=bootstrap,
                seed=seed,
            )
            if t.empty:
                continue
            all_tables.append(t)
            t.to_csv(out_dir / f"mediation_table_hpp_style_{ex}_{out}.csv", index=False)
            tr = t.copy()
            num_cols = [c for c in tr.columns if c.endswith(("Estimate", "Lower", "Upper"))]
            tr[num_cols] = tr[num_cols].round(6)
            tr.to_csv(out_dir / f"mediation_table_hpp_style_{ex}_{out}_rounded.csv", index=False)

    if not all_tables:
        summary = {
            "dataset": ds_name,
            "status": "error",
            "error": "No valid exposure-outcome runs (insufficient complete rows or schema mismatch).",
            "schema": schema,
            "n_exposures": int(len(exposures)),
            "n_outcomes": int(len(outcomes)),
            "n_mediators": int(len(mediators)),
            "n_covars": int(len(covars)),
            "weight_col_used": str(weight_col) if weight_col else "",
        }
        pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
        return summary

    all_df = pd.concat(all_tables, ignore_index=True)
    all_df.to_csv(out_dir / "mediation_table_all_combos.csv", index=False)

    sig_count = int(all_df["NIE_Significant"].sum())
    total_count = int(len(all_df))
    summary = {
        "dataset": ds_name,
        "status": "ok",
        "n_pathways": total_count,
        "nie_significant_count": sig_count,
        "nie_significant_rate": float(sig_count / max(1, total_count)),
        "mean_abs_nie": float(all_df["NIE_Estimate"].abs().mean()),
        "max_abs_nie": float(all_df["NIE_Estimate"].abs().max()),
        "schema": schema,
        "n_exposures": int(len(exposures)),
        "n_outcomes": int(len(outcomes)),
        "n_mediators": int(len(mediators)),
        "n_covars": int(len(covars)),
        "weight_col_used": str(weight_col) if weight_col else "",
    }
    pd.DataFrame([summary]).to_csv(out_dir / "summary.csv", index=False)
    return summary


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)
    out_root = base / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    exposures = [e for e in (args.exposures or []) if e]
    outcomes = [o for o in (args.outcomes or []) if o]
    mediators = [m for m in (args.mediators or []) if m]
    covars = [c for c in (args.covars or []) if c]

    run_cfg = {
        "base_dir": str(base),
        "inputs": list(args.inputs),
        "out_dir": str(args.out_dir),
        "bootstrap": int(args.bootstrap),
        "seed": int(args.seed),
        "min_complete_rows": int(args.min_complete_rows),
        "exposures_cli": exposures,
        "outcomes_cli": outcomes,
        "mediators_cli": mediators,
        "covars_cli": covars,
    }
    (out_root / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    summaries = []
    for in_name in args.inputs:
        fp = base / in_name
        if not fp.exists():
            summaries.append({"dataset": in_name, "status": "error", "error": f"Missing file: {fp}"})
            continue
        ds_name = fp.stem
        df = pd.read_csv(fp)
        summaries.append(
            run_dataset(
                df=df,
                ds_name=ds_name,
                out_root=out_root,
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
                exposures_cli=exposures or None,
                outcomes_cli=outcomes or None,
                mediators_cli=mediators or None,
                covars_cli=covars or None,
                min_complete_rows=int(args.min_complete_rows),
            )
        )

    pd.DataFrame(summaries).to_csv(out_root / "summary_all_datasets.csv", index=False)


if __name__ == "__main__":
    main()

