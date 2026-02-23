#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create a small reproducible synthetic demo dataset for GitHub.")
    ap.add_argument("--generator-dir", required=True, help="Directory containing generate_synthetic_dataset.py")
    ap.add_argument("--out-dir", required=True, help="Output directory for the demo dataset bundle")
    ap.add_argument("--scenario-name", default="Demo-Small")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--p", type=int, default=24)
    ap.add_argument("--rho", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ell", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--rmiss", type=float, default=0.10)
    ap.add_argument("--missing-mechanism", default="mixed", choices=["none", "mcar", "mar", "mixed"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    gen_dir = Path(args.generator_dir)
    if str(gen_dir) not in sys.path:
        sys.path.insert(0, str(gen_dir))

    try:
        from generate_synthetic_dataset import generate_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Cannot import generate_dataset from: {gen_dir}") from exc

    out_root = Path(args.out_dir)
    sc_dir = out_root / args.scenario_name
    sc_dir.mkdir(parents=True, exist_ok=True)

    (
        df_missing,
        df_complete,
        missing_mask,
        missing_meta,
        A,
        W,
        names,
        blocks,
        v_names,
        effects,
    ) = generate_dataset(
        p=int(args.p),
        n=int(args.n),
        rho_nonlin=float(args.rho),
        k=int(args.k),
        ell=int(args.ell),
        seed=int(args.seed),
        rmiss=float(args.rmiss),
        missing_mechanism=str(args.missing_mechanism),
        graph_family="er",
        estimate_effects=False,
    )

    df_missing.to_csv(sc_dir / f"{args.scenario_name}_data.csv", index=False)
    df_complete.to_csv(sc_dir / f"{args.scenario_name}_data_complete.csv", index=False)
    pd.DataFrame(missing_mask.astype(int), columns=names).to_csv(sc_dir / f"{args.scenario_name}_missing_mask.csv", index=False)
    pd.DataFrame(A, index=names, columns=names).to_csv(sc_dir / f"{args.scenario_name}_adjacency.csv")
    pd.DataFrame(W, index=names, columns=names).to_csv(sc_dir / f"{args.scenario_name}_weights.csv")
    (sc_dir / f"{args.scenario_name}_nodes.txt").write_text("\n".join(names), encoding="utf-8")

    meta = {
        "scenario": args.scenario_name,
        "n": int(args.n),
        "p": int(args.p),
        "rho": float(args.rho),
        "k": int(args.k),
        "ell": int(args.ell),
        "seed": int(args.seed),
        "rmiss": float(args.rmiss),
        "missing_mechanism": str(args.missing_mechanism),
        "blocks": blocks,
        "num_outcomes": len(v_names),
        "columns_preview": names[:10],
        "missing_meta": missing_meta,
        "effects_estimated": bool(effects is not None),
    }
    (sc_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(str(sc_dir))


if __name__ == "__main__":
    main()

