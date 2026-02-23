#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot full and lipid-focused route graphs from edge list.")
    ap.add_argument("--edge-list", required=True)
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def node_group(node: str) -> str:
    if node.startswith("G_"):
        return "G"
    if node.startswith("Zfix_") or node.startswith("Znoise_"):
        return "Z"
    if node.startswith("Lt_"):
        return "Lt"
    if node.startswith("Lm_"):
        return "Lm"
    if node.startswith("R_"):
        return "R"
    if node.startswith("V_"):
        return "V"
    return "Other"


def group_x(group: str) -> float:
    order = {"G": 0, "Z": 1, "Lt": 2, "Lm": 3, "R": 4, "V": 5, "Other": 6}
    return float(order.get(group, 6))


def group_color(group: str) -> str:
    cmap = {
        "G": "#f59e0b",
        "Z": "#64748b",
        "Lt": "#2563eb",
        "Lm": "#22c55e",
        "R": "#ef4444",
        "V": "#7c3aed",
        "Other": "#a3a3a3",
    }
    return cmap.get(group, "#a3a3a3")


def layered_positions(nodes: list[str]) -> dict[str, tuple[float, float]]:
    by_group = {}
    for n in nodes:
        by_group.setdefault(node_group(n), []).append(n)
    pos = {}
    for g, arr in by_group.items():
        arr = sorted(arr)
        x = group_x(g)
        m = len(arr)
        for i, n in enumerate(arr):
            y = (m - 1) / 2.0 - i
            pos[n] = (x, y)
    return pos


def plot_graph(df: pd.DataFrame, out_png: Path, title: str) -> None:
    G = nx.DiGraph()
    for _, r in df.iterrows():
        G.add_edge(str(r["source"]), str(r["target"]), weight=float(r.get("abs_weight", 1.0)))

    nodes = list(G.nodes())
    pos = layered_positions(nodes)
    groups = {n: node_group(n) for n in nodes}
    ncolors = [group_color(groups[n]) for n in nodes]

    fig = plt.figure(figsize=(18, 10))
    ax = plt.gca()
    ax.set_title(title)

    widths = []
    for u, v in G.edges():
        w = float(G[u][v].get("weight", 1.0))
        widths.append(0.8 + min(4.0, 1.2 * (w ** 0.5)))

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=12,
        edge_color="#334155",
        width=widths,
        alpha=0.55,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(G, pos, node_color=ncolors, node_size=1050, edgecolors="white", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    for g in ["G", "Z", "Lt", "Lm", "R", "V"]:
        x = group_x(g)
        ax.text(x, max([p[1] for p in pos.values()]) + 1.4, g, ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    edge_fp = Path(args.edge_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(edge_fp)
    if df.empty:
        raise RuntimeError(f"Edge list is empty: {edge_fp}")

    plot_graph(df, out_dir / "updated_route_full_graph.png", "Updated Route Graph (All Blocks)")

    lipids = {"LBXTC", "LBDHDD", "LBXTR", "LBXAPB", "LBDLDL"}
    keep_nodes = set()
    for _, r in df.iterrows():
        s, t = str(r["source"]), str(r["target"])
        keep = False
        if s.startswith("Lt_") and s.split("Lt_", 1)[1] in lipids:
            keep = True
        if t.startswith("Lt_") and t.split("Lt_", 1)[1] in lipids:
            keep = True
        if s.startswith("R_") or t.startswith("R_") or s.startswith("V_") or t.startswith("V_"):
            keep = True
        if keep:
            keep_nodes.add(s)
            keep_nodes.add(t)
    d2 = df[df["source"].isin(keep_nodes) & df["target"].isin(keep_nodes)].copy()
    if not d2.empty:
        plot_graph(d2, out_dir / "updated_route_lipid_vascular_graph.png", "Updated Route Graph (Lipid and Vascular Focus)")


if __name__ == "__main__":
    main()

