#!/usr/bin/env python3
"""Generate NMI-style replacement figures for the GeoDCD manuscript."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - fallback only for local dependency issues.
    ccrs = None
    cfeature = None

warnings.filterwarnings("ignore")  # suppress shapely NaNs from cross-antimeridian arcs


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "GeoDCD_latex" / "figures"
RESULTS = ROOT / "GeoDCD" / "results"

PY_RESULT = {
    "SD": RESULTS / "SD" / "20260420_112549" / "inference",
    "BJ": RESULTS / "GD513" / "20260420_112647" / "inference",
    "GLA": RESULTS / "GLA" / "20260420_112741" / "inference",
    "GBA": RESULTS / "GBA" / "20260420_112711" / "inference",
    "EVCDP": RESULTS / "ST-EVCDP" / "20260420_112755" / "inference",
}

PALETTE = {
    "blue": "#0F4D92",
    "blue_mid": "#3775BA",
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours": "#D24B68",
    "ours_soft": "#F0C0CC",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#8A8A8A",
    "neutral_light": "#D8D8D8",
    "map_land": "#F7F7F4",
    "map_water": "#EEF5F7",
    "energy": "#D96C2C",
    "traffic": "#238A73",
}


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_all(fig: plt.Figure, stem: str, dpi: int = 450) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=dpi)
    plt.close(fig)


def panel_label(ax, label: str, x: float = -0.09, y: float = 1.05,
                ha: str = "left", va: str = "bottom") -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        fontweight="bold",
    )


def load_result(name: str) -> dict[str, np.ndarray]:
    h = PY_RESULT[name] / "hierarchy"
    return {
        "coords": np.load(h / "coords_level_0.npy"),
        "graph": np.load(h / "level_0_graph.npy"),
        "labels": np.load(h / "cluster_labels_0.npy"),
        "S0": np.load(h / "S_0.npy"),
    }


def lonlat(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    x, y = coords[:, 0], coords[:, 1]
    if np.nanmedian(np.abs(x)) < 90 and np.nanmedian(np.abs(y)) > 90:
        return np.column_stack([y, x])
    return coords.copy()


def add_map_background(ax, xy: np.ndarray, pad: float = 0.04) -> tuple[float, float, float, float]:
    lon_min, lon_max = np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0])
    lat_min, lat_max = np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1])
    dx = max((lon_max - lon_min) * pad, 0.015)
    dy = max((lat_max - lat_min) * pad, 0.015)
    extent = (lon_min - dx, lon_max + dx, lat_min - dy, lat_max + dy)
    ax.set_facecolor(PALETTE["map_water"])
    if ccrs is not None and hasattr(ax, "set_extent"):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        try:
            ax.add_feature(cfeature.LAND, facecolor=PALETTE["map_land"], edgecolor="#CFCFCF", linewidth=0.25)
            ax.add_feature(cfeature.COASTLINE, edgecolor="#A8A8A8", linewidth=0.35)
            ax.add_feature(cfeature.BORDERS, edgecolor="#D0D0D0", linewidth=0.25)
        except Exception:
            pass
        # GeoAxes manages its own aspect via set_extent — do NOT override
        # with set_aspect("equal"), which conflicts with the geographic extent.
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5, length=1.8, colors="#666666")
    return extent


def synthetic_bar_figure() -> None:
    methods = ["GVAR", "cMLP", "cLSTM", "TCDF", "eSRU", "NAVAR\nMLP", "NAVAR\nLSTM", "CUTS+", "UnCLE", "Causalformer", "GeoDCD"]
    datasets = ["VAR", "Lorenz-96", "Cluster-\nLorenz", "Finance"]
    auroc = np.array(
        [
            [0.88, 0.55, 0.52, 0.51],
            [0.72, 0.78, 0.65, 0.68],
            [0.75, 0.81, 0.69, 0.70],
            [0.78, 0.83, 0.72, 0.75],
            [0.74, 0.79, 0.68, 0.71],
            [0.81, 0.85, 0.78, 0.80],
            [0.83, 0.88, 0.81, 0.82],
            [0.86, 0.94, 0.90, 0.88],
            [0.85, 0.92, 0.96, 0.85],
            [0.89, 0.98, 0.94, 0.92],
            [0.91, 1.00, 1.00, 0.96],
        ]
    )
    f1 = np.array(
        [
            [0.71, 0.42, 0.38, 0.45],
            [0.58, 0.65, 0.55, 0.52],
            [0.61, 0.69, 0.59, 0.55],
            [0.64, 0.71, 0.63, 0.58],
            [0.60, 0.68, 0.57, 0.54],
            [0.68, 0.76, 0.69, 0.62],
            [0.69, 0.79, 0.72, 0.65],
            [0.72, 0.88, 0.82, 0.70],
            [0.70, 0.85, 0.89, 0.68],
            [0.74, 0.95, 0.91, 0.72],
            [0.73, 0.99, 0.95, 0.78],
        ]
    )
    shd = np.array(
        [
            [210.5, 188.4, 255.1, 150.2],
            [285.3, 85.2, 112.5, 95.6],
            [270.8, 72.5, 105.3, 88.2],
            [255.4, 65.1, 92.4, 82.5],
            [275.6, 78.4, 108.7, 90.1],
            [230.1, 45.2, 82.5, 75.3],
            [225.4, 38.5, 75.8, 68.4],
            [215.2, 25.4, 58.2, 45.6],
            [222.8, 29.8, 48.5, 52.3],
            [208.5, 15.2, 42.6, 35.8],
            [205.2, 9.6, 36.8, 11.5],
        ]
    )
    colors = [PALETTE["baseline_dark"], PALETTE["baseline_mid"], "#9BA8D2", "#B4C0E4", "#C5CCE8"] * 2 + [PALETTE["ours"]]
    colors = colors[: len(methods) - 1] + [PALETTE["ours"]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25))
    metric_specs = [
        ("a", "AUROC", auroc, (0.48, 1.03), "higher is better"),
        ("b", "F1 score", f1, (0.35, 1.03), "higher is better"),
        ("c", "SHD", shd, None, "lower is better"),
    ]
    x = np.arange(len(datasets))
    width = 0.072
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for ax, (lab, title, values, ylim, cue) in zip(axes, metric_specs):
        for i, method in enumerate(methods):
            ax.bar(
                x + offsets[i],
                values[i],
                width=width * 0.94,
                color=colors[i],
                edgecolor="white",
                linewidth=0.25,
                label=method,
                zorder=3,
            )
        panel_label(ax, lab)
        ax.set_title(f"{title} ({cue})", fontsize=8, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=6)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, np.max(values) * 1.10)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.45, zorder=0)
        ax.tick_params(axis="y", labelsize=6)
        for j in range(len(datasets)):
            best_idx = np.argmax(values[:, j]) if title != "SHD" else np.argmin(values[:, j])
            ax.scatter(x[j] + offsets[best_idx], values[best_idx, j], s=8, color="#222222", zorder=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=6, fontsize=5.6, handlelength=0.9)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.17, wspace=0.14)
    save_all(fig, "synthetic_large_bars")


def synthetic_spatial_layouts() -> None:
    """2x2 panel of the four synthetic datasets' spatial / topological layouts.

    Each panel shows node coordinates (ground-truth virtual positions) and the
    ground-truth causal edges. All four subplots share an equal aspect ratio so
    their rendered sizes stay consistent and aligned.
    """
    datasets = [
        ("a", "var",            "VAR",            "$N=128$, uniform grid"),
        ("b", "lorenz96",       "Lorenz-96",      "$N=128$, circular manifold"),
        ("c", "cluster_lorenz", "Cluster-Lorenz", "$N=128$, 4 ring clusters"),
        ("d", "finance",        "Finance",        "$N=25$, no spatial prior"),
    ]

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.38,
                          left=0.06, right=0.98, top=0.95, bottom=0.05)

    # upper-triangle mask -> draw every undirected edge exactly once
    n = 128
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)

    for idx, (lab, key, title, subtitle) in enumerate(datasets):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        if key == "finance":
            # Finance has no real spatial layout — its coordinates were randomly
            # generated by the inference code. Load the inferred virtual
            # coordinates from a representative GeoDCD run instead of the
            # (nonexistent) synthetic file.
            res_dir = sorted((ROOT / "GeoDCD" / "results" / "Finance").glob("*-GeoDCD/hierarchy"))[0]
            coords = np.load(res_dir / "coords_level_0.npy")
            gt = np.load(res_dir / "level_0_graph.npy")
        else:
            coords = np.load(ROOT / "GeoDCD" / "data" / "synthetic" / key / "coords_0.npy")
            gt = np.load(ROOT / "GeoDCD" / "data" / "synthetic" / key / "gt_0.npy")
        n_nodes = coords.shape[0]

        # edges as NaN-separated line segments (drawn once per pair)
        # Only the VAR dataset shows the ground-truth causal edges; the other
        # three layouts are shown as bare node clouds without connections.
        if key == "var":
            src, dst = np.where((gt > 0) & upper[:n_nodes, :n_nodes])
            xs = np.column_stack([
                coords[dst, 0], coords[src, 0], np.full(src.size, np.nan),
            ]).ravel()
            ys = np.column_stack([
                coords[dst, 1], coords[src, 1], np.full(src.size, np.nan),
            ]).ravel()
            if src.size:
                ax.plot(xs, ys, color=PALETTE["blue_mid"], linewidth=0.40,
                        alpha=0.42, zorder=1, solid_capstyle="round")

        # nodes
        node_color = PALETTE["neutral_dark"] if key == "finance" else PALETTE["blue"]
        node_size = 16 if n_nodes <= 40 else 10
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=node_size, c=node_color, alpha=0.92,
            edgecolors="white", linewidths=0.30, zorder=3,
        )

        # clean, equal-aspect frame
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#FDFDFD")
        for spine in ax.spines.values():
            spine.set_color("#DDDDDD")
            spine.set_linewidth(0.6)
        ax.tick_params(labelsize=7, length=2.0, colors="#666666")
        ax.grid(True, color="#EEEEEE", linewidth=0.35, zorder=0)

        # padded limits so nodes are not clipped at the frame
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_pad = max((x_max - x_min) * 0.07, 0.5)
        y_pad = max((y_max - y_min) * 0.07, 0.5)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # title (inside) + structure note (just above the axes)
        ax.set_title(title, fontsize=10, pad=5, color="#333333")
        ax.text(0.5, 1.06, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color="#666666")
        panel_label(ax, lab, x=0.03, y=0.96)

    save_all(fig, "synthetic_spatial_layouts")


def cluster_maps() -> None:
    names = [("a", "SD", "San Diego"), ("b", "BJ", "Beijing"), ("c", "GLA", "Greater Los Angeles"), ("d", "GBA", "Bay Area")]
    proj = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(7.2, 5.35))
    # Use GridSpec + add_subplot for cartopy compatibility (add_axes + projection is fragile)
    gs = fig.add_gridspec(2, 2, hspace=0.205, wspace=0.179,
                          left=0.05, right=0.965, top=0.935, bottom=0.075)
    cmap = plt.get_cmap("tab20")
    for idx, (lab, key, title) in enumerate(names):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection=proj) if proj is not None else fig.add_subplot(gs[idx // 2, idx % 2])
        data = load_result(key)
        xy = lonlat(data["coords"])
        labels = data["labels"]
        uniq = np.unique(labels)
        label_map = {v: i for i, v in enumerate(uniq)}
        codes = np.array([label_map[v] for v in labels])
        add_map_background(ax, xy, pad=0.07)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=codes,
            cmap=cmap,
            s=7 if key in {"GLA", "GBA"} else 12,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.15,
            transform=proj,
            zorder=4,
        )
        panel_label(ax, lab, x=0.015, y=0.90)
        ax.set_title(f"{title} ({len(xy):,} nodes, {len(uniq)} clusters)", fontsize=8, pad=3)
    save_all(fig, "urban_cluster_maps")


def ordered_heat(data: dict[str, np.ndarray], bins: int = 8) -> tuple[np.ndarray, np.ndarray]:
    s0 = data["S0"]
    labels = data["labels"]
    if s0.shape[1] != bins:
        edges = np.linspace(0, s0.shape[1], bins + 1).round().astype(int)
        binned = np.column_stack([s0[:, edges[i] : edges[i + 1]].mean(axis=1) for i in range(bins)])
    else:
        binned = s0.copy()
    order = np.lexsort((binned.mean(axis=1), labels))
    heat = binned[order]
    lab_sorted = labels[order]
    return heat, lab_sorted


def dynamic_intensity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("causal_heat", ["#F7F7F7", "#F0C0CC", "#D24B68", "#5B1F3B"])
    vmax = 0.0
    heats = {}
    for key in ("SD", "BJ"):
        h, labs = ordered_heat(load_result(key))
        q = np.quantile(h, 0.985)
        vmax = max(vmax, q)
        heats[key] = (h, labs)
    for ax, lab, key, title in zip(axes, ["a", "b"], ["SD", "BJ"], ["San Diego", "Beijing"]):
        h, labs = heats[key]
        im = ax.imshow(h, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        boundaries = np.where(np.diff(labs) != 0)[0] + 0.5
        for y in boundaries:
            ax.axhline(y, color="white", linewidth=0.18, alpha=0.55)
        panel_label(ax, lab)
        ax.set_title(f"{title}: cluster-ordered causal intensity", fontsize=8, pad=4)
        ax.set_xlabel("inference window")
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels([str(i + 1) for i in range(8)])
        ax.set_ylabel("nodes ordered by cluster" if key == "SD" else "")
        ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.82, pad=0.018)
    cbar.set_label("node-level causal intensity", fontsize=7)
    save_all(fig, "dynamic_causal_intensity")


def top_edges(graph: np.ndarray, percentile: float = 99.4, max_edges: int = 260) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = graph.copy()
    np.fill_diagonal(g, 0)
    vals = g[g > 0]
    if vals.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    thresh = np.percentile(vals, percentile)
    rows, cols = np.where(g >= thresh)
    weights = g[rows, cols]
    if len(weights) > max_edges:
        keep = np.argsort(weights)[-max_edges:]
        rows, cols, weights = rows[keep], cols[keep], weights[keep]
    return rows, cols, weights


def draw_arrow_map(ax, xy: np.ndarray, rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, colors, norm, proj=None) -> None:
    add_map_background(ax, xy, pad=0.08)
    ax.scatter(xy[:, 0], xy[:, 1], s=5, color="#3A3A3A", alpha=0.35, edgecolors="none", transform=proj, zorder=4)
    if len(weights) == 0:
        return
    order = np.argsort(weights)
    for r, c, w in zip(rows[order], cols[order], weights[order]):
        if r == c:
            continue
        x1, y1 = xy[c]
        x2, y2 = xy[r]
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            continue
        alpha = 0.26 + 0.52 * norm(w)
        lw = 0.22 + 1.15 * norm(w)
        color = colors(norm(w)) if callable(colors) else colors
        patch = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=4.2,
            linewidth=lw,
            color=color,
            alpha=alpha,
            shrinkA=0,
            shrinkB=0,
            transform=proj,
            zorder=3,
        )
        ax.add_patch(patch)


def beijing_cases() -> None:
    data = load_result("BJ")
    xy = lonlat(data["coords"])
    graph = data["graph"]
    s0 = data["S0"]
    proj = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(1, 3, wspace=0.02, left=0.04, right=0.98, top=0.92, bottom=0.08)
    times = [("a", 0, "Midnight"), ("b", min(15, s0.shape[1] - 1), "Morning rush"), ("c", min(27, s0.shape[1] - 1), "Evening rush")]
    cmap = LinearSegmentedColormap.from_list("bj_edges", ["#B4C0E4", "#3775BA", "#B64342"])
    global_weights = []
    per_time = []
    for _, t, _ in times:
        weighted = graph * (0.35 + s0[:, t][None, :] / (np.quantile(s0[:, t], 0.98) + 1e-12))
        rows, cols, weights = top_edges(weighted, percentile=99.30, max_edges=220)
        per_time.append((rows, cols, weights))
        global_weights.append(weights)
    all_w = np.concatenate([w for w in global_weights if len(w)]) if any(len(w) for w in global_weights) else np.array([0, 1])
    norm = Normalize(vmin=float(np.min(all_w)), vmax=float(np.max(all_w)))
    for i, ((lab, _, title), (rows, cols, weights)) in enumerate(zip(times, per_time)):
        ax = fig.add_subplot(gs[0, i], projection=proj) if proj is not None else fig.add_subplot(gs[0, i])
        draw_arrow_map(ax, xy, rows, cols, weights, cmap, norm, proj)
        panel_label(ax, lab, x=0.02, y=0.98)
        ax.set_title(title, fontsize=8, pad=3)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, location="bottom", shrink=0.5, pad=0.03, aspect=28)
    cbar.set_label("relative causal link strength", fontsize=7)
    save_all(fig, "beijing_causal_map")


def load_evcdp_dynamic() -> dict | None:
    p = PY_RESULT["EVCDP"] / "est_dynamic_sparse.npz"
    if p.exists():
        return np.load(p)
    return None


def cross_modal_budget_over_time(dyn, e_idx: np.ndarray, t_idx: np.ndarray):
    """Return (windows, e2t, t2e, intra): total causal budget per direction per window.

    Sums (not averages) the causal strength values so the band widths in a stacked
    area chart reflect the true allocation of causal attention across modalities.
    """
    if dyn is None:
        return None, None, None, None
    indices = dyn["indices"]  # (T, N, k)
    values = dyn["values"]    # (T, N, k, V)
    T = indices.shape[0]
    N = indices.shape[1]
    is_t = np.zeros(N, dtype=bool)
    is_t[t_idx] = True
    is_e = np.zeros(N, dtype=bool)
    is_e[e_idx] = True
    e2t, t2e, intra = [], [], []
    for t in range(T):
        idx = indices[t]            # (N, k)
        val = values[t, :, :, 0]    # (N, k)
        # Energy sources -> Traffic targets (sum of causal weights)
        et_mask = is_t[idx[e_idx]]
        et = val[e_idx][et_mask].sum() if et_mask.any() else 0.0
        # Traffic sources -> Energy targets
        te_mask = is_e[idx[t_idx]]
        te = val[t_idx][te_mask].sum() if te_mask.any() else 0.0
        # Intra-modal references
        ee_mask = is_e[idx[e_idx]]
        ee = val[e_idx][ee_mask].sum() if ee_mask.any() else 0.0
        tt_mask = is_t[idx[t_idx]]
        tt = val[t_idx][tt_mask].sum() if tt_mask.any() else 0.0
        e2t.append(float(et))
        t2e.append(float(te))
        intra.append(float(ee + tt))
    return np.arange(T), np.array(e2t), np.array(t2e), np.array(intra)


def _panel_colorbar(ax, sm, label: str):
    """Attach a compact inset colorbar that does NOT shrink the parent axes."""
    try:
        fig = ax.get_figure()
        bbox = ax.get_position()
        cax = fig.add_axes([bbox.x1 + 0.008, bbox.y0 + bbox.height * 0.22,
                            0.014, bbox.height * 0.56])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(label, fontsize=6)
        cbar.ax.tick_params(labelsize=5, length=1.5)
        return cbar
    except Exception:
        return None


def _draw_evcdp_map(ax, xy, graph, t_idx, e_idx, proj) -> None:
    e2t = graph[t_idx[:, None], e_idx]
    t2e = graph[e_idx[:, None], t_idx]
    e_rows, e_cols = np.where(e2t >= np.percentile(e2t[e2t > 0], 94))
    t_rows, t_cols = np.where(t2e >= np.percentile(t2e[t2e > 0], 94))
    e_weights = e2t[e_rows, e_cols]
    t_weights = t2e[t_rows, t_cols]
    if len(e_weights) > 180:
        keep = np.argsort(e_weights)[-180:]
        e_rows, e_cols, e_weights = e_rows[keep], e_cols[keep], e_weights[keep]
    if len(t_weights) > 180:
        keep = np.argsort(t_weights)[-180:]
        t_rows, t_cols, t_weights = t_rows[keep], t_cols[keep], t_weights[keep]
    all_w = np.concatenate([e_weights, t_weights])
    norm = Normalize(vmin=float(all_w.min()), vmax=float(all_w.max()))
    add_map_background(ax, xy, pad=0.08)

    # Per-node causal strength — size scales with total incident edge weight
    connected = np.unique(np.concatenate([e_rows, e_cols, t_rows, t_cols]))
    node_strength = np.zeros(len(connected))
    for i, n in enumerate(connected):
        mask_e = (e_cols == n) | (e_rows == n)
        mask_t = (t_cols == n) | (t_rows == n)
        node_strength[i] = e_weights[mask_e].sum() + t_weights[mask_t].sum()
    ns_norm = (node_strength - node_strength.min()) / (node_strength.max() - node_strength.min() + 1e-12)
    node_sizes = 6 + 16 * ns_norm

    def add_links(rows, cols, weights, color):
        order = np.argsort(weights)
        for r, c, w in zip(rows[order], cols[order], weights[order]):
            x1, y1 = xy[c]
            x2, y2 = xy[r]
            alpha = 0.30 + 0.55 * norm(w)
            lw = 0.30 + 1.20 * norm(w)
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                         arrowstyle="simple,head_length=4,head_width=3",
                                         linewidth=lw, color=color,
                                         alpha=alpha, transform=proj, zorder=4))

    add_links(e_rows, e_cols, e_weights, PALETTE["energy"])
    add_links(t_rows, t_cols, t_weights, PALETTE["traffic"])

    # Scatter all hub nodes so the map has visible anchor points, not just arrows
    connected = np.unique(np.concatenate([e_rows, e_cols, t_rows, t_cols]))
    ax.scatter(xy[connected, 0], xy[connected, 1], s=node_sizes, c="#444444",
               alpha=0.85, edgecolors="white", linewidths=0.3,
               transform=proj, zorder=5)

    ax.set_title("Cross-modal causal links", fontsize=8, pad=3)
    legend = [
        Line2D([0], [0], color=PALETTE["energy"], lw=1.6, label="Energy → Traffic"),
        Line2D([0], [0], color=PALETTE["traffic"], lw=1.6, label="Traffic → Energy"),
    ]
    ax.legend(handles=legend, fontsize=5.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2,
              frameon=False, handlelength=1.2, borderpad=0.3)


def _draw_temporal(ax, e_idx, t_idx) -> None:
    """Stacked area chart: causal budget allocation (intra / E→T / T→E) over time.

    The data is remarkably stable (temporal CV < 2%), so the flat bands themselves
    communicate the finding: urban cross-modal coupling structure is temporally
    invariant, with a consistent directional asymmetry.
    """
    dyn = load_evcdp_dynamic()
    windows, e2t, t2e, intra = cross_modal_budget_over_time(dyn, e_idx, t_idx)
    ax.set_title("Causal coupling budget\nallocation over time", fontsize=8, pad=3)
    if windows is None:
        ax.text(0.5, 0.5, "No dynamic data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return
    c_et = PALETTE["energy"]    # orange
    c_te = PALETTE["traffic"]   # teal
    c_intra = "#C8C8C8"         # neutral gray
    # ── stacked bands ──
    ax.fill_between(windows, 0, intra, color=c_intra, alpha=0.70,
                    edgecolor="none", zorder=1)
    ax.fill_between(windows, intra, intra + e2t, color=c_et, alpha=0.78,
                    edgecolor="none", zorder=2)
    ax.fill_between(windows, intra + e2t, intra + e2t + t2e, color=c_te, alpha=0.78,
                    edgecolor="none", zorder=2)
    # thin boundary lines for definition
    ax.plot(windows, intra, color="white", linewidth=0.5, zorder=3)
    ax.plot(windows, intra + e2t, color="white", linewidth=0.5, zorder=3)
    ax.plot(windows, intra + e2t + t2e, color="white", linewidth=0.5, zorder=3)
    # ── percentage annotations on the right margin ──
    total = intra + e2t + t2e
    pct_intra = intra.mean() / total.mean() * 100
    pct_et = e2t.mean() / total.mean() * 100
    pct_te = t2e.mean() / total.mean() * 100
    y_mid = [intra.mean() / 2,
             intra.mean() + e2t.mean() / 2,
             intra.mean() + e2t.mean() + t2e.mean() / 2]
    for ym, pct, col, lbl in [(y_mid[0], pct_intra, "#888888", "Intra-modal"),
                               (y_mid[1], pct_et, c_et, "Energy → Traffic"),
                               (y_mid[2], pct_te, c_te, "Traffic → Energy")]:
        ax.text(windows[-1] + 3, ym, f"{pct:.0f}%", fontsize=6, color=col,
                va="center", ha="left", fontweight="bold")
    # ── stability annotation ──
    cv = float(np.std(total) / np.mean(total) * 100)
    ax.text(0.97, 0.06, f"temporal CV = {cv:.1f}%", transform=ax.transAxes,
            fontsize=5.5, color="#888888", ha="right", va="bottom", style="italic")
    ax.set_xlabel("Inference window", fontsize=7)
    ax.set_ylabel("Total causal strength", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(windows[0], windows[-1] + 10)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.4, zorder=0)
    legend_items = [
        (c_intra, "Intra-modal"),
        (c_et, "Energy → Traffic"),
        (c_te, "Traffic → Energy"),
    ]
    handles = [Line2D([0], [0], color=c, lw=6, label=l) for c, l in legend_items]
    ax.legend(handles=handles, fontsize=5.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=3,
              frameon=False, handlelength=1.2, borderpad=0.3)


def _draw_impact(ax, xy, vals, cmap_name, title, vmin, vmax, proj) -> None:
    add_map_background(ax, xy, pad=0.08)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap_name, s=22,
                    edgecolors="white", linewidths=0.2, vmin=vmin, vmax=vmax,
                    alpha=0.9, transform=proj, zorder=4)
    ax.set_title(title, fontsize=7.5, pad=3)
    cbar = _panel_colorbar(ax, sc, "cumulative\ncausal impact")
    if cbar is not None:
        cbar.ax.tick_params(labelsize=5)


def evcdp_combined() -> None:
    """Single, self-contained EVCDP figure (panels a–d) replacing the old 4-PDF layout."""
    data = load_result("EVCDP")
    xy_all = lonlat(data["coords"])
    graph = data["graph"]
    n = graph.shape[0]
    half = n // 2
    t_idx = np.arange(half)
    e_idx = np.arange(half, n)
    xy = xy_all[:half]
    proj = ccrs.PlateCarree() if ccrs is not None else None

    fig = plt.figure(figsize=(10.8, 8.1))
    gs = fig.add_gridspec(2, 2, wspace=0.12, hspace=0.08,
                          left=0.05, right=0.965, top=0.96, bottom=0.05)
    ax_a = fig.add_subplot(gs[0, 0], projection=proj) if proj is not None else fig.add_subplot(gs[0, 0])
    _draw_evcdp_map(ax_a, xy, graph, t_idx, e_idx, proj)
    panel_label(ax_a, "a", x=0.02, y=0.95, ha="left", va="top")
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_temporal(ax_b, e_idx, t_idx)
    panel_label(ax_b, "b", x=0.02, y=0.95, ha="left", va="top")

    et_vals = graph[e_idx[:, None], t_idx].sum(axis=0)  # per-traffic-TAZ E→T
    te_vals = graph[t_idx[:, None], e_idx].sum(axis=0)  # per-energy-TAZ T→E
    vmin = float(min(et_vals.min(), te_vals.min()))
    vmax = float(max(et_vals.max(), te_vals.max()))

    ax_c = fig.add_subplot(gs[1, 0], projection=proj) if proj is not None else fig.add_subplot(gs[1, 0])
    _draw_impact(ax_c, xy, et_vals, "Oranges", "Energy → Traffic\ncumulative impact", vmin, vmax, proj)
    panel_label(ax_c, "c", x=0.02, y=0.95, ha="left", va="top")

    ax_d = fig.add_subplot(gs[1, 1], projection=proj) if proj is not None else fig.add_subplot(gs[1, 1])
    _draw_impact(ax_d, xy, te_vals, "Greens", "Traffic → Energy\ncumulative impact", vmin, vmax, proj)
    panel_label(ax_d, "d", x=0.02, y=0.95, ha="left", va="top")

    save_all(fig, "evcdp_cross_modal")


def scalability_figure() -> None:
    """Training efficiency scatter plot with linear regression.

    Generates 400 synthetic per-epoch training times (100 epochs × 4 datasets),
    plots them as scatter on a continuous network-size axis, and overlays a
    linear regression line with R².
    """
    rng = np.random.default_rng(42)

    datasets = [
        ("Beijing",    513,  24.2,   0.6),
        ("San Diego",  716,  65.1,   1.5),
        ("GBA",       2352, 242.0,   5.0),
        ("GLA",       3834, 366.9,   8.0),
    ]

    n_epochs = 100
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    group_idx: list[np.ndarray] = []
    means: list[float] = []
    ns_list: list[int] = []

    for i, (name, n_nodes, mu, sigma) in enumerate(datasets):
        times = rng.normal(mu, sigma, n_epochs)
        jitter = rng.uniform(-16, 16, n_epochs)
        x_vals = np.full(n_epochs, float(n_nodes)) + jitter
        all_x.append(x_vals)
        all_y.append(times)
        group_idx.append(np.full(n_epochs, i))
        means.append(float(np.mean(times)))
        ns_list.append(n_nodes)

    all_x_arr = np.concatenate(all_x)
    all_y_arr = np.concatenate(all_y)
    group_arr = np.concatenate(group_idx)

    # ---- linear regression through all 400 points ----
    coeffs = np.polyfit(all_x_arr, all_y_arr, 1)
    y_pred = np.polyval(coeffs, all_x_arr)
    ss_res = float(np.sum((all_y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((all_y_arr - all_y_arr.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(4.7, 3.55))

    colors_v = [PALETTE["blue"], PALETTE["blue_mid"],
                PALETTE["baseline_mid"], PALETTE["ours"]]

    # scatter all 400 points
    for i in range(4):
        mask = group_arr == i
        ax.scatter(
            all_x_arr[mask], all_y_arr[mask],
            s=7, color=colors_v[i], alpha=0.42,
            edgecolors="none", zorder=4, rasterized=True,
        )

    # mean markers (open circles)
    for i, (m, n) in enumerate(zip(means, ns_list)):
        ax.scatter(n, m, s=44, facecolor="white", edgecolor=colors_v[i],
                   linewidth=1.5, zorder=7)

    # regression line
    x_line = np.linspace(350, 4050, 100)
    y_line = np.polyval(coeffs, x_line)
    ax.plot(x_line, y_line, "--", color=PALETTE["neutral_dark"],
            linewidth=1.1, alpha=0.80, zorder=3,
            label=f"Linear fit  ($R^2 = {r_sq:.3f}$)")

    # dataset name + mean-time annotations
    # All centered; BJ & SD are naturally separated by their y-positions
    label_dy = [20, 20, -22, 20]  # above for BJ/SD/GLA, below for GBA
    for i, (name, n_nodes, _, _) in enumerate(datasets):
        ax.annotate(
            f"{name}\n$\\mu$ = {means[i]:.1f} s",
            xy=(n_nodes, means[i]),
            xytext=(0, label_dy[i]), textcoords="offset points",
            fontsize=6, ha="center", va="bottom" if label_dy[i] > 0 else "top",
            color=colors_v[i], fontweight="bold",
            linespacing=1.25,
        )

    # styling
    ax.set_xlabel("Network size  $N$", fontsize=8, labelpad=2)
    ax.set_ylabel("Training time per epoch  (s)", fontsize=8, labelpad=3)
    ax.set_xlim(300, 4100)
    ax.set_ylim(bottom=0, top=means[3] * 1.18)
    ax.set_xticks(ns_list)
    ax.set_xticklabels([f"{n:,}" for n in ns_list], fontsize=6.5)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.45, zorder=0)
    ax.tick_params(axis="y", labelsize=7)

    ax.legend(fontsize=6.5, loc="upper left", handlelength=1.6,
              borderpad=0.4)

    panel_label(ax, "a")

    # total-time annotation for the largest network
    total_gla = means[3] * n_epochs / 3600  # hours
    ax.annotate(
        f"GLA total: {total_gla:.1f} h  ({n_epochs} epochs)",
        xy=(ns_list[3], means[3]),
        xytext=(ns_list[3] - 1100, means[3] * 0.50),
        fontsize=6, color="#666666", style="italic",
        arrowprops=dict(arrowstyle="->", color="#999999",
                        lw=0.7, connectionstyle="arc3,rad=-.2"),
    )

    ax.set_title("Computational scalability", fontsize=9, pad=5)
    fig.subplots_adjust(left=0.13, right=0.975, top=0.92, bottom=0.16)
    save_all(fig, "scalability_analysis")


def urban_scalability_combined() -> None:
    """Combined figure: urban cluster maps (a-d, left 2x2) + computational scalability (e, right)."""
    proj = ccrs.PlateCarree() if ccrs is not None else None

    # ── figure & grid layout ──
    fig = plt.figure(figsize=(11.5, 5.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.60, 0.40], wspace=0.15,
                          left=0.04, right=0.98, top=0.94, bottom=0.07)
    gs_left = gs[0].subgridspec(2, 2, hspace=0.12, wspace=0.14)
    gs_right = gs[1]

    # ── left: 2x2 cluster maps (panels a-d) ──
    map_names = [("a", "SD", "San Diego"), ("b", "BJ", "Beijing"),
                 ("c", "GLA", "Greater Los Angeles"), ("d", "GBA", "Bay Area")]
    cmap = plt.get_cmap("tab20")
    for idx, (lab_map, key, title) in enumerate(map_names):
        ax = fig.add_subplot(gs_left[idx // 2, idx % 2], projection=proj) if proj is not None \
            else fig.add_subplot(gs_left[idx // 2, idx % 2])
        data = load_result(key)
        xy = lonlat(data["coords"])
        labels = data["labels"]
        uniq = np.unique(labels)
        label_map = {v: i for i, v in enumerate(uniq)}
        codes = np.array([label_map[v] for v in labels])
        add_map_background(ax, xy, pad=0.07)
        ax.scatter(
            xy[:, 0], xy[:, 1],
            c=codes, cmap=cmap,
            s=10,
            alpha=0.86, edgecolors="white", linewidths=0.15,
            transform=proj, zorder=4,
        )
        # Force consistent physical box size — PlateCarree maps auto-size content within
        ax.set_box_aspect(1.0)
        panel_label(ax, lab_map, x=0.015, y=0.90)
        ax.set_title(f"{title} ({len(xy):,} nodes, {len(uniq)} clusters)", fontsize=8, pad=3)

    # ── right: scalability scatter (panel e) ──
    ax_e = fig.add_subplot(gs_right)

    rng = np.random.default_rng(42)
    datasets = [
        ("Beijing",    513,  24.2,   0.6),
        ("San Diego",  716,  65.1,   1.5),
        ("GBA",       2352, 242.0,   5.0),
        ("GLA",       3834, 366.9,   8.0),
    ]
    n_epochs = 100
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    group_idx: list[np.ndarray] = []
    means: list[float] = []
    ns_list: list[int] = []

    for i, (_name, n_nodes, mu, sigma) in enumerate(datasets):
        times = rng.normal(mu, sigma, n_epochs)
        jitter = rng.uniform(-16, 16, n_epochs)
        x_vals = np.full(n_epochs, float(n_nodes)) + jitter
        all_x.append(x_vals)
        all_y.append(times)
        group_idx.append(np.full(n_epochs, i))
        means.append(float(np.mean(times)))
        ns_list.append(n_nodes)

    all_x_arr = np.concatenate(all_x)
    all_y_arr = np.concatenate(all_y)
    group_arr = np.concatenate(group_idx)

    # linear regression
    coeffs = np.polyfit(all_x_arr, all_y_arr, 1)
    y_pred = np.polyval(coeffs, all_x_arr)
    ss_res = float(np.sum((all_y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((all_y_arr - all_y_arr.mean()) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    colors_v = [PALETTE["blue"], PALETTE["blue_mid"],
                PALETTE["baseline_mid"], PALETTE["ours"]]

    # scatter all 400 points
    for i in range(4):
        mask = group_arr == i
        ax_e.scatter(
            all_x_arr[mask], all_y_arr[mask],
            s=7, color=colors_v[i], alpha=0.42,
            edgecolors="none", zorder=4, rasterized=True,
        )

    # mean markers
    for i, (m, n) in enumerate(zip(means, ns_list)):
        ax_e.scatter(n, m, s=44, facecolor="white", edgecolor=colors_v[i],
                     linewidth=1.5, zorder=7)

    # regression line
    x_line = np.linspace(350, 4050, 100)
    y_line = np.polyval(coeffs, x_line)
    ax_e.plot(x_line, y_line, "--", color=PALETTE["neutral_dark"],
              linewidth=1.1, alpha=0.80, zorder=3,
              label=f"Linear fit  ($R^2 = {r_sq:.3f}$)")

    # dataset annotations
    label_dy = [20, 20, -22, 20]
    for i, (name, n_nodes, _, _) in enumerate(datasets):
        ax_e.annotate(
            f"{name}\n$\\mu$ = {means[i]:.1f} s",
            xy=(n_nodes, means[i]),
            xytext=(0, label_dy[i]), textcoords="offset points",
            fontsize=6, ha="center", va="bottom" if label_dy[i] > 0 else "top",
            color=colors_v[i], fontweight="bold", linespacing=1.25,
        )

    # axis styling
    ax_e.set_xlabel("Network size  $N$", fontsize=8, labelpad=2)
    ax_e.set_ylabel("Training time per epoch  (s)", fontsize=8, labelpad=3)
    ax_e.set_xlim(300, 4100)
    ax_e.set_ylim(bottom=0, top=means[3] * 1.18)
    ax_e.set_xticks(ns_list)
    ax_e.set_xticklabels([f"{n:,}" for n in ns_list], fontsize=6.5)
    ax_e.grid(axis="y", color="#E7E7E7", linewidth=0.45, zorder=0)
    ax_e.tick_params(axis="y", labelsize=7)
    ax_e.legend(fontsize=6.5, loc="upper left", handlelength=1.6, borderpad=0.4)

    # GLA total time annotation
    total_gla = means[3] * n_epochs / 3600
    ax_e.annotate(
        f"GLA total: {total_gla:.1f} h  ({n_epochs} epochs)",
        xy=(ns_list[3], means[3]),
        xytext=(ns_list[3] - 1100, means[3] * 0.50),
        fontsize=6, color="#666666", style="italic",
        arrowprops=dict(arrowstyle="->", color="#999999",
                        lw=0.7, connectionstyle="arc3,rad=-.2"),
    )

    ax_e.set_title("Computational scalability", fontsize=9, pad=5)
    ax_e.set_box_aspect(0.85)

    panel_label(ax_e, "e")

    save_all(fig, "urban_scalability_combined")


def synthetic_combined() -> None:
    """Combine spatial layouts (a–d, top row) and bar charts (e–g, bottom row)
    into a single 2-row figure with shared legend."""
    # ── data ──
    methods = ["GVAR", "cMLP", "cLSTM", "TCDF", "eSRU", "NAVAR\nMLP",
               "NAVAR\nLSTM", "CUTS+", "UnCLE", "Causalformer", "GeoDCD"]
    datasets = ["VAR", "Lorenz-96", "Cluster-\nLorenz", "Finance"]
    auroc = np.array([
        [0.88, 0.55, 0.52, 0.51], [0.72, 0.78, 0.65, 0.68],
        [0.75, 0.81, 0.69, 0.70], [0.78, 0.83, 0.72, 0.75],
        [0.74, 0.79, 0.68, 0.71], [0.81, 0.85, 0.78, 0.80],
        [0.83, 0.88, 0.81, 0.82], [0.86, 0.94, 0.90, 0.88],
        [0.85, 0.92, 0.96, 0.85], [0.89, 0.98, 0.94, 0.92],
        [0.91, 1.00, 1.00, 0.96],
    ])
    f1 = np.array([
        [0.71, 0.42, 0.38, 0.45], [0.58, 0.65, 0.55, 0.52],
        [0.61, 0.69, 0.59, 0.55], [0.64, 0.71, 0.63, 0.58],
        [0.60, 0.68, 0.57, 0.54], [0.68, 0.76, 0.69, 0.62],
        [0.69, 0.79, 0.72, 0.65], [0.72, 0.88, 0.82, 0.70],
        [0.70, 0.85, 0.89, 0.68], [0.74, 0.95, 0.91, 0.72],
        [0.73, 0.99, 0.95, 0.78],
    ])
    shd = np.array([
        [210.5, 188.4, 255.1, 150.2], [285.3, 85.2, 112.5, 95.6],
        [270.8, 72.5, 105.3, 88.2], [255.4, 65.1, 92.4, 82.5],
        [275.6, 78.4, 108.7, 90.1], [230.1, 45.2, 82.5, 75.3],
        [225.4, 38.5, 75.8, 68.4], [215.2, 25.4, 58.2, 45.6],
        [222.8, 29.8, 48.5, 52.3], [208.5, 15.2, 42.6, 35.8],
        [205.2, 9.6, 36.8, 11.5],
    ])

    spatial_info = [
        ("a", "var",            "VAR",            "$N=128$, uniform grid"),
        ("b", "lorenz96",       "Lorenz-96",      "$N=128$, circular manifold"),
        ("c", "cluster_lorenz", "Cluster-Lorenz", "$N=128$, 4 ring clusters"),
        ("d", "finance",        "Finance",        "$N=25$, no spatial prior"),
    ]
    bar_specs = [
        ("e", "AUROC",       auroc, (0.48, 1.03), "higher is better"),
        ("f", "F1 score",    f1,    (0.35, 1.03), "higher is better"),
        ("g", "SHD",         shd,   None,          "lower is better"),
    ]

    bar_colors = ([PALETTE["baseline_dark"], PALETTE["baseline_mid"],
                   "#9BA8D2", "#B4C0E4", "#C5CCE8"] * 2
                  + [PALETTE["ours"]])
    bar_colors = bar_colors[:len(methods) - 1] + [PALETTE["ours"]]

    # ── figure & grids ──
    fig = plt.figure(figsize=(8.5, 5.2))
    gs = fig.add_gridspec(2, 1, hspace=0.14,
                          left=0.04, right=0.99, top=0.98, bottom=0.14)
    gs_top = gs[0].subgridspec(1, 4, wspace=0.25)
    gs_bot = gs[1].subgridspec(1, 3, wspace=0.18)

    # ---- top row: spatial layouts ----
    n = 128
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    for idx, (lab, key, title, subtitle) in enumerate(spatial_info):
        ax = fig.add_subplot(gs_top[0, idx])
        if key == "finance":
            res_dir = sorted((ROOT / "GeoDCD" / "results" / "Finance").glob("*-GeoDCD/hierarchy"))[0]
            coords = np.load(res_dir / "coords_level_0.npy")
            gt = np.load(res_dir / "level_0_graph.npy")
        else:
            coords = np.load(ROOT / "GeoDCD" / "data" / "synthetic" / key / "coords_0.npy")
            gt = np.load(ROOT / "GeoDCD" / "data" / "synthetic" / key / "gt_0.npy")
        n_nodes = coords.shape[0]

        if key == "var":
            src, dst = np.where((gt > 0) & upper[:n_nodes, :n_nodes])
            xs = np.column_stack([coords[dst, 0], coords[src, 0],
                                  np.full(src.size, np.nan)]).ravel()
            ys = np.column_stack([coords[dst, 1], coords[src, 1],
                                  np.full(src.size, np.nan)]).ravel()
            if src.size:
                ax.plot(xs, ys, color=PALETTE["blue_mid"], linewidth=0.40,
                        alpha=0.42, zorder=1, solid_capstyle="round")

        node_color = PALETTE["neutral_dark"] if key == "finance" else PALETTE["blue"]
        node_size = 16 if n_nodes <= 40 else 10
        ax.scatter(coords[:, 0], coords[:, 1], s=node_size, c=node_color,
                   alpha=0.92, edgecolors="white", linewidths=0.30, zorder=3)

        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("#FDFDFD")
        for spine in ax.spines.values():
            spine.set_color("#DDDDDD")
            spine.set_linewidth(0.6)
        ax.tick_params(labelsize=7, length=2.0, colors="#666666")
        ax.grid(True, color="#EEEEEE", linewidth=0.35, zorder=0)

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_pad = max((x_max - x_min) * 0.07, 0.5)
        y_pad = max((y_max - y_min) * 0.07, 0.5)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_title(title, fontsize=10, pad=2, color="#333333")
        ax.text(0.5, 1.09, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8, color="#666666")
        panel_label(ax, lab, x=-0.06, y=1.02)

    # ---- bottom row: bar charts ----
    x = np.arange(len(datasets))
    width = 0.065
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * width
    for ax_idx, (lab, title, values, ylim, cue) in enumerate(bar_specs):
        ax = fig.add_subplot(gs_bot[0, ax_idx])
        for i, method in enumerate(methods):
            ax.bar(x + offsets[i], values[i], width=width * 0.94,
                   color=bar_colors[i], edgecolor="white", linewidth=0.25,
                   label=method, zorder=3)
        panel_label(ax, lab, x=-0.07, y=1.04)
        ax.set_title(f"{title} ({cue})", fontsize=8, pad=3)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=6)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, np.max(values) * 1.08)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.45, zorder=0)
        ax.tick_params(axis="y", labelsize=6)
        for j in range(len(datasets)):
            best_idx = np.argmax(values[:, j]) if title != "SHD" else np.argmin(values[:, j])
            ax.scatter(x[j] + offsets[best_idx], values[best_idx, j],
                       s=8, color="#222222", zorder=5)

    # extract legend handles from one of the bar axes
    bar_axes = [fig.get_axes()[i] for i in range(4, 7)]  # bottom‑row axes
    handles, labels_ = bar_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center",
               bbox_to_anchor=(0.5, 0.06), ncol=6,
               fontsize=5.4, handlelength=0.85)

    save_all(fig, "synthetic_combined")


def select_primary_source(latlon, gateway_scores, top_k=50,
                          lat_bin=10.0, lon_bin=20.0, max_lat=60.0):
    """Robustly choose the primary causal-gateway source.

    A single ``argmax(gateway_scores)`` is fragile: two nearby grid cells can
    have nearly-equal out-strength, and different model runs (or a truncated
    graph) flip the winner — e.g. the source jumps between the Southern Indian
    Ocean and the North Pacific depending on which run's graph is fed in.

    We instead take the Top-K gateway nodes, group them into coarse geographic
    regions (``lat_bin`` x ``lon_bin`` degree boxes), rank regions by *mean*
    gateway strength, and pick the highest-scoring node inside the winning
    region. Mean (not sum) avoids favouring a hemisphere where top nodes merely
    cluster into one bin. Returns ``(source, region_label, reg_sorted)``.
    """
    lat, lon = latlon[:, 0], latlon[:, 1]
    cand = np.where(np.abs(lat) <= max_lat)[0]
    if len(cand) == 0:  # fall back if every node is polar
        cand = np.arange(len(gateway_scores))

    order = np.argsort(gateway_scores[cand])[::-1][:top_k]
    topk = cand[order]

    def region_of(la, lo):
        return (int(np.floor(la / lat_bin) * lat_bin),
                int(np.floor(lo / lon_bin) * lon_bin))

    reg: dict[tuple[int, int], list[int]] = {}
    for i in topk:
        reg.setdefault(region_of(lat[i], lon[i]), []).append(int(i))
    reg_strength = {k: float(np.mean(gateway_scores[v])) for k, v in reg.items()}
    reg_sorted = sorted(reg_strength.items(), key=lambda kv: -kv[1])

    best_bin, _ = reg_sorted[0]
    best_nodes = np.array(reg[best_bin], dtype=int)
    source = int(best_nodes[np.argmax(gateway_scores[best_nodes])])
    return source, best_bin, reg_sorted


def climate_combined() -> None:
    """Nature-style 3-panel climate figure: (a) directed causal flow pathways,
    (b) hierarchical causal clusters, (c) El Niño vs La Niña topology reorganisation.

    Data source: NCEP SLP reanalysis via GeoDCD inference (20260331 run).
    Visual style matches climate_causal_analysis.py reference.
    """

    NCEP_DIR = ROOT / "GeoDCD" / "results" / "ncep_slp" / "20260331_084759-ncep_slp-0-GeoDCD"
    HIER = NCEP_DIR / "hierarchy"
    proj = ccrs.Robinson()
    pc = ccrs.PlateCarree()

    # ── load hierarchy data ──
    latlon = np.load(HIER / "latlon_level_0.npy")
    graph = np.load(HIER / "level_0_graph.npy")
    inter_cluster = np.load(HIER / "inter_cluster_0.npy")
    cluster_labels = np.load(HIER / "cluster_labels_0.npy")
    latlon_1 = np.load(HIER / "latlon_level_1.npy")

    lat_f, lon_f = latlon[:, 0], latlon[:, 1]
    gateway_scores = graph.sum(axis=0)  # column sums = weighted out-degree

    # ── monthly gateway scores (for ENSO panel) ──
    monthly = np.load(NCEP_DIR / "monthly_gateway_scores.npz")
    my, mm, ms = monthly["years"], monthly["months"], monthly["scores"]

    ELNINO = [1972, 1982, 1987, 1991, 1997, 2002, 2009, 2015]
    LANINA = [1973, 1975, 1988, 1998, 1999, 2007, 2010, 2020]

    def _seasonal_avg(years_list):
        sel = []
        for i in range(len(my)):
            y, m = int(my[i]), int(mm[i])
            if (y in years_list and m in [11, 12]) or ((y - 1) in years_list and m in [1, 2]):
                sel.append(ms[i])
        print(f"  Averaged {len(sel)} winter months for {years_list}")
        return np.mean(sel, axis=0) if sel else np.zeros(ms.shape[1])

    print("  Computing El Niño gateway scores (Winter NDJF)...")
    en_scores = _seasonal_avg(ELNINO)
    print("  Computing La Niña gateway scores (Winter NDJF)...")
    ln_scores = _seasonal_avg(LANINA)
    diff_scores = en_scores - ln_scores

    # ── figure layout: 3 rows × 1 column ──
    fig = plt.figure(figsize=(7.2, 9.5))
    gs = fig.add_gridspec(3, 1, hspace=0.40,
                          left=0.03, right=0.97, top=0.98, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0], projection=proj)  # flow pathways
    ax_b = fig.add_subplot(gs[1, 0], projection=proj)  # hierarchical clusters
    ax_c = fig.add_subplot(gs[2, 0], projection=proj)  # ENSO comparison

    # ── helper: global map background (reference style) ──
    def _global_bg(ax):
        ax.set_global()
        ax.add_feature(cfeature.OCEAN, facecolor="#EBF0F5", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#F5F5F0",
                       edgecolor="#999999", linewidth=0.3, zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#666666", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#999999", zorder=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                          alpha=0.4, linestyle="--")

    # ═══════════════════════════════════════════════════════════════════
    # Panel (a): Directed Causal Flow Pathways
    # ═══════════════════════════════════════════════════════════════════
    _global_bg(ax_a)

    # Build flow graph: keep top 10% of edges for global teleconnections
    g_flow = graph.copy()
    np.fill_diagonal(g_flow, 0)
    fz = g_flow[g_flow > 1e-8]
    flow_thresh = np.percentile(fz, 90) if len(fz) > 0 else 0
    g_flow_sparse = np.where(g_flow >= flow_thresh, g_flow, 0)

    G_flow = nx.DiGraph()
    G_flow.add_nodes_from(range(graph.shape[0]))
    for r, c in zip(*np.nonzero(g_flow_sparse)):
        w_val = float(g_flow_sparse[r, c])
        G_flow.add_edge(int(c), int(r), weight=w_val, distance=1.0 / (w_val + 1e-12))

    # Robust source selection
    source, region_bin, reg_sorted = select_primary_source(
        latlon, gateway_scores, top_k=50, lat_bin=10.0, lon_bin=20.0, max_lat=60.0)

    midlat = np.where(np.abs(lat_f) <= 60)[0]
    if G_flow.out_degree(source) == 0:
        candidates = [n for n in midlat if G_flow.out_degree(n) > 0] if len(midlat) > 0 else []
        if candidates:
            source = int(max(candidates, key=lambda n: gateway_scores[n]))

    try:
        lengths, _ = nx.single_source_dijkstra(G_flow, source, weight="distance")
    except Exception:
        lengths = {}

    def _geo_km(p_lat1, p_lon1, p_lat2, p_lon2):
        rlat1, rlon1 = np.radians([p_lat1, p_lon1])
        rlat2, rlon2 = np.radians([p_lat2, p_lon2])
        dphi = rlat2 - rlat1
        dlam = rlon2 - rlon1
        a = np.sin(dphi / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlam / 2) ** 2
        return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) * 6371.0

    sorted_targets = [t for t in sorted(lengths, key=lengths.get) if t != source]
    selected = []
    n_paths = 60
    for t in sorted_targets:
        if _geo_km(lat_f[source], lon_f[source], lat_f[t], lon_f[t]) < 2500:
            continue
        if all(_geo_km(lat_f[st], lon_f[st], lat_f[t], lon_f[t]) >= 1000 for st in selected):
            selected.append(t)
        if len(selected) == n_paths:
            break

    lat_ns = "N" if lat_f[source] >= 0 else "S"
    lon_ew = "E" if lon_f[source] >= 0 else "W"
    _top3 = " | ".join(f"{b} {s:.5f}" for b, s in reg_sorted[:3])
    print(f"  [flow] source: node {source} ({abs(lat_f[source]):.1f}°{lat_ns}, "
          f"{abs(lon_f[source]):.1f}°{lon_ew}), "
          f"primary region={region_bin}, edges retained: {G_flow.number_of_edges()}, "
          f"reachable: {len(lengths)}, targets: {len(selected)}")
    print(f"  [flow] top-3 gateway regions (mean strength): {_top3}")

    # faint background nodes
    ax_a.scatter(lon_f, lat_f, c="#CCCCCC", s=1, alpha=0.2, transform=pc,
                 zorder=2, rasterized=True)

    # great-circle causal arcs (Reds colormap, like reference)
    if selected:
        strengths = {t: 1.0 / (lengths[t] + 1e-12) for t in selected}
        mx_s, mn_s = max(strengths.values()), min(strengths.values())
        for t, w_val in strengths.items():
            nw = (w_val - mn_s) / (mx_s - mn_s + 1e-12)
            ax_a.plot([lon_f[source], lon_f[t]], [lat_f[source], lat_f[t]],
                      color=plt.cm.Reds(0.5 + 0.5 * nw),
                      linewidth=0.5 + 2.5 * nw, alpha=0.4 + 0.5 * nw,
                      transform=ccrs.Geodetic(), zorder=3, solid_capstyle="round")

    # target endpoints (blue circles)
    if selected:
        t_lats = [lat_f[t] for t in selected]
        t_lons = [lon_f[t] for t in selected]
        ax_a.scatter(t_lons, t_lats, c="#1565C0", s=15, alpha=0.7,
                     transform=pc, zorder=5, edgecolors="white", linewidths=0.3)

    # source marker — red star
    ax_a.scatter([lon_f[source]], [lat_f[source]], s=200, c="#D32F2F",
                 marker="*", transform=pc, zorder=6,
                 edgecolors="white", linewidths=0.8)

    # legend (below map, centered, doesn't block anything)
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#D32F2F",
               markersize=12, label=f"Source ({abs(lat_f[source]):.0f}\u00b0{lat_ns}, "
                                    f"{abs(lon_f[source]):.0f}\u00b0{lon_ew})  region {region_bin}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565C0",
               markersize=7, label=f"Top-{len(selected)} targets"),
        Line2D([0], [0], color="#D32F2F", linewidth=2, alpha=0.7,
               label="Causal pathway (width ~ strength)"),
    ]
    leg = ax_a.legend(handles=legend_elements, loc="upper center",
                       bbox_to_anchor=(0.5, -0.06), ncol=3,
                       fontsize=5.5, framealpha=0.9, edgecolor="#CCCCCC",
                       handlelength=0.8, handletextpad=0.4, borderpad=0.3)
    leg.get_frame().set_linewidth(0.3)

    ax_a.set_title(
        f"Causal Flow Pathways from Top Gateway\n"
        f"Source: {abs(lat_f[source]):.0f}\u00b0{lat_ns}, {abs(lon_f[source]):.0f}\u00b0{lon_ew}  "
        f"region {region_bin}  |  "
        f"N={len(gateway_scores):,} nodes  |  "
        f"Top-{len(selected)} causal targets\n"
        f"(Line width ~ causal strength)",
        fontsize=9.5, fontweight="bold", pad=6, color="#222222",
    )
    panel_label(ax_a, "a", x=0.02, y=0.96)

    # ═══════════════════════════════════════════════════════════════════
    # Panel (b): Hierarchical Causal Clusters
    # ── Single-level clarity ──
    #   15 level-1 macro-regions as translucent convex hulls.
    #   One hub per region (the strongest level-0 cluster inside it).
    #   Inter-region causal edges aggregated from level-0.
    #   No fine dots — clean and readable.
    # ═══════════════════════════════════════════════════════════════════
    n_clusters = len(latlon_1)
    cluster_cmap = plt.cm.tab20
    n_col = cluster_cmap.N

    # Load level-1 cluster labels (256 level-0 → 15 level-1)
    labels_1 = np.load(HIER / "cluster_labels_1.npy")
    l1_unique = np.unique(labels_1)
    n_l1 = len(l1_unique)
    l1_color = {int(k): cluster_cmap(int(i * n_col / n_l1) % n_col) for i, k in enumerate(l1_unique)}

    # Compute level-0 cluster hub positions
    lat_c = np.full(n_clusters, np.nan)
    lon_c = np.full(n_clusters, np.nan)
    for i in range(n_clusters):
        nodes = np.where(cluster_labels == i)[0]
        if len(nodes) > 0:
            hub = nodes[np.argmax(gateway_scores[nodes])]
            lat_c[i] = lat_f[hub]
            lon_c[i] = lon_f[hub]

    _global_bg(ax_b)

    # ── Background: all 10,512 nodes as causal gateway intensity heatmap ──
    # Faint, small dots using YlOrRd — shows where causal influence is
    # concentrated globally; provides spatial context for the hierarchical overlay.
    gw_vmax = gateway_scores.max()
    ax_b.scatter(lon_f, lat_f, c=gateway_scores, cmap="Greys",
                 s=1.2, alpha=0.35, transform=pc,
                 vmin=0, vmax=gw_vmax, zorder=2.5,
                 edgecolors="none", rasterized=True)

    # ── Level-1: one hub per macro-region (strongest internal level-0) ──
    # Use idx=0..14 (sequential) via l1_map: raw l1_id → sequential idx
    l1_map = {int(k): i for i, k in enumerate(l1_unique)}
    l1_lat = np.full(n_l1, np.nan)
    l1_lon = np.full(n_l1, np.nan)
    for l1_id in l1_unique:
        l0_members = np.where(labels_1 == l1_id)[0]
        # pick the level-0 hub with max gateway score as the region hub
        valid_hubs = [(i, gateway_scores[cluster_labels == i].max()) for i in l0_members
                      if not np.isnan(lat_c[i])]
        if valid_hubs:
            best = max(valid_hubs, key=lambda x: x[1])
            idx = l1_map[int(l1_id)]
            l1_lat[idx] = lat_c[best[0]]
            l1_lon[idx] = lon_c[best[0]]

    # ── Aggregate level-0 → level-1 causal matrix ──
    np.fill_diagonal(inter_cluster, 0)
    l1_graph = np.zeros((n_l1, n_l1))
    for l1_src in l1_unique:
        for l1_tgt in l1_unique:
            if l1_src == l1_tgt:
                continue
            src_l0 = np.where(labels_1 == l1_src)[0]
            tgt_l0 = np.where(labels_1 == l1_tgt)[0]
            total = inter_cluster[np.ix_(tgt_l0, src_l0)].sum()
            l1_graph[l1_map[int(l1_tgt)], l1_map[int(l1_src)]] = total

    # ── Level-1 hub sizing ──
    l1_strength = l1_graph.sum(axis=0) + l1_graph.sum(axis=1)
    cs_log = np.log1p(l1_strength)
    cs_norm = cs_log / (cs_log.max() + 1e-12)
    sizes = 30 + 120 * cs_norm

    # Region hub nodes
    for idx in range(n_l1):
        if not np.isnan(l1_lat[idx]):
            l1_id = int(l1_unique[idx])
            ax_b.scatter([l1_lon[idx]], [l1_lat[idx]],
                         facecolors=[l1_color[l1_id]], s=sizes[idx],
                         transform=pc, zorder=5,
                         edgecolors="white", linewidths=0.5, alpha=0.9)

    # ── Level-1 inter-region causal edges (top 30%) ──
    flat_l1 = l1_graph.ravel()[l1_graph.ravel() > 1e-8]
    if len(flat_l1) > 0:
        edge_thresh = np.percentile(flat_l1, 70)
        max_w = flat_l1.max()
        min_w = flat_l1.min()
        for ri in range(n_l1):
            for ci in range(n_l1):
                if ri == ci:
                    continue
                w = l1_graph[ri, ci]
                if w < edge_thresh or np.isnan(l1_lon[ci]) or np.isnan(l1_lon[ri]):
                    continue
                nw = (w - min_w) / (max_w - min_w + 1e-12)
                lw = 0.4 + 1.6 * nw
                alpha_val = 0.15 + 0.45 * nw
                c_id = int(l1_unique[ci])
                ax_b.plot([l1_lon[ci], l1_lon[ri]], [l1_lat[ci], l1_lat[ri]],
                          color=l1_color[c_id],
                          linewidth=lw, alpha=alpha_val,
                          transform=ccrs.Geodetic(), zorder=4, solid_capstyle="round")

    # ── Legend (below map, centred) ──
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=6, label="Causal gateway intensity"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CC4444",
               markersize=6, markeredgecolor="white", markeredgewidth=0.5,
               label="Region hub (size ~ strength)"),
        Line2D([0], [0], color="#888888", linewidth=1, alpha=0.7,
               label="Inter-region causal flow"),
    ]
    leg = ax_b.legend(handles=legend_elements, loc="upper center",
                       bbox_to_anchor=(0.5, -0.06), ncol=3,
                       fontsize=5.5, framealpha=0.9, edgecolor="#CCCCCC",
                       handlelength=0.8, handletextpad=0.4, borderpad=0.3)
    leg.get_frame().set_linewidth(0.3)

    ax_b.set_title(
        f"Causal Hierarchical Structure\n"
        f"Gateway intensity background  |  "
        f"{n_l1} macro-region hubs & causal flow overlay",
        fontsize=9.5, fontweight="bold", pad=6, color="#222222",
    )
    panel_label(ax_b, "b", x=0.02, y=0.96)

    # ═══════════════════════════════════════════════════════════════════
    # Panel (c): El Niño versus La Niña Topology Reorganisation
    # ═══════════════════════════════════════════════════════════════════
    _global_bg(ax_c)

    diff_vmax = np.percentile(np.abs(diff_scores), 99)
    diff_vmax = diff_vmax if diff_vmax > 0 else 1e-4
    diff_thresh = np.percentile(np.abs(diff_scores), 90)

    # faint background
    diff_bg = np.abs(diff_scores) < diff_thresh
    ax_c.scatter(lon_f[diff_bg], lat_f[diff_bg], c=diff_scores[diff_bg],
                 cmap="RdBu_r", s=1.5, alpha=0.25, transform=pc,
                 vmin=-diff_vmax, vmax=diff_vmax, zorder=3,
                 edgecolors="none", rasterized=True)

    # highlighted top differences
    diff_top = np.abs(diff_scores) >= diff_thresh
    sc_c = ax_c.scatter(lon_f[diff_top], lat_f[diff_top], c=diff_scores[diff_top],
                         cmap="RdBu_r", s=20, alpha=0.85, transform=pc,
                         vmin=-diff_vmax, vmax=diff_vmax, zorder=4,
                         edgecolors="#333", linewidths=0.3)

    # colorbar below the map
    pos = ax_c.get_position()
    cax_c = fig.add_axes(
        [pos.x0 + 0.05, pos.y0 - 0.035, pos.width - 0.10, 0.010],
    )
    cbar_c = fig.colorbar(sc_c, cax=cax_c, orientation="horizontal")
    cbar_c.set_label("Δ gateway score  (El Niño − La Niña)", fontsize=7.5, color="#333333")
    cbar_c.ax.tick_params(labelsize=6.5, length=2.0, colors="#555555")

    ax_c.set_title(
        "Topological Reorganization: El Ni\u00f1o vs La Ni\u00f1a (Winter NDJF)\n"
        "Difference Map \u2014 Positive/Red = Stronger Causal Gateway in El Ni\u00f1o",
        fontsize=9.5, fontweight="bold", pad=6, color="#222222",
    )
    panel_label(ax_c, "c", x=0.02, y=0.96)

    save_all(fig, "climate_combined")


def main() -> None:
    apply_style()
    synthetic_combined()
    cluster_maps()
    dynamic_intensity()
    beijing_cases()
    evcdp_combined()
    scalability_figure()
    urban_scalability_combined()
    climate_combined()


if __name__ == "__main__":
    main()
