#!/usr/bin/env python3
"""Generate NMI-style replacement figures for the GeoDCD manuscript."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - fallback only for local dependency issues.
    ccrs = None
    cfeature = None


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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=6, fontsize=5.6, handlelength=0.9)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.28, wspace=0.14)
    save_all(fig, "synthetic_large_bars")


def cluster_maps() -> None:
    names = [("a", "SD", "San Diego"), ("b", "BJ", "Beijing"), ("c", "GLA", "Greater Los Angeles"), ("d", "GBA", "Bay Area")]
    proj = ccrs.PlateCarree() if ccrs is not None else None
    fig = plt.figure(figsize=(7.2, 5.35))
    positions = [
        [0.05, 0.545, 0.42, 0.39],
        [0.545, 0.545, 0.42, 0.39],
        [0.05, 0.075, 0.42, 0.39],
        [0.545, 0.075, 0.42, 0.39],
    ]
    cmap = plt.get_cmap("tab20")
    for idx, (lab, key, title) in enumerate(names):
        ax = fig.add_axes(positions[idx], projection=proj) if proj is not None else fig.add_axes(positions[idx])
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
    fig = plt.figure(figsize=(7.2, 5.1), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, wspace=0.02)
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

    def add_links(rows, cols, weights, color):
        order = np.argsort(weights)
        for r, c, w in zip(rows[order], cols[order], weights[order]):
            x1, y1 = xy[c]
            x2, y2 = xy[r]
            alpha = 0.30 + 0.55 * norm(w)
            lw = 0.45 + 1.80 * norm(w)
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-",
                                         linewidth=lw, color=color,
                                         alpha=alpha, transform=proj, zorder=4))

    add_links(e_rows, e_cols, e_weights, PALETTE["energy"])
    add_links(t_rows, t_cols, t_weights, PALETTE["traffic"])
    ax.set_title("Cross-modal causal links", fontsize=8, pad=3)
    legend = [
        Line2D([0], [0], color=PALETTE["energy"], lw=1.6, label="Energy → Traffic"),
        Line2D([0], [0], color=PALETTE["traffic"], lw=1.6, label="Traffic → Energy"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=5.5, framealpha=0.9)


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
    ax.legend(handles=handles, fontsize=5.5, loc="upper left", frameon=False,
              handlelength=1.2, borderpad=0.3)


def _draw_impact(ax, xy, vals, cmap_name, title, vmin, vmax, proj) -> None:
    add_map_background(ax, xy, pad=0.06)
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
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.06,
                          left=0.05, right=0.965, top=0.96, bottom=0.05)
    ax_a = fig.add_subplot(gs[0, 0], projection=proj) if proj is not None else fig.add_subplot(gs[0, 0])
    _draw_evcdp_map(ax_a, xy, graph, t_idx, e_idx, proj)
    panel_label(ax_a, "a", x=0.5, y=-0.18, ha="center", va="top")
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_temporal(ax_b, e_idx, t_idx)
    ax_b.set_box_aspect(1 / 2.388)  # match the w/h ratio that map panels a/c/d naturally adopt
    panel_label(ax_b, "b", x=0.5, y=-0.18, ha="center", va="top")

    et_vals = graph[e_idx[:, None], t_idx].sum(axis=0)  # per-traffic-TAZ E→T
    te_vals = graph[t_idx[:, None], e_idx].sum(axis=0)  # per-energy-TAZ T→E
    vmin = float(min(et_vals.min(), te_vals.min()))
    vmax = float(max(et_vals.max(), te_vals.max()))

    ax_c = fig.add_subplot(gs[1, 0], projection=proj) if proj is not None else fig.add_subplot(gs[1, 0])
    _draw_impact(ax_c, xy, et_vals, "Oranges", "Energy → Traffic\ncumulative impact", vmin, vmax, proj)
    panel_label(ax_c, "c", x=0.5, y=-0.18, ha="center", va="top")

    ax_d = fig.add_subplot(gs[1, 1], projection=proj) if proj is not None else fig.add_subplot(gs[1, 1])
    _draw_impact(ax_d, xy, te_vals, "Greens", "Traffic → Energy\ncumulative impact", vmin, vmax, proj)
    panel_label(ax_d, "d", x=0.5, y=-0.18, ha="center", va="top")

    save_all(fig, "evcdp_cross_modal")


def main() -> None:
    apply_style()
    synthetic_bar_figure()
    cluster_maps()
    dynamic_intensity()
    beijing_cases()
    evcdp_combined()


if __name__ == "__main__":
    main()
