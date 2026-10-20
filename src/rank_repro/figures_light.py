"""Figures 1–6 and S1–S2: theoretical risks, scores, and isotonic geometry.

Figure 1 preserves the archived Student-t phase-grid values. The legacy
100-node grid is not uniformly converged near delta=1 and small nu; see
reproducibility notes. Other Student-t curves are recomputed with 300 nodes.
No artificial data point at delta=1 is inserted.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.ndimage import distance_transform_edt
from scipy.optimize import brentq
from scipy.special import erf
from scipy.stats import norm

from .gaussian import (solve_ls_gaussian, solve_lad_gaussian, solve_rank_gaussian,
                       score_function, rank_atom_mass, verify_self_consistency)
from .student import (classical_risks, _classical_regions, load_interpolated_risks,
                      risk_curve, RankRegressionStudentT, find_c0_boundary_delta,
                      alpha2_lad_standard_t, ls_risk)

NAMES = ("LS", "LAD", "Rank")
LEGEND_LABELS = ("least squares", "least absolute deviations", "rank regression")
COLORS = ("#C43C39", "#3767AF", "#2D7F50")
LINESTYLES = ("--", "-.", "-")

def _ticks(ax, axis, candidates, limits):
    values = [v for v in candidates
              if limits[0] * (1 - 1e-12) <= v <= limits[1] * (1 + 1e-12)]
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(values))
    target.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    target.set_minor_formatter(NullFormatter())


def _winner_labels(ax, x, y, winner, positions=None, fontsize=18):
    """Place each label inside its winning region; positions may override this."""
    positions = positions or {}
    for k, name in enumerate(NAMES):
        region = winner == k
        if not region.any():
            continue
        if name in positions:
            px, py = positions[name]
        else:
            # Distance on the regular log-coordinate grid, padded at all edges.
            distance = distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
            i, j = np.unravel_index(np.argmax(distance), region.shape)
            px, py = x[i, j], y[i, j]
        ax.text(px, py, f"{name} wins", color=COLORS[k], fontsize=fontsize,
                ha="center", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.5))


def _dimension_arrows(ax):
    for tip, start, center, label in (
        (0.96, 0.53, 0.75, "Low dim."),
        (0.04, 0.47, 0.25, "High dim."),
    ):
        ax.annotate("", xy=(-0.24, tip), xytext=(-0.24, start),
                    xycoords="axes fraction", annotation_clip=False,
                    arrowprops=dict(arrowstyle="->", color="0.25", lw=1.3))
        ax.text(-0.285, center, label, transform=ax.transAxes,
                rotation=90, ha="center", va="center", fontsize=11,
                color="0.25", clip_on=False)


def make_figure(csv_path="tmp_student_t_phase.csv", *, figsize=(7.6, 6.8),
                fine_n_df=500, fine_n_delta=500, region_alpha=0.42,
                boundary_lw=1.6, label_positions=None, winner_fontsize=18,
                legend_fontsize=12, height_ratios=(1, 1.35)):
    """Create the two-panel figure and return (fig, (ax_top, ax_bottom)).

    Example manual labels:
        label_positions={"LAD": (1.35, 20), "Rank": (5, 8), "LS": (25, 2)}
    Label positions are actual (nu, delta) coordinates, not axes fractions.
    Fonts use Matplotlib defaults. Adjust winner_fontsize, region_alpha,
    legend_fontsize, and height_ratios to tune the presentation.
    """
    x, y, log_risk = load_interpolated_risks(csv_path, fine_n_df, fine_n_delta)
    winner = np.argmin(log_risk, axis=0)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    palette = ListedColormap(COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], palette.N)
    rc = {"font.size": 12, "axes.labelsize": 14,
          "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
          "axes.facecolor": "white", "figure.facecolor": "white",
          "axes.labelcolor": "0.15", "text.color": "0.15",
          "xtick.color": "0.15", "ytick.color": "0.15",
          "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.style.context("default"), plt.rc_context(rc):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.13},
        )
        # Reserve room for both the y label and the dimension arrows.
        fig.subplots_adjust(left=0.28, right=0.98, bottom=0.10, top=0.95)
        nu_plot = np.geomspace(xmin, xmax, 1600)
        constants = classical_risks(nu_plot)
        for lo, hi, k in _classical_regions(xmin, xmax):
            ax1.axvspan(lo, hi, color=COLORS[k], alpha=region_alpha, linewidth=0)
        for k in range(3):
            ax1.plot(nu_plot, np.where(np.isfinite(constants[k]), constants[k], np.nan),
                     color=COLORS[k], ls=LINESTYLES[k], lw=2.2,
                     label=LEGEND_LABELS[k])
        ax1.legend(loc="upper right", fontsize=legend_fontsize,
                   facecolor="white", framealpha=1, edgecolor="0.75")
        ax1.set_yscale("log")
        ax1.set_ylim(1, 10)
        ax1.set_ylabel("Asymptotic variance")
        ax1.set_title(r"(a) Classical limit $(\delta\to\infty)$", loc="left", fontsize=12, pad=7)
        _ticks(ax1, "y", [1, 2, 5, 10], (1, 10))
        # A categorical mesh avoids interpolation between integer method labels.
        # The fill is rasterized in PDF; boundaries, curves, and text stay vector.
        ax2.pcolormesh(x, y, winner, cmap=palette, norm=norm, shading="nearest",
                       alpha=region_alpha, rasterized=True, zorder=1)
        for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            ri, rj, rk = log_risk[i], log_risk[j], log_risk[k]
            valid = np.isfinite(ri) & np.isfinite(rj)
            valid &= np.maximum(ri, rj) <= rk + 1e-8
            difference = np.full_like(ri, np.nan)
            np.subtract(ri, rj, out=difference, where=valid)
            values = difference[valid]
            if values.size and values.min() < 0 < values.max():
                ax2.contour(x, y, np.ma.masked_invalid(difference), levels=[0],
                            colors="0.15", linewidths=boundary_lw, zorder=4)
        ax2.set_yscale("log")
        ax2.set_ylim(ymin, ymax)
        ax2.set_xlabel(r"Degrees of freedom $\nu$")
        ax2.set_ylabel(r"$\delta=n/p$")
        ax2.set_title("(b) Efficiency phase diagram", loc="left", fontsize=12, pad=7)
        _ticks(ax2, "y", [1.001, 1.5, 2, 5, 10, 20, 50], (ymin, ymax))
        _winner_labels(ax2, x, y, winner, label_positions, winner_fontsize)
        _dimension_arrows(ax2)
        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_xlim(xmin, xmax)
            ax.set_axisbelow(True)
            ax.grid(True, which="major", color="0.80", lw=0.65, alpha=0.6)
            ax.grid(False, which="minor")
            ax.tick_params(which="minor", length=0)
        _ticks(ax2, "x", [1, 2, 5, 10, 20, 50], (xmin, xmax))
        ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    return fig, (ax1, ax2)


def _save(fig, output_dir, name):
    paths = []
    for extension in ("pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        paths.append(path)
    plt.close(fig)
    return paths


def _gaussian_efficiency():
    # Avoid loss of precision in the risk denominator at delta-1 ~ 1e-14.
    deltas = np.unique(np.r_[np.geomspace(1.001, 10, 700), np.sqrt(2)])
    rank = np.asarray([solve_rank_gaussian(d).alpha2 for d in deltas])
    lad = np.asarray([solve_lad_gaussian(d).alpha2 for d in deltas])
    ls = 1 / (deltas - 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.plot(deltas, ls / rank, lw=2.8, color=COLORS[2], label=r"Rank: $\alpha_{\rm LS}^2/\alpha_*^2$")
    ax.plot(deltas, ls / lad, lw=2.3, color=COLORS[1], label=r"LAD: $\alpha_{\rm LS}^2/\alpha_{\rm LAD}^2$")
    lower = deltas[deltas < np.sqrt(2)]
    ax.plot(lower, (3 * lower - np.pi) / (np.pi * (lower - 1)),
            color=COLORS[0], ls=":", lw=2, label="Inactive formula (extension)")
    d = np.sqrt(2)
    ax.scatter([d], [(3*d-np.pi)/(np.pi*(d-1))], s=45, color="black", zorder=5,
               label=r"$\delta_{\rm c}=\sqrt{2}$")
    ax.axhline(3/np.pi, color="0.4", ls="--", lw=1.5, label=r"$3/\pi$")
    ax.axhline(2/np.pi, color="0.5", ls=":", lw=1.5, label=r"$2/\pi$")
    ax.set(xscale="log", xlim=(1, 10), ylim=(0, 1.02), xlabel=r"$\delta=n/p$", ylabel="ARE")
    ax.legend(loc="lower right", facecolor="white", fontsize=9)
    return fig


def _isotonic_figure():
    """Proximal example: c=1/sqrt(2), Z~N(0,1/2), so V_c~N(0,1)."""
    blue, orange, gray = "#0072B2", "#D55E00", "#777777"
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.6), sharex="col", sharey="col")
    fig.subplots_adjust(left=.08, right=.98, bottom=.10, top=.87, wspace=.22, hspace=.26)
    fig.text(.28, .97, "Isotonic projection in quantile space", ha="center", va="top", weight="semibold")
    fig.text(.77, .97, "Proximal score in residual space", ha="center", va="top", weight="semibold")
    q = np.linspace(.003, .997, 3501)
    quantile = norm.ppf(q)
    v = np.linspace(-3, 3, 10001)
    for row, tau in enumerate((.45, .90)):
        cutoff = 0.0 if 4*tau*norm.pdf(0) <= 1 else brentq(lambda x: 4*tau*(norm.cdf(x)-.5)-x, 1e-7, 4*tau)
        raw = quantile - 4*tau*(q-.5)
        fit = np.where(np.abs(quantile) < cutoff, 0, raw)
        unconstrained = 4*tau*(norm.cdf(v)-.5)
        score = np.where(np.abs(v) < cutoff, v, unconstrained)
        left, right = axes[row]
        left.plot(q, quantile, color=gray, ls=":", lw=1.5, label=r"quantile $Q_c$")
        left.plot(q, raw, color=orange, ls="--", lw=1.8, label=r"shifted quantile $Q_c-\tau\omega$")
        left.plot(q, fit, color=blue, lw=2.2, label=r"isotonic fit $h_{c,\tau}$")
        right.plot(v, v, color=gray, ls=":", lw=1.4, label="identity")
        right.plot(v, score, color=blue, lw=2.5, label=r"proximal score $\psi_{c,\tau}$")
        right.plot(v, unconstrained, color=orange, ls="--", lw=1.5, label=r"CDF score $4\tau(\Phi(v)-1/2)$")
        if cutoff:
            left.axvspan(norm.cdf(-cutoff), norm.cdf(cutoff), color="#E4D9F0", alpha=.65)
            right.axvspan(-cutoff, cutoff, color="#E4D9F0", alpha=.65)
            left.text(.5, -1.30, "central interval projected to 0", ha="center", fontsize=9)
            right.text(0, -1.78, r"identity core: $\psi_{c,\tau}(v)=v$", ha="center", fontsize=9)
        arrow_q = .85
        top = norm.ppf(arrow_q)
        bottom = 0 if abs(top)<cutoff else top - 4*tau*(arrow_q-.5)
        left.annotate("", xy=(arrow_q, top), xytext=(arrow_q, bottom),
                      arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
        left.text(arrow_q+.02, (top+bottom)/2, r"$\psi_{c,\tau}(Q_c(q))$", fontsize=9)
        left.set(title=rf"$\tau={tau:.2f}$: {'active' if cutoff else 'inactive'} constraint", ylim=(-1.5, 1.5), xlim=(0,1))
        right.set(title=rf"$C_{{c,\tau}}={cutoff:.3f}$", xlim=(-3,3), ylim=(-2,2), ylabel="score")
        for ax in (left, right):
            ax.axhline(0, color="0.7", lw=.6)
        if row == 0:
            left.legend(loc="lower right", fontsize=9, facecolor="white")
            right.legend(loc="lower right", fontsize=8, facecolor="white")
    axes[1,0].set_xlabel(r"quantile level $q$")
    axes[1,1].set_xlabel(r"effective residual $v$")
    return fig


def _effective_scores():
    deltas = np.round(np.arange(1.1,1.81,.1), 1)
    x = np.linspace(-4.8, 4.8, 2001)
    fig, axes = plt.subplots(2,4,figsize=(16.5,8.3),sharex=True,sharey=True)
    diagnostics = []
    for ax, delta in zip(axes.flat,deltas):
        for name, color, solver, label in zip(NAMES,COLORS,
                (solve_ls_gaussian,solve_lad_gaussian,solve_rank_gaussian),
                (r"$\psi_*^{\rm LS}$",r"$\psi_*^{\rm LAD}$",r"$\psi_*$")):
            solution = verify_self_consistency(solver(delta))
            ax.plot(x, score_function(solution)(x), color=color, lw=2.7, label=label)
            diagnostics.append({"delta":float(delta), "estimator":name,
                                "residual_alpha2": solution.residual_alpha2,
                                "residual_alpha":solution.residual_alpha})
            if name == "Rank":
                c = solution.c
                cx = [0] if c == 0 else [-c,c]
                ax.scatter(cx,cx,color=color,s=95,edgecolors="white",linewidths=1,zorder=5)
        ax.set(title=rf"$\delta={delta:.1f}$",xlim=(-4.8,4.8),ylim=(-4.5,4.5))
        ax.axhline(0,color="0.4",lw=.7)
        ax.axvline(0,color="0.4",lw=.7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=3,fontsize=18,facecolor="white")
    fig.supxlabel(r"$v$")
    fig.supylabel("effective score")
    fig.tight_layout(rect=(.025,.025,1,.92))
    return fig, diagnostics


def _atom_figure():
    dc = np.sqrt(2)
    delta = np.unique(np.r_[np.linspace(1.035,dc-.02,250), dc-np.geomspace(.02,1e-7,200), np.linspace(dc,2.3,250)])
    mass = np.array([rank_atom_mass(d) for d in delta])
    fig, ax = plt.subplots(figsize=(7.2,4.8),layout="constrained")
    ax.plot(delta,mass,color=COLORS[2],lw=2.7,label="Rank")
    ax.plot(delta,1/delta,color=COLORS[1],lw=2.7,ls="--",label=r"LAD: $1/\delta$")
    ax.plot(delta,np.zeros_like(delta),color=COLORS[0],lw=2.7,ls=":",label="LS: 0")
    ax.axvline(dc,color="0.6",ls="-.",lw=1.5)
    ax.text(dc+.02,.89,r"$\delta_{\rm c}=\sqrt{2}$",fontsize=16)
    ax.set(xlabel=r"$\delta=n/p$",ylabel=r"$p_0(\delta)$",ylim=(-.035,1.015))
    ax.legend(facecolor="white",loc="upper right")
    return fig


def _student_figures():
    """Compute once, then share risk curves across the three comparisons."""
    deltas = np.geomspace(1.01,10,100)
    fig_rank, ax_rank = plt.subplots(figsize=(6.4,4.8),layout="constrained")
    fig_lad, ax_lad = plt.subplots(figsize=(6.4,4.8),layout="constrained")
    fig_comparison, ax_comparison = plt.subplots(figsize=(6.4,4.8),layout="constrained")
    colors_all = plt.cm.viridis_r(np.linspace(0,1,10))
    colors_finite = plt.cm.viridis_r(np.linspace(0,1,8))
    rows = []
    for df in range(1,11):
        curve = risk_curve(df,deltas,order=300)
        for row in curve:
            rows.append({"df":df,"delta":row[0],"alpha2_LS":row[1],"alpha2_LAD":row[2],"alpha2_rank":row[3],"rank_cutoff":row[4],"quadrature_order":300})
        label = rf"$\nu={df}$"
        ax_comparison.plot(deltas,curve[:,3]/curve[:,2],color=colors_all[df-1],lw=2.5,label=label)
        if df >= 3:
            color = colors_finite[df-3]
            ax_rank.plot(deltas,curve[:,1]/curve[:,3],color=color,lw=2.5,label=label)
            ax_lad.plot(deltas,curve[:,1]/curve[:,2],color=color,lw=2.5,label=label)
        solver = RankRegressionStudentT(df,n_quad=300)
        boundary = find_c0_boundary_delta(solver,delta_lo=float(deltas[0]),delta_hi=10)
        if boundary is not None:
            rank = solver.solve(boundary)["alpha2"]
            lad = alpha2_lad_standard_t(boundary,df,n_quad=300)
            ax_comparison.scatter([boundary],[rank/lad],s=50,color=colors_all[df-1],edgecolors="0.5",zorder=5)
            if df >= 3:
                ax_rank.scatter([boundary],[ls_risk(boundary,df)/rank],s=50,color=colors_finite[df-3],edgecolors="0.5",zorder=5)
    for ax, ylabel in ((ax_rank,r"$\alpha_{\rm LS}^2/\alpha_*^2$"),
                       (ax_lad,r"$\alpha_{\rm LS}^2/\alpha_{\rm LAD}^2$"),
                       (ax_comparison,r"$\alpha_*^2/\alpha_{\rm LAD}^2$")):
        ax.set(xscale="log",yscale="log",xlim=(1,10),xlabel=r"$\delta=n/p$",ylabel=ylabel)
        ax.axhline(1,color="0.5",ls=":",lw=1.5)
        ax.legend(ncol=2,facecolor="white",fontsize=9)
    return (fig_rank,fig_lad,fig_comparison), rows


def generate_light_figures(data_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all light figures in PDF+PNG; preserve the original phase cache.

    The deterministic Student-t curves take a few minutes on one CPU.
    Gaussian score residuals and computed Student-t curves are also exported.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    paths = []
    with plt.style.context("ggplot"), plt.rc_context({"pdf.fonttype":42,"ps.fonttype":42,
            "font.family":"DejaVu Sans","font.size":11,"axes.labelsize":13,
            "axes.titlesize":14,"figure.facecolor":"white"}):
        meta_path = data_dir / "unpenalized_student_t.metadata.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if metadata.get("complete") is False:
            raise ValueError("The Student-t phase grid has unresolved numerical checks; inspect its metadata before plotting.")
        if not metadata.get("recomputation"):
            warnings.warn("Figure 1 uses the supplied Student-t grid; the bundled 100-node archive is quadrature-sensitive at low df near delta=1. See the reproduction notes.",UserWarning,stacklevel=2)
        fig,_ = make_figure(data_dir/"unpenalized_student_t.csv")
        paths += _save(fig,output_dir,"fig01_student_efficiency")
        paths += _save(_gaussian_efficiency(),output_dir,"fig02_gaussian_efficiency")
        paths += _save(_isotonic_figure(),output_dir,"fig04_isotonic_projection")
        fig, diagnostics = _effective_scores()
        paths += _save(fig,output_dir,"fig05_effective_scores")
        paths += _save(_atom_figure(),output_dir,"fig06_proximal_atom")
        figures, rows = _student_figures()
        for fig,name in zip(figures,("fig03_student_rank_vs_ls","figS01_student_lad_vs_ls","figS02_student_rank_over_lad")):
            paths += _save(fig,output_dir,name)
        pd.DataFrame(rows).to_csv(output_dir/"student_risk_curves.csv",index=False)
        (output_dir/"gaussian_diagnostics.json").write_text(json.dumps(diagnostics,indent=2)+"\n")
    return paths
