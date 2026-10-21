"""Supplementary Figures S3 and S4: unpenalized Cauchy and Laplace ARE.

Run: python3 -m rank_repro.figures_are --output-dir build/are
The design convention is n/p -> delta>1, with entries N(0,1/p).
Both error distributions have location zero and scale one.
All calculations use deterministic one-dimensional quadrature, not simulation.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfcx, ndtr, voigt_profile

SQRT_PI = math.sqrt(math.pi)
PHI0 = 1 / math.sqrt(2 * math.pi)
FIGURE_NAMES = {"cauchy": "figS03_cauchy_are", "laplace": "figS04_laplace_are"}


class SmoothedNoise:
    """Density of W=G+sZ; s=1/alpha keeps the Gaussian scale equal to one."""

    def __init__(self, name: str, s: float, tol: float = 2e-12):
        self.name, self.s, self.tol = name, float(s), tol

    def pdf(self, x):
        x = abs(float(x))
        s = self.s
        if s == 0:
            return PHI0 * math.exp(-x*x/2)
        if self.name == "cauchy":
            return float(voigt_profile(x, 1.0, s))
        a = 1 / s
        gaussian = math.exp(-x*x/2)
        # erfcx avoids subtracting O(1/s^2) logarithms near delta=1.
        if x <= a:
            first = .5*gaussian*float(erfcx((a-x)/math.sqrt(2)))
        else:
            first = math.exp(a*a/2-a*x)*float(ndtr(x-a))
        second = .5*gaussian*float(erfcx((a+x)/math.sqrt(2)))
        return (first + second) / (2*s)

    def integral(self, f, lo, hi):
        return quad(f, lo, hi, epsabs=self.tol, epsrel=self.tol, limit=200)[0]

    def central_probability(self, x):
        if x == 0:
            return 0.0
        # Integrating the positive density avoids cancellation in the CDF near 0.
        return 2 * self.integral(self.pdf, 0, x)

    def central_second_moment(self, x):
        return 2 * self.integral(lambda v: v*v*self.pdf(v), 0, x)

    def squared_density_tail(self, x):
        return self.integral(lambda v: self.pdf(v)**2, x, np.inf)

    def density_at_zero(self):
        if self.s == 0:
            return PHI0
        if self.name == "cauchy":
            return PHI0 * float(erfcx(self.s / math.sqrt(2)))
        return float(erfcx(1 / (math.sqrt(2)*self.s))) / (2*self.s)

    def squared_density_integral(self):
        s = self.s
        if s == 0:
            return 1 / (2*SQRT_PI)
        if self.name == "cauchy":
            return float(erfcx(s)) / (2*SQRT_PI)
        # Parseval: I = pi^{-1} int_0^infty exp(-t^2)/(1+s^2 t^2)^2 dt.
        if s < .3:
            return self.integral(lambda t: math.exp(-t*t)/(1+s*s*t*t)**2,
                                 0, np.inf) / math.pi
        a = 1/s
        return ((1-2*a*a)*float(erfcx(a)) + 2*a/SQRT_PI) / (4*s)


def positive_root(f, start=1.0, xtol=2e-15):
    hi = float(start)
    while f(hi) <= 0:
        hi *= 2
        if hi > 1e8:
            raise RuntimeError("Could not bracket a scalar root")
    return brentq(f, 0, hi, xtol=xtol, rtol=5e-13)


def rank_at_scale(noise: SmoothedNoise, delta: float):
    """Enforce E[psi'(V)]=1/delta, then return E[(psi(V)/alpha)^2]."""
    q = 1/delta
    integral = noise.squared_density_integral()
    if q <= integral/noise.density_at_zero():
        b = q/(4*integral)  # kappa / alpha
        return 4*b*b/3, b, 0.0, q

    def derivative(c):
        if c == 0:
            return integral/noise.density_at_zero()
        p = noise.central_probability(c)
        return p + 4*c*noise.squared_density_tail(c)/p

    c = positive_root(lambda x: derivative(x)-q, start=2)
    p = noise.central_probability(c)
    # Exact integration of the CDF-score tail: int p(v)^2 dp(v).
    tail = c*c*(1-p)*(1+p+p*p)/(3*p*p)
    moment = noise.central_second_moment(c) + tail
    return moment, c/(2*p), c, derivative(c)


def lad_at_scale(noise: SmoothedNoise, delta: float):
    q = 1/delta
    k = positive_root(lambda x: noise.central_probability(x)-q, start=2)
    moment = noise.central_second_moment(k) + k*k*(1-q)
    return moment, k, noise.central_probability(k)


def solve_risk(name: str, method: str, delta: float, tol=2e-12):
    if delta <= 1:
        raise ValueError("Require delta>1")
    if delta-1 < .001:
        tol = min(tol, 5e-14)
    evaluate = rank_at_scale if method == "Rank" else lad_at_scale
    start = min(1.0, delta-1 if name == "cauchy" else math.sqrt(delta-1))
    s = positive_root(lambda x: delta*evaluate(SmoothedNoise(name, x, tol), delta)[0]-1,
                      start=start)
    result = evaluate(SmoothedNoise(name, s, tol), delta)
    alpha = 1/s
    return {
        "alpha2": alpha*alpha,
        "kappa": alpha*result[1],
        "cutoff": alpha*result[2] if method == "Rank" else 0.0,
        "score_moment_relative_residual": delta*result[0]-1,
        "score_derivative_relative_residual": delta*result[-1]-1,
    }


def critical_delta(name):
    # At the transition I/f(0)=12 I^2 in the W=G+sZ coordinates.
    def residual(s):
        noise = SmoothedNoise(name, s)
        return 1-12*noise.squared_density_integral()*noise.density_at_zero()
    s = positive_root(residual)
    noise = SmoothedNoise(name, s)
    return noise.density_at_zero()/noise.squared_density_integral()


def compute_curve(name, deltas):
    rows = []
    for i, delta in enumerate(deltas):
        rank = solve_risk(name, "Rank", delta)
        lad = solve_risk(name, "LAD", delta)
        row = {"delta": float(delta), "alpha2_rank": rank["alpha2"],
               "alpha2_LAD": lad["alpha2"],
               "alpha2_LS": 2/(delta-1) if name == "laplace" else math.inf,
               "rank_cutoff": rank["cutoff"],
               "rank_kappa": rank["kappa"], "lad_kappa": lad["kappa"]}
        row["ARE_Rank_LAD"] = row["alpha2_LAD"]/row["alpha2_rank"]
        row["ARE_Rank_LS"] = row["alpha2_LS"]/row["alpha2_rank"]
        row["ARE_LAD_LS"] = row["alpha2_LS"]/row["alpha2_LAD"]
        row["max_relative_equation_residual"] = max(
            abs(sol[key]) for sol in (rank, lad) for key in
            ("score_moment_relative_residual", "score_derivative_relative_residual"))
        rows.append(row)
        if i % 40 == 0:
            print(f"{name}: {i+1}/{len(deltas)}, delta={delta:.6g}", flush=True)
    return rows


def export_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax, dc, threshold_label_y=.04):
    ax.set_xscale("log")
    ax.set_xlim(.85, 1000)
    ticks = [1, 2, 5, 10, 30, 100, 300, 1000]
    ax.set_xticks(ticks, ["1", "2", "5", "10", "30", "100", "300", "1000"])
    ax.minorticks_off()
    ax.set_xlabel(r"$\delta$", fontsize=30, labelpad=8)
    ax.set_ylabel("ARE", fontsize=30, labelpad=10)
    ax.tick_params(labelsize=16, pad=6)
    ax.grid(True, color="white", linewidth=1.0)
    ax.axhline(1, color="0.35", lw=2.0, ls="--", zorder=1)
    ax.axvline(dc, color="0.48", lw=1.8, ls=":", zorder=1)
    ax.text(dc*1.045, threshold_label_y, rf"$\delta_{{\mathrm{{c}}}}={dc:.3f}$",
            transform=ax.get_xaxis_transform(), fontsize=15, va="bottom", rotation=90)


def draw_panel(ax, rows, name, dc, crossings):
    x = np.array([r["delta"] for r in rows])
    style_axis(ax, dc, threshold_label_y=.68 if name == "laplace" else .18)
    if name == "cauchy":
        y = np.array([r["ARE_Rank_LAD"] for r in rows])
        ax.plot(x, y, color="#2D7F50", lw=4.5,
                label=r"$\alpha_{\mathrm{LAD}}^2/\alpha_*^2$")
        ax.axhline(.75, color="#2D7F50", lw=3.0, ls=":", alpha=.9)
        ax.text(1100, .75, r"$3/4$", va="center", ha="left", fontsize=19,
                color="#2D7F50", clip_on=False)
        ax.set_ylim(.72, max(1.03, y.max()+.025))
        ax.set_title(r"Cauchy$(0,1)$", fontsize=24, pad=16)
        inset = ax.inset_axes([.49, .33, .47, .42])
        inset.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_color("0.65")
        mask = x <= 2.1
        inset.plot(x[mask], y[mask], color="#2D7F50", lw=3.2)
        inset.axhline(1, color="0.35", lw=1.5, ls="--")
        inset.axvline(dc, color="0.5", lw=1.4, ls=":")
        crossing = crossings["Rank_LAD"]
        inset.scatter([crossing], [1], color="#2D7F50", s=55, zorder=5)
        inset.annotate(rf"$\delta={crossing:.3f}$", (crossing, 1), xytext=(1.10, .975),
                       fontsize=14, arrowprops={"arrowstyle":"-", "color":"0.4", "lw":1.1})
        inset.set(xlim=(.97, 2.05), ylim=(.965, 1.018), title="High-dimensional region")
        inset.set_xticks([1, 1.5, 2])
        inset.set_yticks([.97, 1, 1.01])
        inset.tick_params(labelsize=12.5)
        inset.title.set_fontsize(16)
        inset.grid(color="0.90", linewidth=.7)
        inset.scatter([1], [1], s=55, facecolors="white", edgecolors="0.2", linewidths=1.5, zorder=7)
    else:
        specs = [
            ("ARE_Rank_LS", "#2D7F50", "-", 1.5,
             r"$\alpha_{\mathrm{LS}}^2/\alpha_*^2$", r"$3/2$"),
            ("ARE_LAD_LS", "#4C63A6", "--", 2,
             r"$\alpha_{\mathrm{LS}}^2/\alpha_{\mathrm{LAD}}^2$", r"$2$"),
            ("ARE_Rank_LAD", "#B36B35", "-.", .75,
             r"$\alpha_{\mathrm{LAD}}^2/\alpha_*^2$", r"$3/4$"),
        ]
        for key, color, ls, limit, label, text in specs:
            ax.plot(x, [r[key] for r in rows], color=color, ls=ls, lw=4.5, label=label)
            ax.axhline(limit, color=color, ls=":", lw=3.0, alpha=.85)
            ax.text(1100, limit, text, va="center", ha="left", fontsize=19,
                    color=color, clip_on=False)
        ax.set_ylim(.5, 2.15)
        ax.set_title(r"Laplace$(0,1)$", fontsize=24, pad=16)
        for key, color, xytext in [("Rank_LS", "#2D7F50", (1.65, 1.39)),
                                   ("LAD_LS", "#4C63A6", (4.4, .64)),
                                   ("Rank_LAD", "#B36B35", (21, 1.10))]:
            d = crossings[key]
            ax.scatter([d], [1], color=color, s=65, zorder=5)
            ax.annotate(rf"$\delta={d:.3f}$", (d, 1), xytext=xytext,
                        fontsize=16, color=color,
                        arrowprops={"arrowstyle":"-", "color":color, "lw":1.1})
    legend_position = {"loc":"upper left", "bbox_to_anchor":(.11, 1.02)} if name == "laplace" else {"loc":"upper right"}
    ax.legend(**legend_position,
              fontsize=24, facecolor="white", edgecolor="0.8", framealpha=.96,
              labelspacing=.22, borderpad=.35, handlelength=2.0, handletextpad=.6)
    # The open point is a limit of ratios; risks themselves diverge at delta=1.
    # It is not added as a finite data row or connected to the numerical curves.
    ax.scatter([1], [1], s=75, facecolors="white", edgecolors="0.2", linewidths=1.8, zorder=7)


def save_plot(fig, output, name):
    # Publish complete image bytes together, so previews never read a partial PNG.
    for extension in ("pdf", "png"):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=extension, dpi=220, bbox_inches="tight")
        path = output/f"{name}.{extension}"
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(path)


def make_plots(output, curves, thresholds, crossings):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.style.context("ggplot"), plt.rc_context({
        "font.family":"DejaVu Sans", "pdf.fonttype":42, "ps.fonttype":42,
        "figure.facecolor":"white", "axes.facecolor":"#F0F0F0",
        "text.color":"#242424", "axes.labelcolor":"#242424",
        "xtick.color":"#242424", "ytick.color":"#242424",
    }):
        for name in ("cauchy", "laplace"):
            fig, ax = plt.subplots(figsize=(8.0, 6.2), layout="constrained")
            draw_panel(ax, curves[name], name, thresholds[name], crossings[name])
            save_plot(fig, output, FIGURE_NAMES[name])
            paths.extend(output/f"{FIGURE_NAMES[name]}.{ext}" for ext in ("pdf", "png"))
            plt.close(fig)
    return paths


def find_crossings():
    def ratio(name, first, second, delta):
        risk = lambda method: (2/(delta-1) if method == "LS" else
                               solve_risk(name, method, delta)["alpha2"])
        return math.log(risk(first)/risk(second))
    specs = {"cauchy":[("Rank", "LAD", 1.5, 2)],
             "laplace":[("Rank", "LS", 1.5, 2), ("LAD", "LS", 3, 4),
                        ("Rank", "LAD", 10, 30)]}
    return {name:{f"{first}_{second}":brentq(
                lambda d:ratio(name, first, second, d), lo, hi, xtol=1e-9)
                for first, second, lo, hi in cases}
            for name, cases in specs.items()}


def load_are_data(directory: Path):
    """Read the scalar risk tables and plotting parameters for S3 and S4."""
    directory = Path(directory)
    metadata = json.loads((directory/"metadata.json").read_text())
    curves = {}
    for name in FIGURE_NAMES:
        with (directory/f"risk_{name}.csv").open() as handle:
            rows = [{key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(handle)]
        if not rows or any(row["delta"] <= 1 for row in rows):
            raise ValueError(f"Invalid {name} aspect-ratio grid")
        if any(right["delta"] <= left["delta"] for left, right in zip(rows, rows[1:])):
            raise ValueError(f"The {name} grid must be strictly increasing")
        if any(not math.isfinite(row[key]) or row[key] <= 0
               for row in rows for key in ("alpha2_rank", "alpha2_LAD")):
            raise ValueError(f"Invalid {name} risk values")
        curves[name] = rows
    return curves, metadata


def generate_are_figures(data_dir: Path, output_dir: Path) -> list[Path]:
    """Draw Figures S3 and S4 from the supplied scalar risk tables."""
    curves, metadata = load_are_data(Path(data_dir)/"are")
    return make_plots(output_dir, curves, metadata["critical_delta"],
                      metadata["risk_crossings"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/are"))
    parser.add_argument("--points", type=int, default=180)
    parser.add_argument("--plot-only", action="store_true",
                        help="Draw the saved tables in --output-dir without recomputation")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        curves, metadata = load_are_data(args.output_dir)
        make_plots(args.output_dir, curves, metadata["critical_delta"],
                   metadata["risk_crossings"])
        return
    thresholds = {name: critical_delta(name) for name in FIGURE_NAMES}
    crossings = find_crossings()
    curves = {}
    for name in FIGURE_NAMES:
        # Cauchy risk differences below this grid floor are too small to resolve
        # reliably in double precision.
        epsilon_min = 1e-4 if name == "cauchy" else 1e-6
        near_one = 1+np.geomspace(epsilon_min, .01, 33 if name == "cauchy" else 65)
        deltas = np.unique(np.r_[np.geomspace(1.01, 1000, args.points),
                                near_one, np.linspace(1.01, 2, 55), thresholds[name],
                                [1.1, 1.5, 2, 3, 5, 10, 100, 1000]])
        curves[name] = compute_curve(name, deltas)
        export_csv(args.output_dir/f"risk_{name}.csv", curves[name])
    metadata = {"figure_numbers":{"cauchy":"S3", "laplace":"S4"},
                "noise_scale":1, "penalty":"none",
                "delta_min":min(row["delta"] for rows in curves.values() for row in rows),
                "delta_max":1000,
                "delta_min_by_noise":{name:min(row["delta"] for row in rows) for name, rows in curves.items()},
                "x_limits":[.85,1000],
                "open_endpoint":{"delta":1,"ARE":1,"interpretation":"right-hand limit, not a finite-risk data point; not connected to computed curves"},
                "ARE_definition":"ARE(A:B)=risk(B)/risk(A); above one favors A",
                "critical_delta":thresholds,
                "risk_crossings":crossings,
                "classical_scaled_risks":{
                    "cauchy":{"Rank":math.pi**2/3,"LAD":math.pi**2/4,"LS":"infinite"},
                    "laplace":{"Rank":4/3,"LAD":1,"LS":2}},
                "max_relative_equation_residual":{
                    name:max(row["max_relative_equation_residual"] for row in rows)
                    for name, rows in curves.items()}}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    make_plots(args.output_dir, curves, thresholds, crossings)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
