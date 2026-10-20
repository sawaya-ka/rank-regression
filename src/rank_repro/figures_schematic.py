"""Explicit Gaussian realization of the schematic in Supplementary Figure S3.

The supplied archives contain no source for the manuscript's TeX drawing.
This reconstruction uses V ~ N(0,1) and tau=0.9, as in the active example
in Figure 4. It is not a claim of pixel-identical recovery of the original.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def generate_schematic(output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    tau = 0.9
    cutoff = brentq(lambda v: norm.cdf(v) - 0.5 - v / (4 * tau), 1e-6, 4 * tau)
    qlo, qhi = norm.cdf(-cutoff), norm.cdf(cutoff)
    x = np.linspace(0, 3.4, 600)
    q = np.linspace(0.004, 0.996, 1000)
    shifted = norm.ppf(q) - tau * (4 * q - 2)
    output = np.where((q >= qlo) & (q <= qhi), 0.0, shifted)
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.7), layout="constrained")
        axes[0].plot(x, norm.cdf(x) - 0.5 - x / (4 * tau), color="#333333", lw=1.8)
        axes[0].axhline(0, color="0.6", lw=0.8)
        axes[0].scatter([cutoff], [0], color="#333333", s=16, zorder=3)
        axes[0].set(xticks=[0, cutoff], xticklabels=["0", r"$C_{c,\tau}$"],
                    yticks=[], xlabel=r"$x$", ylabel=r"$r_{c,\tau}(x)$")
        axes[1].plot(q, shifted, color="#c57b24", lw=1.6, ls="--", label=r"$\widetilde Q_{c,\tau}$")
        axes[1].plot(q, output, color="#246b93", lw=2, label=r"$h_{c,\tau}$")
        axes[1].axhline(0, color="0.6", lw=0.8, zorder=0)
        axes[1].set(xticks=[qlo, 0.5, qhi],
                    xticklabels=[r"$q^-_{c,\tau}$", r"$1/2$", r"$q^+_{c,\tau}$"],
                    xlim=(0, 1), yticks=[], xlabel=r"$q$")
        axes[1].legend(frameon=False, loc="upper left")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix in ("pdf", "png"):
            path = output_dir / f"figS03_isotonic_schematic.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths
