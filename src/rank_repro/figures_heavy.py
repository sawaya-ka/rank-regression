"""Figures 7–10 from complete, explicitly described numerical result tables.

Inputs must contain complete numerical grids and valid solver results.
Each CSV needs a sibling ``<stem>.metadata.json`` containing ``noise``,
``signal``, and explicit ``grid.df``, ``grid.delta`` (plus ``grid.penalty``
for penalized tables). See ``docs/reproduction.md`` for data conventions.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter


METHODS = ("LS", "LAD", "rank")
# Shared pastel colors for the theory and empirical phase maps.
COLORS = ("#FA9C9E", "#A0A0F3", "#95C89C")
PHASE_ALPHA = 1.0
FILES = {
    "7": ("theory/penalized_phase.csv", "figure_07_penalized_phase.pdf"),
    "8": ("theory/penalized_are.csv", "figure_08_penalized_are.pdf"),
    "9": ("empirical/unpenalized_phase.csv", "figure_09_empirical_unpenalized.pdf"),
    "10": ("empirical/penalized_phase.csv", "figure_10_empirical_penalized.pdf"),
}


class FigureDataError(ValueError):
    """The requested paper figure cannot be reproduced from the supplied data."""


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(".metadata.json")


def _load(path: Path, *, penalized: bool) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FigureDataError(
            f"Required result table is missing: {path}. Provide the complete "
            "table and metadata, or run the corresponding experiment; "
            "the archived PDF cannot substitute for these data."
        )
    meta_path = _metadata_path(path)
    if not meta_path.is_file():
        raise FigureDataError(f"Missing provenance metadata: {meta_path}.")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(path, low_memory=False)
    except (ValueError, pd.errors.ParserError) as exc:
        raise FigureDataError(f"Unreadable or truncated input {path}: {exc}") from exc
    if meta.get("noise") != "standard_student_t":
        raise FigureDataError(f"{path}: noise must be explicitly standard_student_t.")
    if meta.get("complete") is False:
        raise FigureDataError(f"{path}: the generating experiment reports incomplete or failed results.")
    if "signal" not in meta:
        raise FigureDataError(f"{meta_path}: signal distribution is required.")
    keys = ["df", "delta"] + (["penalty"] if penalized else [])
    missing = set(keys) - set(frame.columns)
    if missing:
        raise FigureDataError(f"{path}: missing coordinate columns {sorted(missing)}.")
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise FigureDataError(f"{path}: missing or duplicate grid coordinates.")
    grid = meta.get("grid", {})
    expected_count = 1
    for key in keys:
        if key not in grid:
            raise FigureDataError(f"{meta_path}: grid.{key} must list every expected coordinate.")
        declared = grid[key]
        actual = frame[key].unique()
        if len(declared) < (2 if key != "penalty" else 1):
            raise FigureDataError(f"{meta_path}: at least two df/delta values are needed.")
        if key == "penalty":
            matches = set(actual) == set(declared) == {"ridge", "lasso"}
        else:
            try:
                expected = np.asarray(declared, dtype=float)
                observed = np.asarray(actual, dtype=float)
            except ValueError as exc:
                raise FigureDataError(f"{path}: nonnumeric {key} coordinates.") from exc
            if (not np.isfinite(expected).all() or np.any(expected <= 0)
                    or len(np.unique(expected)) != len(expected)):
                raise FigureDataError(f"{meta_path}: invalid declared {key} grid.")
            matches = len(observed) == len(expected) and np.allclose(
                np.sort(observed), np.sort(expected), rtol=1e-10, atol=1e-12
            )
        if not matches:
            raise FigureDataError(f"{path}: {key} coordinates differ from the declared complete grid.")
        expected_count *= len(declared)
    if len(frame) != expected_count:
        raise FigureDataError(
            f"{path}: incomplete grid ({len(frame)} rows, expected {expected_count}); "
            "missing cells will not be interpolated or filled."
        )
    return frame, meta


def _require_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise FigureDataError(f"{name}: required columns absent: {sorted(missing)}.")
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except ValueError as exc:
            raise FigureDataError(f"{name}: nonnumeric values in {column}.") from exc


def _risk_columns(frame: pd.DataFrame, empirical: bool, *, include_lad: bool = True) -> list[str]:
    methods = METHODS if include_lad else ("LS", "rank")
    columns = ([f"mse_{method}_mean" for method in methods] if empirical else
               [f"alpha2_{method.lower()}_opt" for method in methods])
    _require_numeric(frame, columns, "risk table")
    for column, method in zip(columns, methods):
        value = frame[column].to_numpy(dtype=float)
        allowed_unavailable = (frame["df"].to_numpy() <= 2) if method == "LS" and not empirical else np.zeros(len(frame), dtype=bool)
        bad = ~np.isfinite(value) | (value <= 0)
        if np.any(bad & ~allowed_unavailable):
            raise FigureDataError(f"Invalid or failed risk values in {column}; rerun these cells.")
        # For standard t errors with df <= 2, the finite-variance LS formula
        # is unavailable. It is excluded analytically, never by generic NaN filling.
        if np.any(allowed_unavailable):
            frame.loc[allowed_unavailable, column] = np.inf
    return columns


def _check_empirical(frame: pd.DataFrame, meta: dict, *, penalized: bool) -> None:
    if not penalized and np.any(frame["delta"].to_numpy(dtype=float) <= 1):
        raise FigureDataError("Unpenalized comparisons require delta > 1.")
    for field in ("n", "n_rep"):
        if field not in meta or field not in frame:
            raise FigureDataError(f"Empirical data and metadata must record {field}.")
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy()
        if not np.isfinite(values).all() or not np.all(values == meta[field]):
            raise FigureDataError(f"Empirical {field} values disagree with metadata.")
    for method in METHODS:
        field = f"success_rate_{method}"
        if field not in frame:
            raise FigureDataError(f"Missing estimator availability diagnostic: {field}.")
        values = pd.to_numeric(frame[field], errors="coerce").to_numpy()
        if not np.isfinite(values).all() or not np.allclose(values, 1):
            raise FigureDataError(f"{field} contains failed repetitions; resolve them before plotting.")
    if penalized:
        selection = meta.get("selection", {})
        if not all(field in selection for field in ("criterion", "evaluation", "display_statistic")):
            raise FigureDataError("Penalized empirical metadata must describe tuning criterion, evaluation reuse, and display statistic.")
        if selection["display_statistic"] != "mean":
            raise FigureDataError("Paper figures require arithmetic mean MSE, not a median summary.")


def _edges(values: np.ndarray) -> np.ndarray:
    logged = np.log(values)
    middle = (logged[:-1] + logged[1:]) / 2
    return np.exp(np.r_[2 * logged[0] - middle[0], middle, 2 * logged[-1] - middle[-1]])


def _grids(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.sort(frame["df"].unique())
    y = np.sort(frame["delta"].unique())
    values = np.stack([frame.pivot(index="delta", columns="df", values=column)
                       .reindex(index=y, columns=x).to_numpy(dtype=float) for column in columns])
    return x, y, values


def _phase(ax, frame: pd.DataFrame, columns: list[str], title: str) -> None:
    x, y, risks = _grids(frame, columns)
    winner = np.argmin(risks, axis=0)
    cmap = ListedColormap(COLORS)
    ax.pcolormesh(_edges(x), _edges(y), winner, cmap=cmap,
                  norm=BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N),
                  alpha=PHASE_ALPHA, shading="flat", rasterized=True)
    ax.set(xscale="log", yscale="log", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
           xlabel=r"degrees of freedom $\nu$", ylabel=r"$\delta$", title=title)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(24)
    ax.title.set_size(22)
    ax.tick_params(axis="both", which="both", labelsize=18)
    ax.set_axisbelow(False)
    ax.grid(True, which="both", color="white", alpha=.12, linewidth=.6)


def _phase_legend(fig, *, empirical: bool) -> None:
    """Place one readable legend above the maps, clear of theory boundaries."""
    fig.legend(handles=[Patch(facecolor=color, alpha=PHASE_ALPHA, label=method)
                       for color, method in zip(COLORS, ("LS", "LAD", "Rank"))],
               title="smallest mean MSE" if empirical else "smallest limiting risk",
               loc="outside upper center", ncol=3, frameon=False,
               fontsize=18, title_fontsize=17, handlelength=1.6,
               columnspacing=1.7, borderaxespad=.2)


def _boundaries(ax, frame: pd.DataFrame, columns: list[str]) -> None:
    x, y, risks = _grids(frame, columns)
    for first in range(3):
        for second in range(first + 1, 3):
            other = 3 - first - second
            pair_min = np.minimum(risks[first], risks[second])
            valid = np.isfinite(risks[first]) & np.isfinite(risks[second]) & (risks[other] >= pair_min)
            difference = np.zeros_like(pair_min)
            np.subtract(np.log(risks[first], where=valid, out=np.zeros_like(pair_min)),
                        np.log(risks[second], where=valid, out=np.zeros_like(pair_min)), out=difference)
            values = difference[valid]
            if values.size and values.min() < 0 < values.max():
                ax.contour(x, y, np.ma.masked_where(~valid, difference), levels=[0],
                           colors="black", linewidths=1.8, zorder=10)


def _matched_overlay(ax, theory: pd.DataFrame, theory_meta: dict, empirical_meta: dict,
                     columns: list[str], *, penalized: bool) -> None:
    if theory_meta["noise"] != empirical_meta["noise"]:
        raise FigureDataError("Theory and empirical noise distributions differ; no boundary overlay is valid.")
    if penalized and theory_meta["signal"] != empirical_meta["signal"]:
        raise FigureDataError("Theory and empirical signal distributions differ; regenerate matching theory before overlaying boundaries.")
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for column, limits in (("df", xlim), ("delta", ylim)):
        if theory[column].min() > limits[0] * (1 + 1e-10) or theory[column].max() < limits[1] * (1 - 1e-10):
            raise FigureDataError(f"Theory grid does not cover the empirical {column} range; a partial boundary overlay would be misleading.")
    _boundaries(ax, theory, columns)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _figure8(frame: pd.DataFrame, meta: dict):
    _risk_columns(frame, False, include_lad=False)
    _require_numeric(frame, ["lambda_rank_opt", "lambda_ls_opt"], "ARE table")
    if np.any(frame["df"] <= 2):
        raise FigureDataError("Figure 8 compares finite LS risk and requires df > 2.")
    if not np.isfinite(frame[["lambda_rank_opt", "lambda_ls_opt"]].to_numpy()).all() or np.any(frame[["lambda_rank_opt", "lambda_ls_opt"]].to_numpy() <= 0):
        raise FigureDataError("Figure 8 requires finite positive selected lambdas.")
    notes = []
    for column in ("rank_opt_on_boundary", "ls_opt_on_boundary"):
        if column in frame:
            count = frame[column].astype(str).str.lower().isin(["true", "1"]).sum()
            if count:
                notes.append(f"{int(count)} {column} rows: widen/check lambda search before interpreting a global infimum.")
    settings = meta.get("config", {}).get("settings", {})
    recorded_search = all(key in settings for key in ("lambda_min", "lambda_max", "n_lambda"))
    if "optimization" not in meta and "lambda_search" not in meta and not recorded_search:
        notes.append("Runtime lambda-search settings are not recorded; script defaults are described in metadata. Finite-grid minima are not certified global infima.")
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=2)
    dfs = np.sort(frame["df"].unique())
    normalize = Normalize(dfs[0], dfs[-1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col", sharey="row", layout="constrained")
    for j, penalty in enumerate(("ridge", "lasso")):
        for df in dfs:
            part = frame[(frame["df"] == df) & (frame["penalty"] == penalty)].sort_values("delta")
            x = part["delta"].to_numpy()
            axes[0, j].plot(x, part["alpha2_ls_opt"] / part["alpha2_rank_opt"],
                            color=plt.colormaps["viridis"](normalize(df)), lw=2.0, label=rf"$\nu={df:g}$")
            shade = .35 + .55 * normalize(df)
            axes[1, j].plot(x, part["lambda_rank_opt"], color=plt.colormaps["Greens"](shade), lw=1.5)
            axes[1, j].plot(x, part["lambda_ls_opt"], color=plt.colormaps["Reds"](shade), ls="--", lw=1.3)
        axes[0, j].axhline(1, color=".4", ls=":", lw=1)
        axes[0, j].set(title=penalty.title() + " penalty",
                       ylabel=r"$\frac{\min_{\lambda}\alpha_{\rm LS}^{2}(\lambda)}{\min_{\lambda}\alpha_*^{2}(\lambda)}$")
        axes[1, j].set(xlabel=r"$\delta$", ylabel=r"selected $\lambda$")
        for ax in axes[:, j]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.yaxis.set_minor_formatter(StrMethodFormatter("{x:g}"))
            ax.xaxis.label.set_size(22)
            ax.yaxis.label.set_size(20)
            ax.title.set_size(22)
            ax.tick_params(axis="both", which="both", labelsize=16)
    axes[0, 1].set_ylabel("")
    axes[1, 1].set_ylabel("")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=9,
               fontsize=13, frameon=False, handlelength=1.6,
               columnspacing=.9, labelspacing=.4, handletextpad=.45)
    fig.legend(handles=[Line2D([], [], color=plt.colormaps["Greens"](.8), lw=2, label="Rank"),
                        Line2D([], [], color=plt.colormaps["Reds"](.8), lw=2, ls="--", label="LS")],
               loc="outside lower center", ncol=2, frameon=False,
               fontsize=17, handlelength=2.2, columnspacing=2)
    return fig, notes


def generate_heavy_figures(data_dir: Path, output_dir: Path,
                          figures: list[str] | None = None) -> list[Path]:
    """Generate selected Figures 7–10, refusing incomplete or mismatched data.

    ``figures`` uses manuscript numbers (``["7", "10"]``); ``None`` requests
    all four. Metadata are copied beside outputs with overlay/tuning notes.
    Failure messages identify the missing table rather than copying a reference
    PDF. An omitted optional overlay is recorded explicitly in output metadata.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    selected = [str(value).lower().removeprefix("figure_").lstrip("0") for value in (list(FILES) if figures is None else figures)]
    if any(value not in FILES for value in selected):
        raise ValueError("Heavy figure numbers must be 7, 8, 9, or 10.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.style.context("ggplot"), plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        for number in selected:
            relative, filename = FILES[number]
            frame, meta = _load(data_dir / relative, penalized=number != "9")
            notes: list[str] = []
            overlay_used = False
            if number == "8":
                fig, notes = _figure8(frame, meta)
            else:
                empirical = number in ("9", "10")
                if empirical:
                    _check_empirical(frame, meta, penalized=number == "10")
                columns = _risk_columns(frame, empirical)
                if number == "9":
                    fig, ax = plt.subplots(figsize=(7, 5.8), layout="constrained")
                    _phase(ax, frame, columns, "")
                    theory_path = data_dir / "unpenalized_student_t.csv"
                    if theory_path.exists() and _metadata_path(theory_path).exists():
                        theory, theory_meta = _load(theory_path, penalized=False)
                        theory = theory.rename(columns={f"alpha2_{m}": f"alpha2_{m.lower()}_opt" for m in METHODS})
                        theory_columns = _risk_columns(theory, False)
                        _matched_overlay(ax, theory, theory_meta, meta, theory_columns, penalized=False)
                        overlay_used = True
                    else:
                        notes.append("Theory overlay omitted: complete unpenalized theory data and metadata were not supplied.")
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), layout="constrained")
                    theory = theory_meta = theory_columns = None
                    if empirical:
                        theory_path = data_dir / FILES["7"][0]
                        if theory_path.exists() and _metadata_path(theory_path).exists():
                            theory, theory_meta = _load(theory_path, penalized=True)
                            theory_columns = _risk_columns(theory, False)
                        else:
                            notes.append("Theory overlay omitted: matching penalized theory data and metadata were not supplied.")
                    for ax, penalty in zip(axes, ("ridge", "lasso")):
                        part = frame[frame["penalty"] == penalty]
                        _phase(ax, part, columns, penalty.title() + " penalty")
                        if theory is not None:
                            _matched_overlay(ax, theory[theory["penalty"] == penalty], theory_meta,
                                             meta, theory_columns, penalized=True)
                            overlay_used = True
                _phase_legend(fig, empirical=empirical)
            for note in notes:
                if "overlay" in note:
                    warnings.warn(note, UserWarning, stacklevel=2)
            path = output_dir / filename
            fig.savefig(path, bbox_inches="tight")
            png_path = path.with_suffix(".png")
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            record = {"figure": number, "source": relative, "data_metadata": meta,
                      "theory_overlay": overlay_used, "notes": notes,
                      "grid_rendering": "observed rectangular grid; no missing-cell filling"}
            path.with_suffix(".metadata.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            paths.extend([path, png_path])
    return paths
