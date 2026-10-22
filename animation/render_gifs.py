#!/usr/bin/env python3
"""Animate Student-t regression risks in the style of manuscript Fig. 1(a).

Run after compute_risks.py, or use the supplied risk_grid.npz directly.
No interpolation in delta: every finite frame has its own computed risk curve.
Interpolation in log degrees of freedom is only for drawing smooth curves.
"""
from pathlib import Path
import argparse
import io
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.interpolate import PchipInterpolator
from PIL import Image

from rank_repro.student import classical_risks

ROOT = Path(__file__).resolve().parent
COLORS = ('#C43C39', '#3767AF', '#2D7F50')
LABELS = ('least squares', 'least absolute deviations', 'rank regression')
STYLES = ('--', '-.', '-')


def save_image(image, path, **kwargs):
    """Finish encoding before replacing the output file."""
    buffer = io.BytesIO()
    image.save(buffer, format=path.suffix[1:].upper(), **kwargs)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(buffer.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def delta_label(value):
    if np.isinf(value):
        return r'\infty'
    decimals = max(0, 2 - int(math.floor(math.log10(value))))
    return f'{value:.{decimals}f}'


def tick_label(value, _):
    if value >= 10000:
        exponent = int(np.floor(np.log10(value)))
        mantissa = value / 10**exponent
        return rf'${mantissa:g}\!\times\!10^{{{exponent}}}$' if mantissa != 1 else rf'$10^{{{exponent}}}$'
    return f'{value:g}'


def dense_risks(df, delta, risks, mode):
    x = np.unique(np.r_[np.geomspace(1., 50., 1600), 2.])
    if np.isinf(delta):
        out = classical_risks(x)
        if mode == 'raw':
            out[np.isfinite(out)] = 0.
        return x, out
    out = np.empty((3, len(x)))
    out[0] = np.inf
    valid = x > 2
    out[0, valid] = x[valid] / ((x[valid] - 2) * (delta - 1))
    for k in (1, 2):
        out[k] = np.exp(PchipInterpolator(np.log(df), np.log(risks[k]))(np.log(x)))
    if mode == 'scaled':
        out *= delta
    return x, out


def draw_background(ax, x, risk):
    winner = np.argmin(risk, axis=0)
    cuts = np.r_[0, np.flatnonzero(np.diff(winner)) + 1, len(x)]
    for start, end in zip(cuts[:-1], cuts[1:]):
        lo = x[0] if start == 0 else np.sqrt(x[start-1] * x[start])
        hi = x[-1] if end == len(x) else np.sqrt(x[end-1] * x[end])
        ax.axvspan(lo, hi, color=COLORS[winner[start]], alpha=.35, lw=0, zorder=0)


def ls_plot_curve(delta, mode, ymax):
    """Sample the exact LS curve beyond the top of the visible risk range."""
    factor = 1. if np.isinf(delta) else 1. / (delta - 1.)
    if mode == 'scaled' and np.isfinite(delta):
        factor *= delta
    # R(nu) = factor * nu / (nu - 2). Sampling risk rather than nu avoids
    # truncation when the common df grid does not approach 2 closely enough.
    risk = np.geomspace(1.15 * ymax, factor * 50. / 48., 600)
    df = 2. + 2. * factor / (risk - factor)
    return df, risk, factor


def render_frame(df, delta, risk, mode):
    x, y = dense_risks(df, delta, risk, mode)
    is_zero = mode == 'raw' and np.isinf(delta)
    with plt.rc_context({
        'font.family':'DejaVu Sans', 'font.size':12,
        'axes.labelsize':15, 'axes.labelcolor':'#222222',
        'text.color':'#222222', 'xtick.color':'#333333', 'ytick.color':'#333333',
        'axes.edgecolor':'#333333', 'axes.linewidth':1.,
        'figure.facecolor':'white', 'axes.facecolor':'white',
        'mathtext.fontset':'dejavusans', 'savefig.facecolor':'white',
    }):
        fig = plt.figure(figsize=(8.4, 5.25), dpi=144)
        ax = fig.add_axes([.115, .25, .855, .575])
        fig.text(.115, .91, 'Regression risk under Student-t errors', fontsize=16, weight='semibold')
        subtitle = ('Classical limit' if np.isinf(delta) else 'Proportional high-dimensional limit')
        fig.text(.115, .863, subtitle, fontsize=11.5, color='#555555')
        fig.text(.97, .865, rf'$\delta={delta_label(delta)}$', ha='right', fontsize=20)
        if not is_zero:
            draw_background(ax, x, y)
        ax.set_xscale('log')
        ax.set_xlim(1, 50)
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r'Degrees of freedom $\nu$', labelpad=8)
        if is_zero:
            ax.set_ylim(-.04, 1.05)
            ax.set_yticks([0, .5, 1])
            ax.set_ylabel(r'Risk $\alpha^2(\delta)$')
            ax.text(.48, .43,
                    'Finite-risk limits are zero.\nLAD and Rank: all shown degrees of freedom\nLS: only for '+r'$\nu>2$',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12.5,
                    linespacing=1.7, bbox=dict(facecolor='white', edgecolor='#CCCCCC', boxstyle='round,pad=.7'))
        else:
            ax.set_yscale('log')
            ymin = min(y[0, -1], y[1, -1], y[2, -1]) * .89
            ymax = max(np.max(y[1:]) * 1.25, ymin * 10.8)
            ax.set_ylim(ymin, ymax)
            ticks = [m*10.**e for e in range(-6, 9) for m in (1, 2, 5)
                     if ymin <= m*10.**e <= ymax]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(tick_label))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_ylabel(r'Scaled risk $\delta\,\alpha^2(\delta)$' if mode == 'scaled'
                          else r'Risk $\alpha^2(\delta)$')
        for k in range(3):
            if k == 0 and not is_zero:
                plot_x, plot_y, ls_factor = ls_plot_curve(delta, mode, ymax)
            else:
                plot_x, plot_y = x, np.where(np.isfinite(y[k]), y[k], np.nan)
            ax.plot(plot_x, plot_y, color=COLORS[k], ls=STYLES[k],
                    lw=2.6, zorder=3+k, label=LABELS[k])
        if not is_zero:
            # Follow the same analytic curve with a small continuation arrow.
            arrow_y = ymax * (ymax / ymin) ** np.array([-.045, .008])
            arrow_x = 2. + 2. * ls_factor / (arrow_y - ls_factor)
            ax.annotate('', xy=(arrow_x[1], arrow_y[1]),
                        xytext=(arrow_x[0], arrow_y[0]), annotation_clip=False,
                        arrowprops=dict(arrowstyle='-|>', color=COLORS[0],
                                        lw=1.8, mutation_scale=12,
                                        shrinkA=0, shrinkB=0), zorder=6)
        ax.set_axisbelow(True)
        ax.grid(which='major', color='#FFFFFF', alpha=.58, lw=.8)
        ax.tick_params(axis='both', labelsize=11, length=4)
        ax.legend(loc='upper right', fontsize=11.2, framealpha=1, facecolor='white',
                  edgecolor='#BBBBBB', handlelength=3.1, borderpad=.65, labelspacing=.45)
        ax.text(.012, .09 if is_zero else .027, r'LS: infinite risk for $\nu\leq 2$', transform=ax.transAxes,
                fontsize=9.5, color='#7E2523', bbox=dict(facecolor='white', edgecolor='none', alpha=.86, pad=3))
        fig.text(.115, .116, 'Background: smallest risk' if not is_zero else 'No unique winner at the zero-risk limit',
                 fontsize=9.5, color='#555555')
        fig.text(.97, .116, 'Vertical axis rescales' if not is_zero else 'Zero-risk frame: linear vertical axis',
                 ha='right', fontsize=9.5, color='#555555')
        # A log-delta ruler makes the direction and finite-grid spacing explicit.
        x0, x1, yy = .30, .80, .065
        fig.add_artist(Line2D([x0, x1], [yy, yy], transform=fig.transFigure, color='#AEB4BC', lw=2))
        infinity_pos = x0 - .05
        fig.add_artist(Line2D([infinity_pos, x0], [yy, yy], transform=fig.transFigure,
                             color='#AEB4BC', lw=1, ls=':'))
        fig.text(infinity_pos, yy-.023, r'$\infty$', ha='center', va='top', fontsize=9, color='#666666')
        fig.text(.115, yy, 'Low dim.', va='center', fontsize=10, color='#555555')
        fig.text(.97, yy, 'High dim.', va='center', ha='right', fontsize=10, color='#555555')
        def pos(d):
            return x0 + (np.log(100)-np.log(d))/(np.log(100)-np.log(1.01))*(x1-x0)
        for d in [100, 10, 2, 1.01]:
            xx = pos(d)
            fig.add_artist(Line2D([xx,xx], [yy-.005,yy+.005], transform=fig.transFigure, color='#999999', lw=1))
            fig.text(xx, yy-.023, f'{d:g}', ha='center', va='top', fontsize=8.5, color='#666666')
        dot = infinity_pos if np.isinf(delta) else pos(delta)
        fig.add_artist(Line2D([dot], [yy], marker='o', markersize=6.5, color='#333333',
                             markeredgecolor='white', transform=fig.transFigure))
        fig.canvas.draw()
        picture = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        plt.close(fig)
    return picture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data'/'risk_grid.npz')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent/'assets'/'animations')
    parser.add_argument('--frame-ms', type=int, default=120)
    parser.add_argument('--mode', choices=('scaled', 'raw', 'both'), default='scaled')
    args = parser.parse_args()
    if args.frame_ms < 10 or args.frame_ms % 10:
        parser.error('frame-ms must be a positive multiple of 10 (GIF time resolution)')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    dfs, deltas, risks = data['df'], data['delta'], data['risks']
    assert risks.shape == (len(deltas), 3, len(dfs))
    assert np.all(np.diff(deltas) < 0) and np.all(deltas > 1)
    written = []
    modes = ('scaled', 'raw') if args.mode == 'both' else (args.mode,)
    for mode in modes:
        filename = 'fig1a_scaled_risk.gif' if mode == 'scaled' else 'fig1a_risk.gif'
        frames, snapshots = [], []
        indices = {0, 1+int(np.argmin(abs(deltas-5))), 1+int(np.argmin(abs(deltas-1.5))), len(deltas)}
        series = [(np.inf, None), *zip(deltas, risks)]
        palette = None
        for i, (delta, risk) in enumerate(series):
            pic = render_frame(dfs, delta, risk, mode)
            if i in indices:
                save_image(pic, args.output_dir/f'preview_{mode}_{i:03d}.png')
                snapshots.append(pic.copy())
            # Stable palette throughout the animation, including all three region colors.
            if palette is None:
                palette_source = render_frame(dfs, np.inf, None, 'scaled')
                palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            frames.append(pic.quantize(palette=palette, dither=Image.Dither.NONE))
            if i % 25 == 0:
                print(f'{mode}: rendered {i+1}/{len(series)} frames', flush=True)
        durations = [1800] + [args.frame_ms]*len(deltas)
        durations[-1] = 2400
        path = args.output_dir/filename
        save_image(frames[0], path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=1, optimize=True)
        with Image.open(path) as check:
            assert check.n_frames == len(series)
            assert check.info.get('loop') == 0
            total_ms = 0
            for j in range(check.n_frames):
                check.seek(j)
                check.load()
                total_ms += check.info['duration']
        written.append({'file':filename,'frames':len(series),'duration_ms':total_ms,
                        'width':frames[0].width,'height':frames[0].height,'bytes':path.stat().st_size})
        contact = Image.new('RGB', (snapshots[0].width*2, snapshots[0].height*2), 'white')
        for i, pic in enumerate(snapshots):
            contact.paste(pic, ((i%2)*pic.width, (i//2)*pic.height))
        save_image(contact, args.output_dir/f'contact_{mode}.png')
    (args.output_dir/'render_metadata.json').write_text(json.dumps(written, indent=2)+'\n')
    print(json.dumps(written,indent=2), flush=True)


if __name__ == '__main__':
    main()
