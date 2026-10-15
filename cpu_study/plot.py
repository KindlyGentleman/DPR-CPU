"""Regenerate the two IEEE figures from compact, recorded experimental evidence."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import scienceplots  # registers the actual SciencePlots styles

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'


def configure_style():
    """Require SciencePlots IEEE; explicit small adjustments preserve readability."""
    assert all(s in plt.style.available for s in ('science', 'ieee', 'no-latex'))
    plt.style.use(['science', 'ieee', 'no-latex'])
    assert plt.rcParams['font.size'] == plt.style.library['ieee']['font.size']
    assert not plt.rcParams['text.usetex']
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'xtick.top': False, 'ytick.right': False, 'savefig.bbox': None})


def read_results():
    with (HERE / 'results.csv').open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def subset(rows, section, metric=None):
    return [r for r in rows if r['section'] == section and (metric is None or r['metric'] == metric)]


def title(ax, letter, text):
    ax.set_title(f'({letter}) {text}', loc='left', pad=8, fontweight='bold')


def export(fig, name):
    fig.savefig(HERE / f'{name}.png', dpi=600, facecolor='white')
    fig.savefig(HERE / f'{name}.pdf', facecolor='white',
                metadata={'Creator': 'Matplotlib + SciencePlots science/ieee/no-latex',
                          'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main_figure(rows, saved, meta):
    fig = plt.figure(figsize=(7.16, 6.4))
    gs = fig.add_gridspec(3, 2, left=.035, right=.985, top=.91, bottom=.13,
                          height_ratios=[1.1, 1, 1], wspace=.13, hspace=.28)
    fig.suptitle('Deep phase retrieval on CPU: execution, optimization and reliability', fontsize=10)

    def chart(cell, letter, heading):
        frame = fig.add_subplot(cell); frame.axis('off')
        frame.text(0, 1, f'({letter}) {heading}', transform=frame.transAxes,
                   va='top', fontweight='bold', fontsize=8)
        p = cell.get_position(fig)
        return fig.add_axes([p.x0+.32*p.width, p.y0+.20*p.height, .66*p.width, .64*p.height])

    ax = chart(gs[0, 0], 'a', 'Disjoint profile: mean ± SD, five passes')
    data = subset(rows, 'stage')
    y = np.arange(6)
    values = np.array([float(r['value']) for r in data])
    errors = np.array([float(r['high']) - float(r['value']) for r in data])
    ax.barh(y, values, xerr=errors, height=.65,
            color=[BLUE, BLUE, ORANGE, BLUE, BLUE, BLUE],
            error_kw={'elinewidth': .7, 'capsize': 2})
    labels = ['Normalization', 'WPC encoder', 'FFC / merge', 'Projection', 'Learned refine.', 'Flux scaling']
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlim=(0, 730), xlabel='Disjoint stage wall time (ms)')
    for i, v in enumerate(values):
        ax.text(v + errors[i] + 9, i, f'{v:.3f}', va='center', fontsize=7)

    ax = chart(gs[0, 1], 'b', 'Exact candidates; none promoted')
    data = subset(rows, 'candidate')
    for i, r in enumerate(data):
        v, lo, hi = (float(r[k]) for k in ('value', 'low', 'high'))
        ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt='o', color=BLUE, capsize=2, ms=3.5)
    ax.axvline(1, color='black', ls='--', lw=.8)
    ax.set_yticks(np.arange(5), ['WPC caching', 'Channels-last', 'Complex unpack', 'LFU broadcast', 'Cartesian proj.'])
    ax.invert_yaxis()
    ax.set(xlim=(.88, 2.60), xlabel='Paired speedup (×), 95% CI')
    ax.set_xticks([1, 1.5, 2, 2.5])

    ax = chart(gs[1, 0], 'c', 'Sequential thread screen; 36 calls / setting')
    data = subset(rows, 'thread', 'median')
    threads = np.array([int(r['name']) for r in data])
    medians = np.array([float(r['value']) for r in data])
    p95 = [float(r['value']) for r in subset(rows, 'thread', 'p95')]
    ax.plot(threads, p95, 's--', color='#666666', ms=3, lw=.8, label='p95')
    ax.plot(threads, medians, 'o-', color=BLUE, ms=3, lw=.8, label='Median')
    ax.plot(1, medians[0], 'x', color=ORANGE, ms=7, mew=1.2)
    ax.plot(4, medians[2], 'o', color=ORANGE, ms=6, mfc='white', mew=1.2)
    ax.annotate('4 selected', (4, medians[2]), xytext=(3.1, 1.35),
                arrowprops={'arrowstyle': '-', 'lw': .6}, fontsize=7)
    ax.annotate('6 nominally fastest', (6, medians[3]), xytext=(5.1, .26),
                arrowprops={'arrowstyle': '-', 'lw': .6}, fontsize=7)
    ax.set(xticks=threads, xlabel='Intra-op threads (inter-op = 1)', ylabel='Latency (s)', ylim=(0, 2.5))
    ax.legend(loc='upper right', frameon=False, handlelength=1.7)

    ampmax = max(float(saved[f'{k}_{i}'].max()) for i in (0, 4) for k in ('truth', 'reference'))
    errmax = max(float(np.abs(saved[f'truth_{i}']-saved[f'reference_{i}']).max()) for i in (0, 4))
    anorm, enorm = PowerNorm(.5, 0, ampmax), Normalize(0, errmax)
    for cell, i, letter, heading in [(gs[1, 1], 0, 'd', 'Representative high-photon case'),
                                     (gs[2, 0], 4, 'e', 'Locked worst case: OOD, low photons')]:
        frame = fig.add_subplot(cell); frame.axis('off')
        frame.text(0, 1, f'({letter}) {heading}', transform=frame.transAxes,
                   fontweight='bold', va='top', fontsize=8)
        p = cell.get_position(fig)
        truth, pred = saved[f'truth_{i}'], saved[f'reference_{i}']
        for j, (array, name) in enumerate([(truth, 'Ground truth'), (pred, 'Frozen DPR'), (abs(truth-pred), 'Absolute error')]):
            width = .24*p.width
            height = width * 7.16 / 6.4
            ax = fig.add_axes([p.x0+(.04+j*.35)*p.width, p.y0+.23*p.height, width, height])
            im = ax.imshow(array, cmap='cividis' if j < 2 else 'magma',
                           norm=anorm if j < 2 else enorm, interpolation='nearest')
            ax.set(xticks=[], yticks=[])
            ax.set_title(name, fontsize=7, pad=3)
            if j == 1: amp_image = im
            if j == 2: error_image = im
        case = meta['cases'][i]
        frame.text(0, .18, f"{saved['sample_ids'][i]}\nAligned NRMSE {case['aligned_nrmse']:.3f}; SSIM {case['aligned_ssim']:.3f}",
                   transform=frame.transAxes, fontsize=7, va='top')

    ax = fig.add_subplot(gs[2, 1]); ax.axis('off')
    ax.text(0, 1, '(f) Absolute reference reliability', fontweight='bold', va='top', fontsize=8)
    rel = {r['metric']: r for r in subset(rows, 'reliability')}
    table = []
    for key, label in [('aligned_nrmse', 'Aligned NRMSE'), ('aligned_ssim', 'SSIM'), ('failure', 'Failure fraction')]:
        r = rel[key]
        table.append([label, f"{float(r['value']):.3f}", f"[{float(r['low']):.3f}, {float(r['high']):.3f}]"])
    tab = ax.table(cellText=table, colLabels=['Metric', 'Base mean', '95% base CI'],
                   colWidths=[.40, .23, .37], cellLoc='left', bbox=[0, .43, 1, .43])
    tab.auto_set_font_size(False); tab.set_fontsize(7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(.4); cell.set_edgecolor('#aaaaaa')
        if r == 0: cell.set_text_props(weight='bold')
    ax.text(0, .32, '251 / 570 failures (aligned NRMSE > 0.5)\n15 independent objects; 38 conditions / object\n10 test + 5 OOD objects; synthetic data\nSelected: original FP32 DPR1, 4 / 1 threads',
            va='top', fontsize=7, linespacing=1.6)
    for im, left, label in [(amp_image, .13, 'Physical amplitude (γ=0.5)'),
                             (error_image, .64, 'Absolute raw-coordinate error')]:
        cax = fig.add_axes([left, .047, .27, .012])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .1, .2])
        cb.set_label(label, fontsize=7, labelpad=4)
        cax.xaxis.set_label_position('top')
    export(fig, 'figure_main')


def diagnostic_figure(rows, saved):
    fig = plt.figure(figsize=(7.16, 4.15), layout='constrained')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1], height_ratios=[1.15, 1], hspace=.18, wspace=.18)
    fig.suptitle('CPU kernels and reconstruction fidelity: completed technical comparisons', fontsize=10)
    fft = subset(rows, 'fft')
    backends = ['torch', 'scipy', 'numpy', 'fftw', 'ort']
    # The recorded backend names are read explicitly below, no substituted timings.
    actual = {r['name'].split('|')[0] for r in fft}
    backends = [b for b in backends if b in actual]
    assert len(backends) == 5, actual
    colors = [BLUE, ORANGE, GREEN, '#CC79A7', '#333333']
    marks = ['o', 's', '^', 'D', 'v']
    shapes = ['(1, 16, 32, 32)', '(1, 128, 64, 64)', '(1, 1, 512, 512)']
    sub = gs[0, 0].subgridspec(1, 2, wspace=.15)
    handles = []
    for d, direction in enumerate(('forward', 'inverse')):
        ax = fig.add_subplot(sub[0, d])
        for b, backend in enumerate(backends):
            for category in ('native_call', 'torch_bridge'):
                points = []
                for k, shape in enumerate(shapes):
                    r = next(r for r in fft if r['name'] == '|'.join([backend, direction, category, shape]))
                    x = k + (b-2)*.13 + (-.025 if category=='native_call' else .025)
                    v, lo, hi = (float(r[j]) for j in ('value', 'low', 'high'))
                    h = ax.errorbar(x, v, yerr=[[v-lo], [hi-v]], fmt=marks[b],
                                    color=colors[b], mfc='white' if category=='native_call' else colors[b],
                                    ms=3.2, lw=.55, capsize=1)
                    if category == 'torch_bridge' and backend=='ort' and direction=='forward' and k==2:
                        ax.plot(x, v, 'x', color='black', ms=6, mew=1)
                    points.append(h)
                if d==0 and category=='torch_bridge': handles.append(points[0])
        ax.set(yscale='log', xticks=range(3), xticklabels=['16×32²', '128×64²', '1×512²'],
               xlabel='Channels × H × W (batch = 1)', ylim=(.015, 120), xlim=(-.5, 2.5))
        ax.set_yticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
        if d == 0: ax.set_ylabel('FFT wall time (ms, log scale)')
        ax.set_title(f'(a) {direction.capitalize()} FFT' if d==0 else 'Inverse FFT', loc='left', fontweight='bold')
    handles += [Line2D([], [], marker='o', color='black', mfc='white', ls='none', ms=4),
                Line2D([], [], marker='o', color='black', ls='none', ms=4)]
    fig.legend(handles, ['PyTorch', 'SciPy', 'NumPy', 'FFTW', 'ORT DFT', 'Native', 'Bridge'],
               loc='outside lower center', ncol=7, frameon=False, handlelength=1.1, columnspacing=1)

    sub = gs[0, 1].subgridspec(2, 1, hspace=.16)
    for j, (metric, label) in enumerate([('median_psnr_loss_db', 'Median PSNR loss (dB)'), ('median_ssim_loss', 'Median SSIM loss')]):
        ax = fig.add_subplot(sub[j, 0])
        data = subset(rows, 'int8', metric)
        vals = [float(r['value']) for r in data]
        ax.barh([0, 1], vals, color=[BLUE, ORANGE], height=.55)
        for y, v in enumerate(vals): ax.text(v + max(vals)*.035, y, f'{v:.3f}', va='center', fontsize=7)
        ax.set(yticks=[0, 1], yticklabels=['Spectral', 'Spatial 1×1'], xlabel=label, xlim=(0, max(vals)*1.35))
        ax.invert_yaxis()
        if j==0: title(ax, 'b', 'Grouped INT8 rejection')

    sub = gs[1, :].subgridspec(2, 5, height_ratios=[.16, 1], width_ratios=[1, 1, 1, 1, .035], wspace=.12)
    ax = fig.add_subplot(sub[0, :]); ax.axis('off')
    ax.text(0, .5, '(c) Rejected precision changes: genuine low-photon / large-beamstop development case',
            va='center', fontweight='bold', fontsize=8)
    qmax = max(float(saved[k].max()) for k in ('quant_truth', 'quant_fp32', 'quant_spectral', 'quant_spatial'))
    norm = PowerNorm(.5, 0, qmax)
    for j, (key, label) in enumerate([('quant_truth', 'Ground truth'), ('quant_fp32', 'FP32 DPR'),
                                     ('quant_spectral', 'Spectral INT8'), ('quant_spatial', 'Spatial 1×1 INT8')]):
        ax = fig.add_subplot(sub[1, j])
        im = ax.imshow(saved[key], cmap='cividis', norm=norm, interpolation='nearest')
        ax.set(xticks=[], yticks=[]); ax.set_title(label, fontsize=8, pad=3)
    cax = fig.add_subplot(sub[1, 4])
    cb = fig.colorbar(im, cax=cax); cb.set_label('Amplitude (γ=0.5)', fontsize=7)
    cb.formatter = ScalarFormatter(useMathText=True); cb.update_ticks()
    export(fig, 'figure_diagnostics')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--figure', choices=['main', 'diagnostics', 'all'], default='all')
    args = parser.parse_args()
    configure_style()
    rows = read_results()
    meta = json.loads((HERE / 'provenance.json').read_text())
    with np.load(HERE / 'examples.npz', allow_pickle=False) as saved:
        if args.figure in ('main', 'all'): main_figure(rows, saved, meta)
        if args.figure in ('diagnostics', 'all'): diagnostic_figure(rows, saved)
    print('Rendered SciencePlots science/ieee/no-latex; 7.16 in; 600-dpi PNG and vector PDF.')


if __name__ == '__main__':
    main()
