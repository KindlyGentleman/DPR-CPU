"""Generate a fixed-layout, infinitely looping GIF of five stored DPR cases."""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from PIL import Image
import scienceplots

HERE = Path(__file__).resolve().parent


def main():
    plt.style.use(['science', 'ieee', 'no-latex'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'savefig.bbox': None, 'text.usetex': False})
    meta = json.loads((HERE / 'provenance.json').read_text())
    frames = []
    with np.load(HERE / 'examples.npz', allow_pickle=False) as saved:
        ampmax = max(float(saved[f'{k}_{i}'].max()) for i in range(5) for k in ('truth', 'reference'))
        detector_max = max(float(np.log10(1 + np.maximum(saved[f'intensity_{i}'], 0)).max()) for i in range(5))
        anorm = PowerNorm(.5, 0, ampmax)
        dnorm = Normalize(0, detector_max)
        fig, axes = plt.subplots(1, 4, figsize=(10, 3.4), gridspec_kw={'wspace': .15})
        fig.subplots_adjust(left=.02, right=.98, bottom=.24, top=.80)
        for i in range(5):
            for ax in axes: ax.clear()
            fields = [(np.log10(1 + np.maximum(saved[f'intensity_{i}'], 0)), 'Diffraction intensity', 'magma', dnorm),
                      (saved[f'valid_{i}'], 'Valid detector mask', 'gray', Normalize(0, 1)),
                      (saved[f'reference_{i}'], 'Frozen DPR reconstruction', 'cividis', anorm),
                      (saved[f'truth_{i}'], 'Ground truth', 'cividis', anorm)]
            for ax, (array, heading, cmap, norm) in zip(axes, fields):
                ax.imshow(array, cmap=cmap, norm=norm, interpolation='nearest')
                ax.set(xticks=[], yticks=[]); ax.set_title(heading, fontsize=10, pad=7)
            case = meta['cases'][i]
            fig.suptitle(f"{i+1}/5  {saved['sample_ids'][i]}  |  {case['split']}: {case['family']}  |  aligned NRMSE {case['aligned_nrmse']:.3f}",
                         fontsize=10, y=.98)
            # Fixed numeric scales and coordinate convention on every frame.
            while fig.texts[1:]: fig.texts[-1].remove()
            fig.text(.02, .13, f'Detector: log₁₀(1+max(I,0)), 0–{detector_max:.2f}  |  Mask: white = valid', fontsize=9)
            fig.text(.02, .055, f'Amplitude: 0–{ampmax:.3f}, γ=0.5, shared across all cases  |  Native grids: 512² detector; 64² object; raw coordinates', fontsize=9)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=120, facecolor='white')
            buffer.seek(0)
            frames.append(Image.open(buffer).convert('RGB'))
        plt.close(fig)
    # One shared palette keeps the same color meaning across all frames.
    sheet = Image.new('RGB', (frames[0].width, frames[0].height * len(frames)), 'white')
    for i, frame in enumerate(frames): sheet.paste(frame, (0, i*frame.height))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path = HERE / 'reconstruction.gif'
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=2000,
                   loop=0, disposal=2, optimize=False)
    print(f'{path.name}: {len(frames)} genuine cases; 2 s/case; infinite loop; {path.stat().st_size/1e6:.2f} MB')


if __name__ == '__main__':
    main()
