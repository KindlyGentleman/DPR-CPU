"""Run the unchanged, hash-verified DPR1 checkpoint on stored CPU inputs."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT_SHA256 = '823572aa0c4bc973acf149734b79833cb9fccf708aad23426373b8c62b5dc379'


def verify_checkpoint(path):
    """Check identity before allowing torch to deserialize the checkpoint."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint SHA-256 mismatch: {digest}')
    return digest


def validate_inputs(intensity, valid):
    """Preserve signed measurements; True/1 means a valid detector pixel."""
    x, mask = np.asarray(intensity), np.asarray(valid)
    if x.shape != (512, 512) or mask.shape != (512, 512):
        raise ValueError('Intensity and valid mask must both be 512x512')
    if not np.isfinite(x).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('Finite intensity and a binary valid mask are required')
    if not np.any(np.maximum(x, 0) * mask > 0):
        raise ValueError('At least one valid pixel must have positive intensity')
    return (np.array(x, dtype=np.float32, copy=True)[None, None],
            np.array(mask, dtype=np.float32, copy=True)[None, None])


def validate_output_path(path):
    """Keep fresh demonstrations separate from archived evidence."""
    path = Path(path).resolve()
    if path.suffix != '.npz' or path == (HERE / 'examples.npz').resolve():
        raise ValueError('Choose a new .npz output, separate from archived examples')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    return path


def verify_reference(prediction, reference):
    """Fail reproduction before saving when stored-output parity is lost."""
    delta = float(np.max(np.abs(prediction - reference)))
    if not np.allclose(prediction, reference, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f'Stored-output parity failed: max_abs={delta:.7g}')
    return delta


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--case', type=int, default=0, help='Stored case index, 0 to 4')
    parser.add_argument('--all', action='store_true', help='Run all five stored inputs')
    parser.add_argument('--output', type=Path, help='Optional new NPZ for fresh outputs')
    args = parser.parse_args()
    if not 0 <= args.case < 5:
        parser.error('--case must be between 0 and 4')
    output = validate_output_path(args.output) if args.output else None
    checkpoint = ROOT / 'pretrained/param_dpr1.pt'
    verify_checkpoint(checkpoint)

    import torch
    sys.path.insert(0, str(ROOT))
    from deeppr import Network

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)  # fresh process, before parallel work
    model = Network(ngf=64, max_features=1024, weight_model=True,
                    downsample_FFC=False, refinement=True).cpu().float()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(weights['model_state_dict'])
    model.eval()
    print(f'CPU FP32 DPR1; intra-op=4, inter-op=1; torch={torch.__version__}')
    fresh = {}
    indices = range(5) if args.all else [args.case]
    with np.load(HERE / 'examples.npz', allow_pickle=False) as saved:
        for i in indices:
            x, mask = validate_inputs(saved[f'intensity_{i}'], saved[f'valid_{i}'])
            start = time.perf_counter()
            with torch.inference_mode():
                pred = model(torch.from_numpy(x), torch.from_numpy(mask), false_scale=False)
            elapsed = time.perf_counter() - start
            pred = pred.numpy()[0, 0]
            if pred.shape != (64, 64) or not np.isfinite(pred).all():
                raise RuntimeError('DPR must return a finite 64x64 physical-amplitude array')
            reference = saved[f'reference_{i}']
            delta = verify_reference(pred, reference)
            print(f'{saved["sample_ids"][i]}: {elapsed:.4f} s; '
                  f'max_abs={delta:.7g}; stored-output allclose=True')
            fresh[f'prediction_{i}'] = pred
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects a path that appeared while inference ran.
        with output.open('xb') as handle:
            np.savez_compressed(handle, **fresh)
        print(f'Fresh demonstration saved: {output}')
    print('These calls include first-use effects; they do not replace historical warmed timings.')


if __name__ == '__main__':
    main()
