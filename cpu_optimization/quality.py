"""Physical-scale metrics with integer translation/inversion and outside penalty."""

import numpy as np
from scipy.signal import correlate
from scipy.ndimage import shift, binary_erosion, binary_dilation
from skimage.metrics import structural_similarity


def image_metrics(truth, prediction):
    t, p = np.asarray(truth, float), np.asarray(prediction, float)
    if (
        t.shape != (64, 64)
        or p.shape != (64, 64)
        or not np.isfinite(p).all()
        or not np.isfinite(t).all()
        or t.max() <= 0
    ):
        raise ValueError("Finite 64x64 images with positive truth required")
    target = np.pad(t, 32)
    best = None
    for inv in (False, True):
        a = np.pad(np.flip(p, (0, 1)) if inv else p, 32)
        corr = correlate(target, a, mode="full", method="fft")[95:160, 95:160]
        dy, dx = (int(v) - 32 for v in np.unravel_index(np.argmax(corr), corr.shape))
        a = shift(a, (dy, dx), order=0, mode="constant", prefilter=False)
        err = float(np.sum((a - target) ** 2))
        if best is None or err < best[0]:
            best = (err, a[32:96, 32:96], dy, dx, inv)
    err, a, dy, dx, inv = best
    nrmse = float(np.sqrt(err / np.sum(t * t)))
    rmse = float(np.sqrt(err / t.size))
    dr = float(t.max())
    gt = t > 0
    support = a > 0.1 * dr
    union = (gt | support).sum()
    bg = gt ^ binary_erosion(gt)
    bp = support ^ binary_erosion(support)
    prec = (bp & binary_dilation(bg)).sum() / max(1, bp.sum())
    rec = (bg & binary_dilation(bp)).sum() / max(1, bg.sum())
    return dict(
        aligned_nrmse=nrmse,
        aligned_ssim=float(structural_similarity(t, a, data_range=dr)),
        aligned_psnr=float(20 * np.log10(dr / max(rmse, 1e-30))),
        failure=int(nrmse > 0.5),
        raw_nrmse=float(np.linalg.norm(p - t) / np.linalg.norm(t)),
        support_iou=float((gt & support).sum() / max(1, union)),
        boundary_f1=float(2 * prec * rec / max(1e-30, prec + rec)),
        dy=dy,
        dx=dx,
        inverted=inv,
    )


def compare_outputs(reference, candidate):
    b, c = np.asarray(reference), np.asarray(candidate)
    valid = c.shape == b.shape and bool(np.isfinite(c).all())
    if not valid:
        return dict(
            allclose=False, finite=bool(np.isfinite(c).all()), shape=list(c.shape)
        )
    delta = np.abs(c.astype(float) - b)
    return dict(
        allclose=bool(np.allclose(c, b, rtol=1e-4, atol=1e-5)),
        finite=True,
        shape=list(c.shape),
        max_abs=float(delta.max()),
        max_relative=float((delta / np.maximum(np.abs(b), 1e-12)).max()),
        relative_l2=float(np.linalg.norm(delta) / max(1e-30, np.linalg.norm(b))),
        violating_elements=int(np.count_nonzero(delta > 1e-5+1e-4*np.abs(b))),
        elements=int(b.size),
        bitwise=bool(np.array_equal(b, c)),
    )


def physics_metrics(prediction, intensity, valid):
    modeled = np.abs(np.fft.fft2(np.pad(prediction, 224))) ** 2
    modeled = np.fft.fftshift(modeled)
    observed = np.maximum(intensity, 0)
    v = valid.astype(bool)
    return dict(
        valid_amplitude_residual=float(
            np.linalg.norm((np.sqrt(modeled) - np.sqrt(observed))[v])
            / max(1e-30, np.linalg.norm(np.sqrt(observed)[v]))
        ),
        valid_flux_relative_error=float(
            abs(modeled[v].sum() - observed[v].sum()) / max(1e-30, observed[v].sum())
        ),
    )
