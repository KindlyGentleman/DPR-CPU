"""CPU-only inference; auto remains reference until all promotion gates pass."""

import hashlib
from pathlib import Path
import threading
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "823572aa0c4bc973acf149734b79833cb9fccf708aad23426373b8c62b5dc379"


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def default_checkpoint():
    source = ROOT / "pretrained/param_dpr1.pt"
    return source if source.is_file() else Path.cwd() / "pretrained/param_dpr1.pt"


def verify_checkpoint(path):
    digest = file_sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {digest}")
    return digest


def prepare_inputs(intensity, valid_mask):
    def array(x):
        if isinstance(x, torch.Tensor):
            if x.device.type != "cpu":
                raise ValueError("CPU input required")
            x = x.detach().numpy()
        return np.asarray(x)

    x, m = array(intensity), array(valid_mask)
    if x.shape != (512, 512) or m.shape != (512, 512):
        raise ValueError("intensity and valid_mask must both be 512x512")
    if np.iscomplexobj(x) or np.iscomplexobj(m):
        raise ValueError("Real detector intensity and binary mask required")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.array(x, dtype=np.float32, copy=True)
        valid = np.isin(m, (0, 1)).all()
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric detector arrays required") from exc
    if not np.isfinite(x).all() or not valid:
        raise ValueError("Finite FP32 intensity and binary mask required")
    m = np.array(m, dtype=np.float32, copy=True)
    if not np.any((x > 0) & (m == 1)):
        raise ValueError("Positive flux on at least one valid pixel required")
    return torch.from_numpy(x[None, None]), torch.from_numpy(m[None, None])


def configure_threads(threads, interop=1):
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if torch.get_num_interop_threads() != interop:
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError as exc:
            raise RuntimeError(
                "Set inter-op threads before parallel work; use a fresh process"
            ) from exc
    torch.set_num_threads(threads)


def configure_affinity(affinity):
    """Optionally restrict this process to explicitly chosen available logical CPUs."""
    if affinity is None:
        return
    if (
        not isinstance(affinity, (list, tuple))
        or not affinity
        or any(type(v) is not int or v < 0 for v in affinity)
    ):
        raise ValueError(
            "affinity must be a nonempty list/tuple of logical CPU integers"
        )
    try:
        import psutil
    except ImportError as exc:
        raise ImportError("Explicit CPU affinity requires optional psutil") from exc
    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        raise RuntimeError("CPU affinity is unavailable on this platform")
    requested = sorted(set(affinity))
    if not set(requested).issubset(process.cpu_affinity()):
        raise ValueError("Requested affinity exceeds the currently available CPU set")
    process.cpu_affinity(requested)


def load_reference(checkpoint):
    from deeppr import Network

    verify_checkpoint(checkpoint)
    model = (
        Network(
            ngf=64,
            max_features=1024,
            weight_model=True,
            downsample_FFC=False,
            refinement=True,
        )
        .cpu()
        .float()
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.eval()


class DPRCPU:
    """Reusable CPU model. Threads/affinity affect the process; calls serialize per instance.

    predict accepts 512x512 real arrays and returns an owned 64x64 FP32 amplitude
    array. Experimental backends are opt-in and do not imply scientific acceptance.
    """

    def __init__(self, model, backend, checkpoint, allow_fallback):
        self.model, self.backend = model, backend
        self.checkpoint, self.allow_fallback = checkpoint, allow_fallback
        self._lock = threading.Lock()

    @staticmethod
    def auto_backend():
        return "reference"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint=None,
        backend="auto",
        threads=4,
        allow_fallback=False,
        affinity=None,
    ):
        from .candidates import NAMES, apply_candidate

        backend = cls.auto_backend() if backend == "auto" else backend
        if backend not in NAMES:
            raise ValueError(f"Unknown backend {backend}; choose from {sorted(NAMES)}")
        configure_affinity(affinity)
        configure_threads(threads)
        checkpoint = Path(checkpoint) if checkpoint else default_checkpoint()
        reference = load_reference(checkpoint)
        try:
            model = apply_candidate(reference, backend)
        except Exception as exc:
            if not allow_fallback:
                raise
            warnings.warn(
                f"{backend} unavailable: {exc}; falling back to reference",
                RuntimeWarning,
            )
            model, backend = load_reference(checkpoint), "reference"
        return cls(model, backend, checkpoint, allow_fallback)

    def predict(self, intensity, valid_mask):
        x, m = prepare_inputs(intensity, valid_mask)
        with (
            self._lock,
            torch.inference_mode(),
            torch.autocast(device_type="cpu", enabled=False),
        ):
            try:
                out = self.model(x, m, false_scale=False)
            except Exception as exc:
                if not self.allow_fallback or self.backend == "reference":
                    raise
                warnings.warn(
                    f"{self.backend} execution failed: {exc}; falling back to reference",
                    RuntimeWarning,
                )
                self.model, self.backend = load_reference(self.checkpoint), "reference"
                out = self.model(x, m, false_scale=False)
            if out.shape != (1, 1, 64, 64) or not torch.isfinite(out).all():
                raise RuntimeError("Backend must return finite 1x1x64x64 amplitude")
            return out.numpy()[0, 0].copy()
