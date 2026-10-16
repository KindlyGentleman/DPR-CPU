"""Explicit reusable real FFTW plans; inverse scratch input is always restored."""

import numpy as np
import torch


class FFTWPlan:
    def __init__(
        self, shape, direction, norm="ortho", threads=1, effort="FFTW_ESTIMATE"
    ):
        import pyfftw

        self.shape, self.direction, self.norm = tuple(shape), direction, norm
        real_shape = tuple(shape)
        complex_shape = real_shape[:-1] + (real_shape[-1] // 2 + 1,)
        self.input = pyfftw.empty_aligned(
            real_shape if direction == "forward" else complex_shape,
            dtype="float32" if direction == "forward" else "complex64",
        )
        self.output = pyfftw.empty_aligned(
            complex_shape if direction == "forward" else real_shape,
            dtype="complex64" if direction == "forward" else "float32",
        )
        self.plan = pyfftw.FFTW(
            self.input,
            self.output,
            axes=(-2, -1),
            direction="FFTW_FORWARD" if direction == "forward" else "FFTW_BACKWARD",
            flags=(effort,),
            threads=threads,
            planning_timelimit=1.0,
        )

    def kernel(self):
        # For real inverse FFTW destroys scratch input. Callers of kernel must
        # restore it before EACH timed kernel; bridge includes this copy.
        # normalise_idft=False selects forward normalization in pyFFTW. The
        # backward convention therefore uses True for BOTH directions.
        return self.plan(
            normalise_idft=self.norm == "backward", ortho=self.norm == "ortho"
        )

    def bridge(self, tensor):
        np.copyto(self.input, tensor.detach().numpy())
        return torch.from_numpy(self.kernel().copy())
