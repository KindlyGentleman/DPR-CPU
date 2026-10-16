# Attribution and optional dependencies

Original `deeppr/`, pretrained weights, LICENSE.txt and LICENSES/ remain unchanged.
The FourierUnit instrumentation preserves the operation sequence of Sung Yun Lee's
DPR/FFC implementation and is subject to the repository's existing license notices.
The new wrappers call PyTorch public APIs (and its version-specific oneDNN helper)
without copying third-party implementation files. No external native DLL is bundled.

Dependencies used for experiments: PyTorch (BSD-style), NumPy/SciPy/scikit-image
(BSD-style), psutil (BSD-3-Clause), Matplotlib (Matplotlib license), SciencePlots
(MIT), pytest (MIT), pyFFTW wrapper (BSD-3-Clause). FFTW itself has separate GPL
license terms. Installing a wheel locally does not establish permission to
redistribute FFTW DLLs under the repository's BSD license. This branch does not
redistribute those libraries or require them for original reference inference.

Scientific metric definitions follow the already completed local study: integer
translation/inversion, outside-window energy, fixed truth-based PSNR/SSIM range and
no fitted amplitude. External implementation references and adaptation details are
recorded in the separately preserved local campaign `reports/REFERENCES.md`;
the compact public report is `../docs/CPU_RESULTS.md`. No published speedup is
imported as a DPR result.
