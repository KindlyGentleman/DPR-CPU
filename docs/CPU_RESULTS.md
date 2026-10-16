# Completed frozen-model CPU results

The model is the original FP32 DPR1, checkpoint SHA-256
`823572aa0c4bc973acf149734b79833cb9fccf708aad23426373b8c62b5dc379`.
This release changes communication, packaging and maintenance tooling, not the
upstream network, checkpoint, historical measurements, selection lock or tests.

## Scheduling result

The two fixed-process trials `fixed_t6_p_only_v1` and
`fixed_t6_p_only_repeat72_v1` support the explicitly selected configuration:
six intra-op threads, one inter-op, reference backend, logical CPUs 0–11 on the
tested i7-13620H topology (six P cores, both SMT siblings). The repeat has 72
randomized pairs: reference median 0.77705005 s, candidate median 0.70511025 s,
candidate p95 0.879229935 s, paired ratio 1.1020263143, 95% CI
[1.0676809002, 1.1295978659]. This is about 9.26% lower median latency, not 10.2%
lower latency. The historical protocol's numeric gate is speedup ≥1.10.

Each arm runs in its own persistent worker. Thread-related environment settings
are fixed at worker creation. The isolated comparison supports a configuration
effect, but does not separate core placement, thread count, scheduling overhead,
memory behavior or thermal effects. Temperatures were unavailable. The portable
four-thread/inherited-affinity default remains unchanged.

The raw trial summaries retain their original `inconclusive` status, which
required independent repetition. The subsequent immutable selection lock refers
to the screen and repeat hashes; it is the source of the configuration-level
acceptance. The five-example locked replay then passed with no new failures and
maximum absolute amplitude difference 7.65547e-7. This was a validation replay,
not a new timing speedup or full 570-case evaluation.

## Compiler and convolution extension

V3 supplied characterization, profiling and local prototypes. V4 recovered a
working MSVC native toolchain and completed selective compilation; the first
full-model attempt was blocked before paired clocks. V5 completed the full
wrapper using the explicitly reused native cache. Graph capture alone was not
counted as native execution. Campaign IDs, original manifest hashes, terminal
commands and summaries are in the compact campaign index.

Selective compilation: 36 pairs, 0.924032014× [0.728673767, 1.038126951], quality
passed. Full compilation: 36 fresh pairs, 0.932621935× (see exact source CSV for
unrounded value), CI [0.90681, 0.97152], strict parity failed on two development
measurements / 11 pixels. Neither was promoted. These trials compare against
their own four-thread reference arms; they are not direct comparisons against
the selected six-P-core configuration. Native execution success establishes
toolchain feasibility, not faster or numerically acceptable DPR inference.

All four local prototypes were slower than native convolution; tile-16 alone
passed strict FP32 operator parity. Algebraic equivalence alone does not ensure
FP32 parity. Primitive timings include input transforms/tiling/reassembly and
cannot establish a whole-model gain. Warm compiler clocks exclude startup and
warmup; v5 is not a cold compilation measurement.

## Actual bottleneck and fidelity

The saved `profile_v2` contains three instrumented forwards after ten warmups,
with bitwise instrumented output agreement. Operator self CPU time totals
2,415,575 μs. The repeated `[1,256,66,66]` input / `[256,256,3,3]` MKLDNN group
accounts for 1,029,147.4 μs, or 42.6046%; FFT kernel self-time is 2.28827%.
The remaining share is 55.1071%. WPC/FFC include substantial spatial convolution;
the name FFC does not mean its full stage time is FFT work. Inclusive scopes
overlap operator calls and are not added to these percentages.

With that profile held fixed, making FFT kernels free gives an ideal Amdahl bound
`1/(1-0.0228827091) = 1.02342×`. This diagnostic bound is not an observed
optimization and assumes no other workload changes.

Quality uses the original 18 development measurements from seven base objects:
elementwise reference-anchored `rtol=1e-4, atol=1e-5`, per-case NRMSE increase
and SSIM loss ≤1e-4, no new NRMSE>0.5 failures. No amplitude fitting is performed.
Model-to-model parity does not prove reconstruction correctness against truth;
the unchanged historical reference has 251/570 aligned NRMSE>0.5 failures.

## Evidence scope and preservation

`cpu_study/evidence/manifest.json` hashes byte-preserved summaries, paired rows,
quality records, profile operators and convolution rows. The plotting script
replays each interval using its saved seed/draw/block parameters. The compact
campaign index is an explicit extraction; it retains the hashes of the original
v3/v4/v5 manifests but omits machine-local commands and unrelated research data.
The original manifests, raw prediction arrays, source ZIPs and compiler logs are
preserved in the local research tree and its separately hashed backup. They are
not all distributed. Hashes identify unavailable records; they do not make those
records independently reconstructible from the public subset.

The public subset reproduces plotted paired statistics and operator shares,
and includes five genuine archived demonstration inputs. It does not recreate
the unavailable full historical detector population, reproduce compiler binaries,
or independently revalidate the entire historical campaign. The recorded
extension audits were parent artifact checks, not independent scientific review.
No synthetic policy-selection inference is part of these CPU findings.

Original config provenance/selection records retain intentional historical
absolute paths and hashes as immutable metadata. None is a required execution
path in the documented portable commands. The separate research project is not
included. License texts and the original upstream test tolerance are preserved.

Visualization/review workflow guidance: Kassis, Agarwal, He, Patel and Brueckner
(2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
