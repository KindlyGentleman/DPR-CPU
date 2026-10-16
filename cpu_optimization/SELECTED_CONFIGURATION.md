# Selected machine-specific configuration

The unchanged `config/selection.json` locks reference FP32 DPR1, six intra-op,
one inter-op thread and logical CPUs 0–11 on the tested i7-13620H topology.
The raw screen/repeat status is inconclusive pending repetition; the subsequent
lock records configuration-level acceptance using both experiments. Five locked
published examples then passed quality replay. This does not select a compiler
or convolution replacement. Portable auto/default remains four-thread reference.

See [completed scientific results](../docs/CPU_RESULTS.md) for exact paired values,
the negative compiler findings and limitations. Do not edit this lock to promote
a future result; a new campaign requires its own selection and replay evidence.
