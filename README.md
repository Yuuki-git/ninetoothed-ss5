# NineToothed SS5 gfx936 build dependency

This repository contains the complete Python package from the original
NineToothed 0.26.0 wheel. The first commit imports its package files byte for
byte, with the original Apache 2.0 license. `UPSTREAM.json` records every
imported file and the wheel SHA256. It is an independently packaged source
snapshot, rather than a GitHub fork of the upstream development branch.

The second commit includes the two source changes used in the SS5 S3/S4
compilation experiments:

* `generation.py` reads `NINETOOTHED_CACHE_DIR`. Set this variable to an existing
  writable directory inside the current workspace before importing NineToothed.
* `aot.py` accepts the vendor Triton compiler's HIP `.cpp` output and a single
  `gfx936` variant. The SS5 build driver provides `_ss5_hip_variant` to convert
  that output into the HIP host launcher. This package alone does not provide
  a standalone HIP AOT command.

The extra `pyproject.toml` packages the unchanged upstream modules plus those
two changes as version `0.26.0+ss5.1`. No vendor Torch, Triton, or FlashAttention
source is included or replaced. Keep the existing vendor Triton installation;
install this repository with `--no-deps --no-build-isolation` into a private
build environment. NumPy 1.26.4 is a build dependency in that environment.

The executable build driver and the three operator definitions are in
[the public entry repository](https://github.com/Yuuki-git/minimax-h3-8gpu).
It compiles device code from these sources during the build. The three resulting
HIP shared libraries are used at model runtime; this Python package is needed
only for compilation.

The checked-in `ninetoothed/aot.py` SHA256 is
`fc86e4aa739ac6bc61d3e638a18ac960ef9a48cc1d5d876716bd4645c5d5efb3`.
