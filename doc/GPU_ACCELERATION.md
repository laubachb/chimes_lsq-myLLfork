# GPU acceleration for A-matrix construction

Optional CUDA support speeds up the Chebyshev derivative step that builds the design matrix during `chimes_lsq` runs. This is distinct from GPU MD forces in [chimes_calculator](https://github.com/rk-lindsey/chimes_calculator); here the target is **fitting**, not simulation.

## Status

| Component | Status |
|-----------|--------|
| 2-body (`Deriv_2B`) CUDA kernel | Implemented |
| 3-body (`Deriv_3B`) CUDA kernel | Implemented |
| 4-body (`Deriv_4B`) CUDA kernel | Implemented |
| MPI rank → GPU device mapping | Implemented |
| Binary A-matrix output (force rows) | Implemented |
| CPU fallback for unsupported cases | Implemented |
| Multi-frame GPU batching | Not implemented |
| Binary A reader in `chimes_lsq.py` | Not implemented |
| CI / automated GPU validation | Not implemented |

Last updated: 2025-06 (branch `laubachb/gpu-acceleartion`).

## Build

Requires CUDA toolkit and a C++ compiler with MPI (same as CPU build).

```bash
export hosttype=UT-TACC-GPU   # loads intel, impi, cmake, python, cuda — see modfiles/UT-TACC-GPU.mod
./install.sh 0 "" 1 1 1       # 5th argument DOGPU=1 enables -DWITH_CUDA=ON
```

Or manually:

```bash
cd build
cmake -DWITH_CUDA=ON -DUSE_MPI=1 ..
make
```

Without `WITH_CUDA`, all GPU code is compiled out and behavior matches the CPU-only tree.

## Runtime

GPU use is **opt-in**. Default installs behave exactly as before.

### Input file (`fm_setup.in`)

```
# USEGPU # true
# BINARYA # true    # optional: write A.NNNN.bin (force rows only)
```

### Environment overrides

| Variable | Effect |
|----------|--------|
| `CHIMES_LSQ_USE_GPU=1` | Enable GPU path (overrides input if set) |
| `CHIMES_LSQ_BINARY_A=1` | Enable binary A output |
| `CHIMES_LSQ_GPU_DEVICE=N` | Pin MPI rank to device `N` (default: `rank % num_devices`) |

If `USE_GPU` is requested but no CUDA device is found, rank 0 prints a warning and the run continues on CPU.

### MPI

Each rank selects `device_id = rank % cudaGetDeviceCount()`. Run on GPU nodes with one rank per GPU (or fewer ranks than GPUs) for best utilization.

## Validation

```bash
./scripts/gpu_validate.sh                          # default: test_suite-lsq/special3b (2B+3B)
./scripts/gpu_validate.sh test_suite-lsq/<case>    # custom case
```

The script runs CPU and GPU builds in a temp directory and compares `A`/`b` text files element-wise (tolerance `1e-10`).

**Not yet done:** validation on Stampede3 GPU compute nodes in CI; 4B-specific regression case; stress/energy-inclusive fits.

## Architecture

```
ZCalc_Deriv (functions.C)
  └─ lsq_gpu_deriv_cheby()          [chimes_lsq_gpu_host.cpp]
       ├─ build_2b_pairs / build_trips / build_quads
       ├─ build_cluster_tables (3B/4B metadata)
       ├─ lsq_gpu_launch_deriv_2b/3b/4b   [chimes_lsq_gpu.cu]
       └─ scatter_to_amat → A_MAT
```

Entry point: `src/chimes_lsq_gpu.cuh`. Shared structs: `src/chimes_lsq_gpu_types.h`.

Kernels mirror CPU `Cheby::Deriv_2B`, `Deriv_3B`, and `Deriv_4B` (cluster cutoffs, `ALLOWED_POWERS`, Chebyshev transforms, `DERIV_CONST` scaling).

## CPU fallback

`lsq_gpu_deriv_cheby()` returns `false` and `ZCalc_Deriv` runs the original CPU path when:

- Not built with `USE_CUDA`, or GPU init failed
- `HIERARCHICAL_FIT` or `FIT_COUL` is enabled
- Pair type is not `CHEBYSHEV`
- Any 2B/3B/4B polynomial order exceeds `LSQ_MAX_POLY_ORDER` (24 in `chimes_lsq_gpu_types.h`)
- A CUDA launch or memcpy fails

**Gap:** inner-cutoff `cheby_fix_type` smoothing (`ZERO_DERIV`, `CONSTANT_DERIV`, `SMOOTH`) is applied on CPU but **not** in GPU kernels. Fits that rely on non-default inner-cutoff derivative fixes may disagree with GPU results until this is implemented.

## Tech debt / follow-ups

Track these when extending or reviewing the GPU path:

1. **Inner-cutoff Cheby fixes** — Port `Cheby::cheby_fix` logic into GPU `set_polys` / derivative evaluation.
2. **Multi-frame batching** — `JOB_CONTROL.GPU_BATCH_FRAMES` is reserved; currently one frame per GPU accumulation.
3. **Binary A format** — Only force rows go to `A.NNNN.bin`; stress and energy rows remain text-only. `chimes_lsq.py` / DLASSO do not read binary A yet.
4. **Host-side caching** — Trip/quad cluster tables and pair params are rebuilt and re-uploaded every frame; cache when geometry/hyperparameters are unchanged across frames.
5. **Polynomial order cap** — Raise or remove `LSQ_MAX_POLY_ORDER` (affects GPU stack arrays in 4B kernel).
6. **Automated testing** — Add GPU-node job to CI or document a manual release checklist; extend `gpu_validate.sh` for 4B and stress/energy fits.
7. **Performance profiling** — Measure PCIe transfer vs kernel time; consider persistent device buffers and CUDA graphs for production campaigns.
8. **Documentation sync** — Keep this file, `doc/source/gpu_acceleration.rst`, and the PR description aligned when behavior changes.

## File index

| Path | Role |
|------|------|
| `src/chimes_lsq_gpu.cu` | CUDA kernels and device memory |
| `src/chimes_lsq_gpu_host.cpp` | Host enumeration, upload, scatter |
| `src/chimes_lsq_gpu.cuh` | Public C++ API |
| `src/chimes_lsq_gpu_types.h` | Host/device structs, internal launch API |
| `src/functions.C` | `ZCalc_Deriv` GPU dispatch |
| `src/chimes_lsq.C` | Per-rank `lsq_gpu_init` / `finalize` |
| `src/A_Matrix.C` | Text + optional binary row output |
| `scripts/gpu_validate.sh` | CPU vs GPU regression helper |
| `modfiles/UT-TACC-GPU.mod` | Stampede3 GPU module stack |
| `CMakeLists.txt` | `WITH_CUDA` option |
