# GPU acceleration for A-matrix construction

Optional CUDA support speeds up the Chebyshev derivative step that builds the design matrix during `chimes_lsq` runs. This is distinct from GPU MD forces in [chimes_calculator](https://github.com/rk-lindsey/chimes_calculator); here the target is **fitting**, not simulation.

## Status

| Component | Status |
|-----------|--------|
| 2-body (`Deriv_2B`) CUDA kernel | Implemented |
| 3-body (`Deriv_3B`) CUDA kernel | Implemented |
| 4-body (`Deriv_4B`) CUDA kernel | Implemented |
| MPI rank → GPU device mapping | Implemented |
| GPU neighbor enumeration (fused 2B/3B/4B) | Implemented |
| Static device table cache | Implemented |
| Inner-cutoff Cheby fixes (`ZERO_DERIV`, `CONSTANT_DERIV`, `SMOOTH`) | Implemented |
| Binary A-matrix output (force/stress/energy rows) | Implemented |
| CPU fallback for unsupported cases | Implemented |
| Multi-frame device batching (single D2H) | Not implemented |
| `CHIMES_LSQ_GPU_BATCH_FRAMES` sync grouping | Partial |
| Binary A reader in `chimes_lsq.py` | Not implemented |
| CI / automated GPU validation | Not implemented |

Last updated: 2026-06 (branch `laubachb/gpu-acceleartion`).

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
# BINARYA # true    # optional: write A.NNNN.bin
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

## Tier 1 pipeline (implemented)

When `USE_GPU` is enabled, the GPU path now:

1. **Skips CPU neighbor-list rebuild** (`DO_UPDATE`) and enumerates pairs/trips/quads on device from coordinates (MIC-aware).
2. **Caches static tables** on device (pair params, cluster metadata, type lookup maps) across frames.
3. **Fuses enumeration + derivative kernels** — no host-side `build_*` pair lists.
4. **Optional binary-only A output** — set `CHIMES_LSQ_BINARY_A=1` and `CHIMES_LSQ_BINARY_ONLY=1` (or `# BINARYA #` + skip text) to write `A.NNNN.bin` without `A.NNNN.txt`. Stress and energy rows are included in binary when fitted.

| Variable | Effect |
|----------|--------|
| `CHIMES_LSQ_GPU_BATCH_FRAMES=N` | Group CUDA sync points (default 1) |
| `CHIMES_LSQ_BINARY_ONLY=1` | Suppress `A.NNNN.txt` when binary is enabled |
| `CHIMES_LSQ_SKIP_TEXT_A=1` | Same as above |

## Architecture

```
ZCalc_Deriv (functions.C)
  └─ lsq_gpu_deriv_cheby()          [chimes_lsq_gpu_host.cpp]
       ├─ upload static tables (once) + frame coords
       ├─ lsq_gpu_enumerate_2b/3b/4b  [GPU neighbor + cluster filter]
       ├─ lsq_gpu_launch_deriv_*_device
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

Inner-cutoff `cheby_fix_type` handling (`ZERO_DERIV`, `CONSTANT_DERIV`, `SMOOTH`) is mirrored in GPU polynomial evaluation for many-body cutoffs.

## Tech debt / follow-ups

Track these when extending or reviewing the GPU path:

1. **True multi-frame device batching** — accumulate N frames on GPU before host download (needs per-frame device buffers).
2. **Binary A reader** — `chimes_lsq.py` / DLASSO do not read `A.NNNN.bin` yet.
3. **Host-side caching** — static tables cached on device; coords still uploaded per frame.
4. **Polynomial order cap** — Raise or remove `LSQ_MAX_POLY_ORDER` (affects GPU stack arrays in 4B kernel).
5. **Automated testing** — Add GPU-node job to CI or document a manual release checklist; extend `gpu_validate.sh` for 4B and stress/energy fits.
6. **Performance profiling** — Measure PCIe transfer vs kernel time; consider persistent device buffers and CUDA graphs for production campaigns.
7. **Documentation sync** — Keep this file, `doc/source/gpu_acceleration.rst`, and the PR description aligned when behavior changes.

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
