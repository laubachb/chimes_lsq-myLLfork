# GPU acceleration for A-matrix construction

Optional CUDA support speeds up the Chebyshev derivative step that builds the design matrix during `chimes_lsq` runs. This is distinct from GPU MD forces in [chimes_calculator](https://github.com/rk-lindsey/chimes_calculator); here the target is **fitting**, not simulation.

## Status

| Component | Status |
|-----------|--------|
| 2-body (`Deriv_2B`) CUDA kernel | Implemented |
| 3-body (`Deriv_3B`) CUDA kernel | Implemented |
| 4-body (`Deriv_4B`) CUDA kernel | Implemented |
| MPI rank → GPU device mapping | Implemented |
| GPU neighbor enumeration (fused 2B/3B/4B) | Implemented (2B brute-force O(N·nall); 3B/4B neighbor-list-bound, see below) |
| Static device table cache | Implemented |
| Inner-cutoff Cheby fixes (`ZERO_DERIV`, `CONSTANT_DERIV`, `SMOOTH`) | Implemented |
| Binary A-matrix output (force/stress/energy rows) | Implemented |
| CPU fallback for unsupported cases | Implemented |
| Multi-frame device batching (single D2H) | Not implemented |
| `CHIMES_LSQ_GPU_BATCH_FRAMES` sync grouping | Partial |
| Binary A reader in `chimes_lsq.py` | Not implemented |
| CI / automated GPU validation | Not implemented |

Last updated: 2026-06 (branch `laubachb/gpu-acceleartion`).

## Prerequisites

- **NVIDIA GPU + driver** compatible with the CUDA Toolkit you build against. Check with `nvidia-smi` (top-right corner shows the max supported CUDA version) and `nvidia-smi --query-gpu=compute_cap --format=csv` (you'll need this compute-capability number for the build step below).
- **CUDA Toolkit** (`nvcc` on `PATH`, or load via your site's module system). The TACC GPU module stack (`modfiles/UT-TACC-GPU.mod`) pins `cuda/12.4`; other toolkit versions newer than ~11.0 should work but aren't routinely tested here.
- **CMake ≥ 3.18** to build the CUDA sources (`enable_language(CUDA)` + `CMAKE_CUDA_ARCHITECTURES` need this; the project's own `cmake_minimum_required` is 3.10 for the CPU-only build, but the GPU path needs a newer CMake). `modfiles/UT-TACC-GPU.mod` loads `cmake/3.28.1`.
- **C++/MPI compiler** — same as the CPU build (`USE_MPI=1` is independent of `WITH_CUDA`; you can build GPU-accelerated + MPI, or GPU-accelerated + serial).
- A test case to validate against once built — see [Validation](#validation) below.

## Build

### On Stampede3 (or any site with a `UT-TACC-GPU`-style module stack)

```bash
export hosttype=UT-TACC-GPU   # loads intel, impi, cmake, python, cuda — see modfiles/UT-TACC-GPU.mod
./install.sh 0 "" 1 1 1       # 5th argument DOGPU=1 enables -DWITH_CUDA=ON
```

### Manual build (any machine with `nvcc` on `PATH`)

`CMakeLists.txt` does **not** currently set `CMAKE_CUDA_ARCHITECTURES`, so without an explicit value CMake/nvcc falls back to a toolkit-default compute capability that may not match your GPU — this can show up later as a *build that succeeds* but a *kernel launch that fails at runtime* (`no kernel image is available for execution on the device`, see [Troubleshooting](#troubleshooting)). Always pass it explicitly:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. "9.0" for H100, "8.0" for A100

cd build
cmake -DWITH_CUDA=ON -DUSE_MPI=1 -DCMAKE_CUDA_ARCHITECTURES=90 ..   # 90 = H100; 80 = A100; 86 = RTX 30xx/A40; 75 = T4/RTX 20xx
make
```

(CMake ≥ 3.24 also accepts `-DCMAKE_CUDA_ARCHITECTURES=native` to auto-detect the architecture of the GPU visible at configure time.)

Confirm the build actually picked up CUDA by checking for `Building CUDA object .../chimes_lsq_gpu.cu.o` in the `make` output, or simply that `chimes_lsq_gpu.cu` and `chimes_lsq_gpu_host.cpp` produced `.o` files under `build/CMakeFiles/`.

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

### Quick sanity check after install

Run any small `fm_setup.in` case with `CHIMES_LSQ_USE_GPU=1` and check stdout for one of:

- `GPU A-matrix build enabled (CUDA; rank 0 -> device N)` — GPU path is active.
- `WARNING: USE_GPU set but no CUDA device found; using CPU` — printed once, by rank 0, when `lsq_gpu_available()` finds no device (see [Troubleshooting](#troubleshooting)).

This is the cheapest way to confirm the binary was actually built with `WITH_CUDA=ON` *and* that a GPU is visible to the process before trusting any performance numbers.

### Tuning the 3B/4B neighbor-list caps

3-body and 4-body GPU enumeration build a per-atom candidate list (`kBuildNeighborList` in `chimes_lsq_gpu.cu`) capped at `LSQ_GPU_MAX_NEIGH3` (256) and `LSQ_GPU_MAX_NEIGH4` (96) neighbors per atom. These are compile-time `#define`s, not runtime flags. If a run prints:

```
GPU neighbor-list overflow (cap=256); increase LSQ_GPU_MAX_NEIGH3/4 or reduce cutoff/system size
```

the actual neighbor count for some atom (within the 3B or 4B cutoff + padding) exceeded the cap. The run still completes correctly — that frame falls back to the CPU path automatically — but raise the relevant `#define` in `src/chimes_lsq_gpu.cu` and rebuild if you want GPU coverage for that system (very dense systems, large cutoffs, or small/thin boxes with heavy ghost replication are the usual cause).

### MPI

Each rank selects `device_id = rank % cudaGetDeviceCount()`. Run on GPU nodes with one rank per GPU (or fewer ranks than GPUs) for best utilization. Pin a specific rank to a specific device with `CHIMES_LSQ_GPU_DEVICE=N` if automatic mapping doesn't match your job's GPU allocation (e.g. under a job scheduler that doesn't expose all node GPUs to every rank).

## Validation

First-time walkthrough, after a GPU build (`build/chimes_lsq` exists and was built with `WITH_CUDA=ON`):

```bash
./scripts/gpu_validate.sh                          # default: test_suite-lsq/special3b (2B+3B)
```

Expected output ends with `PASS: GPU A/b matches CPU` and two `max abs diff` lines near `0e+00` (tolerance is `1e-10`). A `FAIL` here means the GPU and CPU paths disagree numerically — don't trust GPU output for that input style until resolved.

Run against other cases the same way:

```bash
./scripts/gpu_validate.sh test_suite-lsq/<case>    # any case with fm_setup.in + .xyzf
```

The script runs CPU and GPU builds in a temp directory and compares `A`/`b` text files element-wise (tolerance `1e-10`).

**Not yet done:** validation on Stampede3 GPU compute nodes in CI; a dedicated 4B regression case (the 3B/4B enumeration path changed significantly — see [Tier 1 pipeline](#tier-1-pipeline-implemented) below — so a 4B-specific `gpu_validate.sh` run is the highest-priority manual check before relying on 4-body GPU fits); stress/energy-inclusive fits.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` | Built without (or with the wrong) `CMAKE_CUDA_ARCHITECTURES` for your GPU | Rebuild with `-DCMAKE_CUDA_ARCHITECTURES=<your compute capability ×10>` (see [Build](#build)) |
| `WARNING: USE_GPU set but no CUDA device found; using CPU` | No GPU visible to the process | Check `nvidia-smi` runs in the same shell/job; check `CUDA_VISIBLE_DEVICES`; confirm you're on a GPU node/allocation |
| `GPU neighbor-list overflow (cap=...)` | System/cutoff denser than the compiled-in 3B/4B neighbor cap | Falls back to CPU automatically (correct, just slower); raise `LSQ_GPU_MAX_NEIGH3`/`LSQ_GPU_MAX_NEIGH4` in `chimes_lsq_gpu.cu` and rebuild if you need GPU coverage |
| `GPU enum job count too large` | Extremely large frame even after the neighbor-list fix | Falls back to CPU automatically; consider reducing ghost-atom padding/layers if GPU coverage matters for this system |
| `CUDA error at ... — ...` then CPU fallback | Any failed `cudaMalloc`/`cudaMemcpy`/kernel launch (e.g. out of GPU memory) | Check `nvidia-smi` for memory pressure from other jobs; reduce `CHIMES_LSQ_GPU_BATCH_FRAMES` (shouldn't matter today, see below) or system size |
| GPU run completes but isn't faster than CPU | `CHIMES_LSQ_GPU_BATCH_FRAMES` is currently a no-op for the per-frame round trip (see [Status](#status) — "Partial"); per-frame host↔device sync dominates for small/cheap frames | Expected with the current implementation; true multi-frame batching is tracked in [Tech debt](#tech-debt--follow-ups) |
| Numbers differ between CPU and GPU beyond `1e-10` | Possible regression, or a fit option not yet covered by GPU math | Run `scripts/gpu_validate.sh` against the failing case's input style; check [CPU fallback](#cpu-fallback) conditions in case the comparison itself is invalid (e.g. comparing a `HIERARCHICAL_FIT` run, which never uses GPU) |

## Tier 1 pipeline (implemented)

When `USE_GPU` is enabled, the GPU path now:

1. **Skips CPU neighbor-list rebuild** (`DO_UPDATE`) and enumerates pairs/trips/quads on device from coordinates (MIC-aware).
2. **Caches static tables** on device (pair params, cluster metadata, type lookup maps) across frames.
3. **Fuses enumeration + derivative kernels** — no host-side `build_*` pair lists.
4. **3B/4B enumeration is neighbor-list-bound, not brute-force.** Each real atom's
   candidate set is built once per frame (`kBuildNeighborList`, O(natoms·nall),
   same cost class as 2B) at the 3B/4B cutoff radius, capped at
   `LSQ_GPU_MAX_NEIGH3`/`LSQ_GPU_MAX_NEIGH4` (256/96, see `chimes_lsq_gpu.cu`).
   3B/4B combinations are then generated only from that list — O(natoms·k²)/
   O(natoms·k³) with k ≤ the cap, instead of the previous O(natoms·nall²)/
   O(natoms·nall³) all-tuples scan, which could exceed `INT_MAX` threads and
   silently fall back to CPU for non-trivial systems (ghost-inflated `nall` in
   the thousands made 4B enumeration infeasible). This restriction is exact,
   not approximate: any valid cluster containing atom `a1` must have every
   other member within `a1`'s cutoff, so neither candidate set narrowing nor
   the existing per-edge distance checks change which clusters are found —
   only how many wasted candidates are evaluated to find them. If the actual
   neighbor count exceeds the cap, enumeration fails cleanly (logged) and
   falls back to the CPU path rather than truncating results.
5. **Optional binary-only A output** — set `CHIMES_LSQ_BINARY_A=1` and `CHIMES_LSQ_BINARY_ONLY=1` (or `# BINARYA #` + skip text) to write `A.NNNN.bin` without `A.NNNN.txt`. Stress and energy rows are included in binary when fitted.

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
8. **`CMAKE_CUDA_ARCHITECTURES` not set by `CMakeLists.txt`** — `WITH_CUDA` enables `enable_language(CUDA)` without ever setting `CMAKE_CUDA_ARCHITECTURES`, so it silently relies on the toolkit's default and on every builder remembering to pass it manually (see [Build](#build)). Worth setting a sane default (or requiring it explicitly with a clear error) in `CMakeLists.txt` directly rather than only documenting the workaround here.
9. **Per-atom neighbor caps are compile-time constants** — `LSQ_GPU_MAX_NEIGH3`/`LSQ_GPU_MAX_NEIGH4` (256/96) are `#define`s in `chimes_lsq_gpu.cu`, not CLI/input-file tunable. Fine for now (overflow falls back to CPU safely), but a runtime override would avoid needing a rebuild for unusually dense systems.

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
