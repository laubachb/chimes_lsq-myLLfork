.. _page-gpu_acceleration:

GPU acceleration (experimental)
===============================

``chimes_lsq`` can optionally use CUDA to accelerate construction of the Chebyshev design matrix (the :math:`\mathrm{\mathbf{A}}` matrix). This targets the derivative evaluation in force-matching fits and is separate from GPU molecular dynamics in the ChIMES calculator.

Overview
--------

When enabled, 2-, 3-, and 4-body Chebyshev derivatives are computed on the GPU in a single pass per trajectory frame, then scattered into the same ``A_MAT`` structure used by the CPU code. If the GPU path cannot run (unsupported options, no device, build without CUDA), execution falls back to the existing CPU implementation automatically.

Prerequisites
-------------

- An NVIDIA GPU and driver. ``nvidia-smi --query-gpu=compute_cap --format=csv,noheader`` reports the compute capability you'll need below (e.g. ``9.0`` for H100, ``8.0`` for A100).
- A CUDA Toolkit (``nvcc`` on ``PATH``, or loaded via your site's modules).
- CMake **3.18 or newer** for the GPU build specifically (the CPU-only build only requires the project's own ``cmake_minimum_required`` floor of 3.10).

Building with CUDA
------------------

On Stampede3, load the GPU module stack and pass ``DOGPU=1`` as the fifth argument to ``install.sh``:

.. code-block:: bash

   export hosttype=UT-TACC-GPU
   ./install.sh 0 "" 1 1 1

The fifth argument enables ``-DWITH_CUDA=ON`` in CMake. Without it, the install is CPU-only and unchanged from prior releases.

On a machine without a site module stack, build manually and pass your GPU's compute capability explicitly — ``CMakeLists.txt`` does not set ``CMAKE_CUDA_ARCHITECTURES`` on its own, and an unset/mismatched value can produce a binary that builds cleanly but fails at kernel-launch time:

.. code-block:: bash

   nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # e.g. "9.0"

   cd build
   cmake -DWITH_CUDA=ON -DUSE_MPI=1 -DCMAKE_CUDA_ARCHITECTURES=90 ..   # 90=H100, 80=A100, 86=RTX30xx/A40, 75=T4/RTX20xx
   make

(CMake 3.24+ also accepts ``-DCMAKE_CUDA_ARCHITECTURES=native`` to auto-detect from the GPU visible at configure time.)

Enabling at runtime
-------------------

Add to ``fm_setup.in``:

.. code-block:: bash

   # USEGPU # true
   # BINARYA # true    # optional

Or set environment variables before launching ``chimes_lsq``:

.. code-block:: bash

   export CHIMES_LSQ_USE_GPU=1
   export CHIMES_LSQ_BINARY_A=1      # optional
   export CHIMES_LSQ_GPU_DEVICE=0    # optional; default is rank % num_devices

MPI ranks map to GPUs via ``rank % cudaGetDeviceCount()``.

After launching, check stdout for ``GPU A-matrix build enabled (CUDA; rank 0 -> device N)`` to confirm the GPU path is actually active, as opposed to ``WARNING: USE_GPU set but no CUDA device found; using CPU``.

Validation
----------

From the repository root after a GPU build:

.. code-block:: bash

   ./scripts/gpu_validate.sh test_suite-lsq/special3b

This compares CPU and GPU ``A``/``b`` outputs for a 2B+3B test case to a tolerance of ``1e-10``, and should end with ``PASS: GPU A/b matches CPU``. Run it again with other test-suite cases (any directory containing an ``fm_setup.in`` and matching ``.xyzf``) before relying on GPU output for a different fitting configuration — in particular, there isn't yet a dedicated 4-body regression case, so validate manually against a 4B fit before trusting it.

Limitations
-----------

The GPU path is not used when ``HIERARCHICAL_FIT`` or ``FITCOUL`` is enabled, when polynomial orders exceed 24, or when CUDA is unavailable. Inner-cutoff Chebyshev derivative fixes (``ZERO_DERIV``, ``CONSTANT_DERIV``, and ``SMOOTH``) are mirrored on the GPU.

3-body and 4-body GPU enumeration build a per-atom neighbor candidate list capped at compile-time constants (``LSQ_GPU_MAX_NEIGH3``/``LSQ_GPU_MAX_NEIGH4`` in ``src/chimes_lsq_gpu.cu``). If a system's actual neighbor count exceeds the cap for some atom, that frame falls back to the CPU path automatically (correct, just slower) and a ``GPU neighbor-list overflow`` message is printed — raise the constant and rebuild if you need GPU coverage for unusually dense systems or large cutoffs.

Binary ``A`` output writes the generated ``A`` rows as doubles; the Python solver still expects text ``A.*.txt``.

For the full implementation status, troubleshooting table, and tech-debt checklist, see ``doc/GPU_ACCELERATION.md`` in the repository.
