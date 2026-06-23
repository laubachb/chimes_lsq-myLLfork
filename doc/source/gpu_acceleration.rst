.. _page-gpu_acceleration:

GPU acceleration (experimental)
===============================

``chimes_lsq`` can optionally use CUDA to accelerate construction of the Chebyshev design matrix (the :math:`\mathrm{\mathbf{A}}` matrix). This targets the derivative evaluation in force-matching fits and is separate from GPU molecular dynamics in the ChIMES calculator.

Overview
--------

When enabled, 2-, 3-, and 4-body Chebyshev derivatives are computed on the GPU in a single pass per trajectory frame, then scattered into the same ``A_MAT`` structure used by the CPU code. If the GPU path cannot run (unsupported options, no device, build without CUDA), execution falls back to the existing CPU implementation automatically.

Building with CUDA
------------------

On Stampede3, load the GPU module stack and pass ``DOGPU=1`` as the fifth argument to ``install.sh``:

.. code-block:: bash

   export hosttype=UT-TACC-GPU
   ./install.sh 0 "" 1 1 1

The fifth argument enables ``-DWITH_CUDA=ON`` in CMake. Without it, the install is CPU-only and unchanged from prior releases.

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

Validation
----------

From the repository root after a GPU build:

.. code-block:: bash

   ./scripts/gpu_validate.sh test_suite-lsq/special3b

This compares CPU and GPU ``A``/``b`` outputs for a 2B+3B test case.

Limitations
-----------

The GPU path is not used when ``HIERARCHICAL_FIT`` or ``FITCOUL`` is enabled, when polynomial orders exceed 24, or when CUDA is unavailable. Inner-cutoff Chebyshev derivative fixes (``ZERO_DERIV``, ``CONSTANT_DERIV``, and ``SMOOTH``) are mirrored on the GPU.

Binary ``A`` output writes the generated ``A`` rows as doubles; the Python solver still expects text ``A.*.txt``.

For the full implementation status and tech-debt checklist, see ``doc/GPU_ACCELERATION.md`` in the repository.
