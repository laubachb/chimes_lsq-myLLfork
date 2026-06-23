<p style="text-align:center;">
    <img src="./doc/ChIMES-lsq_Github_logo-2.png" alt="" width="250"/>
</p>
<hr>

**chimes_lsq** is the ChIMES parameter generator: it builds the least-squares design matrix (A) and right-hand side (b) from reference trajectories, then solves for Chebyshev coefficients via `chimes_lsq.py`.

<hr>

Building
--------

Standard install (MPI enabled by default):

```bash
export hosttype=UT-TACC    # or UM-ARC, LLNL-LC, etc. — see modfiles/
./install.sh
```

Arguments: `./install.sh <debug 0|1> <install_prefix> <verbosity 0-3> <MPI 0|1> <CUDA 0|1>`

**GPU build** (CUDA A-matrix acceleration; requires a GPU module stack):

```bash
export hosttype=UT-TACC-GPU
./install.sh 0 "" 1 1 1
```

Building manually on a machine without a site module stack? Pass `-DCMAKE_CUDA_ARCHITECTURES=<your GPU's compute capability>` explicitly — it's not set automatically and an unset/mismatched value can build cleanly but fail at runtime.

See [doc/GPU_ACCELERATION.md](doc/GPU_ACCELERATION.md) for prerequisites, manual build instructions, runtime options, validation, and troubleshooting.

<hr>

Documentation
----------------

[**Full documentation**](https://chimes-lsq.readthedocs.io/en/latest/) is available.

<hr>

Community
------------------------

Questions, discussion, and contributions (e.g. bug fixes, documentation, and extensions) are welcome. 

Additional Resources: [ChIMES Google group](https://groups.google.com/g/chimes_software).

<hr>

Contributing
------------------------

Contributions to the ChIMES generator should be made through a pull request, with ``GH-develop`` as the destination branch. A test suite log file should be attached to the PR. For additional contributing guidelines, see the documentation.

<hr>

Releases
--------

For most users, we recommend using the ChIMES calculator stable releases.

[stable releases](https://github.com/rk-lindsey/chimes_lsq/releases).

<hr>

Authors
----------------

The ChIMES generator was developed by Rebecca K. Lindsey, Nir Goldman, and Laurence E Fried.

Contributors can be found [here](https://github.com/rk-lindsey/chimes_lsq/graphs/contributors).

<hr>

Citing
----------------

See [the documentation](https://chimes-lsq.readthedocs.io/en/latest/citing.html) for guidance on referencing ChIMES and the ChIMES calculator in a publication.

<hr>

License
----------------

The ChIMES calculator is distributed under terms of 
[LGPL v3.0 License](https://github.com/rk-lindsey/chimes_lsq/blob/main/LICENSE).

LLNL-CODE-835874
