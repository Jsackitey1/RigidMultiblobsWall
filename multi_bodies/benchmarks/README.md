# Iterative Brownian shear validation and benchmarks

The dense `stochastic_first_order_RFD_dense_algebra` scheme remains the small-system
reference. `stochastic_traction_EM` now supports shear with a single wall and
nonperiodic boundaries. Other stochastic shear schemes remain blocked.

The traction method's auxiliary RFD solve uses an explicit random RHS, without
background flow. Its physical solve evaluates `calc_slip` after restoring the
unperturbed configuration, adding shear together with thermal noise and drift.
The existing algorithm therefore needs no extra shear drift term. Small body
preconditioners remain CPU-based; `pycuda` accelerates global mobility products
and (in the benchmarks) blob pair forces. This is not a fully GPU-resident solver.
GMRES failures and traction Lanczos iteration-limit exhaustion now stop the run
rather than silently accepting an unconverged solution.

From the repository root, in the simulation Python environment:

```sh
export MPLCONFIGDIR=/tmp/rigid-mpl
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=2
python -m unittest discover -s multi_bodies/tests -p 'test_brownian_shear.py' -v
python -m unittest discover -s multi_bodies/tests -p 'test_iterative_brownian_shear.py' -v
```

On the NVIDIA server, run the same reference checks against GPU mobility products:

```sh
RIGID_TEST_GPU=1 python -m unittest discover -s multi_bodies/tests -p 'test_iterative_brownian_shear.py' -v
```

Tests cover two interacting bodies' deterministic response, matched-noise shear
increments, the full translation/rotation covariance versus dense mobility, and
RFD drift versus finite-difference mobility derivatives. Brownian trajectories
from different algorithms need not match pointwise. These are implementation
checks, not a long-time equilibrium or timestep convergence study.

Only after those checks pass, benchmark isolated short runs:

```sh
python multi_bodies/benchmarks/benchmark_brownian_shear.py --backend pycuda --counts 50 100 500 --steps 10 --warmup 2
```

CPU comparison and small dense reference:

```sh
python multi_bodies/benchmarks/benchmark_brownian_shear.py --backend numba --counts 50 100 500 --steps 10 --warmup 2
python multi_bodies/benchmarks/benchmark_brownian_shear.py --backend numba --scheme stochastic_first_order_RFD_dense_algebra --counts 50 --steps 5 --warmup 1
```

Each invocation creates a timestamped `multi_bodies/data/benchmark_brownian_*`
directory containing per-size input snapshots, generated clones, trajectories,
logs, and `summary.json`. Production settings and previous outputs are untouched.
Inputs inherit the physical parameters from `inputfile_dynamic.dat`; geometry
bounds are resized for comparable nominal initial density. Timings exclude the
requested warmup steps. GPU memory is sampled per solver PID approximately once
per second and may miss a short-lived peak. Null memory means unavailable, not
zero usage. Linear runtime projections are only short-run estimates; contacts,
convergence, memory pressure, and shared-machine load can change performance.
Use the saved configs when comparing results; these are not the earlier server
initial conditions. Confirm generated counts and non-overlap reports.

For detached server benchmarking after validation:

```sh
nohup python -u multi_bodies/benchmarks/benchmark_brownian_shear.py --backend pycuda --counts 50 100 500 --steps 10 --warmup 2 > /tmp/rigid-gpu-benchmark.log 2>&1 < /dev/null &
tail -f /tmp/rigid-gpu-benchmark.log
```

After GPU validation and benchmarking, the production input can use:

```text
scheme                                   stochastic_traction_EM
mobility_blobs_implementation            python
mobility_vector_prod_implementation      pycuda
blob_blob_force_implementation           pycuda
periodic_length                          0 0 0
update_PC                                1
```

Do not automatically restart the 50,000-step run. First examine timing, convergence,
and close-contact behavior. Soft repulsion still does not enforce hard-sphere
non-overlap. Keep the dense solver for regression tests.

## Local smoke benchmark, 2026-10-15

Apple M2, Numba mobility, two Numba threads and one BLAS thread; five integration
steps with the first excluded from timing. Iterative traction EM averaged 0.211 s
per step at 50 particles, 0.756 s at 100, and 18.362 s at 500. The matching
50-particle dense reference averaged 1.002 s per step. These four-step averages
are preliminary CPU measurements, not GPU results or long-run performance claims.
All six saved frames per case contained finite coordinates and unit quaternions.
The shear test suite passed 11 CPU tests, with five CUDA tests skipped locally.
GPU validation and memory measurements remain pending on the RTX 4070 Ti server.
