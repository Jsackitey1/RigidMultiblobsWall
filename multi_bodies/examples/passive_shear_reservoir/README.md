# Passive shear: reservoir to initially empty fluid

Minimum end-to-end pilot for issues #7, #5 and #8. This is a reproducible numerical
transport demonstration, not a calibrated experiment or a completed near-wall
sphere benchmark. Both regions contain fluid; only the downstream region is
initially free of particles. The dashed interface is an observation marker,
not a physical wall or gate.

From the repository root:

```sh
python3 multi_bodies/examples/passive_shear_reservoir/run_pilot.py
```

This overwrites only the example's generated `multi_bodies/data/passive_shear/`
outputs. It generates initial conditions, runs the pilot and two controls,
checks the results, and creates `video/spheres_simulation.mp4`. Use the Python
environment with the repository dependencies, NumPy, SciPy, Numba, Matplotlib,
OpenCV, and an available video encoder (see `visualizer/video_writer.py`).

## Pilot choices

All quantities use consistent nondimensional units. `pilot.dat` is the complete
input, shared by the generator, solver and visualizer. Nominal sphere radius is
1; the existing sphere has 12 blobs with radius 0.25 and an envelope radius of
1.25. No claim of spatial convergence is made for this coarse representation.
The viewing window is x=[0,64], y=[0,16]; the reservoir is x=[0,16]. The generator
uses nominal reservoir area fraction 0.15, seed 42, and initial height 2.5.
This dilute pilot avoids using dense packing to mask implementation errors.
Height is an initial condition, not an enforced equilibrium or fixed constraint.

The shear rate is 0.5, viscosity 1, and thermal forcing is disabled. Gravity is
0.01 per unit blob mass; wall and pair repulsion strengths are 0.1, with decay
lengths 0.5 and 0.1 respectively. These are pilot settings, not measured colloid
properties. Pair forces act at blob level, not at a nominal spherical surface.
The simulation runs to time 24 with dt=0.02 and saves every 10 steps.

The floor is no-slip, x is open, and y is periodic in the existing implementation.
The Numba mobility uses nearest translated image boxes (finite image sum), not a
converged infinite periodic Stokes solver. Transverse-size/image convergence and
multiblob resolution studies remain required for quantitative research.

## Physics implementation

`shear_rate` defaults to zero. The general solver subtracts
`u_infinity=(shear_rate*z,0,0)` from the blob slip RHS, using current blob locations:

```text
M lambda - K U = slip - u_infinity
K^T lambda = external force/torque
```

The passive no-slip example has zero physical slip. Body rotation and
shear-induced disturbance forces follow from the coupled solve; they are not
added afterward at body centers. Nonzero shear is currently supported only for
single-wall general multiblob deterministic forward Euler and Adams–Bashforth,
with nonperiodic z and wall-compatible mobility implementations. Roller schemes are rejected. The separate nonperiodic Brownian shear example
also supports the dense first-order RFD stochastic scheme; other stochastic
schemes are rejected. No pressure-driven channel is modeled.

## Initial conditions and observations

`box` denotes the full observation window. Optional `reservoir_end` limits initial
placement to its upstream portion and prohibits x periodicity. Concentration is
computed over that placement slab; the validation JSON stores both bounds and the
initial downstream count. Existing inputs without `reservoir_end` keep their
previous behavior. Use `periodic_length 0 Ly 0` for y-only periodicity; do not use
`--periodic`, which enables both x and y.

The MP4 shows the fixed window, initial interface, periodic image copies, trails,
and number of centers downstream. Color represents x displacement divided by the
saved time interval, not an instantaneous fluid velocity. The final frame repeats
the last measured interval; animation interpolation does not create analysis data.
Time is nondimensional. Particle trajectories are not fluid streamlines.

`video/validation.json` includes raw-frame downstream occupancy, signed endpoint
crossings, interval net flux, and 40-bin longitudinal number density (number per
unit x). Centers on the interface count as downstream. Multiple crossings within
one saved interval are unresolved. Downstream occupancy includes particles beyond
the viewing window; histogram out-of-window counts are reported separately.

## Checks and limits

`run_pilot.py` also runs identical initial conditions with zero shear and with
half dt at matching saved times. `check_runs.py` requires an initially empty
region, positive downstream transfer, no control crossings, maximum positional
difference below 0.01 nominal radii, and no interbody blob overlap or wall
penetration at saved pilot frames. It writes `pilot_checks.json` and per-run
diagnostics. These checks do not assert continuous-time non-overlap.

The automated tests include force/torque balance, zero flow, shear reversal,
far-wall translation/rotation limits, reservoir metadata, and synthetic crossing
counts. Run from the repository root:

```sh
PYTHONPYCACHEPREFIX=/tmp/rigid-pycache MPLCONFIGDIR=/tmp/rigid-mpl python3 -m unittest discover -s multi_bodies/tests -v
```

Do not infer diffusion, migration, shock formation or near-wall quantitative
accuracy from this pilot. The verified finite-sphere benchmark, physical parameter
selection, domain/resolution convergence, and later research stages remain open.
