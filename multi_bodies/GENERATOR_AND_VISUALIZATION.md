# Initial conditions and simulation visualization

Run these commands from `multi_bodies/` with NumPy and Matplotlib installed.
Video export also needs the encoder dependencies described by `visualizer/video_writer.py`.

```sh
python3 generate_sphere_suspension.py --input-file inputfile_suspension.dat --seed 42
python3 multi_bodies.py --input-file inputfile_dynamic.dat
python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode all --diagnostic-plots
```

The generator writes a `.clones` file, an initial XY plot, and
`<output-clones>.validation.json`. Existing simulation output is not regenerated
by the visualizer. Regenerate initial conditions and rerun the solver to produce
a trajectory using the new exclusion policy.

## Radius and exclusion policy

The CLI defaults to `--geometry-mode bounding-sphere`. It requires a vertex file
and the **same positive `blob_radius` used by the solver**. The example generator
input supplies both. `--vertex-file` and `--blob-radius` override the file values.
A single solver `structure` line can also supply the vertex path. Relative vertex
paths are resolved from the working directory, then the input file directory.

- `radius` / `--radius`: nominal radius used for area fraction, not an inferred
  hydrodynamic radius.
- `bounding_radius`: `max(norm(vertex)) + blob_radius`, about the solver's body
  reference point. The reference vertices are not recentered.
- `blob_radius`: radius of each blob envelope.
- `safety_gap`: dimensionless fractional buffer above the exclusion diameter.
- `interaction_clearance`: optional additional separation in length units. It is
  not automatically inferred from the repulsion's Debye length.

Placement requires center separation
`2 * exclusion_radius * (1 + safety_gap) + interaction_clearance`.
The bounding-sphere policy uses `bounding_radius` as its exclusion radius and
checks transformed inter-body blob surfaces and blob-wall surfaces before saving.
It is conservative for nonspherical bodies. It is not an exact-shape packing
algorithm: exact blob checks validate the result, but do not relax the bounding
sphere placement restriction.

`--geometry-mode ideal-sphere` explicitly retains nominal sphere center exclusion.
The Python function also retains that default for compatibility. When vertex
geometry and blob radius are supplied, the report includes actual blob clearances;
inter-body blob overlap does not fail the ideal-sphere policy, but blob-wall
penetration still does. With `domain no_wall`, wall checks are disabled.

These are geometric initialization policies. The solver regularizes some blob
and wall overlaps; reported envelope overlap does not by itself invalidate its
numerical method. Intrabody blob overlaps are excluded from inter-body diagnostics.

Generation validates input, checks necessary packing bounds, tries RSA and bounded
relaxation/retries, and fails before writing if separation remains unresolved.
The hexagonal disk bound including the gap is used only for two periodic axes.
Passing a necessary bound does not establish finite-box packability.
`--max-retries` defaults to 3. Coordinates are saved with round-trip precision and
the clones file is replaced atomically after validation.

`--area-fraction` is the preferred spelling; `--density` and `--fraction` remain
aliases. The report distinguishes requested and achieved nominal area fraction
`N*pi*radius**2/(Lx*Ly)` and number density `N/(Lx*Ly)`. Counts are rounded;
a fraction that rounds to zero is rejected. Explicit CLI count or fraction
replaces the alternate choice in the input file. Supplying both on the CLI fails.

`periodic_length` infers periodicity per axis unless `periodic` or `--periodic`
explicitly selects a policy. Supply a nonzero `box` extent for a nonperiodic axis.
Box lengths must match active periodic lengths. Nonperiodic box edges constrain
initial placement only; they do not add side walls to the dynamics solver.

## Raw trajectories and animation

`positions_unwrapped` preserves solver positions; `positions_wrapped` (and the
compatibility alias `positions`) is for display. Periodic axes are independent.
Raw-frame displacement rates use each actual time interval. MSD is the ensemble
mean squared **3D displacement of the body reference point from the first saved
frame**, including any drift. It is not a time-origin-averaged or drift-subtracted
diffusion estimator.

The main solver writes unwrapped positions. For externally wrapped-only data,
`--coordinates wrapped` explicitly requests minimum-image reconstruction, which
assumes less than half-cell displacement between saved frames. Larger motion
cannot be recovered without additional image information.

`.config` files contain no timestamps. By default times are constructed using
`dt`, `n_save`, and the first saved step at/after `initial_step`. Use
`--timestamps times.txt` for irregular output or a trajectory whose time origin
cannot be inferred; this file must contain one strictly increasing time per frame.

Animation uses linear interpolation in unwrapped position and SLERP in orientation,
then wraps for display. An interpolated trajectory has
`metadata['interpolated_for_visualization'] = True`, references its raw trajectory
through `analysis_source`, and has `msd = None`. Displayed displacement rates and
height summary values reference the preceding saved frame. Reports always use the
raw source, regardless of animation FPS. The final displacement-rate value repeats
the last measured interval for display. Brownian displacement rates are finite-time
measurements, not smooth instantaneous velocities.

Both MP4 and HTML top views draw periodic image copies. MP4 trails split at boundary
jumps. The HTML side view supports blob and envelope geometry and periodic X copies;
wall indicators are disabled for `no_wall`. A displayed envelope is not necessarily
the physical or hydrodynamic surface. `--radius` overrides display size only.
The animation keeps at least all original frames, so a short requested duration
can produce a longer MP4 at the chosen FPS; the CLI prints the actual MP4 duration.

For inputs with several structures, select `--structure-index N` and the matching
`--config` file. The loader does not silently apply the first geometry to another
structure. Combined heterogeneous/articulated files are not supported by this
single-geometry visualizer.

## Diagnostics and tests

Every visualization run writes `validation.json` from raw frames, including MSD,
height statistics, **initial-frame** inter-body and periodic self clearances, and
wall clearance across all saved frames when geometry and a wall are present.
Pairwise checks cost quadratically in the number of bodies and blobs; they are
not performed at every trajectory frame. No claim of all-frame inter-body
non-overlap is made.

`--diagnostic-plots` adds an initial X-Z center view and saved-frame height histogram.
For fully XY-periodic data with at least two bodies, it also plots initial projected
`g(r)` using unordered pairs, the `N*(N-1)/2` normalization, and full annuli only up
to half the shorter cell length. No uncorrected finite-container `g(r)` is produced.
Height histograms weight saved frames equally; irregular timestamps do not imply
time-weighted equilibrium sampling.

```sh
python3 -m unittest discover -s tests -v
```

Tests cover periodic crossings, large unwrapped displacements, irregular times,
restart offsets, raw/animation separation, generator failure, geometry and wall
clearance, reproducibility, quaternion agreement with the solver, and rendering.
They validate these additions, not the entire hydrodynamics solver.
