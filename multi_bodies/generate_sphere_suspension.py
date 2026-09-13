#!/usr/bin/env python3
'''
2D Sphere Suspension Generator for RigidMultiblobsWall.

Generates non-overlapping random sphere suspensions in the xy-plane at a fixed height z
above the floor wall for a target area fraction (density) or exact particle count.
Outputs formatted *.clones files to data/ and visual verification plots (PNG).
'''

import argparse
import sys
import os
import math
import json
import tempfile
from geometry import bounding_radius, geometry_report
from visualizer.trajectory_loader import parse_vertex_file
import numpy as np

# Matplotlib headless backend for plotting
try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.patches import Circle
  has_matplotlib = True
except ImportError:
  has_matplotlib = False


def calculate_num_spheres(bounds, target_fraction, radius):
  '''
  Calculate number of spheres N for a target 2D area fraction:
  phi = (N * pi * R^2) / Area
  '''
  Lx = bounds[1] - bounds[0]
  Ly = bounds[3] - bounds[2]
  area = Lx * Ly
  sphere_area = math.pi * (radius ** 2)
  N = int(round((target_fraction * area) / sphere_area))
  return N


def get_periodic_diff(delta, L):
  '''Apply minimum image convention along one dimension.'''
  return delta - L * np.round(delta / L)


def check_overlap(pos, existing_positions, min_dist, periodic_lengths=None):
  '''
  Check if candidate (x, y) overlaps with any existing sphere centers.
  '''
  if len(existing_positions) == 0:
    return False
  
  diffs = existing_positions - pos
  if periodic_lengths is not None:
    if periodic_lengths[0] > 0:
      diffs[:, 0] = get_periodic_diff(diffs[:, 0], periodic_lengths[0])
    if periodic_lengths[1] > 0:
      diffs[:, 1] = get_periodic_diff(diffs[:, 1], periodic_lengths[1])
  
  dists_sq = np.sum(diffs ** 2, axis=1)
  return np.any(dists_sq < (min_dist ** 2))


def place_spheres_rsa(bounds, N, min_dist, periodic_lengths=None, max_attempts=50000):
  '''
  Random Sequential Addition (RSA) in 2D.
  '''
  positions = []
  attempts = 0
  xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
  
  while len(positions) < N and attempts < max_attempts:
    attempts += 1
    
    rx = np.random.uniform(xmin, xmax)
    ry = np.random.uniform(ymin, ymax)
    pos = np.array([rx, ry])
    
    if len(positions) == 0:
      positions.append(pos)
      continue
    
    pos_arr = np.array(positions)
    if not check_overlap(pos, pos_arr, min_dist, periodic_lengths):
      positions.append(pos)
      attempts = 0 # reset streak
      
  return np.array(positions)


def relax_dense_suspension(bounds, initial_positions, min_dist, periodic_lengths=None, max_iters=2000):
  '''
  Force-bias relaxation to push overlapping spheres apart for dense 2D packings.
  '''
  positions = np.copy(initial_positions)
  N = len(positions)
  xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
  
  for _ in range(max_iters):
    displacements = np.zeros_like(positions)
    overlaps_found = 0
    
    for i in range(N):
      diffs = positions - positions[i]
      if periodic_lengths is not None:
        if periodic_lengths[0] > 0:
          diffs[:, 0] = get_periodic_diff(diffs[:, 0], periodic_lengths[0])
        if periodic_lengths[1] > 0:
          diffs[:, 1] = get_periodic_diff(diffs[:, 1], periodic_lengths[1])
          
      dists = np.linalg.norm(diffs, axis=1)
      dists[i] = np.inf # exclude self
      
      overlap_mask = dists < min_dist
      if np.any(overlap_mask):
        overlaps_found += np.sum(overlap_mask)
        overlap_indices = np.where(overlap_mask)[0]
        for j in overlap_indices:
          d = dists[j]
          if d < 1e-8:
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
          else:
            direction = diffs[j] / d
          push = 0.5 * (min_dist - d) * direction
          displacements[i] -= push
          displacements[j] += push
          
    if overlaps_found == 0:
      break
      
    # Apply displacements
    positions += displacements * 0.4
    
    # Boundary handling
    if periodic_lengths is not None and periodic_lengths[0] > 0:
      positions[:, 0] = xmin + (positions[:, 0] - xmin) % (xmax - xmin)
    else:
      positions[:, 0] = np.clip(positions[:, 0], xmin, xmax)
      
    if periodic_lengths is not None and periodic_lengths[1] > 0:
      positions[:, 1] = ymin + (positions[:, 1] - ymin) % (ymax - ymin)
    else:
      positions[:, 1] = np.clip(positions[:, 1], ymin, ymax)
      
  return positions


def check_all_min_distances(positions, periodic_lengths=None):
  '''Compute minimum pairwise distance among all generated spheres.'''
  N = len(positions)
  if N < 2:
    return np.inf
  min_d = np.inf
  for i in range(N):
    diffs = positions[i+1:] - positions[i]
    if periodic_lengths is not None:
      if periodic_lengths[0] > 0:
        diffs[:, 0] = get_periodic_diff(diffs[:, 0], periodic_lengths[0])
      if periodic_lengths[1] > 0:
        diffs[:, 1] = get_periodic_diff(diffs[:, 1], periodic_lengths[1])
    dists = np.linalg.norm(diffs, axis=1)
    if len(dists) > 0:
      local_min = np.min(dists)
      if local_min < min_d:
        min_d = local_min
  return min_d


def generate_sphere_suspension_2d(bounds, target_count=None, target_fraction=None,
                                  radius=1.0, z_fixed=2.0, safety_gap=0.05,
                                  periodic=False, randomize_quaternions=False, seed=None,
                                  vertex_blobs=None, blob_radius=None, geometry_mode='ideal-sphere',
                                  max_retries=3, wall=True, interaction_clearance=0.0):
  '''
  Place non-overlapping spheres in 2D xy-plane at constant height z.
  '''
  bounds = np.asarray(bounds, dtype=float)
  if bounds.shape != (4,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
    raise ValueError('Box must contain finite, increasing x and y bounds')
  if not np.isfinite([radius, z_fixed, safety_gap, interaction_clearance]).all() or radius <= 0 or safety_gap < 0 or interaction_clearance < 0:
    raise ValueError('Radius must be positive; gaps must be nonnegative and finite')
  if target_count is not None and target_fraction is not None:
    raise ValueError('Specify either count or area fraction, not both')
  if target_fraction is not None and (not np.isfinite(target_fraction) or not 0 < target_fraction <= 1):
    raise ValueError('Area fraction must be in (0, 1]')
  if target_count is not None and (not isinstance(target_count, (int, np.integer)) or target_count < 1):
    raise ValueError('Particle count must be a positive integer')
  if not isinstance(max_retries, int) or max_retries < 1:
    raise ValueError('max_retries must be a positive integer')
  if geometry_mode not in ('ideal-sphere', 'bounding-sphere'):
    raise ValueError('Unknown geometry mode')
  exclusion_radius = radius
  if geometry_mode != 'ideal-sphere':
    if vertex_blobs is None or blob_radius is None:
      raise ValueError('Multiblob placement requires vertex geometry and blob_radius')
    exclusion_radius = bounding_radius(vertex_blobs, blob_radius)
  if wall and geometry_mode == 'ideal-sphere' and z_fixed < radius - 1e-7:
    raise ValueError('Sphere surface would penetrate the wall')
  if seed is not None:
    np.random.seed(seed)
    
  min_dist = 2.0 * exclusion_radius * (1.0 + safety_gap) + interaction_clearance
  
  if target_count is not None:
    N = target_count
  elif target_fraction is not None:
    N = calculate_num_spheres(bounds, target_fraction, radius)
  else:
    raise ValueError("Either target_count or target_fraction must be specified.")
    
  Lx = bounds[1] - bounds[0]
  Ly = bounds[3] - bounds[2]
  if N < 1:
    raise ValueError('Requested area fraction rounds to zero bodies; increase fraction or specify count')
  periodic_axes = [bool(periodic)] * 2 if isinstance(periodic, (bool, np.bool_)) else list(periodic)
  if len(periodic_axes) != 2:
    raise ValueError('periodic must specify x and y')
  periodic_lengths = [Lx if periodic_axes[0] else 0, Ly if periodic_axes[1] else 0]
  # Necessary area bound only; passing it does not guarantee finite-box packing.
  if N * math.pi * exclusion_radius**2 > Lx * Ly + 1e-7:
    raise ValueError('Excluded disk area exceeds box area')
  if all(periodic_axes) and N * math.pi * (min_dist/2)**2 / (Lx*Ly) > math.pi/(2*math.sqrt(3)) + 1e-7:
    raise ValueError('Requested periodic disk packing exceeds the hexagonal packing bound including gaps')
  if any(L > 0 and L < min_dist - 1e-7 for L in periodic_lengths):
    raise ValueError('Body exclusion envelope intersects its periodic image')

  place_bounds = list(bounds)
  for axis in range(2):
    if not periodic_axes[axis]:
      place_bounds[2*axis] += exclusion_radius
      place_bounds[2*axis+1] -= exclusion_radius
    if place_bounds[2*axis] > place_bounds[2*axis+1]:
      raise ValueError('Body does not fit within placement box')

  for attempt in range(max_retries):
    positions = place_spheres_rsa(place_bounds, N, min_dist, periodic_lengths)
    if len(positions) < N:
      needed = N - len(positions)
      extra = np.column_stack([np.random.uniform(place_bounds[0], place_bounds[1], needed),
                               np.random.uniform(place_bounds[2], place_bounds[3], needed)])
      combined = np.vstack([positions, extra]) if len(positions) else extra
      positions = relax_dense_suspension(place_bounds, combined, min_dist, periodic_lengths)
    if positions.shape != (N, 2) or not np.isfinite(positions).all():
      raise RuntimeError('Placement returned invalid coordinates')
    inside = all(np.all((positions[:, axis] >= place_bounds[2*axis]-1e-7) &
                        (positions[:, axis] <= place_bounds[2*axis+1]+1e-7)) for axis in range(2))
    if inside and check_all_min_distances(positions, periodic_lengths) >= min_dist - 1e-7:
      break
  else:
    raise RuntimeError(f'Could not satisfy exclusion distance {min_dist:g} after {max_retries} attempts; no configuration saved')

  # Final 3D coordinates (x, y, z_fixed)
  full_coords = np.zeros((N, 3))
  full_coords[:, 0] = positions[:, 0]
  full_coords[:, 1] = positions[:, 1]
  full_coords[:, 2] = z_fixed

  # Quaternions
  quaternions = np.zeros((N, 4))
  if randomize_quaternions:
    u = np.random.uniform(0, 1, (N, 3))
    quaternions[:, 0] = np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1])
    quaternions[:, 1] = np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1])
    quaternions[:, 2] = np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2])
    quaternions[:, 3] = np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])
  else:
    quaternions[:, 0] = 1.0 # identity quaternion

  if vertex_blobs is not None and blob_radius is not None:
    report = geometry_report(full_coords, quaternions, vertex_blobs, blob_radius, periodic_lengths, wall)
    if report['bodies_with_wall_penetration']:
      raise ValueError('Transformed blob surfaces penetrate the wall; increase z_height')
    if geometry_mode != 'ideal-sphere' and (report['overlapping_interbody_blob_pairs'] or
        (report['minimum_periodic_self_clearance'] is not None and report['minimum_periodic_self_clearance'] < -1e-7)):
      raise RuntimeError('Final multiblob geometry validation failed')

  achieved_fraction = (N * math.pi * (radius ** 2)) / (Lx * Ly)
  min_d_observed = check_all_min_distances(positions, periodic_lengths)

  return full_coords, quaternions, achieved_fraction, min_d_observed, min_dist


def save_clones_file(filepath, positions, quaternions):
  '''Write positions and quaternions to *.clones file.'''
  N = len(positions)
  os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
  # Atomic replacement: never leave a partially written configuration.
  positions = np.asarray(positions, dtype=float)
  quaternions = np.asarray(quaternions, dtype=float)
  if positions.shape != (N, 3) or quaternions.shape != (N, 4) or not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
    raise ValueError('Invalid configuration arrays')
  with tempfile.NamedTemporaryFile(mode='w', dir=os.path.dirname(os.path.abspath(filepath)), delete=False) as f:
    temporary = f.name
    f.write('# The format is:\n')
    f.write('# number of rigid bodies\n')
    f.write('# vector_location_body_0  quaternion_body_0\n')
    f.write('# ...\n')
    f.write(f'{N}\n')
    for i in range(N):
      x, y, z = positions[i]
      q0, q1, q2, q3 = quaternions[i]
      f.write(f'{x:.17g}\t{y:.17g}\t{z:.17g}\t{q0:.17g}\t{q1:.17g}\t{q2:.17g}\t{q3:.17g}\n')

  os.replace(temporary, filepath)

def plot_suspension_2d(bounds, positions, radius, achieved_fraction, min_d, min_target, output_plot):
  '''Generate 2D top-down verification plot with circles drawn to true scale.'''
  if not has_matplotlib:
    print("[!] matplotlib is not installed. Skipping plot generation.")
    return

  N = len(positions)
  fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
  xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
  
  # Draw boundary box
  ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'k--', lw=1.5, label='Domain Boundary')
  
  # Draw spheres
  for i in range(N):
    circle = Circle((positions[i, 0], positions[i, 1]), radius, facecolor='#2b7bba', edgecolor='#104e8b', alpha=0.75, lw=1.0)
    ax.add_patch(circle)
    
  ax.set_xlim(xmin - 2*radius, xmax + 2*radius)
  ax.set_ylim(ymin - 2*radius, ymax + 2*radius)
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel('X Position', fontsize=12)
  ax.set_ylabel('Y Position', fontsize=12)
  ax.set_title(f'2D Sphere Suspension (N = {N}, $\\phi_{{2D}} = {achieved_fraction:.3f}$)\n'
               f'Min Dist = {min_d:.3f} (Req >= {min_target:.3f})', fontsize=13)
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.legend(loc='upper right')

  plt.tight_layout()
  os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
  plt.savefig(output_plot, dpi=200)
  plt.close()
  print(f"[+] Visual plot saved to: {output_plot}")


def read_suspension_input_file(filepath):
  '''
  Read key-value options from a *.dat input file matching RigidMultiblobsWall style.
  '''
  options = {}
  if not os.path.exists(filepath):
    sys.exit(f"Error: Input file not found: {filepath}")
    
  with open(filepath, 'r') as f:
    for line in f:
      if '#' in line:
        line, _ = line.split('#', 1)
      line = line.strip()
      if line != '':
        parts = line.split(None, 1)
        if len(parts) == 2:
          if parts[0] == 'structure' and 'structure' in options:
            raise ValueError('Generator accepts one structure; supply an explicit generator configuration for each body type')
          options[parts[0]] = parts[1].strip()
        elif len(parts) == 1:
          options[parts[0]] = ''
  return options


def main():
  parser = argparse.ArgumentParser(
    description="Generate non-overlapping 2D sphere suspensions for RigidMultiblobsWall."
  )
  parser.add_argument('--input-file', type=str, default=None,
                      help="Path to a *.dat configuration file (e.g. inputfile_suspension.dat)")
  parser.add_argument('--box', nargs='+', type=float, default=None,
                      help="Bounding box: 'xmin xmax ymin ymax' or 'Lx Ly'")
  parser.add_argument('--z-height', type=float, default=None,
                      help="Fixed height z above floor wall (default: 2.0 * radius)")
  parser.add_argument('--area-fraction', '--density', '--fraction', dest='fraction', type=float, default=None,
                      help="Target 2D area fraction (e.g. 0.25)")
  parser.add_argument('--num-bodies', '-N', dest='num_bodies', type=int, default=None,
                      help="Explicit number of spheres to place")
  parser.add_argument('--radius', '-R', type=float, default=None,
                      help="Nominal sphere radius for area fraction (default: 1.0)")
  parser.add_argument('--safety-gap', type=float, default=None,
                      help="Fractional buffer above the exclusion diameter (default: 0.05)")
  parser.add_argument('--periodic', action='store_true', default=None,
                      help="Enable periodic boundary condition wrapping in x and y")
  parser.add_argument('--output-clones', type=str, default=None,
                      help="Path to save the generated *.clones file (default: Structures/generated_spheres.clones)")
  parser.add_argument('--plot-image', type=str, default=None,
                      help="Path to save a visual verification PNG plot (default: data/spheres_plot.png)")
  parser.add_argument('--random-quaternions', action='store_true', default=None,
                      help="Randomize 3D orientations (default: identity quaternion 1,0,0,0)")
  parser.add_argument('--seed', type=int, default=None,
                      help="Random number generator seed")
  parser.add_argument('--vertex-file', type=str, default=None,
                      help="Vertex geometry used for conservative placement and validation")

  parser.add_argument('--blob-radius', type=float)
  parser.add_argument('--geometry-mode', choices=['ideal-sphere', 'bounding-sphere'], default=None,
                      help='Default: conservative bounding-sphere; ideal-sphere explicitly uses nominal radius')
  parser.add_argument('--max-retries', type=int, default=None)
  parser.add_argument('--interaction-clearance', type=float, default=None, help='Additional center separation in length units')
  parser.add_argument('--validation-report', type=str)
  args = parser.parse_args()

  # Load options from input file if provided
  file_opts = {}
  if args.input_file is not None:
    file_opts = read_suspension_input_file(args.input_file)
    print(f"[+] Loaded configuration from: {args.input_file}")

  # Resolve box
  box_val = args.box
  if box_val is None and 'box' in file_opts:
    box_val = [float(x) for x in file_opts['box'].split()]
  elif box_val is None and 'periodic_length' in file_opts:
    p_len = [float(x) for x in file_opts['periodic_length'].split()]
    box_val = [0.0, p_len[0], 0.0, p_len[1]]

  if box_val is None:
    sys.exit("Error: Must specify domain boundary via --box or 'box' in input file.")

  if len(box_val) == 2:
    bounds = [0.0, box_val[0], 0.0, box_val[1]]
  elif len(box_val) == 4:
    bounds = [box_val[0], box_val[1], box_val[2], box_val[3]]
  else:
    sys.exit("Error: box must provide 2 numbers (Lx Ly) or 4 numbers (xmin xmax ymin ymax).")

  # Resolve radius
  radius = args.radius
  if radius is None:
    if 'radius' in file_opts:
      radius = float(file_opts['radius'])
    elif 'sphere_radius' in file_opts:
      radius = float(file_opts['sphere_radius'])
    else:
      radius = 1.0

  # Resolve z_height
  z_fixed = args.z_height
  if z_fixed is None:
    if 'z_height' in file_opts:
      z_fixed = float(file_opts['z_height'])
    else:
      z_fixed = 2.0 * radius


  # Resolve density / num_bodies
  fraction = args.fraction
  if fraction is None:
    if 'area_fraction' in file_opts:
      fraction = float(file_opts['area_fraction'])
    elif 'density' in file_opts:
      fraction = float(file_opts['density'])
    elif 'fraction' in file_opts:
      fraction = float(file_opts['fraction'])

  # Explicit CLI count/fraction overrides the alternate choice in the file.
  if args.num_bodies is not None and args.fraction is None:
    fraction = None
  num_bodies = args.num_bodies
  if num_bodies is None:
    if 'num_bodies' in file_opts:
      num_bodies = int(file_opts['num_bodies'])
    elif 'N' in file_opts:
      num_bodies = int(file_opts['N'])

  if args.fraction is not None and args.num_bodies is None:
    num_bodies = None
  if num_bodies is None and fraction is None:
    sys.exit("Error: Must specify either --density/fraction or --num-bodies/N in CLI or input file.")

  # Resolve safety_gap
  safety_gap = args.safety_gap
  if safety_gap is None:
    if 'safety_gap' in file_opts:
      safety_gap = float(file_opts['safety_gap'])
    else:
      safety_gap = 0.05

  # Resolve periodic
  periodic = args.periodic if args.periodic is not None else False
  if not periodic and 'periodic' in file_opts:
    periodic = file_opts['periodic'].lower() in ['true', '1', 'yes']

  if 'periodic_length' in file_opts:
    declared_lengths = [float(v) for v in file_opts['periodic_length'].split()]
    if len(declared_lengths) < 2 or not np.isfinite(declared_lengths).all() or min(declared_lengths[:2]) < 0:
      raise ValueError('periodic_length requires nonnegative finite x and y lengths')
  if args.periodic is None and 'periodic' not in file_opts and 'periodic_length' in file_opts:
    lengths = [float(v) for v in file_opts['periodic_length'].split()]
    periodic = [lengths[0] > 0, lengths[1] > 0]
  if 'periodic_length' in file_opts:
    lengths = [float(v) for v in file_opts['periodic_length'].split()]
    axes = [periodic]*2 if isinstance(periodic, bool) else periodic
    for axis in range(2):
      if axes[axis] and not np.isclose(lengths[axis], bounds[2*axis+1]-bounds[2*axis]):
        raise ValueError('Placement box must match periodic_length on periodic axes')

  # Resolve outputs
  output_clones = args.output_clones or file_opts.get('output_clones') or 'Structures/generated_spheres.clones'
  plot_image = args.plot_image or file_opts.get('plot_image') or 'data/spheres_plot.png'
  vertex_file = args.vertex_file or file_opts.get('vertex_file') or 'Structures/shell_N_12_Rg_1.vertex'
  
  if 'structure' in file_opts and args.vertex_file is None and 'vertex_file' not in file_opts:
    vertex_file = file_opts['structure'].split()[0]
  if not os.path.exists(vertex_file) and args.input_file:
    vertex_file = os.path.join(os.path.dirname(os.path.abspath(args.input_file)), vertex_file)
  geometry_mode = args.geometry_mode or file_opts.get('geometry_mode', 'bounding-sphere')
  blob_radius = args.blob_radius if args.blob_radius is not None else float(file_opts['blob_radius']) if 'blob_radius' in file_opts else None
  vertices = parse_vertex_file(vertex_file)
  if geometry_mode != 'ideal-sphere' and (vertices is None or blob_radius is None):
    raise ValueError('Bounding-sphere placement requires a valid vertex_file and blob_radius from the simulation input')
  wall = file_opts.get('domain', 'single_wall') != 'no_wall'

  # Resolve random_quaternions
  rand_quat = args.random_quaternions if args.random_quaternions is not None else False
  if not rand_quat and 'random_quaternions' in file_opts:
    rand_quat = file_opts['random_quaternions'].lower() in ['true', '1', 'yes']

  # Resolve seed
  seed = args.seed
  if seed is None and 'seed' in file_opts:
    seed = int(file_opts['seed'])

  print("=" * 60)
  print("2D Sphere Suspension Generator")
  print("=" * 60)
  print(f"Domain Bounds       : [x: {bounds[0]} -> {bounds[1]}, y: {bounds[2]} -> {bounds[3]}]")
  print(f"Nominal Radius (R)  : {radius}")
  print(f"Fixed Z Height      : {z_fixed}")
  print(f"Safety Gap Buffer   : {safety_gap * 100:.1f}% of exclusion diameter")
  print(f"Periodic Boundaries : {periodic}")

  # Generate positions
  positions, quaternions, achieved_fraction, min_d, min_target = generate_sphere_suspension_2d(
    bounds=bounds,
    target_count=num_bodies,
    target_fraction=fraction,
    radius=radius,
    z_fixed=z_fixed,
    safety_gap=safety_gap,
    periodic=periodic,
    randomize_quaternions=rand_quat,
    seed=seed,
    vertex_blobs=vertices, blob_radius=blob_radius, geometry_mode=geometry_mode,
    max_retries=args.max_retries if args.max_retries is not None else int(file_opts.get('max_retries', 3)),
    wall=wall,
    interaction_clearance=args.interaction_clearance if args.interaction_clearance is not None else float(file_opts.get('interaction_clearance', 0))
  )

  N = len(positions)
  print("-" * 60)
  print(f"Placed Spheres (N)  : {N}")
  print(f"Area Fraction (phi) : {achieved_fraction:.4f}")
  print(f"Min Distance Found  : {min_d:.4f} (Required >= {min_target:.4f})")
  
  if min_d >= min_target - 1e-6:
    print("[SUCCESS] Configuration satisfies the selected exclusion policy!")
  else:
    raise RuntimeError("Final center separation check failed")

  if min_d < min_target - 1e-7:
    raise RuntimeError('Final separation validation failed; no configuration saved')
  lengths = [bounds[1]-bounds[0], bounds[3]-bounds[2]]
  axes = [periodic]*2 if isinstance(periodic, bool) else periodic
  lengths = [L if enabled else 0 for L, enabled in zip(lengths, axes)]
  report = geometry_report(positions, quaternions, vertices, blob_radius, lengths, wall) if vertices is not None and blob_radius is not None else {}
  report.update({'placement_policy': geometry_mode, 'nominal_radius': radius,
                 'bounding_radius': bounding_radius(vertices, blob_radius) if vertices is not None and blob_radius is not None else None,
                 'blob_radius': blob_radius, 'seed': seed, 'num_bodies': N,
                 'requested_area_fraction': fraction, 'achieved_nominal_area_fraction': achieved_fraction,
                 'number_density': N / ((bounds[1]-bounds[0])*(bounds[3]-bounds[2])),
                 'minimum_center_distance': min_d if np.isfinite(min_d) else None,
                 'required_center_distance': min_target, 'status': 'valid_under_placement_policy'})

  # Save clones file
  save_clones_file(output_clones, positions, quaternions)
  print(f"[+] Output clones file saved to: {output_clones}")

  report_path = args.validation_report or output_clones + '.validation.json'
  os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
  with open(report_path, 'w') as f:
    json.dump(report, f, indent=2, allow_nan=False)
  print(f'[+] Validation report saved to: {report_path}')

  # Plot image
  if plot_image:
    plot_suspension_2d(bounds, positions, radius if geometry_mode == 'ideal-sphere' else bounding_radius(vertices, blob_radius), achieved_fraction, min_d, min_target, plot_image)

  # Print inputfile snippet
  print("=" * 60)
  print("HOW TO USE IN YOUR SIMULATION:")
  print(f"Add this line to your multi_bodies inputfile (e.g. inputfile_dynamic.dat):")
  print(f"structure {vertex_file} {output_clones}")
  if any(lengths):
    Lx, Ly = lengths
    print(f"periodic_length {Lx:.1f} {Ly:.1f} 0.0")
  print("=" * 60)


if __name__ == '__main__':
  main()

