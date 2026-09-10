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
  return max(1, N)


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
                                  periodic=False, randomize_quaternions=False, seed=None):
  '''
  Place non-overlapping spheres in 2D xy-plane at constant height z.
  '''
  if seed is not None:
    np.random.seed(seed)
    
  min_dist = 2.0 * radius * (1.0 + safety_gap)
  
  if target_count is not None:
    N = target_count
  elif target_fraction is not None:
    N = calculate_num_spheres(bounds, target_fraction, radius)
  else:
    raise ValueError("Either target_count or target_fraction must be specified.")
    
  Lx = bounds[1] - bounds[0]
  Ly = bounds[3] - bounds[2]
  periodic_lengths = [Lx, Ly] if periodic else None

  # Effective placement bounds (shrunk by radius for hard wall boundaries)
  if periodic:
    place_bounds = list(bounds)
  else:
    place_bounds = [bounds[0] + radius, bounds[1] - radius,
                    bounds[2] + radius, bounds[3] - radius]

  # 1. RSA placement
  positions = place_spheres_rsa(place_bounds, N, min_dist, periodic_lengths)
  
  # 2. Dense relaxation if needed
  if len(positions) < N:
    needed = N - len(positions)
    print(f"[*] RSA placed {len(positions)}/{N} spheres. Running relaxation for remaining {needed} spheres...")
    extra = np.zeros((needed, 2))
    extra[:, 0] = np.random.uniform(place_bounds[0], place_bounds[1], needed)
    extra[:, 1] = np.random.uniform(place_bounds[2], place_bounds[3], needed)
    combined = np.vstack([positions, extra]) if len(positions) > 0 else extra
    positions = relax_dense_suspension(place_bounds, combined, min_dist, periodic_lengths)
  
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

  achieved_fraction = (N * math.pi * (radius ** 2)) / (Lx * Ly)
  min_d_observed = check_all_min_distances(positions, periodic_lengths)

  return full_coords, quaternions, achieved_fraction, min_d_observed, min_dist


def save_clones_file(filepath, positions, quaternions):
  '''Write positions and quaternions to *.clones file.'''
  N = len(positions)
  os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
  with open(filepath, 'w') as f:
    f.write('# The format is:\n')
    f.write('# number of rigid bodies\n')
    f.write('# vector_location_body_0  quaternion_body_0\n')
    f.write('# ...\n')
    f.write(f'{N}\n')
    for i in range(N):
      x, y, z = positions[i]
      q0, q1, q2, q3 = quaternions[i]
      f.write(f'{x:.8f}\t{y:.8f}\t{z:.8f}\t{q0:.8f}\t{q1:.8f}\t{q2:.8f}\t{q3:.8f}\n')


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
  parser.add_argument('--density', '--fraction', dest='fraction', type=float, default=None,
                      help="Target 2D area fraction (e.g. 0.25)")
  parser.add_argument('--num-bodies', '-N', dest='num_bodies', type=int, default=None,
                      help="Explicit number of spheres to place")
  parser.add_argument('--radius', '-R', type=float, default=None,
                      help="Sphere hydrodynamic/geometric radius (default: 1.0)")
  parser.add_argument('--safety-gap', type=float, default=None,
                      help="Safety gap buffer fraction above 2R (default: 0.05 -> d_min = 2.1R)")
  parser.add_argument('--periodic', action='store_true', default=None,
                      help="Enable periodic boundary condition wrapping in x and y")
  parser.add_argument('--output-clones', type=str, default=None,
                      help="Path to save the generated *.clones file (default: data/generated_spheres.clones)")
  parser.add_argument('--plot-image', type=str, default=None,
                      help="Path to save a visual verification PNG plot (default: data/spheres_plot.png)")
  parser.add_argument('--random-quaternions', action='store_true', default=None,
                      help="Randomize 3D orientations (default: identity quaternion 1,0,0,0)")
  parser.add_argument('--seed', type=int, default=None,
                      help="Random number generator seed")
  parser.add_argument('--vertex-file', type=str, default=None,
                      help="Reference vertex file to recommend in inputfile snippet")

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
    elif 'blob_radius' in file_opts:
      radius = float(file_opts['blob_radius'])
    else:
      radius = 1.0

  # Resolve z_height
  z_fixed = args.z_height
  if z_fixed is None:
    if 'z_height' in file_opts:
      z_fixed = float(file_opts['z_height'])
    else:
      z_fixed = 2.0 * radius

  if z_fixed < radius:
    print(f"[!] Warning: z_height ({z_fixed}) is less than sphere radius ({radius}). Floor wall is at z=0.")

  # Resolve density / num_bodies
  fraction = args.fraction
  if fraction is None:
    if 'density' in file_opts:
      fraction = float(file_opts['density'])
    elif 'fraction' in file_opts:
      fraction = float(file_opts['fraction'])

  num_bodies = args.num_bodies
  if num_bodies is None:
    if 'num_bodies' in file_opts:
      num_bodies = int(file_opts['num_bodies'])
    elif 'N' in file_opts:
      num_bodies = int(file_opts['N'])

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

  # Resolve outputs
  output_clones = args.output_clones or file_opts.get('output_clones') or 'data/generated_spheres.clones'
  plot_image = args.plot_image or file_opts.get('plot_image') or 'data/spheres_plot.png'
  vertex_file = args.vertex_file or file_opts.get('vertex_file') or 'Structures/shell_N_12_Rg_1.vertex'
  
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
  print(f"Sphere Radius (R)   : {radius}")
  print(f"Fixed Z Height      : {z_fixed}")
  print(f"Safety Gap Buffer   : {safety_gap * 100:.1f}% -> d_min = {2.0 * radius * (1.0 + safety_gap):.4f}")
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
    seed=seed
  )

  N = len(positions)
  print("-" * 60)
  print(f"Placed Spheres (N)  : {N}")
  print(f"Area Fraction (phi) : {achieved_fraction:.4f}")
  print(f"Min Distance Found  : {min_d:.4f} (Required >= {min_target:.4f})")
  
  if min_d >= min_target - 1e-6:
    print("[SUCCESS] All spheres strictly obey non-overlapping criteria!")
  else:
    print("[WARNING] Some spheres have distance close to cutoff. Consider increasing domain size.")

  # Save clones file
  save_clones_file(output_clones, positions, quaternions)
  print(f"[+] Output clones file saved to: {output_clones}")

  # Plot image
  if plot_image:
    plot_suspension_2d(bounds, positions, radius, achieved_fraction, min_d, min_target, plot_image)

  # Print inputfile snippet
  print("=" * 60)
  print("HOW TO USE IN YOUR SIMULATION:")
  print(f"Add this line to your multi_bodies inputfile (e.g. inputfile_dynamic.dat):")
  print(f"structure {vertex_file} {output_clones}")
  if periodic:
    Lx = bounds[1] - bounds[0]
    Ly = bounds[3] - bounds[2]
    print(f"periodic_length {Lx:.1f} {Ly:.1f} 0.0")
  print("=" * 60)


if __name__ == '__main__':
  main()

