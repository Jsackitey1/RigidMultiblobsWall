#!/usr/bin/env python3
'''
Sphere Suspension Generator (2D and 3D) for RigidMultiblobsWall.

Generates non-overlapping random sphere suspensions in 2D (fixed height z)
or 3D (bounding slab/box) for a target number of spheres or area/volume fraction.
Outputs formatted *.clones files and visual verification plots (PNG).
'''

import argparse
import sys
import os
import math
import numpy as np

# Set matplotlib headless backend if plotting
try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.patches import Circle
  has_matplotlib = True
except ImportError:
  has_matplotlib = False


def calculate_num_spheres(dim, bounds, target_fraction, radius):
  '''
  Calculate the number of spheres needed for a target packing fraction.
  dim: 2 or 3
  bounds: [xmin, xmax, ymin, ymax] or [xmin, xmax, ymin, ymax, zmin, zmax]
  '''
  Lx = bounds[1] - bounds[0]
  Ly = bounds[3] - bounds[2]
  
  if dim == 2:
    area = Lx * Ly
    sphere_area = math.pi * (radius ** 2)
    N = int(round((target_fraction * area) / sphere_area))
  else:
    Lz = bounds[5] - bounds[4]
    volume = Lx * Ly * Lz
    sphere_volume = (4.0 / 3.0) * math.pi * (radius ** 3)
    N = int(round((target_fraction * volume) / sphere_volume))
  return max(1, N)


def get_periodic_diff(delta, L):
  '''Apply minimum image convention along one dimension.'''
  return delta - L * np.round(delta / L)


def check_overlap(pos, existing_positions, min_dist, periodic_lengths=None):
  '''
  Check if a candidate position overlaps with any existing positions.
  '''
  if len(existing_positions) == 0:
    return False
  
  diffs = existing_positions - pos
  if periodic_lengths is not None:
    for d in range(len(periodic_lengths)):
      if periodic_lengths[d] > 0:
        diffs[:, d] = get_periodic_diff(diffs[:, d], periodic_lengths[d])
  
  dists_sq = np.sum(diffs ** 2, axis=1)
  return np.any(dists_sq < (min_dist ** 2))


def place_spheres_rsa(dim, bounds, N, min_dist, periodic_lengths=None, max_attempts=50000):
  '''
  Random Sequential Addition (RSA) with spatial sampling.
  '''
  positions = []
  attempts = 0
  
  xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
  if dim == 3:
    zmin, zmax = bounds[4], bounds[5]
  
  while len(positions) < N and attempts < max_attempts:
    attempts += 1
    
    # Sample candidate
    rx = np.random.uniform(xmin, xmax)
    ry = np.random.uniform(ymin, ymax)
    if dim == 2:
      pos = np.array([rx, ry])
    else:
      rz = np.random.uniform(zmin, zmax)
      pos = np.array([rx, ry, rz])
    
    if len(positions) == 0:
      positions.append(pos)
      continue
    
    pos_arr = np.array(positions)
    if not check_overlap(pos, pos_arr, min_dist, periodic_lengths):
      positions.append(pos)
      attempts = 0 # reset streak
  
  return np.array(positions)


def relax_dense_suspension(dim, bounds, initial_positions, min_dist, periodic_lengths=None, max_iters=2000):
  '''
  Force-bias Monte Carlo relaxation to push overlapping spheres apart in dense packings.
  '''
  positions = np.copy(initial_positions)
  N = len(positions)
  
  xmin, xmax, ymin, ymax = bounds[0], bounds[1], bounds[2], bounds[3]
  if dim == 3:
    zmin, zmax = bounds[4], bounds[5]
  
  for iteration in range(max_iters):
    displacements = np.zeros_like(positions)
    overlaps_found = 0
    
    for i in range(N):
      diffs = positions - positions[i]
      if periodic_lengths is not None:
        for d in range(len(periodic_lengths)):
          if periodic_lengths[d] > 0:
            diffs[:, d] = get_periodic_diff(diffs[:, d], periodic_lengths[d])
      
      dists = np.linalg.norm(diffs, axis=1)
      # Exclude self
      dists[i] = np.inf
      
      overlap_mask = dists < min_dist
      if np.any(overlap_mask):
        overlaps_found += np.sum(overlap_mask)
        # Repulsive push
        overlap_indices = np.where(overlap_mask)[0]
        for j in overlap_indices:
          d = dists[j]
          if d < 1e-8:
            # random direction if right on top of each other
            direction = np.random.randn(dim)
            direction /= np.linalg.norm(direction)
          else:
            direction = diffs[j] / d
          push = 0.5 * (min_dist - d) * direction
          displacements[i] -= push
          displacements[j] += push
          
    if overlaps_found == 0:
      break
      
    # Apply displacements with boundary clamping/wrapping
    positions += displacements * 0.4
    
    # Boundary handling for non-periodic boundaries
    if periodic_lengths is not None and periodic_lengths[0] > 0:
      positions[:, 0] = xmin + (positions[:, 0] - xmin) % (xmax - xmin)
    else:
      positions[:, 0] = np.clip(positions[:, 0], xmin, xmax)
      
    if periodic_lengths is not None and periodic_lengths[1] > 0:
      positions[:, 1] = ymin + (positions[:, 1] - ymin) % (ymax - ymin)
    else:
      positions[:, 1] = np.clip(positions[:, 1], ymin, ymax)
      
    if dim == 3:
      positions[:, 2] = np.clip(positions[:, 2], zmin, zmax)
      
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
      for d in range(len(periodic_lengths)):
        if periodic_lengths[d] > 0:
          diffs[:, d] = get_periodic_diff(diffs[:, d], periodic_lengths[d])
    dists = np.linalg.norm(diffs, axis=1)
    if len(dists) > 0:
      local_min = np.min(dists)
      if local_min < min_d:
        min_d = local_min
  return min_d


def generate_sphere_suspension(dim, bounds, target_count=None, target_fraction=None,
                               radius=1.0, z_fixed=2.0, safety_gap=0.05,
                               periodic=False, randomize_quaternions=False, seed=None):
  '''
  Main placement orchestrator.
  '''
  if seed is not None:
    np.random.seed(seed)
    
  min_dist = 2.0 * radius * (1.0 + safety_gap)
  
  # Determine N
  if target_count is not None:
    N = target_count
  elif target_fraction is not None:
    N = calculate_num_spheres(dim, bounds, target_fraction, radius)
  else:
    raise ValueError("Either target_count or target_fraction must be specified.")
    
  # Set up periodic length vector
  Lx = bounds[1] - bounds[0]
  Ly = bounds[3] - bounds[2]
  if periodic:
    periodic_lengths = [Lx, Ly] if dim == 2 else [Lx, Ly, 0.0]
  else:
    periodic_lengths = None

  # Effective placement bounds (shrunk by radius for hard wall boundaries)
  if periodic:
    place_bounds = list(bounds)
  else:
    place_bounds = [bounds[0] + radius, bounds[1] - radius,
                    bounds[2] + radius, bounds[3] - radius]
    if dim == 3:
      place_bounds.extend([bounds[4] + radius, bounds[5] - radius])

  # 1. Attempt RSA
  positions = place_spheres_rsa(dim, place_bounds, N, min_dist, periodic_lengths)
  
  # 2. If RSA could not place all spheres (e.g. dense suspension), use force-bias relaxation
  if len(positions) < N:
    needed = N - len(positions)
    print(f"[*] RSA placed {len(positions)}/{N} spheres. Running relaxation for remaining {needed} spheres...")
    
    # Place random particles for the remainder and relax
    extra = np.zeros((needed, dim))
    extra[:, 0] = np.random.uniform(place_bounds[0], place_bounds[1], needed)
    extra[:, 1] = np.random.uniform(place_bounds[2], place_bounds[3], needed)
    if dim == 3:
      extra[:, 2] = np.random.uniform(place_bounds[4], place_bounds[5], needed)
    
    combined = np.vstack([positions, extra]) if len(positions) > 0 else extra
    positions = relax_dense_suspension(dim, place_bounds, combined, min_dist, periodic_lengths)
  
  # Final 3D coordinates (x, y, z)
  full_coords = np.zeros((N, 3))
  if dim == 2:
    full_coords[:, 0] = positions[:, 0]
    full_coords[:, 1] = positions[:, 1]
    full_coords[:, 2] = z_fixed
  else:
    full_coords = positions

  # Quaternions (q0, q1, q2, q3)
  quaternions = np.zeros((N, 4))
  if randomize_quaternions:
    # Uniform random quaternions (Shoemake algorithm)
    u = np.random.uniform(0, 1, (N, 3))
    quaternions[:, 0] = np.sqrt(1 - u[:, 0]) * np.sin(2 * np.pi * u[:, 1])
    quaternions[:, 1] = np.sqrt(1 - u[:, 0]) * np.cos(2 * np.pi * u[:, 1])
    quaternions[:, 2] = np.sqrt(u[:, 0]) * np.sin(2 * np.pi * u[:, 2])
    quaternions[:, 3] = np.sqrt(u[:, 0]) * np.cos(2 * np.pi * u[:, 2])
  else:
    quaternions[:, 0] = 1.0 # identity quaternion

  # Compute actual achieved metrics
  if dim == 2:
    achieved_fraction = (N * math.pi * (radius ** 2)) / (Lx * Ly)
  else:
    Lz = bounds[5] - bounds[4]
    achieved_fraction = (N * (4.0 / 3.0) * math.pi * (radius ** 3)) / (Lx * Ly * Lz)

  min_d_observed = check_all_min_distances(full_coords, [Lx, Ly, 0.0] if periodic else None)

  return full_coords, quaternions, achieved_fraction, min_d_observed, min_dist


def save_clones_file(filepath, positions, quaternions):
  '''Write coordinates and quaternions to *.clones file.'''
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


def plot_suspension(dim, bounds, positions, radius, achieved_fraction, min_d, min_target, output_plot):
  '''Generate 2D or 3D visualization plot of the sphere suspension.'''
  if not has_matplotlib:
    print("[!] matplotlib is not installed. Skipping plot generation.")
    return

  N = len(positions)
  fig = plt.figure(figsize=(8, 8 if dim == 2 else 7), dpi=150)
  
  if dim == 2:
    ax = fig.add_subplot(111)
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
    
  else:
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(111, projection='3d')
    xmin, xmax, ymin, ymax, zmin, zmax = bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
    
    # Draw 3D scatter colored by z-height
    p = ax.scatter(xs=positions[:, 0], ys=positions[:, 1], zs=positions[:, 2],
                   c=positions[:, 2], cmap='viridis', s=60 * (radius**2),
                   edgecolors='k', alpha=0.85)
    
    # Draw 3D wireframe box
    for s, e in [([xmin, ymin, zmin], [xmax, ymin, zmin]),
                ([xmin, ymax, zmin], [xmax, ymax, zmin]),
                ([xmin, ymin, zmax], [xmax, ymin, zmax]),
                ([xmin, ymax, zmax], [xmax, ymax, zmax]),
                ([xmin, ymin, zmin], [xmin, ymax, zmin]),
                ([xmax, ymin, zmin], [xmax, ymax, zmin]),
                ([xmin, ymin, zmax], [xmin, ymax, zmax]),
                ([xmax, ymin, zmax], [xmax, ymax, zmax]),
                ([xmin, ymin, zmin], [xmin, ymin, zmax]),
                ([xmax, ymin, zmin], [xmax, ymin, zmax]),
                ([xmin, ymax, zmin], [xmin, ymax, zmax]),
                ([xmax, ymax, zmin], [xmax, ymax, zmax])]:
      ax.plot3D(*zip(s, e), color="gray", linestyle="--", linewidth=1.0)
      
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel('X', fontsize=11)
    ax.set_ylabel('Y', fontsize=11)
    ax.set_zlabel('Z', fontsize=11)
    cbar = plt.colorbar(p, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Z Height', fontsize=10)
    ax.set_title(f'3D Sphere Suspension (N = {N}, $\\phi_{{3D}} = {achieved_fraction:.3f}$)\n'
                 f'Min Dist = {min_d:.3f} (Req >= {min_target:.3f})', fontsize=12)

  plt.tight_layout()
  plt.savefig(output_plot, dpi=200)
  plt.close()
  print(f"[+] Visual plot saved to: {output_plot}")


def main():
  parser = argparse.ArgumentParser(
    description="Generate non-overlapping 2D or 3D sphere suspensions for RigidMultiblobsWall."
  )
  parser.add_argument('--box', nargs='+', type=float, required=True,
                      help="Bounding box: 'xmin xmax ymin ymax' or 'Lx Ly'")
  parser.add_argument('--z-height', type=float, default=None,
                      help="Fixed height z above wall (activates 2D mode)")
  parser.add_argument('--z-bounds', nargs=2, type=float, default=None,
                      help="Z range 'zmin zmax' (activates 3D mode)")
  parser.add_argument('--density', '--fraction', dest='fraction', type=float, default=None,
                      help="Target packing fraction (area fraction in 2D, volume fraction in 3D)")
  parser.add_argument('--num-bodies', '-N', dest='num_bodies', type=int, default=None,
                      help="Explicit number of spheres to place")
  parser.add_argument('--radius', '-R', type=float, default=1.0,
                      help="Sphere hydrodynamic/geometric radius (default: 1.0)")
  parser.add_argument('--safety-gap', type=float, default=0.05,
                      help="Safety gap buffer fraction above 2R (default: 0.05 -> d_min = 2.1R)")
  parser.add_argument('--periodic', action='store_true',
                      help="Enable periodic boundary condition wrapping in x and y")
  parser.add_argument('--output-clones', type=str, default='Structures/generated_spheres.clones',
                      help="Path to save the generated *.clones file")
  parser.add_argument('--plot-image', type=str, default=None,
                      help="Path to save a visual verification PNG plot")
  parser.add_argument('--random-quaternions', action='store_true',
                      help="Randomize 3D orientations (default: identity quaternion 1,0,0,0)")
  parser.add_argument('--seed', type=int, default=None,
                      help="Random number generator seed")
  parser.add_argument('--vertex-file', type=str, default='Structures/shell_N_12_Rg_1.vertex',
                      help="Reference vertex file to recommend in inputfile snippet")

  args = parser.parse_args()

  # Parse box
  if len(args.box) == 2:
    bounds_2d = [0.0, args.box[0], 0.0, args.box[1]]
  elif len(args.box) == 4:
    bounds_2d = [args.box[0], args.box[1], args.box[2], args.box[3]]
  else:
    sys.exit("Error: --box must provide 2 numbers (Lx Ly) or 4 numbers (xmin xmax ymin ymax).")

  # Determine 2D vs 3D
  if args.z_bounds is not None:
    dim = 3
    bounds = bounds_2d + [args.z_bounds[0], args.z_bounds[1]]
    if bounds[4] < args.radius:
      print(f"[!] Warning: zmin ({bounds[4]}) is less than sphere radius ({args.radius}). Bottom wall is at z=0.")
    z_fixed = None
  else:
    dim = 2
    bounds = bounds_2d
    z_fixed = args.z_height if args.z_height is not None else 2.0 * args.radius
    if z_fixed < args.radius:
      print(f"[!] Warning: z_height ({z_fixed}) is less than sphere radius ({args.radius}). Bottom wall is at z=0.")

  if args.num_bodies is None and args.fraction is None:
    sys.exit("Error: Must specify either --density/--fraction (e.g. 0.25) or --num-bodies (e.g. 50).")

  print("=" * 60)
  print(f"Sphere Suspension Generator ({'2D Mode' if dim == 2 else '3D Mode'})")
  print("=" * 60)
  print(f"Domain Bounds       : {bounds}")
  print(f"Sphere Radius (R)   : {args.radius}")
  if dim == 2:
    print(f"Fixed Z Height      : {z_fixed}")
  print(f"Safety Gap Buffer   : {args.safety_gap * 100:.1f}% -> d_min = {2.0 * args.radius * (1.0 + args.safety_gap):.4f}")
  print(f"Periodic Boundaries : {args.periodic}")

  # Generate positions
  positions, quaternions, achieved_fraction, min_d, min_target = generate_sphere_suspension(
    dim=dim,
    bounds=bounds,
    target_count=args.num_bodies,
    target_fraction=args.fraction,
    radius=args.radius,
    z_fixed=z_fixed,
    safety_gap=args.safety_gap,
    periodic=args.periodic,
    randomize_quaternions=args.random_quaternions,
    seed=args.seed
  )

  N = len(positions)
  print("-" * 60)
  print(f"Placed Spheres (N)  : {N}")
  print(f"Packing Fraction    : {achieved_fraction:.4f} ({'Area Fraction' if dim == 2 else 'Volume Fraction'})")
  print(f"Min Distance Found  : {min_d:.4f} (Required >= {min_target:.4f})")
  
  if min_d >= min_target - 1e-6:
    print("[SUCCESS] All spheres strictly obey non-overlapping criteria!")
  else:
    print("[WARNING] Some spheres have distance close to cutoff. Consider increasing domain size.")

  # Save clones file
  save_clones_file(args.output_clones, positions, quaternions)
  print(f"[+] Output clones file saved to: {args.output_clones}")

  # Plot image if requested
  if args.plot_image:
    plot_suspension(dim, bounds, positions, args.radius, achieved_fraction, min_d, min_target, args.plot_image)

  # Print inputfile snippet
  print("=" * 60)
  print("HOW TO USE IN YOUR SIMULATION:")
  print(f"Add this line to your multi_bodies inputfile (e.g. inputfile_dynamic.dat):")
  print(f"structure {args.vertex_file} {args.output_clones}")
  if args.periodic:
    Lx = bounds[1] - bounds[0]
    Ly = bounds[3] - bounds[2]
    print(f"periodic_length {Lx:.1f} {Ly:.1f} 0.0")
  print("=" * 60)


if __name__ == '__main__':
  main()
