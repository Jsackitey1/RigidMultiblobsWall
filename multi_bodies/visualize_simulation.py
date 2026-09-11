#!/usr/bin/env python3
'''
Simulation Video Visualization Tool for RigidMultiblobsWall.
Generates 2D MP4 videos (H.264 / yuv420p) and standalone interactive 2D HTML5 players.
Works with any simulation run, any particle count, and any body radius.

Usage:
  # Generate both Spheres and Multi-Blobs MP4 videos + Interactive HTML player:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode all

  # Generate only the Spheres video:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode spheres --duration 8.0 --fps 24

  # Generate only the Multi-Blobs video:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode blobs --duration 8.0 --fps 24

  # Generate only the Interactive 2D HTML player:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode html
'''

import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualizer.trajectory_loader import load_simulation_data
from visualizer.render_2d import render_2d_video
from visualizer.export_html import export_interactive_html


def main():
    parser = argparse.ArgumentParser(description='Visualize RigidMultiblobsWall simulation in 2D MP4 video and HTML formats.')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to simulation .config file (default: resolved from input-file or data/run.generated_spheres.config)')
    parser.add_argument('--input-file', '-i', type=str, default='inputfile_dynamic.dat',
                        help='Path to simulation input file (default: inputfile_dynamic.dat)')
    parser.add_argument('--vertex-file', '-v', type=str, default=None,
                        help='Path to body .vertex file (optional, auto-detected from input file)')
    parser.add_argument('--mode', '-m', type=str, default='all',
                        choices=['spheres', 'blobs', 'html', 'interactive', 'all'],
                        help='Visualization mode: spheres, blobs, html, or all (default: all)')
    parser.add_argument('--radius', '-r', type=float, default=None,
                        help='Particle radius override (optional, auto-computed from vertex file by default)')
    parser.add_argument('--duration', '-d', type=float, default=10.0,
                        help='Target video and HTML playback duration in seconds (default: 10.0s)')
    parser.add_argument('--fps', type=int, default=24,
                        help='Video frame rate for smooth playback (default: 24 fps)')
    parser.add_argument('--output-dir', '-o', type=str, default='data/visualizations',
                        help='Output directory for generated MP4 videos and HTML (default: data/visualizations)')
    parser.add_argument('--no-interpolate', action='store_true',
                        help='Disable SLERP sub-frame interpolation and render raw simulation frames only')
    parser.add_argument('--no-trails', action='store_true',
                        help='Disable trajectory motion trails')
    parser.add_argument('--no-vectors', action='store_true',
                        help='Disable velocity vectors in spheres view')

    args = parser.parse_args()

    # Resolve config file path
    config_file = args.config
    if config_file is None:
        if args.input_file and os.path.exists(args.input_file):
            from visualizer.trajectory_loader import parse_input_file
            opts = parse_input_file(args.input_file)
            out_prefix = opts.get('output_name', 'data/run')
            struct_str = opts.get('structure', '')
            struct_name = 'generated_spheres'
            if struct_str:
                clones_path = struct_str.split()[-1]
                clones_base = os.path.basename(clones_path)
                if clones_base.endswith('.clones'):
                    struct_name = clones_base[:-7]
            candidate = f"{out_prefix}.{struct_name}.config"
            if os.path.exists(candidate):
                config_file = candidate
            else:
                config_file = 'data/run.generated_spheres.config'
        else:
            config_file = 'data/run.generated_spheres.config'

    if not os.path.exists(config_file):
        sys.exit(f"Error: Config file not found: {config_file}\nMake sure to run multi_bodies.py first!")

    print("=" * 70)
    print("RigidMultiblobsWall 2D Simulation Video Visualizer")
    print("=" * 70)
    print(f"Config File     : {config_file}")
    print(f"Input File      : {args.input_file}")
    print(f"Mode            : {args.mode}")
    print(f"Target Duration : {args.duration} s")
    print(f"Target FPS      : {args.fps} fps")
    print(f"Output Dir      : {args.output_dir}")
    print("=" * 70)

    # 1. Load Raw Trajectory Data
    raw_traj = load_simulation_data(
        config_file,
        input_file=args.input_file,
        vertex_file=args.vertex_file,
        manual_radius=args.radius
    )
    print(f"[+] Loaded simulation: {raw_traj.num_frames} raw frames, {raw_traj.num_bodies} bodies, {raw_traj.num_blobs_per_body} blobs/body.")
    print(f"[+] Resolved Body Radius: R = {raw_traj.sphere_radius:.4f} (Blob Radius a = {raw_traj.blob_radius:.4f})")
    print(f"[+] Domain Envelope: [x: {raw_traj.domain_bounds[0]:.1f} -> {raw_traj.domain_bounds[1]:.1f}, y: {raw_traj.domain_bounds[2]:.1f} -> {raw_traj.domain_bounds[3]:.1f}]")
    print(f"[+] Physical Duration: {raw_traj.time_array[-1]:.3f} s (dt = {raw_traj.dt:.4f} s)")

    # 2. Smooth SLERP Sub-Frame Interpolation for Longer Video Duration
    if not args.no_interpolate and raw_traj.num_frames > 1:
        traj = raw_traj.get_interpolated_trajectory(target_duration=args.duration, fps=args.fps)
        print(f"[+] Interpolated to {traj.num_frames} smooth frames ({args.fps} fps -> {args.duration:.1f}s video duration).")
    else:
        traj = raw_traj
        print(f"[!] Rendering {traj.num_frames} raw frames directly without interpolation.")

    os.makedirs(args.output_dir, exist_ok=True)
    generated_all = []

    # 1. 2D Spheres / Bodies Video
    if args.mode in ['spheres', 'all']:
        print("\n--- 1. Generating 2D Spheres Suspension Video ---")
        spheres_path = os.path.join(args.output_dir, 'spheres_simulation.mp4')
        out_spheres = render_2d_video(traj, spheres_path, mode='spheres',
                                      fps=args.fps, show_vectors=not args.no_vectors,
                                      show_trails=not args.no_trails)
        generated_all.extend(out_spheres)

    # 2. 2D Multi-Blobs Discretization Video
    if args.mode in ['blobs', 'all']:
        print("\n--- 2. Generating 2D Multi-Blobs Discretization Video ---")
        blobs_path = os.path.join(args.output_dir, 'multiblobs_simulation.mp4')
        out_blobs = render_2d_video(traj, blobs_path, mode='blobs',
                                    fps=args.fps, show_vectors=False,
                                    show_trails=not args.no_trails)
        generated_all.extend(out_blobs)

    # 3. Standalone Interactive 2D HTML5 Player
    if args.mode in ['html', 'interactive', 'all']:
        print("\n--- 3. Generating Interactive 2D HTML5 Simulation Player ---")
        html_path = os.path.join(args.output_dir, 'interactive_simulation_player.html')
        out_html = export_interactive_html(traj, html_path, target_duration=args.duration)
        generated_all.append(out_html)

    print("\n" + "=" * 70)
    print("All requested visualizations successfully generated!")
    print("=" * 70)
    for f in generated_all:
        print(f"  -> {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
