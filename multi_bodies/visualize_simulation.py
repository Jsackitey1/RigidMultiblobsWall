#!/usr/bin/env python3
'''
Simulation Video Visualization Tool for RigidMultiblobsWall.
Generates 2D MP4 videos (H.264 / yuv420p) and standalone interactive 2D HTML5 players.
Works with any simulation run, any particle count, and any body radius.

Usage:
  # Generate all 2D MP4 videos (8 seconds long at 24 fps) and 2D interactive HTML player:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode all

  # Generate 2D top-down view with custom duration:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode 2d_top --duration 10.0 --fps 30

  # Generate 2D side elevation view:
  python3 visualize_simulation.py --input-file inputfile_dynamic.dat --mode 2d_side
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
from visualizer.render_graphs import render_graph_video
from visualizer.render_composite import render_composite_video
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
                        choices=['2d_top', '2d_side', '2d_dual', '2d_blobs', 'graphs', 'composite', 'interactive_html', 'all'],
                        help='Visualization mode (default: all)')
    parser.add_argument('--radius', '-r', type=float, default=None,
                        help='Particle radius override (optional, auto-computed from vertex file by default)')
    parser.add_argument('--duration', '-d', type=float, default=8.0,
                        help='Target video duration in seconds (default: 8.0s)')
    parser.add_argument('--fps', type=int, default=24,
                        help='Video frame rate for smooth playback (default: 24 fps)')
    parser.add_argument('--output-dir', '-o', type=str, default='data/visualizations',
                        help='Output directory for generated MP4 videos and HTML (default: data/visualizations)')
    parser.add_argument('--no-interpolate', action='store_true',
                        help='Disable SLERP sub-frame interpolation and render raw simulation frames only')
    parser.add_argument('--no-trails', action='store_true',
                        help='Disable trajectory motion trails')
    parser.add_argument('--no-vectors', action='store_true',
                        help='Disable velocity vectors in 2D views')

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

    # 1. 2D Top-Down Suspension View (X-Y)
    if args.mode in ['2d_top', 'all']:
        print("\n--- 1. Generating 2D Top-Down Suspension Video (X-Y Plane) ---")
        top_path = os.path.join(args.output_dir, 'suspension_2d_topdown.mp4')
        out_top = render_2d_video(traj, top_path, view='top', mode='spheres',
                                  fps=args.fps, show_vectors=not args.no_vectors,
                                  show_trails=not args.no_trails)
        generated_all.extend(out_top)

    # 2. 2D Side Elevation Profile View (X-Z)
    if args.mode in ['2d_side', 'all']:
        print("\n--- 2. Generating 2D Side Elevation Profile Video (X-Z Plane) ---")
        side_path = os.path.join(args.output_dir, 'suspension_2d_side_elevation.mp4')
        out_side = render_2d_video(traj, side_path, view='side', mode='spheres',
                                   fps=args.fps)
        generated_all.extend(out_side)

    # 3. 2D Dual View (X-Y Top-Down + X-Z Side Elevation Side-by-Side)
    if args.mode in ['2d_dual', 'all']:
        print("\n--- 3. Generating 2D Dual-View Video (X-Y Top + X-Z Side) ---")
        dual_path = os.path.join(args.output_dir, 'suspension_2d_dual_view.mp4')
        out_dual = render_2d_video(traj, dual_path, view='dual', mode='spheres',
                                   fps=args.fps)
        generated_all.extend(out_dual)

    # 4. 2D Multi-Blobs Discretization View
    if args.mode in ['2d_blobs', 'all']:
        print("\n--- 4. Generating 2D Multi-Blob Discretization Video ---")
        b2d_path = os.path.join(args.output_dir, 'multiblobs_2d_topdown.mp4')
        out_b2d = render_2d_video(traj, b2d_path, view='top', mode='blobs',
                                  fps=args.fps, show_vectors=False,
                                  show_trails=not args.no_trails)
        generated_all.extend(out_b2d)

    # 5. Analytical Metrics Graph Video
    if args.mode in ['graphs', 'all']:
        print("\n--- 5. Generating Dynamic Analytical Metrics Graph Video ---")
        g_path = os.path.join(args.output_dir, 'graph_metrics_dashboard.mp4')
        out_g = render_graph_video(traj, g_path, fps=args.fps, format='mp4')
        generated_all.extend(out_g)

    # 6. Synchronized 2D Composite Dashboard Video
    if args.mode in ['composite', 'all']:
        print("\n--- 6. Generating Synchronized 2D Composite Dashboard Video ---")
        c_path = os.path.join(args.output_dir, 'composite_2d_dashboard.mp4')
        out_c = render_composite_video(traj, c_path, view_2d='top', fps=args.fps)
        generated_all.extend(out_c)

    # 7. Standalone Interactive 2D HTML5 Player
    if args.mode in ['interactive_html', 'all']:
        print("\n--- 7. Generating Interactive 2D HTML5 Player ---")
        h_path = os.path.join(args.output_dir, 'interactive_2d_player.html')
        out_h = export_interactive_html(raw_traj, h_path)
        generated_all.append(out_h)

    print("\n" + "=" * 70)
    print("All 2D visualizations successfully generated!")
    print("=" * 70)
    for f in generated_all:
        print(f"  -> {f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
