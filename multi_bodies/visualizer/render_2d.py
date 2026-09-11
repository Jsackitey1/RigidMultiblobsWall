'''
High-Quality 2D Simulation Video Renderer for RigidMultiblobsWall.
Renders Rigid Spheres / Bodies and Discretized Multi-Blobs in motion.
Encodes universal H.264 MP4 videos.
'''

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .video_writer import save_video, VideoStreamWriter


def render_2d_frame(traj, frame_idx, fig=None, mode='spheres', show_vectors=True, 
                    show_trails=True, trail_length=20, color_by='height'):
    '''
    Render a single top-down (X-Y plane) 2D simulation frame.
    
    Parameters
    ----------
    traj : SimulationTrajectory
        Trajectory data object.
    frame_idx : int
        Index of the frame to render.
    fig : matplotlib.figure.Figure, optional
        Re-usable figure object for performance.
    mode : str
        'spheres' (rigid solid discs with orientation needle) or
        'blobs' (exact constituent hydrodynamic multiblobs).
    show_vectors : bool
        Whether to overlay velocity vectors.
    show_trails : bool
        Whether to show past motion trail paths.
    trail_length : int
        Number of previous frames to include in motion trails.
    color_by : str
        Colormapping criterion: 'height' (z-coord), 'speed' (|v|), or 'index'.
    '''
    if fig is None:
        fig = plt.figure(figsize=(10, 10), dpi=100, facecolor='#0b0f19')
    else:
        fig.clf()

    fig.patch.set_facecolor('#0b0f19')
    ax = fig.add_subplot(111, facecolor='#111827')

    t_curr = traj.time_array[frame_idx]
    pos = traj.positions[frame_idx]           # (N, 3)
    speeds = traj.speeds[frame_idx]           # (N,)
    angles = traj.get_inplane_orientation_angles(frame_idx)  # (N,)
    R = traj.sphere_radius

    xmin, xmax = traj.domain_bounds[0], traj.domain_bounds[1]
    ymin, ymax = traj.domain_bounds[2], traj.domain_bounds[3]

    # --- Draw Trajectory Trails ---
    if show_trails and frame_idx > 0:
        start_t = max(0, frame_idx - trail_length)
        for b in range(traj.num_bodies):
            trail = traj.positions[start_t:frame_idx + 1, b]
            ax.plot(trail[:, 0], trail[:, 1], color='#38bdf8', alpha=0.35, linewidth=1.2, zorder=2)

    # --- Color Mapping ---
    if color_by == 'speed':
        max_speed = max(np.max(traj.speeds), 0.01)
        c_vals = speeds
        cmap = 'plasma'
        c_label = 'Translational Speed $|v|$'
        norm = plt.Normalize(vmin=0, vmax=max_speed)
    elif color_by == 'index':
        c_vals = np.arange(traj.num_bodies)
        cmap = 'tab20'
        c_label = 'Body ID'
        norm = plt.Normalize(vmin=0, vmax=traj.num_bodies)
    else:  # height default
        c_vals = pos[:, 2]
        cmap = 'viridis'
        c_label = 'Height $z$ above wall'
        z_min = min(0.0, np.min(traj.positions[..., 2]))
        z_max = max(np.max(traj.positions[..., 2]) + 0.5, R * 3.0)
        norm = plt.Normalize(vmin=z_min, vmax=z_max)

    colormap = plt.get_cmap(cmap)

    # --- Mode: 2D Discs + Orientation Director Pointers ---
    if mode == 'spheres':
        # Bug 6: Sort by ascending z for correct depth ordering (painter's algorithm)
        z_order = np.argsort(pos[:, 2])

        patches = []
        ordered_colors = []
        for i in z_order:
            circle = Circle((pos[i, 0], pos[i, 1]), radius=R)
            patches.append(circle)
            ordered_colors.append(colormap(norm(c_vals[i])))

        p_coll = PatchCollection(patches, facecolors=ordered_colors, edgecolors='#ffffff', linewidths=1.0, alpha=0.9, zorder=4)
        ax.add_collection(p_coll)

        # In-plane orientation pointer (radial needle from center to boundary)
        for i in z_order:
            th = angles[i]
            end_x = pos[i, 0] + R * math.cos(th)
            end_y = pos[i, 1] + R * math.sin(th)
            ax.plot([pos[i, 0], end_x], [pos[i, 1], end_y], color='#facc15', linewidth=2.0, zorder=6)
            ax.scatter([pos[i, 0]], [pos[i, 1]], color='#ffffff', s=10, zorder=7)

        # Velocity vectors
        if show_vectors and traj.num_frames > 1:
            vels = traj.velocities[frame_idx]
            v_max = max(np.max(traj.speeds), 1e-4)
            scale_factor = R / (v_max * 2.0)
            ax.quiver(pos[:, 0], pos[:, 1], vels[:, 0] * scale_factor, vels[:, 1] * scale_factor,
                      color='#38bdf8', angles='xy', scale_units='xy', scale=1.0, width=0.004, alpha=0.8, zorder=5)

    # --- Mode: 2D Multi-Blobs ---
    elif mode == 'blobs':
        all_blobs = traj.get_blobs_for_frame(frame_idx)
        if all_blobs is not None:
            a = traj.blob_radius
            # Bug 6: Sort blobs by ascending z for correct depth ordering
            blob_z_order = np.argsort(all_blobs[:, 2])
            blob_patches = []
            blob_ordered_colors = []
            for b in blob_z_order:
                blob_circle = Circle((all_blobs[b, 0], all_blobs[b, 1]), radius=a)
                blob_patches.append(blob_circle)
                blob_ordered_colors.append(colormap(norm(all_blobs[b, 2])))
            
            bp_coll = PatchCollection(blob_patches, facecolors=blob_ordered_colors, edgecolors='#ffffff',
                                      linewidths=0.5, alpha=0.9, zorder=4)
            ax.add_collection(bp_coll)
            ax.scatter(pos[:, 0], pos[:, 1], color='#38bdf8', s=12, zorder=6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(c_label, color='#e2e8f0', fontsize=10)
    cbar.ax.tick_params(colors='#e2e8f0', labelsize=8)

    # Axis and Domain Formatting
    pad = R * 1.5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', color='#1f2937', alpha=0.6)

    text_color = '#e2e8f0'
    ax.set_xlabel('X Coordinate', color=text_color, fontsize=11)
    ax.set_ylabel('Y Coordinate', color=text_color, fontsize=11)
    ax.tick_params(colors=text_color, labelsize=9)

    mode_title = f"2D Rigid Bodies Suspension (Radius R={R:.2f})" if mode == 'spheres' else f"2D Constituent Multi-Blobs (Blob Radius a={traj.blob_radius:.2f})"
    title = (f"{mode_title}\n"
             f"Time: {t_curr:.3f} s  |  Frame: {frame_idx + 1}/{traj.num_frames}  |  "
             f"Bodies: {traj.num_bodies}  |  Mean $\\langle z \\rangle$: {traj.mean_z[frame_idx]:.4f}")
    ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=10)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def render_2d_video(traj, output_path, mode='spheres', fps=24, 
                    show_vectors=True, show_trails=True, show_progress=True):
    '''
    Generate a 2D simulation MP4 video for spheres or multiblobs.
    Streams frames directly to disk to avoid buffering the entire video in RAM.
    
    Parameters
    ----------
    traj : SimulationTrajectory
        Trajectory data object.
    output_path : str
        Target output filepath (e.g. data/visualizations/spheres_simulation.mp4).
    mode : str
        'spheres' or 'blobs'
    fps : int
        Frame rate for video encoding.
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if show_progress:
        print(f"[*] Rendering {traj.num_frames} 2D frames (mode={mode})...")

    fig = plt.figure(figsize=(10, 10), dpi=100, facecolor='#0b0f19')

    # Bug 7: Stream frames directly to disk instead of buffering in RAM
    with VideoStreamWriter(output_path, fps=fps) as writer:
        for i in range(traj.num_frames):
            frame = render_2d_frame(traj, i, fig=fig, mode=mode,
                                    show_vectors=show_vectors, show_trails=show_trails)
            writer.write_frame(frame)
            if show_progress:
                print(f"    Rendered 2D frame {i+1}/{traj.num_frames}", end='\r')

    plt.close(fig)
    if show_progress:
        print()
        print(f"[+] Saved 2D MP4 Video: {writer.path}")

    return [writer.path]
