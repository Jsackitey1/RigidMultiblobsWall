'''
High-Quality 2D Simulation Video Renderer for RigidMultiblobsWall.
Renders Top-Down (X-Y), Side-Elevation (X-Z), Dual-Plane, and 2D Discretized Multi-Blob views.
Encodes strictly universal H.264 MP4 videos.
'''

import os
import cv2
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .video_writer import save_video


def render_2d_topdown_frame(traj, frame_idx, fig=None, mode='spheres', show_vectors=True, 
                            show_trails=True, trail_length=20, color_by='height'):
    '''
    Render a single top-down (X-Y plane) 2D frame.
    mode: 'spheres' (2D solid discs with rotation pointers) or 'blobs' (constituent multiblobs)
    color_by: 'height' (z-coordinate), 'speed' (|v|), or 'index'
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
            ax.plot(trail[:, 0], trail[:, 1], color='#38bdf8', alpha=0.3, linewidth=1.0, zorder=2)

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
        c_label = 'Height $z$'
        z_min = min(0.0, np.min(traj.positions[..., 2]))
        z_max = max(np.max(traj.positions[..., 2]) + 0.5, R * 3.0)
        norm = plt.Normalize(vmin=z_min, vmax=z_max)

    colormap = plt.get_cmap(cmap)

    # --- Mode: 2D Discs + In-Plane Rotation Pointers ---
    if mode == 'spheres':
        patches = []
        for i in range(traj.num_bodies):
            circle = Circle((pos[i, 0], pos[i, 1]), radius=R)
            patches.append(circle)

        colors = colormap(norm(c_vals))
        p_coll = PatchCollection(patches, facecolors=colors, edgecolors='#ffffff', linewidths=0.8, alpha=0.9, zorder=4)
        ax.add_collection(p_coll)

        # In-plane orientation pointer (radial needle)
        for i in range(traj.num_bodies):
            th = angles[i]
            end_x = pos[i, 0] + R * math.cos(th)
            end_y = pos[i, 1] + R * math.sin(th)
            ax.plot([pos[i, 0], end_x], [pos[i, 1], end_y], color='#facc15', linewidth=1.8, zorder=6)
            ax.scatter([pos[i, 0]], [pos[i, 1]], color='#ffffff', s=8, zorder=7)

        # Velocity vectors
        if show_vectors and traj.num_frames > 1:
            vels = traj.velocities[frame_idx]
            v_max = max(np.max(traj.speeds), 1e-4)
            scale_factor = R / (v_max * 2.0)
            ax.quiver(pos[:, 0], pos[:, 1], vels[:, 0] * scale_factor, vels[:, 1] * scale_factor,
                      color='#38bdf8', angles='xy', scale_units='xy', scale=1.0, width=0.0035, alpha=0.75, zorder=5)

    # --- Mode: 2D Multi-Blobs ---
    elif mode == 'blobs':
        all_blobs = traj.get_blobs_for_frame(frame_idx)
        if all_blobs is not None:
            a = traj.blob_radius
            blob_patches = []
            for b in range(len(all_blobs)):
                blob_circle = Circle((all_blobs[b, 0], all_blobs[b, 1]), radius=a)
                blob_patches.append(blob_circle)
            
            blob_colors = colormap(norm(all_blobs[:, 2]))
            bp_coll = PatchCollection(blob_patches, facecolors=blob_colors, edgecolors='#ffffff',
                                      linewidths=0.3, alpha=0.85, zorder=4)
            ax.add_collection(bp_coll)
            ax.scatter(pos[:, 0], pos[:, 1], color='#38bdf8', s=10, zorder=6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(c_label, color='#e2e8f0', fontsize=10)
    cbar.ax.tick_params(colors='#e2e8f0', labelsize=8)

    # Formatting
    pad = R * 1.5
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', color='#1f2937', alpha=0.6)

    text_color = '#e2e8f0'
    ax.set_xlabel('X Coordinate', color=text_color, fontsize=11)
    ax.set_ylabel('Y Coordinate', color=text_color, fontsize=11)
    ax.tick_params(colors=text_color, labelsize=9)

    mode_title = f"2D Top-Down Suspension (Radius R={R:.2f})" if mode == 'spheres' else f"2D Multi-Blob Discretization (Blob Radius a={traj.blob_radius:.2f})"
    title = (f"{mode_title}\n"
             f"Time: {t_curr:.3f} s  |  Frame: {frame_idx + 1}/{traj.num_frames}  |  "
             f"Bodies: {traj.num_bodies}  |  Mean $\\langle z \\rangle$: {traj.mean_z[frame_idx]:.4f}")
    ax.set_title(title, color=text_color, fontsize=12, fontweight='bold', pad=10)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def render_2d_side_elevation_frame(traj, frame_idx, fig=None):
    '''
    Render a 2D Side-Elevation profile (X-Z plane) showing sedimentation towards the wall at z = 0.
    '''
    if fig is None:
        fig = plt.figure(figsize=(12, 6), dpi=100, facecolor='#0b0f19')
    else:
        fig.clf()

    fig.patch.set_facecolor('#0b0f19')
    ax = fig.add_subplot(111, facecolor='#111827')

    t_curr = traj.time_array[frame_idx]
    pos = traj.positions[frame_idx]
    R = traj.sphere_radius
    xmin, xmax = traj.domain_bounds[0], traj.domain_bounds[1]

    # Ground floor wall at z = 0
    ax.axhline(0.0, color='#f43f5e', linewidth=2.5, label='Floor Wall ($z=0$)', zorder=3)
    ax.axhline(R, color='#fbbf24', linestyle='--', linewidth=1.2, label=f'Contact Clearance ($z=R={R:.2f}$)', zorder=3)

    # Shaded floor
    ax.fill_between([xmin - 3*R, xmax + 3*R], -1.0, 0.0, color='#1f2937', alpha=0.8, zorder=2)

    # Draw sphere cross-sections in X-Z
    patches = []
    for i in range(traj.num_bodies):
        circle = Circle((pos[i, 0], pos[i, 2]), radius=R)
        patches.append(circle)

    norm = plt.Normalize(vmin=0, vmax=max(np.max(traj.positions[..., 2]) + 0.5, R * 3.0))
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(pos[:, 2]))
    p_coll = PatchCollection(patches, facecolors=colors, edgecolors='#ffffff', linewidths=0.6, alpha=0.85, zorder=5)
    ax.add_collection(p_coll)

    ax.scatter(pos[:, 0], pos[:, 2], color='#facc15', s=10, zorder=6)

    # Mean height line
    mean_z = traj.mean_z[frame_idx]
    ax.axhline(mean_z, color='#34d399', linestyle='-.', linewidth=1.8, label=f'Mean Height $\\langle z \\rangle = {mean_z:.3f}$', zorder=4)

    text_color = '#e2e8f0'
    pad = R * 1.5
    ax.set_xlim(xmin - pad, xmax + pad)
    z_max_plot = max(np.max(traj.positions[..., 2]) + 2 * R, R * 4.0)
    ax.set_ylim(-0.4, z_max_plot)
    ax.set_xlabel('X Coordinate', color=text_color, fontsize=11)
    ax.set_ylabel('Height Z (above wall)', color=text_color, fontsize=11)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.grid(True, linestyle='--', color='#1f2937', alpha=0.5)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor=text_color, fontsize=9, loc='upper right')

    ax.set_title(f"2D Side Elevation Profile (X-Z Plane)  |  Time: {t_curr:.3f} s  |  Frame: {frame_idx + 1}/{traj.num_frames}",
                 color=text_color, fontsize=12, fontweight='bold', pad=10)

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def render_2d_dual_frame(traj, frame_idx, fig_top=None, fig_side=None):
    '''
    Render synchronized Dual 2D View: Left Top-Down (X-Y) + Right Side Elevation (X-Z).
    '''
    frame_top = render_2d_topdown_frame(traj, frame_idx, fig=fig_top, mode='spheres')
    frame_side = render_2d_side_elevation_frame(traj, frame_idx, fig=fig_side)

    h_top, w_top, _ = frame_top.shape
    h_side, w_side, _ = frame_side.shape

    target_h = max(h_top, h_side)
    w_top_res = int(w_top * (target_h / h_top))
    w_side_res = int(w_side * (target_h / h_side))

    f_top_res = cv2.resize(frame_top, (w_top_res, target_h), interpolation=cv2.INTER_AREA)
    f_side_res = cv2.resize(frame_side, (w_side_res, target_h), interpolation=cv2.INTER_AREA)

    sep = np.full((target_h, 4, 3), 40, dtype=np.uint8)
    dual = np.hstack([f_top_res, sep, f_side_res])
    return dual


def render_2d_video(traj, output_path, view='top', mode='spheres', fps=24, 
                    show_vectors=True, show_trails=True, show_progress=True):
    '''
    Generate a 2D simulation MP4 video.
    view: 'top' (X-Y), 'side' (X-Z), or 'dual' (X-Y + X-Z side-by-side)
    mode: 'spheres' or 'blobs'
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frames = []

    if show_progress:
        print(f"[*] Rendering {traj.num_frames} 2D frames (view={view}, mode={mode})...")

    fig1 = plt.figure(figsize=(10, 10), dpi=100, facecolor='#0b0f19')
    fig2 = plt.figure(figsize=(12, 6), dpi=100, facecolor='#0b0f19') if view == 'dual' else None

    for i in range(traj.num_frames):
        if view == 'side':
            frame = render_2d_side_elevation_frame(traj, i, fig=fig1)
        elif view == 'dual':
            frame = render_2d_dual_frame(traj, i, fig_top=fig1, fig_side=fig2)
        else:  # top-down default
            frame = render_2d_topdown_frame(traj, i, fig=fig1, mode=mode,
                                           show_vectors=show_vectors, show_trails=show_trails)
        frames.append(frame)
        if show_progress:
            print(f"    Rendered 2D frame {i+1}/{traj.num_frames}", end='\r')

    plt.close(fig1)
    if fig2 is not None:
        plt.close(fig2)
    if show_progress:
        print()

    generated_files = save_video(frames, output_path, fps=fps, format='mp4')
    if show_progress:
        for f in generated_files:
            print(f"[+] Saved 2D MP4 Video: {f}")

    return generated_files
