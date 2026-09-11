'''
Graph Video & Dynamic Metrics Dashboard Generator.
'''

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_graph_dashboard_frame(traj, frame_idx, fig=None, trail_length=10):
    '''
    Render a single high-quality 4-panel metrics dashboard frame for the given frame_idx.
    Returns: RGB numpy array (H, W, 3) uint8.
    '''
    if fig is None:
        fig = plt.figure(figsize=(14, 9), dpi=100, facecolor='#12141d')
    else:
        fig.clf()

    fig.patch.set_facecolor('#12141d')
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28, left=0.08, right=0.94, top=0.88, bottom=0.09)

    t_curr = traj.time_array[frame_idx]
    pos = traj.positions[frame_idx]
    speeds = traj.speeds[frame_idx]
    z_vals = pos[:, 2]

    # Style colors
    text_color = '#e2e8f0'
    grid_color = '#2d3748'
    accent_blue = '#38bdf8'
    accent_emerald = '#34d399'
    accent_amber = '#fbbf24'
    accent_rose = '#f43f5e'

    # --- PANEL 1 (Top-Left): 2D Top-Down Motion (X-Y) ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#1e2230')
    ax1.set_title('Top-Down Particle Distribution (X-Y Plane)', color=text_color, fontsize=12, fontweight='bold', pad=8)
    
    # Trails
    start_t = max(0, frame_idx - trail_length)
    if frame_idx > 0:
        for b in range(traj.num_bodies):
            trail_pos = traj.positions[start_t:frame_idx + 1, b]
            ax1.plot(trail_pos[:, 0], trail_pos[:, 1], color=accent_blue, alpha=0.25, linewidth=1.0)

    # Current body positions colored by height z
    sc1 = ax1.scatter(pos[:, 0], pos[:, 1], c=z_vals, cmap='viridis', s=45, edgecolors='white', linewidths=0.5, alpha=0.9, zorder=5)
    cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Height $z$', color=text_color, fontsize=9)
    cbar1.ax.tick_params(colors=text_color, labelsize=8)

    ax1.set_xlim(traj.domain_bounds[0], traj.domain_bounds[1])
    ax1.set_ylim(traj.domain_bounds[2], traj.domain_bounds[3])
    ax1.set_xlabel('X Position', color=text_color, fontsize=10)
    ax1.set_ylabel('Y Position', color=text_color, fontsize=10)
    ax1.tick_params(colors=text_color, labelsize=8)
    ax1.grid(True, linestyle='--', alpha=0.3, color=grid_color)

    # --- PANEL 2 (Top-Right): Height Evolution & Sedimentation ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#1e2230')
    ax2.set_title('Sedimentation Height vs Time', color=text_color, fontsize=12, fontweight='bold', pad=8)

    # Historical mean z curve up to current frame
    times = traj.time_array[:frame_idx + 1]
    mean_z_hist = traj.mean_z[:frame_idx + 1]
    std_z_hist = traj.std_z[:frame_idx + 1]

    # All frames envelope for context
    ax2.plot(traj.time_array, traj.mean_z, color='#475569', linestyle=':', alpha=0.6, label='Full Mean Z')
    
    if len(times) > 0:
        ax2.fill_between(times, mean_z_hist - std_z_hist, mean_z_hist + std_z_hist, color=accent_emerald, alpha=0.15)
        ax2.plot(times, mean_z_hist, color=accent_emerald, linewidth=2.5, label=r'$\langle z(t) \rangle \pm \sigma_z$')
        ax2.scatter([t_curr], [traj.mean_z[frame_idx]], color=accent_emerald, s=60, zorder=6)

    # Wall boundary line
    ax2.axhline(0.0, color=accent_rose, linestyle='-', linewidth=1.5, alpha=0.8, label='Wall ($z=0$)')
    ax2.axhline(traj.sphere_radius, color=accent_amber, linestyle='--', linewidth=1.0, alpha=0.7, label=f'Contact ($z=R$)')

    ax2.set_xlim(traj.time_array[0] - 0.05, traj.time_array[-1] + 0.05)
    z_min_plot = min(0.0, np.min(traj.min_z) - 0.2)
    z_max_plot = max(np.max(traj.max_z) + 0.5, traj.sphere_radius * 2.5)
    ax2.set_ylim(z_min_plot, z_max_plot)
    ax2.set_xlabel('Time $t$', color=text_color, fontsize=10)
    ax2.set_ylabel('Height $z$', color=text_color, fontsize=10)
    ax2.tick_params(colors=text_color, labelsize=8)
    ax2.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    ax2.legend(facecolor='#1e2230', edgecolor=grid_color, labelcolor=text_color, fontsize=8, loc='upper right')

    # --- PANEL 3 (Bottom-Left): Speed Distribution ---
    ax3 = fig.add_subplot(gs[1, 0], facecolor='#1e2230')
    ax3.set_title('Instantaneous Speed Distribution', color=text_color, fontsize=12, fontweight='bold', pad=8)

    max_speed = max(np.max(traj.speeds), 0.01)
    bins = np.linspace(0, max_speed * 1.05, 20)
    n_counts, _, _ = ax3.hist(speeds, bins=bins, color=accent_blue, edgecolor='#0284c7', alpha=0.75, rwidth=0.85)

    mean_speed = np.mean(speeds)
    ax3.axvline(mean_speed, color=accent_amber, linestyle='--', linewidth=2.0, label=f'Mean Speed: {mean_speed:.3f}')
    ax3.set_xlabel('Translational Speed $|v|$', color=text_color, fontsize=10)
    ax3.set_ylabel('Particle Count', color=text_color, fontsize=10)
    ax3.tick_params(colors=text_color, labelsize=8)
    ax3.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    ax3.legend(facecolor='#1e2230', edgecolor=grid_color, labelcolor=text_color, fontsize=8, loc='upper right')

    # --- PANEL 4 (Bottom-Right): Mean Squared Displacement (MSD) ---
    ax4 = fig.add_subplot(gs[1, 1], facecolor='#1e2230')
    ax4.set_title('Mean Squared Displacement (MSD)', color=text_color, fontsize=12, fontweight='bold', pad=8)

    ax4.plot(traj.time_array, traj.msd, color='#475569', linestyle=':', alpha=0.6)
    if len(times) > 0:
        ax4.plot(times, traj.msd[:frame_idx + 1], color=accent_amber, linewidth=2.5, label='MSD$(t)$')
        ax4.scatter([t_curr], [traj.msd[frame_idx]], color=accent_amber, s=60, zorder=6)

    ax4.set_xlim(traj.time_array[0] - 0.05, traj.time_array[-1] + 0.05)
    max_msd = max(np.max(traj.msd) * 1.1, 0.01)
    ax4.set_ylim(0, max_msd)
    ax4.set_xlabel('Time $t$', color=text_color, fontsize=10)
    ax4.set_ylabel(r'MSD $\langle |\Delta \mathbf{R}(t)|^2 \rangle$', color=text_color, fontsize=10)
    ax4.tick_params(colors=text_color, labelsize=8)
    ax4.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    ax4.legend(facecolor='#1e2230', edgecolor=grid_color, labelcolor=text_color, fontsize=8, loc='upper left')

    # --- TOP HUD HEADER ---
    title_text = (f"RigidMultiblobsWall Simulation Dashboard  |  "
                  f"Frame: {frame_idx + 1}/{traj.num_frames}  |  "
                  f"Time: {t_curr:.3f} s  |  "
                  f"Bodies: {traj.num_bodies}  |  "
                  f"Mean $\\langle z \\rangle$: {traj.mean_z[frame_idx]:.4f}")
    fig.suptitle(title_text, color=text_color, fontsize=13, fontweight='bold', y=0.96)

    # Convert plot to RGB array
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:, :, :3].copy()
    return rgb


def render_graph_video(traj, output_path, fps=10, format='mp4', trail_length=10, show_progress=True):
    '''
    Generate an animated graph video or GIF for the entire simulation trajectory.
    output_path: Path to save the video file (e.g. 'data/graph_video.mp4')
    fps: Frames per second
    format: 'mp4', 'gif', or 'both'
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_name, _ = os.path.splitext(output_path)

    fig = plt.figure(figsize=(14, 9), dpi=100, facecolor='#12141d')
    frames = []

    if show_progress:
        print(f"[*] Rendering {traj.num_frames} graph dashboard frames...")

    for i in range(traj.num_frames):
        rgb_frame = create_graph_dashboard_frame(traj, i, fig=fig, trail_length=trail_length)
        frames.append(rgb_frame)
        if show_progress:
            print(f"    Rendered graph frame {i+1}/{traj.num_frames}", end='\r')

    plt.close(fig)
    if show_progress:
        print()

    from .video_writer import save_video
    generated_files = save_video(frames, output_path, fps=fps, format=format)
    if show_progress:
        for f in generated_files:
            print(f"[+] Saved Graph Video: {f}")
    return generated_files
