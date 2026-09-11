'''
Synchronized 2D Composite Dashboard Video Generator (2D Motion View + Dynamic Metrics).
'''

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .render_2d import render_2d_topdown_frame, render_2d_dual_frame
from .render_graphs import create_graph_dashboard_frame
from .video_writer import save_video


def render_2d_composite_frame(traj, frame_idx, fig_2d=None, fig_graph=None, view_2d='top'):
    '''
    Render a synchronized dual-pane composite frame:
    Left: 2D Physical Motion Scene (Top-down XY or Dual XY+XZ)
    Right: Dynamic Physical Metrics Dashboard
    '''
    if view_2d == 'dual':
        frame_2d = render_2d_dual_frame(traj, frame_idx)
    else:
        frame_2d = render_2d_topdown_frame(traj, frame_idx, fig=fig_2d, mode='spheres')

    frame_graph = create_graph_dashboard_frame(traj, frame_idx, fig=fig_graph)

    h2, w2, _ = frame_2d.shape
    hg, wg, _ = frame_graph.shape

    target_h = max(h2, hg)
    w2_resized = int(w2 * (target_h / h2))
    wg_resized = int(wg * (target_h / hg))

    frame_2d_res = cv2.resize(frame_2d, (w2_resized, target_h), interpolation=cv2.INTER_AREA)
    frame_graph_res = cv2.resize(frame_graph, (wg_resized, target_h), interpolation=cv2.INTER_AREA)

    sep = np.full((target_h, 4, 3), 40, dtype=np.uint8)
    composite = np.hstack([frame_2d_res, sep, frame_graph_res])
    return composite


def render_composite_video(traj, output_path, view_2d='top', fps=24, show_progress=True):
    '''
    Generate high-definition side-by-side composite MP4 video.
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig_2d = plt.figure(figsize=(10, 10), dpi=100, facecolor='#0b0f19')
    fig_graph = plt.figure(figsize=(12, 10), dpi=100, facecolor='#12141d')
    frames = []

    if show_progress:
        print(f"[*] Rendering {traj.num_frames} composite 2D dashboard frames...")

    for i in range(traj.num_frames):
        comp_frame = render_2d_composite_frame(traj, i, fig_2d=fig_2d, fig_graph=fig_graph, view_2d=view_2d)
        frames.append(comp_frame)
        if show_progress:
            print(f"    Rendered composite frame {i+1}/{traj.num_frames}", end='\r')

    plt.close(fig_2d)
    plt.close(fig_graph)
    if show_progress:
        print()

    generated_files = save_video(frames, output_path, fps=fps, format='mp4')
    if show_progress:
        for f in generated_files:
            print(f"[+] Saved Composite 2D Video: {f}")

    return generated_files
