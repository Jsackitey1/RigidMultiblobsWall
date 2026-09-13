"""Reports use saved solver frames only, independent of animation sampling."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from geometry import geometry_report, minimum_image


def write_diagnostics(traj, output_dir, plots=False):
    raw = traj.analysis_source or traj
    os.makedirs(output_dir, exist_ok=True)
    report = {
        'source': 'raw_solver_frames', 'frames': raw.num_frames, 'bodies': raw.num_bodies,
        'first_time': float(raw.time_array[0]), 'last_time': float(raw.time_array[-1]),
        'periodic_lengths': raw.periodic_lengths.tolist(),
        'bounding_radius': raw.bounding_radius, 'display_radius': raw.sphere_radius,
        'blob_radius': raw.blob_radius, 'nominal_radius': raw.nominal_radius,
        'displacement_rate_definition': 'forward saved-interval displacement / interval duration; final frame repeats last interval',
        'msd_definition': 'ensemble mean squared 3D displacement of body reference point from first saved frame; includes drift',
        'time': raw.time_array.tolist(), 'msd': raw.msd.tolist(),
        'mean_height': raw.mean_z.tolist(),
    }
    wall = raw.metadata.get('domain', 'single_wall') != 'no_wall'
    if raw.vertex_blobs is not None:
        # Pairwise checks are explicitly limited to the initial saved frame.
        report['initial_frame_geometry'] = geometry_report(raw.positions_unwrapped[0], raw.quaternions[0],
                            raw.vertex_blobs, raw.blob_radius, raw.periodic_lengths, wall)
        if wall:
            minima = [float(raw.get_blobs_for_frame(i)[:, 2].min()-raw.blob_radius) for i in range(raw.num_frames)]
            report['minimum_blob_wall_clearance_by_frame'] = minima
            report['minimum_blob_wall_clearance'] = min(minima)
    report_path = os.path.join(output_dir, 'validation.json')
    with open(report_path, 'w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    paths = [report_path]
    if plots:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(raw.positions[0, :, 0], raw.positions[0, :, 2], label='Initial body centers')
        if wall:
            axes[0].axhline(0, color='black', label='Wall')
        axes[0].set(xlabel='x', ylabel='z', title='Initial X–Z view')
        axes[0].legend()
        axes[1].hist(raw.positions_unwrapped[..., 2].ravel(), bins=25, density=True)
        axes[1].set(xlabel='Body center height z', ylabel='Probability density', title='Height distribution (saved-frame samples)')
        fig.tight_layout()
        path = os.path.join(output_dir, 'height_diagnostics.png')
        fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
        if np.all(raw.periodic_lengths > 0) and raw.num_bodies > 1:
            # Unordered pairs, full annuli only: valid for a rectangular 2D torus.
            centers = raw.positions[0, :, :2]
            delta = minimum_image(centers[:, None, :] - centers[None, :, :], raw.periodic_lengths)
            distances = np.linalg.norm(delta, axis=-1)[np.triu_indices(raw.num_bodies, 1)]
            edges = np.linspace(0, min(raw.periodic_lengths)/2, 31)
            counts, _ = np.histogram(distances, edges)
            expected = raw.num_bodies*(raw.num_bodies-1)/2 * np.pi*np.diff(edges**2)/np.prod(raw.periodic_lengths)
            fig, ax = plt.subplots()
            ax.plot((edges[:-1]+edges[1:])/2, counts/expected)
            ax.set(xlabel='Projected center separation r', ylabel='g(r)', title='Initial projected pair distribution')
            path = os.path.join(output_dir, 'pair_distribution.png')
            fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return paths
