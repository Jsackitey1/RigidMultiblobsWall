"""Geometry policies shared by initial-condition generation and visualization.

Blob surfaces are conservative geometric envelopes, not calibrated hydrodynamic
surfaces. Checks exclude pairs within the same rigid body.
"""
import itertools
import numpy as np


def bounding_radius(vertices, blob_radius):
    if not np.isfinite(blob_radius) or blob_radius <= 0:
        raise ValueError('blob_radius must be finite and positive')
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('vertices must be a nonempty finite (N, 3) array')
    return float(np.linalg.norm(vertices, axis=1).max() + blob_radius)


def minimum_image(delta, lengths):
    delta = np.array(delta, dtype=float, copy=True)
    for axis, length in enumerate(lengths[:2]):
        if length > 0:
            delta[..., axis] -= length * np.round(delta[..., axis] / length)
    return delta


def periodic_copies(center, radius, lengths, bounds):
    """Offsets for every image whose bounding disk intersects the display cell."""
    shifts = []
    for axis, length in enumerate(lengths[:2]):
        if length > 0:
            lo, hi = bounds[2*axis:2*axis+2]
            first = int(np.ceil((lo - radius - center[axis]) / length))
            last = int(np.floor((hi + radius - center[axis]) / length))
            shifts.append([k * length for k in range(first, last + 1)])
        else:
            shifts.append([0.0])
    return [np.array([x, y, 0.0]) for x, y in itertools.product(*shifts)]


def geometry_report(positions, quaternions, vertices, blob_radius, lengths=(0, 0), wall=True, tolerance=1e-7):
    from visualizer.trajectory_loader import batch_quaternion_to_rot_matrices
    rots = batch_quaternion_to_rot_matrices(quaternions)
    blobs = np.einsum('nij,bj->nbi', rots, vertices) + positions[:, None, :]
    minimum = float('inf')
    overlaps = 0
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            delta = minimum_image(blobs[i, :, None, :] - blobs[j, None, :, :], lengths)
            clearance = np.linalg.norm(delta, axis=-1) - 2 * blob_radius
            minimum = min(minimum, float(clearance.min()))
            overlaps += int(np.count_nonzero(clearance < -tolerance))
    # A body can intersect its own periodic image in a small cell. Check all
    # image shifts within its diameter, excluding the original body itself.
    radius = bounding_radius(vertices, blob_radius)
    axes = [range(-int(np.ceil(2*radius/L)), int(np.ceil(2*radius/L))+1) if L > 0 else [0] for L in lengths[:2]]
    self_min = float('inf')
    for ix, iy in itertools.product(*axes):
        if (ix, iy) <= (0, 0):
            continue  # count each opposite image pair once
        shift = np.array([ix*lengths[0], iy*lengths[1], 0.0])
        for body in blobs:
            distance = np.linalg.norm(body[:, None, :] - body[None, :, :] - shift, axis=-1) - 2*blob_radius
            self_min = min(self_min, float(distance.min()))
    heights = blobs[..., 2] - blob_radius
    return {
        'geometry_policy': 'blob_surface_envelope',
        'minimum_interbody_blob_clearance': minimum if np.isfinite(minimum) else None,
        'overlapping_interbody_blob_pairs': overlaps,
        'minimum_periodic_self_clearance': self_min if np.isfinite(self_min) else None,
        'minimum_wall_clearance': float(heights.min()) if wall else None,
        'bodies_with_wall_penetration': int(np.count_nonzero(np.any(heights < -tolerance, axis=1))) if wall else 0,
    }
