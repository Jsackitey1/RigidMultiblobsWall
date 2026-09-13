'''
Trajectory loader, kinematics engine, and SLERP interpolation for RigidMultiblobsWall.
Supports arbitrary simulations, body shapes, and particle radii without hardcoded parameters.
'''

import os
import re
import math
import warnings
import numpy as np
from geometry import bounding_radius


def quaternion_to_rot_matrix(q):
    '''
    Convert quaternion [q0, q1, q2, q3] to 3x3 rotation matrix.
    q0 is scalar part (s), (q1, q2, q3) is vector part (p).
    '''
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    norm = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion must be finite and nonzero")
    if norm > 1e-12:
        q0, q1, q2, q3 = q0 / norm, q1 / norm, q2 / norm, q3 / norm
    
    diag = q0 * q0 - 0.5
    rot = 2.0 * np.array([
        [q1*q1 + diag,       q1*q2 - q0*q3, q1*q3 + q0*q2],
        [q2*q1 + q0*q3,      q2*q2 + diag,  q2*q3 - q0*q1],
        [q3*q1 - q0*q2,      q3*q2 + q0*q1, q3*q3 + diag]
    ], dtype=np.float64)
    return rot


def batch_quaternion_to_rot_matrices(quats):
    '''
    Vectorized conversion of (N, 4) quaternions to (N, 3, 3) rotation matrices.
    '''
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("Quaternions must be finite and nonzero")
    q = quats / norms

    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]

    diag = q0 * q0 - 0.5
    N = quats.shape[0]
    rots = np.zeros((N, 3, 3), dtype=np.float64)

    rots[:, 0, 0] = 2.0 * (q1 * q1 + diag)
    rots[:, 0, 1] = 2.0 * (q1 * q2 - q0 * q3)
    rots[:, 0, 2] = 2.0 * (q1 * q3 + q0 * q2)

    rots[:, 1, 0] = 2.0 * (q2 * q1 + q0 * q3)
    rots[:, 1, 1] = 2.0 * (q2 * q2 + diag)
    rots[:, 1, 2] = 2.0 * (q2 * q3 - q0 * q1)

    rots[:, 2, 0] = 2.0 * (q3 * q1 - q0 * q2)
    rots[:, 2, 1] = 2.0 * (q3 * q2 + q0 * q1)
    rots[:, 2, 2] = 2.0 * (q3 * q3 + diag)

    return rots


def slerp(q0, q1, alpha):
    '''
    Spherical Linear Interpolation between two quaternions q0 and q1 for parameter alpha in [0, 1].
    q: (4,) unit quaternion array.
    '''
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # If negative dot product, invert one quaternion to take shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Clamp dot product to prevent numerical issues with arccos
    dot = min(1.0, max(-1.0, dot))

    if dot > 0.9995:
        # Quaternions are very close; linear interpolation with normalization
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return result / np.linalg.norm(result)


def batch_slerp(quats0, quats1, alpha):
    '''
    Vectorized SLERP for arrays of quaternions: (N, 4) -> (N, 4).
    '''
    # Normalize
    q0 = quats0 / np.linalg.norm(quats0, axis=-1, keepdims=True)
    q1 = quats1 / np.linalg.norm(quats1, axis=-1, keepdims=True)

    dots = np.sum(q0 * q1, axis=-1, keepdims=True)  # (N, 1)

    # Shortest path flip
    neg_mask = dots < 0.0
    q1 = np.where(neg_mask, -q1, q1)
    dots = np.where(neg_mask, -dots, dots)
    dots = np.clip(dots, -1.0, 1.0)

    # Close angle linear fallback
    close_mask = dots > 0.9995
    linear_interp = q0 + alpha * (q1 - q0)
    linear_interp = linear_interp / np.linalg.norm(linear_interp, axis=-1, keepdims=True)

    # Spherical interpolation
    theta_0 = np.arccos(dots)
    sin_theta_0 = np.sin(theta_0)
    sin_theta_0 = np.where(sin_theta_0 == 0, 1.0, sin_theta_0)
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)

    s0 = np.cos(theta) - dots * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    spherical_interp = s0 * q0 + s1 * q1
    spherical_interp = spherical_interp / np.linalg.norm(spherical_interp, axis=-1, keepdims=True)

    return np.where(close_mask, linear_interp, spherical_interp)


class SimulationTrajectory:
    '''
    Encapsulates all trajectory frames, body geometries, blob discretizations,
    and computed physical metrics for any RigidMultiblobsWall simulation.
    '''

    def __init__(self, positions, quaternions, time_array, dt, vertex_blobs=None, 
                 blob_radius=0.25, sphere_radius=1.0, domain_bounds=None, metadata=None, periodic_lengths=(0, 0),
                 analysis_source=None):
        positions = np.array(positions, dtype=float, copy=True)
        quaternions = np.array(quaternions, dtype=float, copy=True)
        time_array = np.asarray(time_array, dtype=float)
        if positions.ndim != 3 or positions.shape[-1] != 3 or not all(positions.shape[:2]) or not np.isfinite(positions).all():
            raise ValueError('Positions must be finite, nonempty (frames, bodies, 3)')
        if quaternions.shape != positions.shape[:2] + (4,):
            raise ValueError('Quaternion shape must match positions')
        norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise ValueError('Quaternions must be finite and nonzero')
        quaternions /= norms
        if time_array.shape != (len(positions),) or not np.isfinite(time_array).all() or np.any(np.diff(time_array) <= 0):
            raise ValueError('Timestamps must be finite and strictly increasing')
        self.periodic_lengths = np.asarray(periodic_lengths[:2], dtype=float)
        if self.periodic_lengths.shape != (2,) or not np.isfinite(self.periodic_lengths).all() or np.any(self.periodic_lengths < 0):
            raise ValueError('Periodic lengths must be two nonnegative finite values')
        self.positions_unwrapped = positions
        self.positions_wrapped = positions.copy()
        for axis, length in enumerate(self.periodic_lengths):
            if length > 0:
                origin = domain_bounds[2*axis] if domain_bounds is not None else 0.0
                self.positions_wrapped[..., axis] = origin + (positions[..., axis] - origin) % length
        self.positions = self.positions_wrapped  # compatibility: display coordinates
        self.analysis_source = analysis_source
        self.bounding_radius = compute_effective_radius(vertex_blobs, blob_radius)
        self.nominal_radius = (metadata or {}).get('nominal_radius')
        self.quaternions = quaternions       # (num_frames, num_bodies, 4)
        self.time_array = time_array         # (num_frames,)
        self.dt = dt
        self.num_frames, self.num_bodies, _ = positions.shape
        self.vertex_blobs = vertex_blobs
        self.num_blobs_per_body = len(vertex_blobs) if vertex_blobs is not None else 0
        self.blob_radius = blob_radius
        self.sphere_radius = sphere_radius
        self.domain_bounds = domain_bounds
        self.metadata = dict(metadata or {})

        # Auto-compute domain bounds if not provided
        if self.domain_bounds is None:
            pad = max(self.sphere_radius * 2.5, 2.0)
            xmin = float(np.min(self.positions[..., 0]) - pad)
            xmax = float(np.max(self.positions[..., 0]) + pad)
            ymin = float(np.min(self.positions[..., 1]) - pad)
            ymax = float(np.max(self.positions[..., 1]) + pad)
            zmin = 0.0
            zmax = float(max(np.max(self.positions[..., 2]) + pad, self.sphere_radius * 3.5))
            self.domain_bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
            for axis, length in enumerate(self.periodic_lengths):
                if length > 0:
                    self.domain_bounds[2*axis:2*axis+2] = [0.0, length]

        if analysis_source is None:
            self._compute_kinematics()
        else:
            # Animation samples reference raw-frame analysis; never differentiate
            # interpolated paths or present their MSD as a new measurement.
            indices = np.clip(np.searchsorted(analysis_source.time_array, time_array, side='right') - 1,
                              0, analysis_source.num_frames - 1)
            for name in ('velocities', 'displacement_rate', 'speeds', 'mean_z', 'min_z', 'max_z', 'std_z'):
                setattr(self, name, getattr(analysis_source, name)[indices])
            self.msd = None

    def _compute_kinematics(self):
        '''Pre-compute velocities, height statistics, MSD, and in-plane 2D angles.'''
        self.velocities = np.zeros_like(self.positions_unwrapped)
        if self.num_frames > 1:
            intervals = np.diff(self.time_array)
            self.velocities[:-1] = np.diff(self.positions_unwrapped, axis=0) / intervals[:, None, None]
            self.velocities[-1] = self.velocities[-2]  # display last measured interval
        self.displacement_rate = self.velocities

        self.speeds = np.linalg.norm(self.velocities, axis=-1)  # (num_frames, num_bodies)

        # Height statistics
        z_coords = self.positions[..., 2]
        self.mean_z = np.mean(z_coords, axis=-1)
        self.min_z = np.min(z_coords, axis=-1)
        self.max_z = np.max(z_coords, axis=-1)
        self.std_z = np.std(z_coords, axis=-1)

        # Mean Squared Displacement (MSD)
        disp = self.positions_unwrapped - self.positions_unwrapped[0:1]
        self.msd = np.mean(np.sum(disp**2, axis=-1), axis=-1)

    def get_interpolated_trajectory(self, target_duration=20.0, fps=30):
        '''
        Generate a smoothly interpolated trajectory suitable for the requested playback duration and FPS.
        target_duration: Target video length in seconds (default: 20.0s)
        fps: Target frames per second (default: 30 fps -> 600 total frames)
        '''
        if not np.isfinite(target_duration) or target_duration <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Playback duration and fps must be positive')
        if self.num_frames <= 1:
            return self

        total_target_frames = int(round(target_duration * fps))
        if total_target_frames < 2:
            raise ValueError('Duration and fps must allow at least two frames to preserve both endpoints')

        # Original frame timestamps normalized in [0, 1]
        orig_indices = (self.time_array - self.time_array[0]) / (self.time_array[-1] - self.time_array[0])
        target_indices = np.linspace(0, 1, total_target_frames)

        new_positions = np.zeros((total_target_frames, self.num_bodies, 3), dtype=np.float64)
        new_quaternions = np.zeros((total_target_frames, self.num_bodies, 4), dtype=np.float64)
        new_time = np.linspace(self.time_array[0], self.time_array[-1], total_target_frames)

        # Interpolate each target frame
        for k, u in enumerate(target_indices):
            # Find the segment [i, i+1]
            seg_idx = np.searchsorted(orig_indices, u) - 1
            seg_idx = max(0, min(self.num_frames - 2, seg_idx))
            
            u0 = orig_indices[seg_idx]
            u1 = orig_indices[seg_idx + 1]
            alpha = (u - u0) / (u1 - u0) if u1 > u0 else 0.0
            alpha = min(1.0, max(0.0, alpha))

            # Linear interpolation for positions
            pos0 = self.positions_unwrapped[seg_idx]
            pos1 = self.positions_unwrapped[seg_idx + 1]
            new_positions[k] = (1.0 - alpha) * pos0 + alpha * pos1

            # SLERP for quaternions
            quat0 = self.quaternions[seg_idx]
            quat1 = self.quaternions[seg_idx + 1]
            new_quaternions[k] = batch_slerp(quat0, quat1, alpha)

        dt_interp = (self.time_array[-1] - self.time_array[0]) / max(1, total_target_frames - 1)

        return SimulationTrajectory(
            positions=new_positions,
            quaternions=new_quaternions,
            time_array=new_time,
            dt=dt_interp,
            vertex_blobs=self.vertex_blobs,
            blob_radius=self.blob_radius,
            sphere_radius=self.sphere_radius,
            domain_bounds=self.domain_bounds,
            metadata={**self.metadata, 'interpolated_for_visualization': True},
            periodic_lengths=self.periodic_lengths,
            analysis_source=self.analysis_source or self
        )

    def get_blobs_for_frame(self, frame_idx):
        '''
        Compute lab coordinates of all blobs in specified frame: (num_bodies * num_blobs, 3).
        '''
        if self.vertex_blobs is None or self.num_blobs_per_body == 0:
            return None

        pos = self.positions[frame_idx]
        quats = self.quaternions[frame_idx]
        rots = batch_quaternion_to_rot_matrices(quats)

        rotated_blobs = np.einsum('nij,bj->nbi', rots, self.vertex_blobs)
        all_blobs = rotated_blobs + pos[:, np.newaxis, :]
        return all_blobs.reshape(-1, 3)

    def get_body_orientation_vectors(self, frame_idx):
        '''
        Compute local 3x3 rotation matrices for all bodies in frame_idx.
        '''
        quats = self.quaternions[frame_idx]
        return batch_quaternion_to_rot_matrices(quats)

    def get_inplane_orientation_angles(self, frame_idx):
        '''
        Compute the 2D in-plane orientation angle theta in radians for all bodies in frame_idx.
        theta = atan2(u_x_y, u_x_x) from the rotated body local x-axis.
        '''
        rots = self.get_body_orientation_vectors(frame_idx)
        # Local x-axis rotated into lab frame: rots[:, :, 0] -> (N, 3)
        ux = rots[:, 0, 0]
        uy = rots[:, 1, 0]
        return np.arctan2(uy, ux)


def compute_effective_radius(vertex_blobs, blob_radius, fallback=1.0):
    '''
    Dynamically compute outer bounding radius of a multiblob body from its vertex geometry.
    R = max_k ||r_k|| + a
    '''
    if vertex_blobs is not None and len(vertex_blobs) > 0:
        return bounding_radius(vertex_blobs, blob_radius)
    return float(blob_radius if blob_radius > 0 else fallback)


def parse_input_file(input_file_path):
    '''Parse key-value parameters from a RigidMultiblobsWall input file.'''
    params = {}
    if not input_file_path or not os.path.exists(input_file_path):
        return params

    # Track multi-occurrence keys (structure, obstacle, articulated)
    num_structures = 0
    num_obstacles = 0
    num_articulated = 0

    with open(input_file_path, 'r') as f:
        for line in f:
            # Strip inline comments before parsing (Bug 3)
            if '#' in line:
                line = line.split('#', 1)[0]
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                key = parts[0]
                val = parts[1].strip()
                # Index multi-occurrence keys to avoid overwriting (Bug 9)
                if key == 'structure':
                    key = 'structure' + str(num_structures)
                    num_structures += 1
                elif key == 'obstacle':
                    key = 'obstacle' + str(num_obstacles)
                    num_obstacles += 1
                elif key == 'articulated':
                    key = 'articulated' + str(num_articulated)
                    num_articulated += 1
                params[key] = val
    return params


def parse_vertex_file(vertex_file_path):
    '''
    Parse reference multiblob vertices from a .vertex file.
    Strips # comments before parsing, matching the repo convention.
    Returns (blobs_array, num_blobs) — does NOT read a radius from the header.
    '''
    if not vertex_file_path or not os.path.exists(vertex_file_path):
        return None

    blobs = []
    num_blobs = None
    with open(vertex_file_path, 'r') as f:
        for line in f:
            # Strip comments (Bug 2: handles comment-headed .vertex files)
            if '#' in line:
                line = line.split('#', 1)[0]
            line = line.strip()
            if not line:
                continue
            if num_blobs is None:
                # First non-comment, non-blank line is the header with blob count
                num_blobs = int(line.split()[0])
            else:
                parts = line.split()
                if len(parts) >= 3:
                    blobs.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(blobs) >= num_blobs:
                    break

    if not blobs:
        return None
    if len(blobs) != num_blobs or not np.isfinite(blobs).all():
        raise ValueError('Vertex file must contain the declared number of finite vertices')
    return np.array(blobs, dtype=np.float64)


def parse_config_file(config_file_path):
    '''
    Parse all valid frames from a .config simulation output file.
    Warns on truncated/malformed frames instead of silently discarding them.
    '''
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    frames_positions = []
    frames_quaternions = []

    with open(config_file_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    total_lines = len(lines)
    frame_number = 0

    while idx < total_lines:
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        try:
            num_bodies = int(line)
        except ValueError:
            idx += 1
            continue

        if num_bodies <= 0 or (frames_positions and num_bodies != len(frames_positions[0])):
            raise ValueError('Frame body count must be positive and constant')
        idx += 1
        if idx + num_bodies > total_lines:
            warnings.warn(
                f"Config file truncated: frame {frame_number} declares {num_bodies} bodies "
                f"but only {total_lines - idx} lines remain. "
                f"Keeping {len(frames_positions)} complete frames.",
                RuntimeWarning
            )
            break

        pos_list = []
        quat_list = []
        frame_valid = True

        for b in range(num_bodies):
            data_line = lines[idx + b].strip()
            try:
                parts = [float(x) for x in data_line.split()]
            except ValueError as e:
                warnings.warn(
                    f"Config file: non-numeric data in frame {frame_number}, body {b}: "
                    f"{data_line!r} ({e}). Keeping {len(frames_positions)} complete frames.",
                    RuntimeWarning
                )
                frame_valid = False
                break
            if len(parts) < 7:
                warnings.warn(
                    f"Config file: frame {frame_number}, body {b} has only {len(parts)} values "
                    f"(need 7). Keeping {len(frames_positions)} complete frames.",
                    RuntimeWarning
                )
                frame_valid = False
                break
            pos_list.append(parts[0:3])
            quat_list.append(parts[3:7])

        if frame_valid and len(pos_list) == num_bodies:
            frames_positions.append(pos_list)
            frames_quaternions.append(quat_list)
            idx += num_bodies
            frame_number += 1
        else:
            break

    if not frames_positions:
        raise ValueError(f"No valid frames found in config file: {config_file_path}")

    positions = np.array(frames_positions, dtype=np.float64)
    quaternions = np.array(frames_quaternions, dtype=np.float64)
    return positions, quaternions


def load_simulation_data(config_file, input_file=None, vertex_file=None, manual_radius=None,
                         structure_index=None, timestamps=None, coordinates="unwrapped"):
    '''
    High-level factory function to load any simulation run and construct
    a complete SimulationTrajectory instance with dynamic radius resolution.
    '''
    positions, quaternions = parse_config_file(config_file)
    num_frames, num_bodies, _ = positions.shape

    # Parse input file metadata
    params = parse_input_file(input_file) if input_file else {}
    dt = float(params.get('dt', 0.1))
    n_save = int(params.get('n_save', 1))
    blob_radius = float(params.get('blob_radius', 0.25))

    structures = [key for key in params if re.fullmatch(r'structure\d+', key)]
    if structure_index is None:
        if len(structures) > 1 and vertex_file is None:
            raise ValueError('Multiple structures: select structure_index or provide vertex_file explicitly')
        structure_index = 0
    if structure_index < 0 or (structures and 'structure'+str(structure_index) not in params):
        raise ValueError('Selected structure index does not exist')
    structure_key = 'structure' + str(structure_index)
    # Resolve vertex file from selected structure (structure0) (Bug 9)
    if vertex_file is None and structure_key in params:
        struct_parts = params[structure_key].split()
        if len(struct_parts) >= 1:
            candidate_v = struct_parts[0]
            if os.path.exists(candidate_v):
                vertex_file = candidate_v
            else:
                base_dir = os.path.dirname(input_file) if input_file else ''
                candidate_rel = os.path.join(base_dir, candidate_v)
                if os.path.exists(candidate_rel):
                    vertex_file = candidate_rel

    if vertex_file is None and structure_key in params:
        raise FileNotFoundError('Cannot resolve selected structure vertex file')
    vertex_blobs = parse_vertex_file(vertex_file) if vertex_file else None
    if vertex_file is not None and vertex_blobs is None:
        raise ValueError('Selected vertex file is missing or empty')
    if not np.isfinite(blob_radius) or blob_radius <= 0:
        raise ValueError('blob_radius must be finite and positive')
    if manual_radius is not None and (not np.isfinite(manual_radius) or manual_radius <= 0):
        raise ValueError('Display radius must be finite and positive')

    # Dynamic radius calculation (Bug 1: always use compute_effective_radius)
    if manual_radius is not None and manual_radius > 0:
        sphere_radius = float(manual_radius)
    else:
        sphere_radius = compute_effective_radius(vertex_blobs, blob_radius, fallback=1.0)

    # Build time array
    dt_effective = dt * n_save
    if not np.isfinite(dt_effective) or dt <= 0 or n_save <= 0:
        raise ValueError('dt and n_save must be positive')
    first_step = max(0, int(params.get('initial_step', 0)))
    first_step = ((first_step + n_save - 1) // n_save) * n_save
    time_array = np.asarray(timestamps, dtype=float) if timestamps is not None else first_step * dt + np.arange(num_frames) * dt_effective

    periodic_length = np.zeros(2)
    if 'periodic_length' in params:
        values = [float(x) for x in params['periodic_length'].split()]
        if len(values) < 2 or not np.isfinite(values).all() or min(values[:2]) < 0:
            raise ValueError('Invalid periodic_length')
        periodic_length = np.array(values[:2])
    pad = max(sphere_radius * 2.5, 2.0)
    domain_bounds = [float(positions[..., 0].min()-pad), float(positions[..., 0].max()+pad),
                     float(positions[..., 1].min()-pad), float(positions[..., 1].max()+pad),
                     0.0, float(max(positions[..., 2].max()+pad, sphere_radius*3.5))]
    if 'box' in params:
        box = [float(x) for x in params['box'].split()]
        if len(box) == 2:
            domain_bounds[:4] = [0, box[0], 0, box[1]]
        elif len(box) == 4:
            domain_bounds[:4] = box
    for axis, length in enumerate(periodic_length):
        if length > 0:
            domain_bounds[2*axis:2*axis+2] = [0.0, length]
    if coordinates not in ('wrapped', 'unwrapped'):
        raise ValueError('coordinates must be wrapped or unwrapped')
    if coordinates == 'wrapped':
        warnings.warn('Unwrapping assumes displacement below half a periodic cell per saved interval; larger motion cannot be recovered.', RuntimeWarning)
        delta = np.diff(positions, axis=0)
        for axis, length in enumerate(periodic_length):
            if length > 0:
                delta[..., axis] -= length * np.round(delta[..., axis] / length)
        positions = np.concatenate([positions[:1], positions[:1] + np.cumsum(delta, axis=0)], axis=0)
    params['coordinate_source'] = coordinates

    return SimulationTrajectory(
        positions=positions,
        quaternions=quaternions,
        time_array=time_array,
        dt=dt_effective,
        vertex_blobs=vertex_blobs,
        blob_radius=blob_radius,
        sphere_radius=sphere_radius,
        domain_bounds=domain_bounds,
        metadata=params,
        periodic_lengths=periodic_length
    )
