'''
Trajectory loader, kinematics engine, and SLERP interpolation for RigidMultiblobsWall.
Supports arbitrary simulations, body shapes, and particle radii without hardcoded parameters.
'''

import os
import re
import math
import numpy as np


def quaternion_to_rot_matrix(q):
    '''
    Convert quaternion [q0, q1, q2, q3] to 3x3 rotation matrix.
    q0 is scalar part (s), (q1, q2, q3) is vector part (p).
    '''
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    norm = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
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
    norms[norms == 0] = 1.0
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
                 blob_radius=0.25, sphere_radius=1.0, domain_bounds=None, metadata=None):
        self.positions = positions           # (num_frames, num_bodies, 3)
        self.quaternions = quaternions       # (num_frames, num_bodies, 4)
        self.time_array = time_array         # (num_frames,)
        self.dt = dt
        self.num_frames, self.num_bodies, _ = positions.shape
        self.vertex_blobs = vertex_blobs
        self.num_blobs_per_body = len(vertex_blobs) if vertex_blobs is not None else 0
        self.blob_radius = blob_radius
        self.sphere_radius = sphere_radius
        self.domain_bounds = domain_bounds
        self.metadata = metadata or {}

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

        self._compute_kinematics()

    def _compute_kinematics(self):
        '''Pre-compute velocities, height statistics, MSD, and in-plane 2D angles.'''
        self.velocities = np.zeros_like(self.positions)
        if self.num_frames > 1:
            dt_step = self.time_array[1] - self.time_array[0] if len(self.time_array) > 1 else self.dt
            if dt_step <= 0:
                dt_step = self.dt
            self.velocities[:-1] = (self.positions[1:] - self.positions[:-1]) / dt_step
            self.velocities[-1] = self.velocities[-2] if self.num_frames > 2 else self.velocities[0]

        self.speeds = np.linalg.norm(self.velocities, axis=-1)  # (num_frames, num_bodies)

        # Height statistics
        z_coords = self.positions[..., 2]
        self.mean_z = np.mean(z_coords, axis=-1)
        self.min_z = np.min(z_coords, axis=-1)
        self.max_z = np.max(z_coords, axis=-1)
        self.std_z = np.std(z_coords, axis=-1)

        # Mean Squared Displacement (MSD)
        disp = self.positions - self.positions[0:1]
        self.msd = np.mean(np.sum(disp**2, axis=-1), axis=-1)

    def get_interpolated_trajectory(self, target_duration=8.0, fps=24):
        '''
        Generate a smoothly interpolated trajectory suitable for higher FPS and longer video playback.
        target_duration: Target video length in seconds (e.g. 8.0s)
        fps: Target frames per second (e.g. 24 fps -> 192 total frames)
        '''
        if self.num_frames <= 1:
            return self

        total_target_frames = max(self.num_frames, int(round(target_duration * fps)))
        if total_target_frames == self.num_frames:
            return self

        # Original frame timestamps normalized in [0, 1]
        orig_indices = np.linspace(0, 1, self.num_frames)
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
            pos0 = self.positions[seg_idx]
            pos1 = self.positions[seg_idx + 1]
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
            metadata=self.metadata
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
        max_dist = float(np.max(np.linalg.norm(vertex_blobs, axis=1)))
        return max_dist + float(blob_radius)
    return float(blob_radius if blob_radius > 0 else fallback)


def parse_input_file(input_file_path):
    '''Parse key-value parameters from a RigidMultiblobsWall input file.'''
    params = {}
    if not input_file_path or not os.path.exists(input_file_path):
        return params

    with open(input_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                val = parts[1] if len(parts) == 2 else ' '.join(parts[1:])
                params[key] = val
    return params


def parse_vertex_file(vertex_file_path):
    '''Parse reference multiblob vertices from a .vertex file.'''
    if not vertex_file_path or not os.path.exists(vertex_file_path):
        return None, 0.0

    blobs = []
    with open(vertex_file_path, 'r') as f:
        first_line = f.readline().strip().split()
        num_blobs = int(first_line[0])
        geom_radius = float(first_line[1]) if len(first_line) > 1 else 1.0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(x) for x in line.split()]
            if len(parts) >= 3:
                blobs.append(parts[:3])
            if len(blobs) >= num_blobs:
                break
    return np.array(blobs, dtype=np.float64), geom_radius


def parse_config_file(config_file_path):
    '''
    Parse all valid frames from a .config simulation output file.
    '''
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    frames_positions = []
    frames_quaternions = []

    with open(config_file_path, 'r') as f:
        lines = f.readlines()

    idx = 0
    total_lines = len(lines)

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

        idx += 1
        if idx + num_bodies > total_lines:
            break

        pos_list = []
        quat_list = []
        frame_valid = True

        for b in range(num_bodies):
            data_line = lines[idx + b].strip()
            parts = [float(x) for x in data_line.split()]
            if len(parts) < 7:
                frame_valid = False
                break
            pos_list.append(parts[0:3])
            quat_list.append(parts[3:7])

        if frame_valid and len(pos_list) == num_bodies:
            frames_positions.append(pos_list)
            frames_quaternions.append(quat_list)
            idx += num_bodies
        else:
            break

    if not frames_positions:
        raise ValueError(f"No valid frames found in config file: {config_file_path}")

    positions = np.array(frames_positions, dtype=np.float64)
    quaternions = np.array(frames_quaternions, dtype=np.float64)
    return positions, quaternions


def load_simulation_data(config_file, input_file=None, vertex_file=None, manual_radius=None):
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

    # Resolve vertex file
    if vertex_file is None and 'structure' in params:
        struct_parts = params['structure'].split()
        if len(struct_parts) >= 1:
            candidate_v = struct_parts[0]
            if os.path.exists(candidate_v):
                vertex_file = candidate_v
            else:
                base_dir = os.path.dirname(input_file) if input_file else ''
                candidate_rel = os.path.join(base_dir, candidate_v)
                if os.path.exists(candidate_rel):
                    vertex_file = candidate_rel

    vertex_blobs, geom_radius = parse_vertex_file(vertex_file) if vertex_file else (None, 0.0)

    # Dynamic radius calculation
    if manual_radius is not None and manual_radius > 0:
        sphere_radius = float(manual_radius)
    elif geom_radius > 0:
        sphere_radius = geom_radius
    else:
        sphere_radius = compute_effective_radius(vertex_blobs, blob_radius, fallback=1.0)

    # Build time array
    dt_effective = dt * n_save
    time_array = np.arange(num_frames) * dt_effective

    # Determine domain bounds
    domain_bounds = None
    if 'periodic_length' in params:
        p_len = [float(x) for x in params['periodic_length'].split()]
        domain_bounds = [0.0, p_len[0], 0.0, p_len[1], 0.0, max(np.max(positions[..., 2]) + 3.0, sphere_radius * 4.0)]
    elif 'box' in params:
        b_parts = [float(x) for x in params['box'].split()]
        if len(b_parts) == 2:
            domain_bounds = [0.0, b_parts[0], 0.0, b_parts[1], 0.0, max(np.max(positions[..., 2]) + 3.0, sphere_radius * 4.0)]
        elif len(b_parts) >= 4:
            domain_bounds = [b_parts[0], b_parts[1], b_parts[2], b_parts[3], 0.0, max(np.max(positions[..., 2]) + 3.0, sphere_radius * 4.0)]

    return SimulationTrajectory(
        positions=positions,
        quaternions=quaternions,
        time_array=time_array,
        dt=dt_effective,
        vertex_blobs=vertex_blobs,
        blob_radius=blob_radius,
        sphere_radius=sphere_radius,
        domain_bounds=domain_bounds,
        metadata=params
    )
