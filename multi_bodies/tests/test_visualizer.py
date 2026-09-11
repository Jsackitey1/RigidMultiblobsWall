'''
Automated test suite for RigidMultiblobsWall 2D visualizer.
'''

import os
import sys
import unittest
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
mb_dir = os.path.abspath(os.path.join(test_dir, '..'))
if mb_dir not in sys.path:
    sys.path.insert(0, mb_dir)

from visualizer.trajectory_loader import (
    quaternion_to_rot_matrix,
    batch_quaternion_to_rot_matrices,
    slerp,
    batch_slerp,
    compute_effective_radius,
    load_simulation_data,
    parse_config_file,
    parse_vertex_file
)
from visualizer.render_2d import (
    render_2d_topdown_frame,
    render_2d_side_elevation_frame,
    render_2d_dual_frame
)
from visualizer.render_graphs import create_graph_dashboard_frame
from visualizer.export_html import export_interactive_html


class TestVisualizer2D(unittest.TestCase):

    def setUp(self):
        self.config_path = os.path.join(mb_dir, 'data', 'run.generated_spheres.config')
        self.input_path = os.path.join(mb_dir, 'inputfile_dynamic.dat')
        self.vertex_path = os.path.join(mb_dir, 'Structures', 'shell_N_12_Rg_1.vertex')

    def test_slerp_quaternion_interpolation(self):
        '''Test spherical linear interpolation between quaternions.'''
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        # 90 degree rotation about z
        q1 = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])

        # At alpha=0 -> q0
        np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-7)
        # At alpha=1 -> q1
        np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-7)

        # Midpoint alpha=0.5 -> 45 degree rotation about z
        q_mid = slerp(q0, q1, 0.5)
        expected_mid = np.array([np.cos(np.pi/8), 0.0, 0.0, np.sin(np.pi/8)])
        np.testing.assert_allclose(q_mid, expected_mid, atol=1e-7)

        # Batch slerp
        b_q0 = np.array([q0, q0])
        b_q1 = np.array([q1, q1])
        b_mid = batch_slerp(b_q0, b_q1, 0.5)
        self.assertEqual(b_mid.shape, (2, 4))
        np.testing.assert_allclose(b_mid[0], expected_mid, atol=1e-7)

    def test_dynamic_radius_calculation(self):
        '''Test dynamic radius resolution for arbitrary multiblob geometries.'''
        # 12-blob unit sphere with blob radius 0.25
        blobs, geom_r = parse_vertex_file(self.vertex_path) if os.path.exists(self.vertex_path) else (None, 1.0)
        eff_r = compute_effective_radius(blobs, blob_radius=0.25, fallback=1.0)
        self.assertGreater(eff_r, 0.5)

        # Single blob with radius 0.5
        eff_single = compute_effective_radius(None, blob_radius=0.5)
        self.assertAlmostEqual(eff_single, 0.5)

    def test_trajectory_interpolation_for_longer_videos(self):
        '''Test expanding a short trajectory into a smooth high-frame-count trajectory.'''
        if os.path.exists(self.config_path):
            raw_traj = load_simulation_data(self.config_path, input_file=self.input_path)
            # Interpolate to 8.0 seconds at 24 fps (192 frames)
            interp_traj = raw_traj.get_interpolated_trajectory(target_duration=8.0, fps=24)
            self.assertEqual(interp_traj.num_frames, 192)
            self.assertEqual(interp_traj.num_bodies, raw_traj.num_bodies)
            self.assertAlmostEqual(interp_traj.time_array[0], raw_traj.time_array[0])
            self.assertAlmostEqual(interp_traj.time_array[-1], raw_traj.time_array[-1])

    def test_2d_frame_renderers(self):
        '''Test rendering 2D frames without error.'''
        if os.path.exists(self.config_path):
            traj = load_simulation_data(self.config_path, input_file=self.input_path)
            
            # Top-down frame
            f_top = render_2d_topdown_frame(traj, 0, mode='spheres')
            self.assertEqual(f_top.ndim, 3)
            self.assertEqual(f_top.shape[2], 3)

            # Side-elevation frame
            f_side = render_2d_side_elevation_frame(traj, 0)
            self.assertEqual(f_side.ndim, 3)

            # Dual frame
            f_dual = render_2d_dual_frame(traj, 0)
            self.assertEqual(f_dual.ndim, 3)

    def test_2d_html_export(self):
        '''Test interactive 2D HTML player generation.'''
        if os.path.exists(self.config_path):
            traj = load_simulation_data(self.config_path, input_file=self.input_path)
            out_html = '/tmp/test_2d_player.html'
            export_interactive_html(traj, out_html)
            self.assertTrue(os.path.exists(out_html))
            self.assertGreater(os.path.getsize(out_html), 1000)
            os.remove(out_html)


if __name__ == '__main__':
    unittest.main()
