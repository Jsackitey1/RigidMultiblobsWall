'''
Automated test suite for RigidMultiblobsWall 2D visualizer.
Tests cover all major bugs identified in the code audit.
'''

import os
import sys
import tempfile
import unittest
import warnings
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
    parse_vertex_file,
    parse_input_file
)
from visualizer.render_2d import (
    render_2d_frame,
    render_2d_video
)
from visualizer.export_html import export_interactive_html


class TestQuaternionMath(unittest.TestCase):

    def test_identity_quaternion(self):
        '''Identity quaternion [1,0,0,0] should produce identity rotation matrix.'''
        R = quaternion_to_rot_matrix([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_batch_matches_scalar(self):
        '''batch_quaternion_to_rot_matrices should match quaternion_to_rot_matrix.'''
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R_scalar = quaternion_to_rot_matrix(q)
        R_batch = batch_quaternion_to_rot_matrices(q.reshape(1, 4))[0]
        np.testing.assert_allclose(R_scalar, R_batch, atol=1e-12)


class TestSLERP(unittest.TestCase):

    def test_slerp_endpoints(self):
        '''SLERP at alpha=0 and alpha=1 should return the input quaternions.'''
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        np.testing.assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-7)
        np.testing.assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-7)

    def test_slerp_midpoint(self):
        '''SLERP midpoint for 90-degree z rotation should be 45-degree rotation.'''
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        q_mid = slerp(q0, q1, 0.5)
        expected_mid = np.array([np.cos(np.pi/8), 0.0, 0.0, np.sin(np.pi/8)])
        np.testing.assert_allclose(q_mid, expected_mid, atol=1e-7)

    def test_batch_slerp_matches_scalar(self):
        '''Batch SLERP should match scalar SLERP.'''
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)])
        expected = slerp(q0, q1, 0.5)
        b_result = batch_slerp(q0.reshape(1, 4), q1.reshape(1, 4), 0.5)
        np.testing.assert_allclose(b_result[0], expected, atol=1e-7)


class TestBug1_RadiusCalculation(unittest.TestCase):
    '''Bug 1: Bodies were drawn at 42% of true size because parse_vertex_file
    read the second header token as a radius. Now it should use compute_effective_radius.'''

    vertex_path = os.path.join(mb_dir, 'Structures', 'shell_N_12_Rg_1.vertex')

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'Structures', 'shell_N_12_Rg_1.vertex')),
        'shell_N_12_Rg_1.vertex not found'
    )
    def test_radius_uses_effective_not_geom(self):
        '''sphere_radius should be ~1.25 (max|r|+a), not ~0.526 (old geom_radius).'''
        blobs = parse_vertex_file(self.vertex_path)
        self.assertIsNotNone(blobs)
        eff_r = compute_effective_radius(blobs, blob_radius=0.25)
        # Effective radius for shell_N_12_Rg_1 with a=0.25 is about 1.25
        self.assertGreater(eff_r, 1.0, f'Effective radius {eff_r} should be > 1.0')
        self.assertAlmostEqual(eff_r, 1.25, places=1)
        # Must NOT be the old bogus geom_radius value
        self.assertNotAlmostEqual(eff_r, 0.526, places=1,
                                  msg='Radius is still using the vertex header token, not compute_effective_radius')

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'Structures', 'shell_N_12_Rg_1.vertex')),
        'shell_N_12_Rg_1.vertex not found'
    )
    def test_parse_vertex_returns_no_geom_radius(self):
        '''parse_vertex_file should return only blobs array, not a (blobs, radius) tuple.'''
        result = parse_vertex_file(self.vertex_path)
        # Should be an ndarray, NOT a tuple
        self.assertIsInstance(result, np.ndarray,
                             'parse_vertex_file should return ndarray, not tuple with geom_radius')

    def test_single_blob_effective_radius(self):
        '''Single blob with radius 0.5 should have effective radius 0.5.'''
        eff = compute_effective_radius(None, blob_radius=0.5)
        self.assertAlmostEqual(eff, 0.5)

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'data', 'run.generated_spheres.config'))
        and os.path.exists(os.path.join(mb_dir, 'inputfile_dynamic.dat')),
        'Simulation data not found'
    )
    def test_loaded_trajectory_uses_effective_radius(self):
        '''Full load_simulation_data should use effective radius, not vertex header token.'''
        config_path = os.path.join(mb_dir, 'data', 'run.generated_spheres.config')
        input_path = os.path.join(mb_dir, 'inputfile_dynamic.dat')
        traj = load_simulation_data(config_path, input_file=input_path)
        self.assertGreater(traj.sphere_radius, 1.0,
                           f'traj.sphere_radius = {traj.sphere_radius}, expected > 1.0')


class TestBug2_CommentHeaderVertex(unittest.TestCase):
    '''Bug 2: parse_vertex_file crashed on .vertex files with # comment headers.'''

    boomerang_path = os.path.join(mb_dir, 'Structures', 'boomerang_N_15.vertex')
    rod_path = os.path.join(mb_dir, 'Structures', 'rod_Lg_1.845_Rg_0.1308_Nx_16_Ntheta_6.vertex')

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'Structures', 'boomerang_N_15.vertex')),
        'boomerang_N_15.vertex not found'
    )
    def test_boomerang_vertex_parses(self):
        '''boomerang_N_15.vertex has a comment header and must not crash.'''
        blobs = parse_vertex_file(self.boomerang_path)
        self.assertIsNotNone(blobs)
        self.assertEqual(blobs.shape[1], 3)
        self.assertEqual(blobs.shape[0], 15, 'Expected 15 blobs in boomerang')

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'Structures', 'rod_Lg_1.845_Rg_0.1308_Nx_16_Ntheta_6.vertex')),
        'rod vertex file not found'
    )
    def test_rod_vertex_parses(self):
        '''rod vertex file has a comment header and must not crash.'''
        blobs = parse_vertex_file(self.rod_path)
        self.assertIsNotNone(blobs)
        self.assertEqual(blobs.shape[1], 3)

    def test_comment_only_vertex(self):
        '''A vertex file that is all comments should return None.'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vertex', delete=False) as f:
            f.write('# this is only comments\n')
            f.write('# no data\n')
            tmp_path = f.name
        try:
            result = parse_vertex_file(tmp_path)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp_path)


class TestBug3_InlineComments(unittest.TestCase):
    '''Bug 3: parse_input_file choked on inline # comments like "dt 0.01 # timestep".'''

    def test_inline_comment_stripped(self):
        '''Input file with inline # comments should parse correctly.'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write('# Full-line comment\n')
            f.write('dt 0.01 # timestep\n')
            f.write('n_steps 100 # number of steps\n')
            f.write('blob_radius 0.25\n')
            f.write('\n')
            tmp_path = f.name
        try:
            params = parse_input_file(tmp_path)
            self.assertEqual(params['dt'], '0.01',
                             f"Expected '0.01' but got '{params.get('dt')}'")
            self.assertEqual(params['n_steps'], '100')
            self.assertEqual(params['blob_radius'], '0.25')
            # Verify we can actually convert to float (the original bug)
            self.assertAlmostEqual(float(params['dt']), 0.01)
        finally:
            os.unlink(tmp_path)


class TestBug9_MultiStructure(unittest.TestCase):
    '''Bug 9: parse_input_file overwrote 'structure' key — should index as structure0, structure1.'''

    def test_multi_structure_indexing(self):
        '''Two structure lines should be retained as structure0 and structure1.'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write('dt 0.1\n')
            f.write('structure vertex_a.vertex clones_a.clones\n')
            f.write('structure vertex_b.vertex clones_b.clones\n')
            tmp_path = f.name
        try:
            params = parse_input_file(tmp_path)
            self.assertIn('structure0', params, 'First structure missing')
            self.assertIn('structure1', params, 'Second structure missing')
            self.assertNotIn('structure', params, 'Raw "structure" key should not exist')
            self.assertIn('vertex_a.vertex', params['structure0'])
            self.assertIn('vertex_b.vertex', params['structure1'])
        finally:
            os.unlink(tmp_path)


class TestBug10_TruncatedConfig(unittest.TestCase):
    '''Bug 10: parse_config_file silently truncated on malformed frames.'''

    def test_truncated_frame_warns(self):
        '''A config with an incomplete final frame should issue a warning.'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as f:
            # Frame 0: valid (2 bodies)
            f.write('2\n')
            f.write('1.0 2.0 3.0 1.0 0.0 0.0 0.0\n')
            f.write('4.0 5.0 6.0 1.0 0.0 0.0 0.0\n')
            # Frame 1: declares 2 bodies but only has 1 line
            f.write('2\n')
            f.write('7.0 8.0 9.0 1.0 0.0 0.0 0.0\n')
            tmp_path = f.name
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                pos, quats = parse_config_file(tmp_path)
                # Should have exactly 1 valid frame
                self.assertEqual(pos.shape[0], 1, 'Should keep only the 1 complete frame')
                # Should have warned about truncation
                self.assertTrue(len(w) > 0, 'Expected a warning about truncation')
                self.assertTrue(any('truncated' in str(warning.message).lower() for warning in w))
        finally:
            os.unlink(tmp_path)

    def test_non_numeric_data_warns(self):
        '''A config with non-numeric data should warn, not crash with uncaught ValueError.'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.config', delete=False) as f:
            # Frame 0: valid
            f.write('1\n')
            f.write('1.0 2.0 3.0 1.0 0.0 0.0 0.0\n')
            # Frame 1: garbage data
            f.write('1\n')
            f.write('bad data here\n')
            tmp_path = f.name
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                pos, quats = parse_config_file(tmp_path)
                self.assertEqual(pos.shape[0], 1, 'Should keep only the 1 complete frame')
                self.assertTrue(len(w) > 0, 'Expected a warning about non-numeric data')
        finally:
            os.unlink(tmp_path)


class TestBug6_DepthSorting(unittest.TestCase):
    '''Bug 6: Bodies should be painted in ascending z order (painter's algorithm).'''

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'data', 'run.generated_spheres.config'))
        and os.path.exists(os.path.join(mb_dir, 'inputfile_dynamic.dat')),
        'Simulation data not found'
    )
    def test_2d_frame_renders_without_error(self):
        '''Depth-sorted rendering should produce a valid RGB frame.'''
        config_path = os.path.join(mb_dir, 'data', 'run.generated_spheres.config')
        input_path = os.path.join(mb_dir, 'inputfile_dynamic.dat')
        traj = load_simulation_data(config_path, input_file=input_path)
        frame = render_2d_frame(traj, 0, mode='spheres')
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)

        frame_blobs = render_2d_frame(traj, 0, mode='blobs')
        self.assertEqual(frame_blobs.ndim, 3)
        self.assertEqual(frame_blobs.shape[2], 3)


class TestTrajectoryInterpolation(unittest.TestCase):
    '''Test SLERP-based trajectory interpolation.'''

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'data', 'run.generated_spheres.config'))
        and os.path.exists(os.path.join(mb_dir, 'inputfile_dynamic.dat')),
        'Simulation data not found'
    )
    def test_interpolation_frame_count_and_time(self):
        '''Interpolated trajectory should have the correct number of frames and preserve time range.'''
        config_path = os.path.join(mb_dir, 'data', 'run.generated_spheres.config')
        input_path = os.path.join(mb_dir, 'inputfile_dynamic.dat')
        raw_traj = load_simulation_data(config_path, input_file=input_path)
        interp_traj = raw_traj.get_interpolated_trajectory(target_duration=8.0, fps=24)
        self.assertEqual(interp_traj.num_frames, 192)
        self.assertEqual(interp_traj.num_bodies, raw_traj.num_bodies)
        self.assertAlmostEqual(interp_traj.time_array[0], raw_traj.time_array[0])
        self.assertAlmostEqual(interp_traj.time_array[-1], raw_traj.time_array[-1])


class TestHTMLExport(unittest.TestCase):
    '''Test HTML player export.'''

    @unittest.skipUnless(
        os.path.exists(os.path.join(mb_dir, 'data', 'run.generated_spheres.config'))
        and os.path.exists(os.path.join(mb_dir, 'inputfile_dynamic.dat')),
        'Simulation data not found'
    )
    def test_html_export_contains_viridis_lut(self):
        '''Exported HTML should contain the viridis LUT, not the polynomial approximation.'''
        config_path = os.path.join(mb_dir, 'data', 'run.generated_spheres.config')
        input_path = os.path.join(mb_dir, 'inputfile_dynamic.dat')
        traj = load_simulation_data(config_path, input_file=input_path)

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            tmp_html = f.name
        try:
            export_interactive_html(traj, tmp_html)
            self.assertTrue(os.path.exists(tmp_html))
            with open(tmp_html, 'r') as f:
                html_content = f.read()
            self.assertGreater(len(html_content), 1000)
            # Bug 5: Should contain viridis LUT, not polynomial
            self.assertIn('viridis_lut', html_content)
            # Bug 4: Should have dynamic z_min/z_max
            self.assertIn('z_min', html_content)
            self.assertIn('z_max', html_content)
            # Minor: No LaTeX in HTML
            self.assertNotIn('$\\langle', html_content)
            # Minor: No dead velocities payload
            self.assertNotIn('"velocities"', html_content)
        finally:
            if os.path.exists(tmp_html):
                os.unlink(tmp_html)


if __name__ == '__main__':
    unittest.main()
