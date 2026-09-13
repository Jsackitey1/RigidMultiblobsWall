"""Regression checks for IC geometry and representation, not solver dynamics."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'multi_bodies'))
import generate_sphere_suspension as generator
from geometry import geometry_report, periodic_copies
from visualizer.trajectory_loader import SimulationTrajectory, load_simulation_data, quaternion_to_rot_matrix, batch_quaternion_to_rot_matrices
from visualizer.diagnostics import write_diagnostics
from visualizer.render_2d import render_2d_frame
from quaternion_integrator.quaternion import Quaternion


def trajectory(xs, times=None, lengths=(20, 0)):
    p = np.zeros((len(xs), 1, 3)); p[:, 0, 0] = xs; p[:, 0, 2] = 2
    q = np.zeros((len(xs), 1, 4)); q[..., 0] = 1
    return SimulationTrajectory(p, q, np.array(times if times is not None else np.arange(len(xs))), 1,
             vertex_blobs=np.zeros((1, 3)), blob_radius=.25, periodic_lengths=lengths,
             domain_bounds=[0, 20, 0, 20, 0, 4])


class TrajectoryPhysics(unittest.TestCase):
    def test_periodic_translation_and_interpolation(self):
        t = trajectory([19.9, 20.1])
        np.testing.assert_allclose(t.positions[:, 0, 0], [19.9, .1])
        np.testing.assert_allclose(t.displacement_rate[:, 0, 0], .2)
        self.assertAlmostEqual(t.msd[-1], .04)
        animation = t.get_interpolated_trajectory(3, 1)
        self.assertAlmostEqual(animation.positions_unwrapped[1, 0, 0], 20)
        self.assertAlmostEqual(animation.positions[1, 0, 0], 0)
        self.assertIs(animation.analysis_source, t)
        self.assertIsNone(animation.msd)
        self.assertTrue(animation.metadata['interpolated_for_visualization'])
        self.assertNotIn('interpolated_for_visualization', t.metadata)

    def test_default_presentation_timing(self):
        raw = trajectory([0, 1, 4])
        animation = raw.get_interpolated_trajectory()
        self.assertEqual(animation.num_frames, 600)
        self.assertEqual(animation.num_frames / 30, 20)
        np.testing.assert_allclose(animation.positions_unwrapped[[0,-1]], raw.positions_unwrapped[[0,-1]])

    def test_dense_output_resampled_to_requested_duration(self):
        raw = trajectory(np.linspace(0, 10, 1001), np.linspace(0, 1, 1001))
        animation = raw.get_interpolated_trajectory()
        self.assertEqual(animation.num_frames, 600)
        np.testing.assert_allclose(animation.positions_unwrapped[:,0,0], 10*animation.time_array)
        self.assertEqual(raw.num_frames, 1001)
        self.assertAlmostEqual(raw.msd[-1], 100)

    def test_irregular_times(self):
        t = trajectory([0, 2, 6], [0, 1, 3])
        np.testing.assert_allclose(t.displacement_rate[:, 0, 0], 2)
        a = t.get_interpolated_trajectory(7, 1)
        np.testing.assert_allclose(a.positions_unwrapped[:, 0, 0], 2*a.time_array)

    def test_large_displacement_preserved(self):
        t = trajectory([0, 45])
        self.assertEqual(t.displacement_rate[0, 0, 0], 45)
        self.assertEqual(t.msd[-1], 2025)

    def test_invalid_time_and_quaternion(self):
        for ts in ([0, 0], [1, 0], [0, float('nan')]):
            with self.assertRaises(ValueError):
                trajectory([0, 1], ts)
        with self.assertRaises(ValueError):
            quaternion_to_rot_matrix([0, 0, 0, 0])
        with self.assertRaises(ValueError):
            batch_quaternion_to_rot_matrices(np.zeros((1, 4)))

    def test_loader_single_axis_restart(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'in').write_text('dt 0.5\nn_save 2\ninitial_step 3\nperiodic_length 20 0 0\n')
            (p/'config').write_text('1\n19.9 -5 2 1 0 0 0\n1\n20.1 -4 2 1 0 0 0\n')
            t = load_simulation_data(p/'config', p/'in')
            np.testing.assert_allclose(t.time_array, [2, 3])
            np.testing.assert_allclose(t.positions[:, 0, 1], [-5, -4])
            self.assertAlmostEqual(t.displacement_rate[0, 0, 0], .2)

    def test_wrapped_input_requires_explicit_mode(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'in').write_text('dt 1\nperiodic_length 20 0 0\n')
            (p/'config').write_text('1\n19.9 0 2 1 0 0 0\n1\n0.1 0 2 1 0 0 0\n')
            with self.assertWarns(RuntimeWarning):
                t = load_simulation_data(p/'config', p/'in', coordinates='wrapped')
            self.assertAlmostEqual(t.msd[-1], .04)

    def test_structure_selection(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'small.vertex').write_text('1\n0 0 0\n')
            (p/'large.vertex').write_text('1\n2 0 0\n')
            (p/'in').write_text('structure small.vertex small.clones\nstructure large.vertex large.clones\n')
            (p/'config').write_text('1\n0 0 3 1 0 0 0\n')
            with self.assertRaisesRegex(ValueError, 'Multiple structures'):
                load_simulation_data(p/'config', p/'in')
            selected = load_simulation_data(p/'config', p/'in', structure_index=1)
            self.assertAlmostEqual(selected.bounding_radius, 2.25)
            with self.assertRaises(ValueError):
                load_simulation_data(p/'config', p/'in', structure_index=2)

    def test_raw_report_independent_of_animation(self):
        t = trajectory([0, 1, 4])
        with tempfile.TemporaryDirectory() as d:
            first = json.loads(Path(write_diagnostics(t, d)[0]).read_text())
            second = json.loads(Path(write_diagnostics(t.get_interpolated_trajectory(10, 10), d)[0]).read_text())
            self.assertEqual(first, second)

    def test_rotation_matches_solver_and_is_rigid(self):
        rng = np.random.default_rng(32)
        vertices = rng.normal(size=(10, 3))
        for q in rng.normal(size=(40, 4)):
            q /= np.linalg.norm(q)
            rot = quaternion_to_rot_matrix(q)
            np.testing.assert_allclose(rot, Quaternion(q).rotation_matrix(), atol=1e-14)
            np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rot), 1)
            np.testing.assert_allclose(np.linalg.norm(vertices-vertices[0], axis=1),
                                       np.linalg.norm((vertices-vertices[0]) @ rot.T, axis=1), atol=1e-14)


class GeneratorPhysics(unittest.TestCase):
    def test_failed_relaxation_rejected(self):
        with patch.object(generator, 'place_spheres_rsa', return_value=np.array([[1., 1.]])), \
             patch.object(generator, 'relax_dense_suspension', return_value=np.ones((2, 2))):
            with self.assertRaises(RuntimeError):
                generator.generate_sphere_suspension_2d([0, 10, 0, 10], target_count=2)

    def test_geometry_placement_and_seed(self):
        vertices = np.array([[1, 0, 0], [-1, 0, 0]])
        kwargs = dict(bounds=[0, 20, 0, 20], target_count=12, seed=42, vertex_blobs=vertices,
                      blob_radius=.25, geometry_mode='bounding-sphere', periodic=True)
        first = generator.generate_sphere_suspension_2d(**kwargs)
        second = generator.generate_sphere_suspension_2d(**kwargs)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertAlmostEqual(first[4], 2.625)
        self.assertGreaterEqual(first[3], first[4]-1e-7)
        report = geometry_report(first[0], first[1], vertices, .25, [20, 20])
        self.assertEqual(report['overlapping_interbody_blob_pairs'], 0)
        self.assertAlmostEqual(first[2], 12*np.pi/400)

    def test_invalid_inputs(self):
        for kwargs in (dict(radius=0), dict(target_count=0), dict(target_count=-2),
                       dict(safety_gap=-1), dict(bounds=[0, 1, 0, 1]), dict(z_fixed=.5),
                       dict(target_count=None, target_fraction=0), dict(target_fraction=.2)):
            base = dict(bounds=[0, 10, 0, 10], target_count=2)
            base.update(kwargs)
            with self.assertRaises(ValueError):
                generator.generate_sphere_suspension_2d(**base)

    def test_impossible_periodic_packing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'hexagonal'):
            generator.generate_sphere_suspension_2d([0, 20, 0, 20], target_fraction=.95,
                                                  periodic=True, safety_gap=0)

    def test_self_image_rejected(self):
        with self.assertRaises(ValueError):
            generator.generate_sphere_suspension_2d([0, 1.5, 0, 10], target_count=1, periodic=True)

    def test_rotated_wall_geometry(self):
        vertices = np.array([[1, 0, 0], [-1, 0, 0]])
        pos = np.array([[0., 0., .5]])
        identity = geometry_report(pos, np.array([[1, 0, 0, 0]]), vertices, .25)
        rotated = geometry_report(pos, np.array([[2**-.5, 0, 2**-.5, 0]]), vertices, .25)
        self.assertEqual(identity['bodies_with_wall_penetration'], 0)
        self.assertEqual(rotated['bodies_with_wall_penetration'], 1)
        self.assertAlmostEqual(rotated['minimum_wall_clearance'], -.75)
        self.assertIsNone(geometry_report(pos, np.array([[1, 0, 0, 0]]), vertices, .25, wall=False)['minimum_wall_clearance'])

    def test_periodic_blob_overlap(self):
        r = geometry_report(np.array([[.1, 0, 2], [9.9, 0, 2]]), np.array([[1,0,0,0]]*2),
                            np.zeros((1,3)), .25, [10,0])
        self.assertAlmostEqual(r['minimum_interbody_blob_clearance'], -.3)

    def test_cli_failure_preserves_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'out.clones'; out.write_text('existing')
            argv = ['generator', '--box', '10', '10', '-N', '2', '--geometry-mode', 'ideal-sphere', '--output-clones', str(out)]
            with patch.object(sys, 'argv', argv), patch.object(generator, 'generate_sphere_suspension_2d', side_effect=RuntimeError('failed')):
                with self.assertRaises(RuntimeError):
                    generator.main()
            self.assertEqual(out.read_text(), 'existing')


class RenderingPhysics(unittest.TestCase):
    def test_single_frame_video_duration(self):
        from visualizer.render_2d import render_2d_video
        with tempfile.TemporaryDirectory() as d, \
             patch('visualizer.render_2d.VideoStreamWriter') as writer, \
             patch('visualizer.render_2d.render_2d_frame', return_value=np.zeros((2,2,3), dtype=np.uint8)):
            render_2d_video(trajectory([0]), os.path.join(d, 'static.mp4'), fps=30,
                            target_duration=20, show_progress=False)
            self.assertEqual(writer.return_value.__enter__.return_value.write_frame.call_count, 600)

    def test_corner_copies(self):
        copies = periodic_copies([.1,.1,2], 1, [20,20], [0,20,0,20])
        self.assertEqual(len(copies), 4)
        self.assertTrue(any(np.array_equal(c, [20,20,0]) for c in copies))

    def test_periodic_render_trail_split(self):
        import matplotlib.pyplot as plt
        t = trajectory([19.9,20.1])
        fig = plt.figure()
        render_2d_frame(t, 1, fig=fig)
        trail = fig.axes[0].lines[0]
        self.assertTrue(np.isnan(trail.get_xdata()).any())
        self.assertEqual(fig.axes[0].get_xlim(), (0,20))
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
