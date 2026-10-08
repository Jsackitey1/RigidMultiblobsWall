"""Shear RHS, force-free rigid response, reservoir geometry and transport checks."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'multi_bodies'), str(ROOT)]
import generate_sphere_suspension as generator
from body.body import Body
from quaternion_integrator.quaternion import Quaternion
from visualizer.trajectory_loader import SimulationTrajectory
from visualizer.diagnostics import write_diagnostics
spec = importlib.util.spec_from_file_location('shear_solver', ROOT/'multi_bodies/multi_bodies.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

class PassiveShear(unittest.TestCase):
    def test_force_free_response_and_sign(self):
        vertices = np.loadtxt(ROOT/'multi_bodies/Structures/shell_N_12_Rg_1.vertex', skiprows=1)
        body = Body([0, 0, 100], Quaternion([1, 0, 0, 0]), vertices, .25)
        r = body.get_r_vectors()
        M = solver.mb.single_wall_fluid_mobility(r, 1., .25)
        K = body.calc_K_matrix()
        A = np.block([[M, -K], [-K.T, np.zeros((6, 6))]])
        velocities = []
        for rate in (0., .5, -.5):
            slip = solver.calc_slip([body], body.Nblobs, shear_rate=rate, domain='single_wall', g=0., blob_radius=.25)
            rhs = np.r_[slip.ravel(), np.zeros(6)]
            sol = np.linalg.solve(A, rhs)
            np.testing.assert_allclose(A@sol, rhs, atol=1e-10)
            np.testing.assert_allclose(K.T@sol[:-6], 0, atol=1e-10)
            velocities.append(sol[-6:])
        np.testing.assert_allclose(velocities[0], 0, atol=1e-12)
        np.testing.assert_allclose(velocities[1], -velocities[2], atol=1e-10)
        self.assertAlmostEqual(velocities[1][0]/50., 1., places=4)
        self.assertAlmostEqual(velocities[1][4]/.25, 1., places=3)
        # Shear is evaluated across the sphere, not only at its center.
        self.assertGreater(np.ptp(solver.calc_slip([body], body.Nblobs, shear_rate=.5, g=0., blob_radius=.25)[:, 0]), .1)

    def test_reservoir_cli_metadata_and_invalid_periodicity(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            config = p/'input.dat'
            config.write_text(f'box 0 40 0 20\nreservoir_end 10\nperiodic_length 0 20 0\nnum_bodies 4\nblob_radius .25\nvertex_file {ROOT}/multi_bodies/Structures/shell_N_12_Rg_1.vertex\noutput_clones {p}/reservoir.clones\nplot_image {p}/initial.png\nseed 2\n')
            with patch.object(sys, 'argv', ['generator', '--input-file', str(config)]), patch.object(generator, 'plot_suspension_2d'):
                generator.main()
            report = json.loads((p/'reservoir.clones.validation.json').read_text())
            self.assertEqual(report['initial_downstream_centers'], 0)
            self.assertEqual(report['placement_bounds'], [0, 10, 0, 20])
            self.assertEqual(report['observation_bounds'], [0, 40, 0, 20])
            self.assertAlmostEqual(report['achieved_nominal_area_fraction'], 4*np.pi/200)
            with patch.object(sys, 'argv', ['generator', '--input-file', str(config), '--periodic']):
                with self.assertRaises(ValueError):
                    generator.main()

    def test_crossings_include_recrossing_and_outside_window(self):
        positions = np.zeros((3, 2, 3))
        positions[..., 0] = [[1, 2], [6, 3], [4, 12]]
        positions[..., 2] = 2
        q = np.zeros((3, 2, 4)); q[..., 0] = 1
        t = SimulationTrajectory(positions, q, np.array([0., 1., 3.]), 1.,
            domain_bounds=[0, 10, 0, 4, 0, 4], metadata={'reservoir_end':'5'})
        with tempfile.TemporaryDirectory() as directory:
            report = json.loads(Path(write_diagnostics(t, directory)[0]).read_text())['transport']
        self.assertEqual(report['downstream_count'], [0, 1, 1])
        self.assertEqual(report['forward_crossings'], [1, 1])
        self.assertEqual(report['backward_crossings'], [0, 1])
        self.assertEqual(report['net_flux_per_time'], [1., 0.])
        self.assertEqual(report['outside_x_window'], [0, 0, 1])
        counts = np.asarray(report['number_per_unit_x'])*np.diff(report['density_bin_edges'])
        np.testing.assert_allclose(counts.sum(axis=1)+report['outside_x_window'], 2)
        from visualizer.render_2d import render_2d_frame
        import matplotlib.pyplot as plt
        fig = plt.figure()
        image = render_2d_frame(t, 1, fig=fig, color_by='vx')
        self.assertEqual(image.shape[-1], 3)
        self.assertTrue(any('Downstream centers: 1/2' in text.get_text() for text in fig.axes[0].texts))
        plt.close(fig)

if __name__ == '__main__':
    unittest.main()
