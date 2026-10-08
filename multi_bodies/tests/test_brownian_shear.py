"""Validate combined drift/noise through the existing dense RFD integrator."""
import unittest
from functools import partial
from types import SimpleNamespace
import numpy as np
from test_passive_shear import solver, ROOT, Body, Quaternion


class BrownianShear(unittest.TestCase):
    def make_integrator(self, rate, kT):
        vertices = np.loadtxt(ROOT/'multi_bodies/Structures/shell_N_12_Rg_1.vertex', skiprows=1)
        body = Body([0., 0., 4.], Quaternion([1, 0, 0, 0]), vertices, .25)
        body.calc_body_length()
        it = solver.QuaternionIntegrator([body], body.Nblobs, 'stochastic_first_order_RFD_dense_algebra')
        it.eta, it.a, it.kT, it.rf_delta = 1., .25, kT, 1e-6
        it.get_blobs_r_vectors = solver.get_blobs_r_vectors
        it.mobility_blobs = solver.mb.single_wall_fluid_mobility
        it.calc_K_matrix = solver.calc_K_matrix
        it.force_torque_calculator = lambda *args: np.zeros((2, 3))
        it.calc_slip = partial(solver.calc_slip, shear_rate=rate, g=0., blob_radius=.25, domain='single_wall')
        it.preprocess = it.postprocess = lambda bodies: None
        return it, body

    def test_same_noise_adds_correct_flow_translation_and_rotation(self):
        for kT in (0., 1.):
            increments = []
            for rate in (0., .5):
                it, b = self.make_integrator(rate, kT)
                np.random.seed(71)
                it.advance_time_step(.001)
                q = b.orientation
                angle = 2*np.arctan2(np.linalg.norm(q.p), q.s)
                rot = q.p * angle / np.linalg.norm(q.p) if np.linalg.norm(q.p) else np.zeros(3)
                increments.append(np.r_[b.location-[0,0,4], rot])
            it, b = self.make_integrator(.5, 0.)
            r = b.get_r_vectors(); M = it.mobility_blobs(r, 1., .25); K = b.calc_K_matrix()
            A = np.block([[M, -K], [-K.T, np.zeros((6,6))]])
            expected = np.linalg.solve(A, np.r_[it.calc_slip([b], b.Nblobs).ravel(), np.zeros(6)])[-6:]
            np.testing.assert_allclose(increments[1]-increments[0], .001*expected, atol=1e-11)
            self.assertGreater(expected[0], 0)
            self.assertGreater(expected[4], 0)

    def test_brownian_translation_and_rotation_covariance(self):
        it, b = self.make_integrator(.5, 1.)
        _, N = it.solve_mobility_problem_dense_algebra()
        np.random.seed(123)
        increments = []
        dt = 1e-4
        for _ in range(1500):
            b.location[:] = [0,0,4]
            b.orientation = Quaternion([1,0,0,0])
            it.advance_time_step(dt)
            q = b.orientation
            angle = 2*np.arctan2(np.linalg.norm(q.p), q.s)
            increments.append(np.r_[b.location-[0,0,4], q.p*angle/np.linalg.norm(q.p)])
        cov = np.cov(np.asarray(increments), rowvar=False)
        target = 2*dt*N
        np.testing.assert_allclose(np.diag(cov), np.diag(target), rtol=.15)
        scale = np.sqrt(np.outer(np.diag(target), np.diag(target)))
        self.assertLess(np.max(np.abs((cov-target)/scale)), .15)

    def test_configuration_limits(self):
        config = SimpleNamespace(shear_rate=.5, scheme='stochastic_first_order_RFD_dense_algebra', domain='single_wall', periodic_length=[0,0,0], mobility_blobs_implementation='python', mobility_vector_prod_implementation='numba')
        solver.validate_shear_configuration(config)
        config.periodic_length = [0,16,0]
        with self.assertRaisesRegex(ValueError, 'nonperiodic'):
            solver.validate_shear_configuration(config)
        config.periodic_length = [0,0,0]
        config.scheme = 'stochastic_EM'
        with self.assertRaises(ValueError):
            solver.validate_shear_configuration(config)
