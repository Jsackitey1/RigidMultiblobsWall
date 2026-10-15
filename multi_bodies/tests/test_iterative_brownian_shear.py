"""Compare iterative traction EM with dense reference mobilities."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import test_brownian_shear as dense_tests
from test_passive_shear import solver, Body, Quaternion


def setup(rate=.35, kT=1., backend='numba', count=1):
    it, first = dense_tests.BrownianShear().make_integrator(rate, kT)
    bodies = [first]
    for j in range(1, count):
        b = Body([4.*j, 1., 5.], Quaternion([1,0,0,0]), first.reference_configuration, .25)
        b.calc_body_length()
        bodies.append(b)
    it.bodies, it.Nblobs = bodies, sum(b.Nblobs for b in bodies)
    it.scheme = 'stochastic_traction_EM'
    it.tolerance = 1e-10
    it.constraints, it.articulated = [], []
    it.periodic_length = np.zeros(3)
    it.update_PC, it.print_residual = 1, False
    it.first_guess = np.zeros(3*it.Nblobs+6*count)
    it.force_torque_calculator = lambda *args: np.zeros((2*count,3))
    for b in bodies:
        b.mobility_blobs = it.mobility_blobs
    it.mobility_vector_prod = solver.set_mobility_vector_prod(backend)
    solver.mobility_vector_prod = it.mobility_vector_prod
    it.calc_K_matrix_bodies = solver.calc_K_matrix_bodies
    it.linear_operator = solver.linear_operator_rigid
    it.K_matrix_vector_prod = solver.K_matrix_vector_prod
    it.K_matrix_T_vector_prod = solver.K_matrix_T_vector_prod
    it.build_block_diagonal_preconditioners_det_stoch = solver.build_block_diagonal_preconditioners_det_stoch
    return it


def increment(it, locations):
    result = []
    for b, old in zip(it.bodies, locations):
        q = b.orientation
        norm = np.linalg.norm(q.p)
        rotation = q.p*(2*np.arctan2(norm,q.s)/norm) if norm else np.zeros(3)
        result.extend(np.r_[b.location-old,rotation])
    return np.array(result)


class IterativeBrownianShear(unittest.TestCase):
    backend = 'numba'

    def test_two_body_shear_response(self):
        for thermal in (0., 1.):
            steps=[]
            for rate in (0., .35):
                it=setup(rate,thermal,self.backend,2)
                locations=[b.location.copy() for b in it.bodies]
                np.random.seed(73)
                it.advance_time_step(.001,step=0)
                steps.append(increment(it,locations))
            ref=setup(.35,0.,self.backend,2)
            expected,_=ref.solve_mobility_problem_dense_algebra()
            np.testing.assert_allclose(steps[1]-steps[0],.001*expected,atol=2e-9)

    def test_covariance_against_dense(self):
        it=setup(.35,1.,self.backend,2)
        _,N=it.solve_mobility_problem_dense_algebra()
        initial=[b.location.copy() for b in it.bodies]
        dt=1e-5
        samples=[]
        np.random.seed(321)
        for _ in range(1200):
            for b,x in zip(it.bodies,initial):
                b.location=x.copy(); b.orientation=Quaternion([1,0,0,0])
            it.advance_time_step(dt,step=0)
            samples.append(increment(it,initial))
        target=2*dt*N
        covariance=np.cov(samples,rowvar=False)
        scale=np.sqrt(np.outer(np.diag(target),np.diag(target)))
        self.assertLess(np.max(np.abs((covariance-target)/scale)),.16)
        np.testing.assert_allclose(np.diag(covariance),np.diag(target),rtol=.16)

    def test_thermal_drift_against_mobility_derivative(self):
        # Symmetric deterministic quadrature of RFD noises has E[WW^T]=I.
        it=setup(0.,1.,self.backend)
        b=it.bodies[0]; initial=b.location.copy()
        h=1e-4; target=np.zeros(6)
        for j in range(6):
            cols=[]
            for sign in (-1,1):
                b.location=initial.copy(); b.orientation=Quaternion([1,0,0,0])
                if j<3: b.location[j]+=sign*h
                else:
                    rot=np.zeros(3);rot[j-3]=sign*h
                    b.orientation=Quaternion.from_rotation(rot)
                _,N=it.solve_mobility_problem_dense_algebra()
                cols.append(N[:,j])
            target+=(cols[1]-cols[0])/(2*h)
        measured=[]
        for j in range(6):
            for sign in (-1,1):
                b.location=initial.copy();b.orientation=Quaternion([1,0,0,0])
                noise=np.zeros(6);noise[j]=sign*np.sqrt(6)
                with patch.object(solver.np.random,'normal',return_value=noise), patch('quaternion_integrator.quaternion_integrator_multi_bodies.stochastic.stochastic_forcing_lanczos',return_value=(np.zeros(36),0)):
                    it.advance_time_step(1e-3,step=0)
                measured.append(increment(it,[initial])/1e-3)
        np.testing.assert_allclose(np.mean(measured,axis=0),target,atol=2e-6,rtol=.02)

    def test_lanczos_limit_is_not_silently_accepted(self):
        it=setup(.35,1.,self.backend)
        with patch('quaternion_integrator.quaternion_integrator_multi_bodies.stochastic.stochastic_forcing_lanczos',return_value=(np.zeros(36),1000)):
            with self.assertRaisesRegex(RuntimeError,'Lanczos'):
                it.advance_time_step(.001,step=0)

    def test_gmres_failure_is_not_silently_accepted(self):
        it=setup(.35,0.,self.backend)
        with patch.object(solver.utils,'gmres',return_value=(np.zeros(42),1)):
            with self.assertRaisesRegex(RuntimeError,'GMRES'):
                it.advance_time_step(.001,step=0)


@unittest.skipUnless(os.environ.get('RIGID_TEST_GPU')=='1','Set RIGID_TEST_GPU=1 on CUDA host')
class GPUBrownianShear(IterativeBrownianShear):
    backend='pycuda'
