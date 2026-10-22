"""Stable frame layout when the same Matplotlib figure is reused."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualizer.trajectory_loader import SimulationTrajectory
from visualizer.render_2d import render_2d_frame

class Layout(unittest.TestCase):
    def test_reused_figure_has_stable_left_anchored_plot(self):
        for xmax in (90,300):
            traj=SimulationTrajectory([[[5,0,4],[55,10,4]]],[[[1,0,0,0],[1,0,0,0]]],[0],.01,
                sphere_radius=1.25,domain_bounds=[0,xmax,-50,50,0,10],metadata={'reservoir_end':'60','time_unit':'nondimensional'})
            fig=plt.figure(figsize=(14,6),dpi=100)
            first=render_2d_frame(traj,0,fig=fig,show_vectors=False)
            bounds=fig.axes[0].get_position().bounds
            for _ in range(3):
                current=render_2d_frame(traj,0,fig=fig,show_vectors=False)
                np.testing.assert_array_equal(first,current)
                np.testing.assert_allclose(fig.axes[0].get_position().bounds,bounds)
            self.assertLess(bounds[0],.1)
            self.assertEqual(len(fig.axes),2)
            plt.close(fig)
