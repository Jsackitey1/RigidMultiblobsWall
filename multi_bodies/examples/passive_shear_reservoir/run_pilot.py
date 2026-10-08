#!/usr/bin/env python3
"""Reproduce pilot, controls, numerical checks, and the reservoir-transfer MP4."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
env = os.environ.copy()
env.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'rigid-mpl'))
env.setdefault('PYTHONPYCACHEPREFIX', str(Path(tempfile.gettempdir())/'rigid-pycache'))
env.setdefault('NUMBA_NUM_THREADS', '2')
env.setdefault('OPENBLAS_NUM_THREADS', '1')
def run(*args):
    subprocess.run([sys.executable, *args], cwd=root, env=env, check=True)

if __name__ == '__main__':
    input_path = 'examples/passive_shear_reservoir/pilot.dat'
    run('generate_sphere_suspension.py', '--input-file', input_path)
    run('multi_bodies.py', '--input-file', input_path)
    original = (root/input_path).read_text()
    for name, text in [
        ('control', original.replace('shear_rate 0.5', 'shear_rate 0')),
        ('refined', original.replace('dt 0.02', 'dt 0.01').replace('n_steps 1200', 'n_steps 2400').replace('n_save 10', 'n_save 20')),
    ]:
        path = f'data/passive_shear/{name}.dat'
        (root/path).write_text(text.replace('output_name data/passive_shear/run', f'output_name data/passive_shear/{name}'))
        run('multi_bodies.py', '--input-file', path)
    run('examples/passive_shear_reservoir/check_runs.py')
    run('visualize_simulation.py', '--input-file', input_path, '--mode', 'spheres',
        '--color-by', 'vx', '--trail-length', '35', '--duration', '12', '--fps', '20',
        '--output-dir', 'data/passive_shear/video')
