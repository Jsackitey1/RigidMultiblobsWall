"""Run the research solver and render saved positions/orientations in two views."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'rigid-mpl'))
os.environ.setdefault('NUMBA_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')


def render():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from quaternion_integrator.quaternion import Quaternion
    from visualizer.video_writer import VideoStreamWriter
    out = ROOT/'data/brownian_shear'
    lines = (out/'run.particle.config').read_text().splitlines()
    assert all(int(n) == 1 for n in lines[::2]), 'Single-particle renderer only'
    frames = np.loadtxt(lines[1::2])
    assert np.isfinite(frames).all()
    positions = frames[:, :3]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor='#0b0f19')
    lo, hi = positions[:, 0].min()-2, positions[:, 0].max()+2
    artists = []
    for ax, vertical, label in zip(axes, [1,2], ['Top view: X–Y', 'Side view: X–Z']):
        ax.set_facecolor('#111827')
        ax.set_xlim(lo, hi)
        ax.set_ylim((-4,4) if vertical == 1 else (0, max(7,positions[:,2].max()+2)))
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(label, color='white')
        ax.set_xlabel('X', color='white'); ax.set_ylabel('Y' if vertical == 1 else 'Z', color='white')
        ax.tick_params(colors='white'); ax.grid(alpha=.15)
        trail, = ax.plot([], [], color='#38bdf8', lw=1, alpha=.8)
        disc = Circle((0,0), 1.25, color='#818cf8', alpha=.8)
        ax.add_patch(disc)
        marker, = ax.plot([], [], color='#facc15', lw=3, marker='o', markersize=3)
        if vertical == 2:
            ax.axhline(0, color='white', lw=2)
            # Reference background shear arrows; these are not the disturbed fluid field.
            z = np.array([1.,3.,5.])
            ax.quiver(np.full(3,lo+.5),z,.35*z,np.zeros(3),angles='xy',scale_units='xy',scale=1,color='#94a3b8')
        artists.append((vertical, trail, disc, marker))
    title = fig.suptitle('', color='white', fontsize=14)
    fig.text(.5,.025,'Blue: particle path   |   Yellow: projected body-fixed axis   |   Gray arrows: imposed shear only',ha='center',color='white')
    fig.subplots_adjust(top=.88,bottom=.13,hspace=.55)
    path=out/'combined_motion.mp4'
    with VideoStreamWriter(str(path), fps=20) as writer:
        for i, frame in enumerate(frames):
            pos=frame[:3]
            # Project the actual 3D orientation; do not normalize its projection.
            direction=Quaternion(frame[3:]).rotation_matrix()[:,0]*1.25
            for vertical, trail, disc, marker in artists:
                trail.set_data(positions[:i+1,0],positions[:i+1,vertical])
                disc.center=(pos[0],pos[vertical])
                marker.set_data([pos[0],pos[0]+direction[0]],[pos[vertical],pos[vertical]+direction[vertical]])
            title.set_text(f'Shear + Brownian motion | t = {i*.1:.1f} (nondimensional)\nOne rigid particle · shear rate 0.35 · kT = 1')
            fig.canvas.draw()
            writer.write_frame(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    out=ROOT/'data/brownian_shear'
    out.mkdir(parents=True,exist_ok=True)
    (out/'particle.clones').write_text('1\n0 0 4 1 0 0 0\n')
    with (out/'simulation.log').open('w') as log:
        subprocess.run([sys.executable,'multi_bodies.py','--input-file','examples/brownian_shear/input.dat'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'visualize_simulation.py','--input-file','examples/brownian_shear/input.dat','--mode','spheres','--no-interpolate','--fps','20','--no-vectors','--trail-length','200','--output-dir','data/brownian_shear/video'],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
    render()
