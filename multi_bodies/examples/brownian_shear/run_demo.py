"""Run the research solver and render saved positions/orientations in a top view."""
import argparse
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


def render(out, particle_count):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from quaternion_integrator.quaternion import Quaternion
    from visualizer.video_writer import VideoStreamWriter
    lines = (out/'run.particle.config').read_text().splitlines()
    stride = particle_count + 1
    assert len(lines) % stride == 0
    assert all(int(n) == particle_count for n in lines[::stride])
    frames = np.array([np.loadtxt(lines[i+1:i+stride]).reshape(particle_count, 7)
                       for i in range(0, len(lines), stride)])
    assert np.isfinite(frames).all()
    positions = frames[:, :, :3]
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#0b0f19')
    lo, hi = positions[:, :, 0].min()-2, positions[:, :, 0].max()+2
    artists = []
    ax.set_facecolor('#111827')
    ax.set_xlim(lo, hi)
    ax.set_ylim(positions[:,:,1].min()-2, positions[:,:,1].max()+2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Top view: X–Y', color='white')
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.tick_params(colors='white')
    ax.grid(alpha=.15)
    for particle in range(particle_count):
        color = plt.get_cmap('tab10')(particle % 10)
        trail, = ax.plot([], [], color=color, lw=1, alpha=.8)
        disc = Circle((0,0), 1.25, color=color, alpha=.8)
        ax.add_patch(disc)
        marker, = ax.plot([], [], color='#facc15', lw=2, marker='o', markersize=2)
        artists.append((particle, trail, disc, marker))
    title = fig.suptitle('', color='white', fontsize=14)
    fig.text(.5,.025,'Colored lines: particle paths   |   Yellow: projected body-fixed axis',ha='center',color='white')
    fig.subplots_adjust(top=.80,bottom=.16)
    path=out/'combined_motion.mp4'
    with VideoStreamWriter(str(path), fps=20) as writer:
        for i, frame in enumerate(frames):
            for particle, trail, disc, marker in artists:
                pos = frame[particle, :3]
                direction = Quaternion(frame[particle, 3:]).rotation_matrix()[:,0]*1.25
                trail.set_data(positions[:i+1,particle,0],positions[:i+1,particle,1])
                disc.center=(pos[0],pos[1])
                marker.set_data([pos[0],pos[0]+direction[0]],[pos[1],pos[1]+direction[1]])
            title.set_text(f'Shear + Brownian motion | t = {i*.1:.1f} (nondimensional)\n{particle_count} rigid particle(s) · shear rate 0.35 · kT = 1')
            fig.canvas.draw()
            writer.write_frame(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, choices=[1, 10], default=1)
    args = parser.parse_args()
    out = ROOT/('data/brownian_shear' if args.particles == 1 else 'data/brownian_shear_10')
    out.mkdir(parents=True,exist_ok=True)
    centers = [(0,0,4)] if args.particles == 1 else [(x,y,z) for z in (4,8) for x,y in ((0,-6),(0,0),(0,6),(5,-3),(5,3))]
    (out/'particle.clones').write_text(str(args.particles)+'\n'+''.join(f'{x} {y} {z} 1 0 0 0\n' for x,y,z in centers))
    config = (ROOT/'examples/brownian_shear/input.dat').read_text()
    config = config.replace('data/brownian_shear/', str(out.relative_to(ROOT))+'/')
    if args.particles == 10:
        config = config.replace('# Single-particle', '# Ten-particle').replace('box -3 40 -5 5','box -3 75 -12 12')
    input_path = out/'input.dat'
    input_path.write_text(config)
    print(f'Running {args.particles} particles; log: {out/"simulation.log"}', flush=True)
    with (out/'simulation.log').open('w') as log:
        subprocess.run([sys.executable,'multi_bodies.py','--input-file',str(input_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Simulation complete; rendering videos.', flush=True)
    subprocess.run([sys.executable,'visualize_simulation.py','--input-file',str(input_path),'--mode','spheres','--no-interpolate','--fps','20','--no-vectors','--trail-length','200','--output-dir',str(out/'video')],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
    render(out, args.particles)
