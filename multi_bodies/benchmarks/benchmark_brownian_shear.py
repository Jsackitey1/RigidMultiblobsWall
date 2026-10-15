"""Isolated short shear/Brownian benchmarks; never changes the production input."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def gpu_memory(pid):
    try:
        r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        values = [float(line.split(',')[1]) for line in r.stdout.splitlines()
                  if line.split(',')[0].strip() == str(pid)]
        return max(values) if values else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['numba','pycuda'], default='numba')
    p.add_argument('--scheme', choices=['stochastic_traction_EM','stochastic_first_order_RFD_dense_algebra'], default='stochastic_traction_EM')
    p.add_argument('--counts', type=int, nargs='+', default=[50,100,500])
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--input-file', type=Path, default=ROOT/'inputfile_dynamic.dat')
    args=p.parse_args()
    if min(args.counts)<1 or args.warmup<0 or args.steps<=args.warmup:
        p.error('Positive counts and steps > warmup >= 0 are required')
    if args.scheme.endswith('dense_algebra') and max(args.counts)>50:
        p.error('Dense reference benchmarks are limited to 50 particles')
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out=ROOT/'data'/('benchmark_brownian_'+stamp)
    out.mkdir(parents=True)
    env=os.environ.copy()
    env.setdefault('MPLCONFIGDIR','/tmp/rigid-mpl')
    env.setdefault('NUMBA_NUM_THREADS','2')
    env.setdefault('OPENBLAS_NUM_THREADS','1')
    env.setdefault('VECLIB_MAXIMUM_THREADS','1')
    results=[]
    metadata={'python':sys.version,'backend':args.backend,'scheme':args.scheme,
              'steps':args.steps,'warmup':args.warmup,
              'threads':{k:env.get(k) for k in ['NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
              'timing':'Consecutive solver step wallclock differences after warmup; excludes process startup/JIT before step 0',
              'gpu_memory':'Peak sampled per-process allocation, MiB; sampled approximately once per second; null means unavailable',
              'runs':results}
    try:
        metadata['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        metadata['git_diff']=subprocess.check_output(['git','diff'],cwd=ROOT,text=True)
        metadata['gpu']=subprocess.check_output(['nvidia-smi'],text=True) if args.backend=='pycuda' else None
    except (OSError,subprocess.CalledProcessError):
        pass
    for count in args.counts:
        run=out/str(count); run.mkdir()
        # Keep nominal areal density comparable at each size, with downstream space.
        width=(count*3.141592653589793/.15)**.5+3
        prefix=str(run.relative_to(ROOT))
        updates={'num_bodies':str(count),'n_steps':str(args.steps),'n_save':'1',
                 'scheme':args.scheme,'mobility_vector_prod_implementation':args.backend,
                 'mobility_blobs_implementation':'python','blob_blob_force_implementation':args.backend,
                 'periodic_length':'0 0 0','domain':'single_wall',
                 'box':f'0 {3*width} {-width/2} {width/2}','reservoir_end':str(width),
                 'output_clones':prefix+'/generated_spheres.clones','plot_image':prefix+'/initial.png',
                 'output_name':prefix+'/run','structure':'Structures/shell_N_12_Rg_1.vertex '+prefix+'/generated_spheres.clones',
                 'update_PC':'1'}
        lines=[]
        for line in args.input_file.read_text().splitlines():
            key=line.split()[0] if line.split() else ''
            lines.append(f'{key:<41}{updates.pop(key)}' if key in updates else line)
        lines.extend(f'{k:<41}{v}' for k,v in updates.items())
        config=run/'input.dat'; config.write_text('\n'.join(lines)+'\n')
        print(f'{count} particles: {run}',flush=True)
        with (run/'generation.log').open('w') as log:
            subprocess.run([sys.executable,'generate_sphere_suspension.py','--input-file',str(config)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        peak=None; started=time.monotonic()
        with (run/'simulation.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u','multi_bodies.py','--input-file',str(config)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                value=gpu_memory(proc.pid) if args.backend=='pycuda' else None
                if value is not None: peak=max(peak or 0,value)
                time.sleep(1)
        logtext=(run/'simulation.log').read_text()
        matches=re.findall(r'step =\s+(\d+).*?wallclock time =\s+([\d.eE+-]+)',logtext)
        times={int(s):float(t) for s,t in matches}
        durations=[times[i]-times[i-1] for i in range(args.warmup+1,args.steps+1) if i in times and i-1 in times]
        record={'particles':count,'returncode':proc.returncode,'total_wall_seconds':time.monotonic()-started,
                'peak_sampled_gpu_MiB':peak,'step_seconds':durations,
                'completed':proc.returncode==0 and args.steps in times and '# End' in logtext}
        if durations:
            record['mean_step_seconds']=sum(durations)/len(durations)
            record['estimated_50000_step_days']=record['mean_step_seconds']*50000/86400
        results.append(record)
        (out/'summary.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(record),flush=True)
        if not record['completed']:
            raise RuntimeError(f'Benchmark failed; inspect {run/"simulation.log"}')
    print('Results:',out/'summary.json',flush=True)


if __name__=='__main__':
    main()
