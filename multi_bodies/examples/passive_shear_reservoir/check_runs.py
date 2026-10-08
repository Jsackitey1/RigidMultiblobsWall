"""Run from multi_bodies after pilot, control and refined simulations."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualizer.trajectory_loader import load_simulation_data
from visualizer.diagnostics import write_diagnostics
from geometry import geometry_report

base = Path('data/passive_shear')
def load(name, input_path):
    return load_simulation_data(str(base/f'{name}.reservoir.config'), input_path)
pilot = load('run', 'examples/passive_shear_reservoir/pilot.dat')
control = load('control', str(base/'control.dat'))
refined = load('refined', str(base/'refined.dat'))
np.testing.assert_allclose(pilot.time_array, refined.time_array)
np.testing.assert_allclose(pilot.time_array, control.time_array)
for name, traj in [('run', pilot), ('control', control), ('refined', refined)]:
    write_diagnostics(traj, str(base/name))
reports = [geometry_report(p, q, pilot.vertex_blobs, pilot.blob_radius, pilot.periodic_lengths, True)
           for p, q in zip(pilot.positions_unwrapped, pilot.quaternions)]
error = np.linalg.norm(pilot.positions_unwrapped-refined.positions_unwrapped, axis=-1)
interface = float(pilot.metadata['reservoir_end'])
summary = {
    'status': 'pilot numerical checks; not a calibrated physical benchmark',
    'time_end': float(pilot.time_array[-1]),
    'initial_downstream': int((pilot.positions_unwrapped[0,:,0]>=interface).sum()),
    'final_downstream': int((pilot.positions_unwrapped[-1,:,0]>=interface).sum()),
    'control_final_downstream': int((control.positions_unwrapped[-1,:,0]>=interface).sum()),
    'mean_downstream_displacement': float(np.mean(pilot.positions_unwrapped[-1,:,0]-pilot.positions_unwrapped[0,:,0])),
    'control_mean_downstream_displacement': float(np.mean(control.positions_unwrapped[-1,:,0]-control.positions_unwrapped[0,:,0])),
    'maximum_position_difference_dt_halved': float(error.max()),
    'nominal_radius': 1.,
    'position_difference_tolerance': .01,
    'maximum_interbody_blob_overlaps_saved_frames': max(r['overlapping_interbody_blob_pairs'] for r in reports),
    'maximum_bodies_penetrating_wall_saved_frames': max(r['bodies_with_wall_penetration'] for r in reports),
}
summary['checks_passed'] = bool(summary['initial_downstream']==0 and summary['final_downstream']>0 and
    summary['control_final_downstream']==0 and error.max()<.01 and
    summary['maximum_interbody_blob_overlaps_saved_frames']==0 and
    summary['maximum_bodies_penetrating_wall_saved_frames']==0)
(base/'pilot_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
if not summary['checks_passed']:
    raise SystemExit('Pilot checks failed')
