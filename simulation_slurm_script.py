# This script will run the simulation for different parameters. 
# This script uses Slurm Workload Manager (a job scheduler) to run
# each simulation for each parameter set in parallel to save time.
from simulation_fun import simulation_fun
import numpy as np
import sys

corr_fs_vec = np.arange(0, 5e-3, step=5e-3, dtype=np.float64)
corr_amp_vec = np.arange(0, .7 + .1, step=.1, dtype=np.float64)
tc_wp_vec = np.arange(.1, .5, step=.1, dtype=np.float64)
SWPC_winsize_vec = np.arange(5, 100, step=2, dtype=int)
# SWPC_winsize_vec = np.arange(5, 100, step=2, dtype=int)

# LUT = np.meshgrid(corr_fs_vec, corr_amp_vec, tc_wp_vec, SWPC_winsize_vec)
# LUT = np.array(LUT1).T.reshape(-1,4)
cnt = 0
sz0 = corr_fs_vec.shape[0]
sz1 = corr_amp_vec.shape[0]
sz2 = tc_wp_vec.shape[0]
sz3 = SWPC_winsize_vec.shape[0]
LUT_row_num = sz0 * sz1 * sz2 * sz3
LUT = np.zeros([LUT_row_num, 4])
for cn0 in corr_fs_vec:
    for cn1 in corr_amp_vec:
        for cn2 in tc_wp_vec:
            for cn3 in SWPC_winsize_vec:
                LUT[cnt, 0] = cn0
                LUT[cnt, 1] = cn1
                LUT[cnt, 2] = cn2
                LUT[cnt, 3] = cn3
                cnt = cnt + 1

job_id = sys.argv[1]
job_id = np.int32(job_id)


print(f'job id is {job_id:04d}')
print(f'Job parameters are:')
print(f'correlation fs = {LUT[job_id, 0]}')
print(f'correlation amp = {LUT[job_id, 1]}')
print(f'activity filter cutoff = {LUT[job_id, 2]}')
print(f'Window size = {LUT[job_id, 2]}')
print(f'*'*50)
print(f'*'*50)
print(f'\n'*4)

print(LUT[job_id, 0])
print(LUT[job_id, 1])
simulation_fun(Fs=2, T=500, corr_fs=LUT[job_id, 0], corr_amp=LUT[job_id, 1],
               simulation_num=1000, winsize=np.int32(LUT[job_id, 3]),signal_Wp=LUT[job_id, 2],
               signal_Ws=LUT[job_id, 2]+0.05, mod_n=20)


print(f'job {job_id:04d} is finished.')