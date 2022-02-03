import numpy as np

N_cpu = 4

#N_cpu = 1      ##test##
#Number of lattice vertices
# N = 200        ##Run_1##
# N = 100        ##Run_2##
# N = 200        ##Run_3##
N =11

#Number of runs
#N_runs = N_cpu
# N_runs = N_cpu*100     ##Run_1##
# N_runs = N_cpu*25     ##Run_2##
# N_runs = N_cpu*100     ##Run_3##
N_runs = 1

#Box Length
L = N
#Lattice spacing
a = L/N

#Phi Mag
eta = 1.

#Lattice arrays
x_arr = np.arange(N)
y_arr = np.arange(N)
z_arr = np.arange(N)
phi_arr = np.ndarray(shape=(N,N,N,2), dtype=complex)
Hoft_arr = np.ndarray(shape=(N,N,N,3), dtype=float)
diff_arr = np.ndarray(shape=(N,N,N,2,3), dtype=complex)
em_arr = np.ndarray(shape=(N,N,N,3,3), dtype=complex)

err_mar = 0.1

#path_prompt = input('Select path(PC or Work):')
path_prompt = 'PC'
#path_prompt = 'Work'
#path_prompt = 'cluster'
if path_prompt == 'PC':
    Master_path = "/media/teerthal/Repo/Kibble/run_1"
    Master_path = "/media/teerthal/Repo/Kibble/run_2"
    Master_path = "/media/teerthal/Repo/Kibble/run_3"
    Master_path = "/media/teerthal/Repo/Kibble/test"
    Master_path = "/media/teerthal/Repo/Kibble/test_10"
    #Master_path = "/media/teerthal/Repo 2/Kibble Spectrum/test"
if path_prompt == 'Work':
    Master_path = "/home/cuddlypuff/Repo/Kibble/test"
if path_prompt == 'cluster':
    Master_path = "/home/tpatel28/Kibble/run_1"