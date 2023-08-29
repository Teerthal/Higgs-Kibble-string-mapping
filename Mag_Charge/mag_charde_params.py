import numpy as np

N_cpu = 4

N = 301      ##run_1##
N_runs = 100        ##run_1##


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
    Master_path = "/media/teerthal/Repo/Kibble/mag_charge/run_1"
if path_prompt == 'Work':
    Master_path = "/media/cuddlypuff/HDPH-UT/Kibble/mag_charge/run_1"
if path_prompt == 'cluster':
    Master_path = "/home/tpatel28/Kibble/mag_charge/run_1"