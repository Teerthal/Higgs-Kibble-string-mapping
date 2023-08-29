import numpy as np

N_cpu = 4

#N_cpu = 1      ##test##
#Number of lattice vertices

N = 51      ##RUN 1##

#Number of runs
#N_runs = N_cpu
N_runs = 100        ##RUN 1##

N = 27
N_runs = 1          ##BOX##


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

rel_tol = 1e-5

iter_cap = 1000

sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
I = np.array([[1,0],[0,1]])

#path_prompt = input('Select path(PC or Work):')
path_prompt = 'PC'
# path_prompt = 'Home_PC'
#path_prompt = 'Work_local'
#path_prompt = 'Work'
#path_prompt = 'cluster'
path_prompt = 'Dropbox(ASU)'
# path_prompt = 'Home Dropbox ASU'
if path_prompt == 'PC':
    Master_path = "/media/cuddlypuff/HDPH-UT/dumbell_MHD/topological_collapse/test"        ##RUN 1##
    # Master_path = "/media/teerthal/Repo/Kibble/test_7"      ##BOX##
    #Master_path = "/media/teerthal/Repo 2/Kibble Spectrum/test"
if path_prompt=='Work_local':
    Master_path="/home/cuddlypuff/Repo/dumbbell_MHD/topological_collapse/test"
if path_prompt == 'Home_PC':
    Master_path = "/media/teerthal/HDPH-UT1/dumbell_MHD/topological_collapse/test"
if path_prompt == 'Work':
    Master_path = "/home/cuddlypuff/Repo/Kibble/test"
if path_prompt == 'cluster':
    Master_path = "/home/tpatel28/Kibble/run_1"
if path_prompt == 'Dropbox(ASU)':
    Master_path="/home/cuddlypuff/Dropbox (ASU)/dumbell_MHD/topological_collapse/test"
if path_prompt=='Home Dropbox ASU':
    Master_path='/home/teerthal/Dropbox (ASU)/dumbell_MHD/topological_collapse/test'

Plot_path = '%s/plots'%Master_path
