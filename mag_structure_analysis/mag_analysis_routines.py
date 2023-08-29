import random

import matplotlib.pyplot as plt
import numpy as np

from Lattice import *
from tets import *
# from collapser import *
import matplotlib.animation as animation
import plotly.graph_objects as go
# import pyvista as pv
from scipy.io import FortranFile
import h5py

def initialize_topo(i):
    print('')
    print('----------------------------------------------------------')
    print('Generating randomized lattices')
    print('')
    start = time.time()
    data = jit_populate(phi_arr, Hoft_arr)
    write("phi_arr_%s" % (i), Master_path, data[0])
    write("hoft_arr_%s" % (i), Master_path, data[1])
    print('Time Elapsed', np.abs(time.time() - start))

    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Computing Topological quantities')
    start = time.time()
    no_mn,no_amn=run_tets2(i)
    print('Time for computing topological quantities:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Finding Pairs')
    start = time.time()
    tracing_test(i)
    print('Time for tracing:%s'%(abs(start-time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Plotting Network')
    tet_2_plot(i)
    print('----------------------------------------------------------')
    print('')
    return no_mn,no_amn

def initialize_topo_nopair(i):
    print('')
    print('----------------------------------------------------------')
    print('Generating randomized lattices')
    print('')
    start = time.time()
    data = jit_populate(phi_arr, Hoft_arr)
    write("phi_arr_%s" % (i), Master_path, data[0])
    write("hoft_arr_%s" % (i), Master_path, data[1])
    print('Time Elapsed', np.abs(time.time() - start))

    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Computing Topological quantities')
    start = time.time()
    no_mn,no_amn=run_tets2(i)
    print('Time for computing topological quantities:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    return no_mn,no_amn
