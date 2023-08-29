import numpy as np

from Lattice import *
from tets import *
from collapser import *

def run_multiple_all(i):

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
    run_tets2(i)
    print('Time for computing topological quantities:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Plotting monopoles')
    mono_scatter_plot(i)
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Finding Pairs')
    start = time.time()
    tracing(i)
    print('Time for tracing:%s'%(abs(start-time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Finding Loops')
    start = time.time()
    looping(i)
    print('Time for loops:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Plotting Network')
    tet_2_plot(i)
    print('----------------------------------------------------------')
    print('')

    return

run_multiple_all(0)
tet_2_plot(0);exit()
# pool = Pool(N_cpu)
# pool.map(run_multiple_all, range(N_runs))
# #run_multiple_all()
# pool.close()

new_phi_arr = collapse(0)

write("phi_arr_%s" % (2), Master_path, new_phi_arr)
print('Time Elapsed', np.abs(time.time() - start))

print('----------------------------------------------------------')
print('')

print('----------------------------------------------------------')
print('Computing Topological quantities')
start = time.time()
no_mn,no_amn=run_tets2(2)
print('Time for computing topological quantities:%s' % (abs(start - time.time())))
print('----------------------------------------------------------')
print('')

print('----------------------------------------------------------')
print('Plotting monopoles')
mono_scatter_plot(2)
print('----------------------------------------------------------')
print('')

tracing(2)
looping(2)
tet_2_plot(2)