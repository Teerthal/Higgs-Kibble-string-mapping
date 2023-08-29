import numpy as np

from Lattice import *
from tets import *
from collapser import *

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

it=2
while no_mn[1]!=0:
# for it in range(it,20):
    new_phi_arr=collapse(it)
    it += 1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)
    if no_mn[1]!=no_amn[1]:
        print('Big error monopole-antimonopole number discrepancy')
        break
    print('-------------')
    print('Iteration:',it)
    print(no_mn,no_amn)
    print('-------------')
print('-------------')
print(it)
print('-------------')
tracing(it)
looping(it)
tet_2_plot(it)

print('----------------------------')
print('Finished collapsing')
print('----------------------------')