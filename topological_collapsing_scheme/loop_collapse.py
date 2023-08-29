import numpy as np

from Lattice import *
from tets import *
from collapser import *

it =28
tet_2_plot(it)

new_phi_arr=loop_collapser(it)
it+=1
write("phi_arr_%s" % (it), Master_path, new_phi_arr)
no_mn, no_amn = run_tets2(it)

tracing(it)
looping(it)

tet_2_plot(it)#;exit()

new_phi_arr=collapse(it)
it += 1
write("phi_arr_%s" % (it), Master_path, new_phi_arr)
no_mn, no_amn = run_tets2(it)

tracing(it)
looping(it)
tet_2_plot(it)

while no_mn[1]!=0:
# for it in range(it,20):
    new_phi_arr=collapse(it)
    it += 1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)
    if no_mn[1]!=no_amn[1]:
        print('Big error monopole-antimonopole number discrepancy')
        break

tracing(it)
looping(it)
tet_2_plot(it)

new_phi_arr=loop_collapser(it)
it+=1
write("phi_arr_%s" % (it), Master_path, new_phi_arr)
no_mn, no_amn = run_tets2(it)

tracing(it)
no_loops=looping(it)
tet_2_plot(it)



# for j in range(100):
while no_loops!=0 and it <= iter_cap:
    new_phi_arr = collapse(it)
    it += 1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)

    tracing(it)
    looping(it)
    # tet_2_plot(it)

    while no_mn[1] != 0:
        # for it in range(it,20):
        new_phi_arr = collapse(it)
        it += 1
        write("phi_arr_%s" % (it), Master_path, new_phi_arr)
        no_mn, no_amn = run_tets2(it)
        if no_mn[1] != no_amn[1]:
            print('Big error monopole-antimonopole number discrepancy')
            break

    tracing(it)
    looping(it)
    # tet_2_plot(it)

    new_phi_arr = loop_collapser(it)
    it += 1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)

    tracing(it)
    no_loops=looping(it)
    print('**************')
    print('#loop collapse iteration%i'%it)
    print('#Loops:%i'%(len(no_loops)))
    print('**************')

print('---------------------------')
print('Total iterations:%i'%it)
if it==iter_cap:print('Exceeded allowed iterations !Quitting!')
print('---------------------------')
tet_2_plot(it)

