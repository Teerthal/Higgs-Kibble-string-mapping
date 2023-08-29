import numpy as np

from mag_lattice import *

def run_multiple_all(i):

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('Generating randomized lattices')
    print('');       print('');        print('')
    start = time.time()
    data = jit_populate(phi_arr, Hoft_arr)
    write("phi_arr_%s" % (i), Master_path, data[0])
    write("hoft_arr_%s" % (i), Master_path, data[1])
    print('Time Elapsed', np.abs(time.time() - start))
    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')
    #print(np.shape(data[0]))
    #H_arr = data[1]
    #plt.hist(np.ravel(H_arr[:,:,:,0]), bins=100);plt.show()
    #plt.hist(np.ravel(H_arr[:,:,:,1]), bins=100);plt.show()
    #plt.hist(np.ravel(H_arr[:,:,:,2]), bins=100);plt.show()
    #plt.show()
    print('Computing Derivatives')
    start = time.time()
    phi_arr_i = read("phi_arr_%s"%i,Master_path)
    write("diff_arr_%s" %i, Master_path, derivatives(phi_arr_i,diff_arr,i))
    print('Time Elapsed', np.abs(time.time() - start))

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')

    return

pool = Pool(N_cpu)
pool.map(run_multiple_all, range(N_runs))
#run_multiple_all()
pool.close()
