import numpy as np

from Lattice import *

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

    print('Computing EM Tensors')
    start = time.time()
    diff_arr_i = read("diff_arr_%s" % i, Master_path)
    write("em_tensor_%s" % i, Master_path, em_tensor(diff_arr_i, diff_arr))
    print('Time Elapsed', np.abs(time.time() - start))

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')

    print('Computing integrated B')
    start = time.time()
    write("B_integ_%s" % i, Master_path, B_integ(diff_arr_i, stack_x, stack_y, stack_z))
    print('Time Elapsed', np.abs(time.time() - start))

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')

    print('Computing B across lattice')
    start = time.time()

    stack = np.zeros((N, N, N, 3), dtype=complex)
    stack_mag = np.zeros((N, N, N), dtype=complex)
    B_vec = np.zeros(3, dtype=complex)

    B_arr_i, B_mag_arr_i = B_stack(diff_arr_i, stack, stack_mag, B_vec)
    write("B_arr_%s" % i, Master_path, B_arr_i)
    write("B_arr_%s" % i, Master_path, B_mag_arr_i)

    print('Time Elapsed for B arr', np.abs(time.time() - start))
    print('');        print('');        print('')
    print('----------------------------------------------------------')

    print('Computing fourier transfrom')
    start = time.time()
    B_x_fft, B_y_fft,B_z_fft, B_mag_fft = B_spec(B_arr_i,B_mag_arr_i)
    print(np.shape(B_x_fft),np.shape(B_mag_fft))
    write("B_mag_fft_%s" % i, Master_path, B_mag_fft)
    write("B_x_fft_%s" % i, Master_path, B_x_fft)
    write("B_y_fft_%s" % i, Master_path, B_y_fft)
    write("B_z_fft_%s" % i, Master_path, B_z_fft)

    #fourier_trnsfm_plot(B_mag_fft)

    print('');
    print('');
    print('')
    print('Computing Energy Spectrum')
    print('----------------------------------------------------------')
    start = time.time()
    ###Defined as in 1902.02751, eq 24-33
    # K = np.arange(0, N)
    # k = np.zeros((N), dtype=float)
    # k[:int(N / 2)] = K[:int(N / 2)]
    # k[int(N / 2):] = K[:int(N / 2)] - N

    dummy = np.zeros((3),dtype=complex)
    spec_dum = np.zeros((int(N**3)),dtype=complex)
    K_dum = np.zeros((int(N**3)),dtype=complex)
    N_bins = K_c_bin_num(N)
    E_M_dum = np.zeros((N_bins),dtype=complex)
    sorted_dum = np.zeros((N**3,2),dtype=complex)
    bin_size = 1

    E_M = spec_convolve(B_x_fft,B_y_fft,B_z_fft,spec_dum,dummy,K_dum, E_M_dum,sorted_dum,bin_size)

    print('Shape convolved spec arr', np.shape(E_M))

    write('stack_spec_%s'%i, Master_path, E_M)

    print('Time Elapsed for stacking spectrum', np.abs(time.time() - start))
    print('');
    print('');
    print('')
    print('----------------------------------------------------------')

    print('Time Elapsed for B fourier transform', np.abs(time.time() - start))
    print('');        print('');        print('')
    print('----------------------------------------------------------')

    # sigma = np.array([ [[0,1],[1,0]] , [[0,-1j],[1j, 0]] , [[1,0],[0,-1]] ],dtype=complex)
    # print('Computing Higgs direction')
    # start = time.time()
    # n = np.zeros((N, N, N, 3), dtype=complex)
    # Directions_i = Higgs_direction(phi_arr_i,n,sigma)
    # write('Higgs_directions_%s'%i, Master_path, Directions_i)
    #
    # print('Time Elapsed for Higgs Direction', np.abs(time.time() - start))
    # print('');        print('');        print('')
    # print('----------------------------------------------------------')

    return

pool = Pool(N_cpu)
pool.map(run_multiple_all, range(N_runs))
#run_multiple_all()
pool.close()
