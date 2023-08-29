import matplotlib.pyplot as plt
import numpy as np

from B_smear import *
import os



def run_multiple_all(i):

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('Generating randomized lattices')
    print('');       print('');        print('')
    start = time.time()
    data = jit_populate(phi_arr, Hoft_arr)
    write("phi_arr_%s" % (i), Master_path, data[0])
    #write("hoft_arr_%s" % (i), Master_path, data[1])
    print('Time Elapsed', np.abs(time.time() - start))
    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')

    print('Computing Derivatives')
    start = time.time()
    phi_arr_i = read("phi_arr_%s"%i,Master_path)
    write("diff_arr_%s" %i, Master_path, derivatives(phi_arr_i,diff_arr,i))
    print('Time Elapsed', np.abs(time.time() - start))

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')

    diff_arr_i = read("diff_arr_%s" % i, Master_path)
    # print('Computing integrated B')
    # start = time.time()
    # int_stack = []
    # for L in np.arange(0,N,5):
    #     stack_x = np.zeros((L, 3), dtype=complex)
    #     stack_y = np.zeros((L, 3), dtype=complex)
    #     stack_z = np.zeros(3, dtype=complex)
    #     B_int = B_integ(diff_arr_i, stack_x, stack_y, stack_z,L)
    #     int_stack.append(B_int)
    # write("B_integ_%s" % i, Master_path, int_stack)
    # print('Time Elapsed', np.abs(time.time() - start))
    # print('Integrating Time', np.abs(start-time.time()))

    print('-------------------')
    print('Surface integration')
    start=time.time()
    int_stack = []
    for L in np.arange(3,N,2):
        stack_x = np.zeros((L, 6), dtype=complex)
        stack_y = np.zeros((L, 6), dtype=complex)
        stack_z = np.zeros((L,6), dtype=complex)
        #stack = np.zeros((int(N**3),2,3),dtype=complex)
        B_surf = B_sur(diff_arr_i,phi_arr_i, stack_x,stack_y,stack_z,L)
        int_stack.append(B_surf)
    print('Time for surface integration:%s'%(abs(start-time.time())))
    print('B surface integration shape')
    print(np.shape(int_stack))
    print('--------------------')
    os.remove('%s/phi_arr_%s.npy' % (Master_path, i))
    os.remove('%s/diff_arr_%s.npy' % (Master_path, i))
    # lens = np.arange(3,N,2)
    # print(np.shape([np.sqrt(np.dot(np.conj(x),x)) for x in int_stack]))
    # plt.plot(lens, np.real([np.sqrt(np.dot(np.conj(x),x)) for x in int_stack])/lens**3)
    # x = lens
    # y = np.real([np.sqrt(np.dot(np.conj(x),x)) for x in int_stack])/lens**3
    # popt, cov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
    # lin_fit = np.poly1d(popt)
    # print(popt)
    # plt.plot(x,np.exp(lin_fit(np.log(x))))
    # plt.show()

    
    # stack_x = np.zeros((N, 3), dtype=complex)
    # stack_y = np.zeros((N, 3), dtype=complex)
    # stack_z = np.zeros(3, dtype=complex)
    # stack = np.zeros((int(N**3),2,3),dtype=complex)
    # ana_stack = B_ana(diff_arr_i, phi_arr_i, stack_x, stack_y, stack_z, N, stack)
    # print(np.shape(ana_stack[:,1,:]))
    # grad_pdag_grad_p = np.real([np.sqrt(np.dot(np.conj(x),x)) for x in ana_stack[:,0,:]])
    # pdag_grad_p = np.real([np.sqrt(np.dot(np.conj(x),x)) for x in ana_stack[:,1,:]])
    # #plt.hist(grad_pdag_grad_p)
    # plt.hist(pdag_grad_p,bins=100)
    # plt.yscale('log')
    # plt.xlabel(r'${\Phi}^\dagger\nabla\Phi$')
    # plt.show()
    return

pool = Pool(N_cpu)
pool.map(run_multiple_all, range(N_runs))
pool.close()

def smear_plot():
    for i in range(N_runs):
        stack = read('B_integ_%s'%i, Master_path)
        print(np.shape(stack))
        B_mag = [np.sqrt(np.dot(np.conj(x),x)) for x in stack]
        print(np.shape(B_mag))
        plt.plot(np.arange(N), B_mag)
        plt.plot(np.arange(N),np.absolute(stack[:,0]))
        plt.plot(np.arange(N), np.absolute(stack[:, 1]))
        plt.plot(np.arange(N), np.absolute(stack[:, 2]))
        plt.show()
    return
#smear_plot()