import numpy as np
import random
import time
from numba import njit,prange, jit
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from numpy import ndarray
from multiprocessing import Pool
from Parameters import *
from itertools import permutations,combinations

sin = np.sin;cos = np.cos;exp = np.exp; pi = np.pi

def write(filename,path,data):
    np.save("%s/%s.npy"%(path,filename), data,allow_pickle=True)
    return

def read(filename,path):
    data = np.load("%s/%s.npy"%(path,filename),allow_pickle=True)
    return data
@jit
def phi(eta,alpha,beta,gamma):
    return [eta*sin(alpha)*exp(1j*beta),eta*cos(alpha)*exp(1j*gamma)]

#@jit
def phi_mag(vec):
    return np.sqrt(np.dot(np.conj(vec),vec))

@jit
def rand_coords():
    alpha=1/2*np.arccos(random.uniform(-1.,1.))
    alpha = random.uniform(0,pi/2)
    beta = random.uniform(0,2*pi)
    gamma = random.uniform(0,2*pi)
    return [alpha,beta,gamma]

#Populate random phi in lattice
def populate(latt_arr):
    #for x_idx, x in enumerate(x_arr):
    for x_idx in prange(N):
        #for y_idx, y in enumerate(y_arr):
        for y_idx in prange(N):
            #for z_idx, z in enumerate(z_arr):
            for z_idx in prange(N):
                coords = rand_coords()
                phi_vec = phi(eta, coords[0], coords[1], coords[2])
                latt_arr[x_idx, y_idx, z_idx, 0] = phi_vec[0]
                latt_arr[x_idx, y_idx, z_idx, 1] = phi_vec[1]
    return latt_arr

#@njit(parallel=True)
@jit()
def jit_populate(latt_arr, Hoft_arr):
    for x_idx in prange(N):
        for y_idx in prange(N):
            for z_idx in prange(N):
                alpha = 1 / 2 * np.arccos(random.uniform(-1., 1.))
                #alpha = random.uniform(0, pi / 2)
                beta = random.uniform(0, 2 * pi)
                gamma = random.uniform(0, 2 * pi)
                #alpha,beta,gamma = rand_coords()

                ##Write angles to Hfot array
                Hoft_arr[x_idx,y_idx,z_idx] = [alpha,beta,gamma]

                phi_vec = [eta*cos(alpha)*exp(1j*beta),eta*sin(alpha)*exp(1j*gamma)]

                #phi_vec = [cos(beta/2),sin(beta/2)*exp(1j*gamma/2)]

                latt_arr[x_idx, y_idx, z_idx, 0] = phi_vec[0]
                latt_arr[x_idx, y_idx, z_idx, 1] = phi_vec[1]

                #Periodic Boundary Condition
                if z_idx == int(N-1):
                    latt_arr[x_idx, y_idx, z_idx, 0] = latt_arr[x_idx, y_idx, 0, 0]
                    latt_arr[x_idx, y_idx, z_idx, 1] = latt_arr[x_idx, y_idx, 0, 1]
                if y_idx == int(N-1):
                    latt_arr[x_idx, y_idx, z_idx, 0] = latt_arr[x_idx, 0, z_idx, 0]
                    latt_arr[x_idx, y_idx, z_idx, 1] = latt_arr[x_idx, 0, z_idx, 1]
                if x_idx == int(N-1):
                    latt_arr[x_idx, y_idx, z_idx, 0] = latt_arr[0, y_idx, z_idx, 0]
                    latt_arr[x_idx, y_idx, z_idx, 1] = latt_arr[0, y_idx, z_idx, 1]

    return latt_arr, Hoft_arr


#Run populate several times
# def multiple_populate():
#     for i in range(N_runs):
#         data = jit_populate(phi_arr,Hoft_arr)
#         write("phi_arr_%s"%(i),Master_path,data[0])
#         write("hoft_arr_%s"%(i),Master_path,data[1])
#     return
# multiple_populate()

# test_read = read("phi_arr_0",Master_path)
# print(np.shape(test_read))

def Hoft_sample_hist():

    plt.hist(np.cos(2*np.ravel(Hoft_arr[:,:,:,0])), bins=100);plt.show()
    plt.hist(np.ravel(Hoft_arr[:,:,:,1]), bins=100);plt.show()
    plt.hist(np.ravel(Hoft_arr[:,:,:,2]), bins=100);plt.show()
    return

#Hoft_sample_hist()

# @njit(parallel=True)
# @jit()
def derivatives(phi_arr_i, diff_arr, run_idx):
    arr = phi_arr_i
    # for x_idx in prange(N-1):
    #     for y_idx in prange(N-1):
    #         for z_idx in prange(N-1):
    #             ##Derivatives using central differences
    #
    #             dphi_0_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 0] - arr[x_idx-1, y_idx, z_idx, 0])
    #             dphi_1_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 0] - arr[x_idx-1, y_idx, z_idx, 1])
    #
    #             dphi_0_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 0] - arr[x_idx, y_idx-1, z_idx, 0])
    #             dphi_1_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 0] - arr[x_idx, y_idx-1, z_idx, 1])
    #
    #             dphi_0_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 0] - arr[x_idx, y_idx, z_idx - 1, 0])
    #             dphi_1_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 0] - arr[x_idx, y_idx, z_idx - 1, 1])
    #
    #             diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
    #             diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]
    #
    # #Setting last element derivative to that of the first one
    # diff_arr[-1,:,:,:,:] = diff_arr[0,:,:,:,:]
    # diff_arr[:, -1, :, :, :] = diff_arr[:, 0, :, :, :]
    # diff_arr[:, :, -1, :, :] = diff_arr[:, :, 0, :, :]

    # Right: x_idx = N-1 yz-surface
    # Top: z_idx = N-1 xy-surface
    # Back: y_idx = N-1 xz-surface
    ##supercell is the 3^3 lattice that is divided into tetrahera
    ##Loops over all the vertices of the supercell
    ##Does not to go the right top and back surfaces
    for x_idx in range(0,N-1,2):
        for y_idx in range(0,N-1,2):
            for z_idx in range(0,N-1,2):
                # print(x_idx,y_idx,z_idx)
                bdx = 0
                bdy = 0
                bdz = 0
                if x_idx == 0: bdx = 1
                if y_idx == 0: bdy = 1
                if z_idx == 0: bdz = 1

                dphi_0_x = (1 / 2 * (2*a)) * (arr[x_idx+2, y_idx, z_idx, 0] - arr[x_idx-(2+bdx), y_idx, z_idx, 0])
                dphi_1_x = (1 / 2 * (2*a)) * (arr[x_idx+2, y_idx, z_idx, 1] - arr[x_idx-(2+bdx), y_idx, z_idx, 1])

                dphi_0_y = (1 / 2 * (2*a)) * (arr[x_idx, y_idx+2, z_idx, 0] - arr[x_idx, y_idx-(2+bdy), z_idx, 0])
                dphi_1_y = (1 / 2 * (2*a)) * (arr[x_idx, y_idx+2, z_idx, 1] - arr[x_idx, y_idx-(2+bdy), z_idx, 1])

                dphi_0_z = (1/2*(2*a))*(arr[x_idx, y_idx, z_idx + 2, 0] - arr[x_idx, y_idx, z_idx - (2+bdz), 0])
                dphi_1_z = (1/2*(2*a))*(arr[x_idx, y_idx, z_idx + 2, 1] - arr[x_idx, y_idx, z_idx - (2+bdz), 1])

                diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
                diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]
                
    ##Scans over the supercell face centered points starting from the bottom surface but omitting the top
    ##Does not loop over the face points on the Left or Front surfaces or the supercell centers
    for x_idx in range(1,N-1,2):
        for y_idx in range(1,N-1,2):
            for z_idx in range(0,N-1,2):
                # print(x_idx,y_idx,z_idx)
                bdx = 0;bdfx = 0
                bdy = 0;bdfy = 0
                bdz = 0;bdfz = 0
                if x_idx == 1: bdx = 1
                if y_idx == 1: bdy = 1
                if z_idx == 0: bdz = 1
                if x_idx == N-2: bdfx = -(1+x_idx)
                if y_idx == N-2: bdfy = -(1+y_idx)

                dphi_0_x = (1 / 2 * (2*a)) * (arr[x_idx+2+bdfx, y_idx, z_idx, 0] - arr[x_idx-(2+bdx), y_idx, z_idx, 0])
                dphi_1_x = (1 / 2 * a) * (arr[x_idx+2+bdfx, y_idx, z_idx, 1] - arr[x_idx-(2+bdx), y_idx, z_idx, 1])

                dphi_0_y = (1 / 2 * (2*a)) * (arr[x_idx, y_idx+2+bdfy, z_idx, 0] - arr[x_idx, y_idx-(2+bdy), z_idx, 0])
                dphi_1_y = (1 / 2 * (2*a)) * (arr[x_idx, y_idx+2+bdfy, z_idx, 1] - arr[x_idx, y_idx-(2+bdy), z_idx, 1])

                dphi_0_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 0] - arr[x_idx, y_idx, z_idx - (1+bdz), 0])
                dphi_1_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 1] - arr[x_idx, y_idx, z_idx - (1+bdz), 1])

                diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
                diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]
    #Loop for the face centered points on the left surface
    for x_idx in [0]:
        for y_idx in range(1,N-1,2):
            for z_idx in range(1,N-1,2):
                # print(x_idx,y_idx,z_idx)
                bdx = 0;bdfx = 0
                bdy = 0;bdfy = 0
                bdz = 0;bdfz = 0
                if x_idx == 0: bdx = 1
                if y_idx == 1: bdy = 1
                if z_idx == 1: bdz = 1
                if y_idx == N-2: bdfy = -(1+y_idx)
                if z_idx == N-2: bdfz = -(1+z_idx)

                dphi_0_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 0] - arr[x_idx-(1+bdx), y_idx, z_idx, 0])
                dphi_1_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 1] - arr[x_idx-(1+bdx), y_idx, z_idx, 1])

                dphi_0_y = (1 / 2 * (2*a)) * (arr[x_idx, y_idx+2+bdfy, z_idx, 0] - arr[x_idx, y_idx-(2+bdy), z_idx, 0])
                dphi_1_y = (1 / 2 * (2*a)) * (arr[x_idx, y_idx+2+bdfy, z_idx, 1] - arr[x_idx, y_idx-(2+bdy), z_idx, 1])

                dphi_0_z = (1/2*(2*a))*(arr[x_idx, y_idx, z_idx + 2+bdfz, 0] - arr[x_idx, y_idx, z_idx - (2+bdz), 0])
                dphi_1_z = (1/2*(2*a))*(arr[x_idx, y_idx, z_idx + 2+bdfz, 1] - arr[x_idx, y_idx, z_idx - (2+bdz), 1])

                diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
                diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]
    # Loop for the face centered points on the Front surface
    for x_idx in range(1,N-1,2):
        for y_idx in [0]:
            for z_idx in range(1,N-1,2):
                # print(x_idx,y_idx,z_idx)
                bdx = 0;bdfx = 0
                bdy = 0;bdfy = 0
                bdz = 0;bdfz = 0
                if x_idx == 0: bdx = 1
                if y_idx == 1: bdy = 1
                if z_idx == 1: bdz = 1
                if x_idx == N-2: bdfx = -(1+x_idx)
                if z_idx == N-2: bdfz = -(1+z_idx)
                dphi_0_x = (1 / 2 * (2*a)) * (arr[x_idx+2+bdfx, y_idx, z_idx, 0] - arr[x_idx-(2+bdx), y_idx, z_idx, 0])
                dphi_1_x = (1 / 2 * (2*a)) * (arr[x_idx+2+bdfx, y_idx, z_idx, 1] - arr[x_idx-(2+bdx), y_idx, z_idx, 1])

                dphi_0_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 0] - arr[x_idx, y_idx-(1+bdy), z_idx, 0])
                dphi_1_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 1] - arr[x_idx, y_idx-(1+bdy), z_idx, 1])

                dphi_0_z = (1/2*(2*a))*(arr[x_idx, y_idx, z_idx + 2+bdfz, 0] - arr[x_idx, y_idx, z_idx - (2+bdz), 0])
                dphi_1_z = (1/2*(2*a))*(arr[x_idx, y_idx, z_idx + 2+bdfz, 1] - arr[x_idx, y_idx, z_idx - (2+bdz), 1])

                diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
                diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]
    #Loops over the supercell centers
    for x_idx in range(1,N-1,2):
        for y_idx in range(1,N-1,2):
            for z_idx in range(1,N-1,2):
                # print(x_idx,y_idx,z_idx)
                dphi_0_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 0] - arr[x_idx-1, y_idx, z_idx, 0])
                dphi_1_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 1] - arr[x_idx-1, y_idx, z_idx, 1])

                dphi_0_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 0] - arr[x_idx, y_idx-1, z_idx, 0])
                dphi_1_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 1] - arr[x_idx, y_idx-1, z_idx, 1])

                dphi_0_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 0] - arr[x_idx, y_idx, z_idx - 1, 0])
                dphi_1_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 1] - arr[x_idx, y_idx, z_idx - 1, 1])

                diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
                diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]
    #Setting last element derivative to that of the first one
    
    for x_idx in range(N):
        for y_idx in range(N):
            for z_idx in range(N):
                if x_idx == N-1:
                    diff_arr[x_idx,y_idx,z_idx,:,:] = diff_arr[0,y_idx,z_idx,:,:]
                if y_idx == N-1:
                    diff_arr[x_idx,y_idx,z_idx,:,:] = diff_arr[x_idx,0,z_idx,:,:]
                if z_idx == N-1:
                    diff_arr[x_idx,y_idx,z_idx,:,:] = diff_arr[x_idx,y_idx,0,:,:]

    #Could be elegant with nested for loops but meh
    diff_arr[N-1,0,0]=diff_arr[0,0,0]
    diff_arr[N-1,N-1,0]=diff_arr[0,0,0]
    diff_arr[N-1,0,N-1]=diff_arr[0,0,0]
    diff_arr[N-1,N-1,N-1]=diff_arr[0,0,0]
    diff_arr[0,N-1,0]=diff_arr[0,0,0]
    diff_arr[0,N-1,N-1]=diff_arr[0,0,0]
    diff_arr[0,0,N-1]=diff_arr[0,0,0]

    return diff_arr

# def multiple_diff_dump():
#     for i in range(N_runs):
#         phi_arr_i = read("phi_arr_%s"%i,Master_path)
#         write("diff_arr_%s" %i, Master_path, derivatives(phi_arr_i,diff_arr,i))
#     return
#
# print('');print('');print('')
# print('----------------------------------------------------------')
# print('');print('');print('')
# start = time.time()
#
# multiple_diff_dump()
#
# print('Time Elapsed for derivatives', np.abs(time.time()-start))
# print('');print('');print('')
# print('----------------------------------------------------------')

#@njit(parallel=True)
def boundary_derivs_check(diff_arr, latt_arr, stack):
    idx = 0
    for x_idx in prange(N-1):
        for y_idx in prange(N-1):
            diff_0 = (latt_arr[x_idx, y_idx, 0, 0] - latt_arr[x_idx, y_idx, -1, 0])
            deriv_diff_0 = abs(diff_arr[x_idx, y_idx, 0, 0] - diff_arr[x_idx, y_idx, -1, 0])
            stack[idx,0] = diff_0; stack[idx,1] = deriv_diff_0
            print(diff_0)
            idx = idx + 1
    return

#test_diff_arr = boundary_derivs_check(diff_arr,phi_arr,stack = np.ndarray(shape=(int(N*N), 2), dtype=complex) )
#print(np.shape(test_diff_arr))
#plt.hist(test_diff_arr)


@njit(parallel=True)
def em_tensor(diff_arr, em_arr):
    for x_idx in prange(N):
        for y_idx in prange(N):
            for z_idx in prange(N):
                for i in [0, 1, 2]:
                    for j in [0, 1, 2]:
                        mag = np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,i]),diff_arr[x_idx,y_idx,z_idx,:,j])-\
                              np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,j]),diff_arr[x_idx,y_idx,z_idx,:,i])
                        em_arr[x_idx,y_idx,z_idx, i, j] = mag


    return em_arr

def multiple_em_tensor_dump():
    for i in range(N_runs):
        dif_arr_i = read("diff_arr_%s"%i,Master_path)
        write("em_tensor_%s" %i, Master_path, em_tensor(dif_arr_i,diff_arr))
    return


print('');print('');print('')
print('----------------------------------------------------------')
print('');print('');print('')
start = time.time()
#multiple_em_tensor_dump()
print('');print('');print('')
print('----------------------------------------------------------')

stack_x = np.zeros((N, 3), dtype=complex)
stack_y = np.zeros((N, 3), dtype=complex)
stack_z = np.zeros(3, dtype=complex)

@njit(parallel=True)
def B_integ(diff_arr,stack_x,stack_y,stack_z):
    for x_idx in prange(N):
        for y_idx in prange(N):
            for i,j in zip([1,2,0],[2,0,1]):

                var = [np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,i]),diff_arr[x_idx,y_idx,z_idx,:,j])-\
                              np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,j]),diff_arr[x_idx,y_idx,z_idx,:,i]) for
                                z_idx in range(N)]

                stack_z[i - 1] = np.trapz(var)
            stack_y[y_idx] = stack_z
        stack_x[x_idx] = [np.trapz(stack_y[:,l]) for l in [0,1,2]]
    return [np.trapz(stack_x[:,l]) for l in [0,1,2]]

# print('');print('');print('')
# print('----------------------------------------------------------')
# print('');print('');print('')
# start = time.time()
# def B_integ_all():
#     for i in range(N_runs):
#         diff_arr_i = read("diff_arr_%s"%i,Master_path)
#         write("B_integ_%s"%i, Master_path, B_integ(diff_arr_i,stack_x,stack_y,stack_z))
#     return
# #B_integ_all()
# print('Time Elapsed for B integration tensor', np.abs(time.time()-start))
# print('');print('');print('')
# print('----------------------------------------------------------')

# stack = np.zeros((N, N, N, 3), dtype=complex)
# stack_mag = np.zeros((N, N, N), dtype=complex)
# B_vec = np.zeros(3, dtype=complex)


# @njit(parallel=True)
def B_stack(diff_arr, stack, stack_mag,B_vec):
    for x_idx in prange(0,N,1):
        for y_idx in prange(0,N,1):
            for z_idx in prange(0,N,1):
                for i,j in zip([1,2,0],[2,0,1]):
                    var = np.real(1j*(np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,i]),diff_arr[x_idx,y_idx,z_idx,:,j])-\
                                  np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,j]),diff_arr[x_idx,y_idx,z_idx,:,i])))
                    stack[x_idx,y_idx,z_idx,i-1] = var
                    if np.isnan(np.real(var)):
                        print(diff_arr[x_idx,y_idx,z_idx])

                    B_vec[i-1] = var
                stack_mag[x_idx,y_idx,z_idx] = np.sqrt(np.dot(np.conj(B_vec), B_vec))

    ##Following are loops to replace nonsense values at unused points on the edges of the supercells##
    ##with interpolated values##

    #Loops over the unused points on the edges parallel to x axis#
    for x_idx in range(1,N-1,2):
        for y_idx in range(0,N,2):
            for z_idx in range(0,N,2):

                #current periodic boundary interpolating
                #no need to worry about hitting the boundary along in x-direction
                bdyf = 0
                bdyb = 0
                bdzb = 0
                bdzt = 0
                if y_idx == 0: bdyf = 1
                if z_idx == 0: bdzb = 1
                if y_idx == N-1: bdyb = -(N-1)
                if z_idx == N-1: bdzt = -(N-1)

                for j in range(3):
                    stack[x_idx,y_idx,z_idx,j] = np.mean([stack[x_idx+1,y_idx,z_idx,j]
                                                             ,stack[x_idx-1,y_idx,z_idx,j]
                                                             ,stack[x_idx,y_idx+(1+bdyb),z_idx,j]
                                                             ,stack[x_idx,y_idx-(1+bdyf),z_idx,j]
                                                             ,stack[x_idx,y_idx,z_idx+(1+bdzt),j]
                                                             ,stack[x_idx,y_idx,z_idx-(1+bdzb),j]])

    # Loops over the unused points on the edges parallel to y axis
    for x_idx in range(0,N,2):
        for y_idx in range(1,N-1,2):
            for z_idx in range(0,N,2):

                #current periodic boundary interpolating
                #no need to worry about hitting the boundary along in y-direction
                bdxl = 0
                bdxr = 0
                bdzb = 0
                bdzt = 0
                if x_idx == 0: bdxl = 1
                if z_idx == 0: bdzb = 1
                if x_idx == N-1: bdxr = -(N-1)
                if z_idx == N-1: bdzt = -(N-1)
                for j in range(3):
                    stack[x_idx,y_idx,z_idx,j] = np.mean([stack[x_idx+(1+bdxr),y_idx,z_idx,j]
                                                             ,stack[x_idx-(1+bdxl),y_idx,z_idx,j]
                                                             ,stack[x_idx,y_idx+1,z_idx,j]
                                                             ,stack[x_idx,y_idx-1,z_idx,j]
                                                             ,stack[x_idx,y_idx,z_idx+(1+bdzt),j]
                                                             ,stack[x_idx,y_idx,z_idx-(1+bdzb),j]])

    # Loops over the unused points on the edges parallel to z axis
    for x_idx in range(0,N,2):
        for y_idx in range(0,N,2):
            for z_idx in range(1,N-1,2):

                #current periodic boundary interpolating
                #no need to worry about hitting the boundary along in z-direction
                bdxl = 0
                bdxr = 0
                bdyf = 0
                bdyb = 0
                if x_idx == 0: bdxl = 1
                if y_idx == 0: bdyf = 1
                if x_idx == N-1: bdxr = -(N-1)
                if y_idx == N-1: bdyb = -(N-1)
                for j in range(3):
                    stack[x_idx,y_idx,z_idx,j] = np.mean([stack[x_idx+(1+bdxr),y_idx,z_idx,j]
                                                             ,stack[x_idx-(1+bdxl),y_idx,z_idx,j]
                                                             ,stack[x_idx,y_idx+(1+bdyb),z_idx,j]
                                                             ,stack[x_idx,y_idx-(1+bdyf),z_idx,j]
                                                             ,stack[x_idx,y_idx,z_idx+1,j]
                                                             ,stack[x_idx,y_idx,z_idx-1,j]])
    return stack, stack_mag

def A_stack(diff_arr, phi_arr, stack,interp):
    #Vector potential
    for x_idx in prange(0,N,1):
        for y_idx in prange(0,N,1):
            for z_idx in prange(0,N,1):
                stack[x_idx,y_idx,z_idx,0] = 1j*(np.dot(np.conj(phi_arr[x_idx,y_idx,z_idx,:]),diff_arr[x_idx,y_idx,z_idx,:,0]))
                stack[x_idx,y_idx,z_idx,1] = 1j*(np.dot(np.conj(phi_arr[x_idx,y_idx,z_idx,:]),diff_arr[x_idx,y_idx,z_idx,:,2]))
                stack[x_idx,y_idx,z_idx,2] = 1j*(np.dot(np.conj(phi_arr[x_idx,y_idx,z_idx,:]),diff_arr[x_idx,y_idx,z_idx,:,1]))

    ##Following are loops to replace nonsense values at unused points on the edges of the supercells##
    ##with interpolated values##
    if interp==True:
        #Loops over the unused points on the edges parallel to x axis#
        for x_idx in range(1,N-1,2):
            for y_idx in range(0,N,2):
                for z_idx in range(0,N,2):

                    #current periodic boundary interpolating
                    #no need to worry about hitting the boundary along in x-direction
                    bdyf = 0
                    bdyb = 0
                    bdzb = 0
                    bdzt = 0
                    if y_idx == 0: bdyf = 1
                    if z_idx == 0: bdzb = 1
                    if y_idx == N-1: bdyb = -(N-1)
                    if z_idx == N-1: bdzt = -(N-1)

                    for j in range(3):
                        stack[x_idx,y_idx,z_idx,j] = np.mean([stack[x_idx+1,y_idx,z_idx,j]
                                                                ,stack[x_idx-1,y_idx,z_idx,j]
                                                                ,stack[x_idx,y_idx+(1+bdyb),z_idx,j]
                                                                ,stack[x_idx,y_idx-(1+bdyf),z_idx,j]
                                                                ,stack[x_idx,y_idx,z_idx+(1+bdzt),j]
                                                                ,stack[x_idx,y_idx,z_idx-(1+bdzb),j]])

        # Loops over the unused points on the edges parallel to y axis
        for x_idx in range(0,N,2):
            for y_idx in range(1,N-1,2):
                for z_idx in range(0,N,2):

                    #current periodic boundary interpolating
                    #no need to worry about hitting the boundary along in y-direction
                    bdxl = 0
                    bdxr = 0
                    bdzb = 0
                    bdzt = 0
                    if x_idx == 0: bdxl = 1
                    if z_idx == 0: bdzb = 1
                    if x_idx == N-1: bdxr = -(N-1)
                    if z_idx == N-1: bdzt = -(N-1)
                    for j in range(3):
                        stack[x_idx,y_idx,z_idx,j] = np.mean([stack[x_idx+(1+bdxr),y_idx,z_idx,j]
                                                                ,stack[x_idx-(1+bdxl),y_idx,z_idx,j]
                                                                ,stack[x_idx,y_idx+1,z_idx,j]
                                                                ,stack[x_idx,y_idx-1,z_idx,j]
                                                                ,stack[x_idx,y_idx,z_idx+(1+bdzt),j]
                                                                ,stack[x_idx,y_idx,z_idx-(1+bdzb),j]])

        # Loops over the unused points on the edges parallel to z axis
        for x_idx in range(0,N,2):
            for y_idx in range(0,N,2):
                for z_idx in range(1,N-1,2):

                    #current periodic boundary interpolating
                    #no need to worry about hitting the boundary along in z-direction
                    bdxl = 0
                    bdxr = 0
                    bdyf = 0
                    bdyb = 0
                    if x_idx == 0: bdxl = 1
                    if y_idx == 0: bdyf = 1
                    if x_idx == N-1: bdxr = -(N-1)
                    if y_idx == N-1: bdyb = -(N-1)
                    for j in range(3):
                        stack[x_idx,y_idx,z_idx,j] = np.mean([stack[x_idx+(1+bdxr),y_idx,z_idx,j]
                                                                ,stack[x_idx-(1+bdxl),y_idx,z_idx,j]
                                                                ,stack[x_idx,y_idx+(1+bdyb),z_idx,j]
                                                                ,stack[x_idx,y_idx-(1+bdyf),z_idx,j]
                                                                ,stack[x_idx,y_idx,z_idx+1,j]
                                                                ,stack[x_idx,y_idx,z_idx-1,j]])
    return np.real(stack)

# for i in range(N_runs):
#     diff_arr_i = read("diff_arr_%s"%i,Master_path)
#     B_arr_i, B_mag_arr_i = B_stack(diff_arr_i,stack, stack_mag,B_vec)
#     write("B_arr_%s"%i, Master_path, B_arr_i)
#     write("B_arr_%s" % i, Master_path, B_mag_arr_i)
#     #print('shape B array',np.shape(B_arr_i))
#     #print('shape B mag array',np.shape(B_mag_arr_i))

def curl_A(A_arr,curl_A,interp):
    if interp==True:
        for i in range(N-1):
            for j in range(N-1):
                for k in range(N-1):
                    i_f=1;j_f=1;k_f=1
                    if i==0: 
                        i_b=2 
                    else: 
                        i_b = 1
                    if j==0: 
                        j_b=2 
                    else: 
                        j_b = 1
                    if k==0: 
                        k_b=2 
                    else: 
                        k_b = 1

                    dAy_dx = (0.5/a)*(A_arr[i-i_b,j,k,1]-A_arr[i+i_f,j,k,1])
                    dAz_dx = (0.5/a)*(A_arr[i-i_b,j,k,2]-A_arr[i+i_f,j,k,2])
                    dAx_dy = (0.5/a)*(A_arr[i,j-j_b,k,0]-A_arr[i,j+j_f,k,0])
                    dAz_dy = (0.5/a)*(A_arr[i,j-j_b,k,2]-A_arr[i,j+j_f,k,2])
                    dAx_dz = (0.5/a)*(A_arr[i,j,k-k_b,0]-A_arr[i,j,k+k_f,0])
                    dAy_dz = (0.5/a)*(A_arr[i,j,k-k_b,1]-A_arr[i,j,k+k_f,1])
                    
                    curl_A[i,j,k,0] = dAz_dy - dAy_dz
                    curl_A[i,j,k,1] = dAx_dz - dAz_dx
                    curl_A[i,j,k,2] = dAy_dx - dAx_dy

        for x_idx in range(N):
            for y_idx in range(N):
                for z_idx in range(N):
                    if x_idx == N-1:
                        curl_A[x_idx,y_idx,z_idx] = curl_A[0,y_idx,z_idx]
                    if y_idx == N-1:
                        curl_A[x_idx,y_idx,z_idx] = curl_A[x_idx,0,z_idx]
                    if z_idx == N-1:
                        curl_A[x_idx,y_idx,z_idx] = curl_A[x_idx,y_idx,0]

        for x_idx in [0,N-1]:
            for y_idx in [0,N-1]:
                for z_idx in [0,N-1]:
                    curl_A[x_idx,y_idx,z_idx] = curl_A[0,0,0]
    if interp==False:
        for x_idx in range(0,N-1,2):
            for y_idx in range(0,N-1,2):
                for z_idx in range(0,N-1,2):
                    # print(x_idx,y_idx,z_idx)
                    bdx = 0
                    bdy = 0
                    bdz = 0
                    if x_idx == 0: bdx = 1
                    if y_idx == 0: bdy = 1
                    if z_idx == 0: bdz = 1

                    dAy_dx = (1 / 2 * (2*a)) * (A_arr[x_idx+2, y_idx, z_idx, 1] - A_arr[x_idx-(2+bdx), y_idx, z_idx, 1])
                    dAz_dx = (1 / 2 * (2*a)) * (A_arr[x_idx+2, y_idx, z_idx, 2] - A_arr[x_idx-(2+bdx), y_idx, z_idx, 2])

                    dAx_dy = (1 / 2 * (2*a)) * (A_arr[x_idx, y_idx+2, z_idx, 0] - A_arr[x_idx, y_idx-(2+bdy), z_idx, 0])
                    dAz_dy = (1 / 2 * (2*a)) * (A_arr[x_idx, y_idx+2, z_idx, 2] - A_arr[x_idx, y_idx-(2+bdy), z_idx, 2])

                    dAx_dz = (1/2*(2*a))*(A_arr[x_idx, y_idx, z_idx + 2, 0] - A_arr[x_idx, y_idx, z_idx - (2+bdz), 0])
                    dAy_dz = (1/2*(2*a))*(A_arr[x_idx, y_idx, z_idx + 2, 1] - A_arr[x_idx, y_idx, z_idx - (2+bdz), 1])

                    curl_A[x_idx,y_idx,z_idx,0] = dAz_dy - dAy_dz
                    curl_A[x_idx,y_idx,z_idx,1] = dAx_dz - dAz_dx
                    curl_A[x_idx,y_idx,z_idx,2] = dAy_dx - dAx_dy
                    
        ##Scans over the supercell face centered points starting from the bottom surface but omitting the top
        ##Does not loop over the face points on the Left or Front surfaces or the supercell centers
        for x_idx in range(1,N-1,2):
            for y_idx in range(1,N-1,2):
                for z_idx in range(0,N-1,2):
                    # print(x_idx,y_idx,z_idx)
                    bdx = 0;bdfx = 0
                    bdy = 0;bdfy = 0
                    bdz = 0;bdfz = 0
                    if x_idx == 1: bdx = 1
                    if y_idx == 1: bdy = 1
                    if z_idx == 0: bdz = 1
                    if x_idx == N-2: bdfx = -(1+x_idx)
                    if y_idx == N-2: bdfy = -(1+y_idx)

                    dAy_dx = (1 / 2 * (2*a)) * (A_arr[x_idx+2+bdfx, y_idx, z_idx, 1] - A_arr[x_idx-(2+bdx), y_idx, z_idx, 1])
                    dAz_dx = (1 / 2 * a) * (A_arr[x_idx+2+bdfx, y_idx, z_idx, 2] - A_arr[x_idx-(2+bdx), y_idx, z_idx, 2])

                    dAx_dy = (1 / 2 * (2*a)) * (A_arr[x_idx, y_idx+2+bdfy, z_idx, 0] - A_arr[x_idx, y_idx-(2+bdy), z_idx, 0])
                    dAz_dy = (1 / 2 * (2*a)) * (A_arr[x_idx, y_idx+2+bdfy, z_idx, 2] - A_arr[x_idx, y_idx-(2+bdy), z_idx, 2])

                    dAx_dz = (1/2*a)*(A_arr[x_idx, y_idx, z_idx + 1, 0] - A_arr[x_idx, y_idx, z_idx - (1+bdz), 0])
                    dAy_dz = (1/2*a)*(A_arr[x_idx, y_idx, z_idx + 1, 1] - A_arr[x_idx, y_idx, z_idx - (1+bdz), 1])

                    curl_A[x_idx,y_idx,z_idx,0] = dAz_dy - dAy_dz
                    curl_A[x_idx,y_idx,z_idx,1] = dAx_dz - dAz_dx
                    curl_A[x_idx,y_idx,z_idx,2] = dAy_dx - dAx_dy
        #Loop for the face centered points on the left surface
        for x_idx in [0]:
            for y_idx in range(1,N-1,2):
                for z_idx in range(1,N-1,2):
                    # print(x_idx,y_idx,z_idx)
                    bdx = 0;bdfx = 0
                    bdy = 0;bdfy = 0
                    bdz = 0;bdfz = 0
                    if x_idx == 0: bdx = 1
                    if y_idx == 1: bdy = 1
                    if z_idx == 1: bdz = 1
                    if y_idx == N-2: bdfy = -(1+y_idx)
                    if z_idx == N-2: bdfz = -(1+z_idx)

                    dAy_dx = (1 / 2 * a) * (A_arr[x_idx+1, y_idx, z_idx, 1] - A_arr[x_idx-(1+bdx), y_idx, z_idx, 1])
                    dAz_dx = (1 / 2 * a) * (A_arr[x_idx+1, y_idx, z_idx, 2] - A_arr[x_idx-(1+bdx), y_idx, z_idx, 2])

                    dAx_dy = (1 / 2 * (2*a)) * (A_arr[x_idx, y_idx+2+bdfy, z_idx, 0] - A_arr[x_idx, y_idx-(2+bdy), z_idx, 0])
                    dAz_dy = (1 / 2 * (2*a)) * (A_arr[x_idx, y_idx+2+bdfy, z_idx, 2] - A_arr[x_idx, y_idx-(2+bdy), z_idx, 2])

                    dAx_dz = (1/2*(2*a))*(A_arr[x_idx, y_idx, z_idx + 2+bdfz, 0] - A_arr[x_idx, y_idx, z_idx - (2+bdz), 0])
                    dAy_dz = (1/2*(2*a))*(A_arr[x_idx, y_idx, z_idx + 2+bdfz, 1] - A_arr[x_idx, y_idx, z_idx - (2+bdz), 1])

                    curl_A[x_idx,y_idx,z_idx,0] = dAz_dy - dAy_dz
                    curl_A[x_idx,y_idx,z_idx,1] = dAx_dz - dAz_dx
                    curl_A[x_idx,y_idx,z_idx,2] = dAy_dx - dAx_dy
        # Loop for the face centered points on the Front surface
        for x_idx in range(1,N-1,2):
            for y_idx in [0]:
                for z_idx in range(1,N-1,2):
                    # print(x_idx,y_idx,z_idx)
                    bdx = 0;bdfx = 0
                    bdy = 0;bdfy = 0
                    bdz = 0;bdfz = 0
                    if x_idx == 0: bdx = 1
                    if y_idx == 1: bdy = 1
                    if z_idx == 1: bdz = 1
                    if x_idx == N-2: bdfx = -(1+x_idx)
                    if z_idx == N-2: bdfz = -(1+z_idx)
                    dAy_dx = (1 / 2 * (2*a)) * (A_arr[x_idx+2+bdfx, y_idx, z_idx, 1] - A_arr[x_idx-(2+bdx), y_idx, z_idx, 1])
                    dAz_dx = (1 / 2 * (2*a)) * (A_arr[x_idx+2+bdfx, y_idx, z_idx, 2] - A_arr[x_idx-(2+bdx), y_idx, z_idx, 2])

                    dAx_dy = (1 / 2 * a) * (A_arr[x_idx, y_idx+1, z_idx, 0] - A_arr[x_idx, y_idx-(1+bdy), z_idx, 0])
                    dAz_dy = (1 / 2 * a) * (A_arr[x_idx, y_idx+1, z_idx, 2] - A_arr[x_idx, y_idx-(1+bdy), z_idx, 2])

                    dAx_dz = (1/2*(2*a))*(A_arr[x_idx, y_idx, z_idx + 2+bdfz, 0] - A_arr[x_idx, y_idx, z_idx - (2+bdz), 0])
                    dAy_dz = (1/2*(2*a))*(A_arr[x_idx, y_idx, z_idx + 2+bdfz, 1] - A_arr[x_idx, y_idx, z_idx - (2+bdz), 1])

                    curl_A[x_idx,y_idx,z_idx,0] = dAz_dy - dAy_dz
                    curl_A[x_idx,y_idx,z_idx,1] = dAx_dz - dAz_dx
                    curl_A[x_idx,y_idx,z_idx,2] = dAy_dx - dAx_dy
        #Loops over the supercell centers
        for x_idx in range(1,N-1,2):
            for y_idx in range(1,N-1,2):
                for z_idx in range(1,N-1,2):
                    # print(x_idx,y_idx,z_idx)
                    dAy_dx = (1 / 2 * a) * (A_arr[x_idx+1, y_idx, z_idx, 1] - A_arr[x_idx-1, y_idx, z_idx, 1])
                    dAz_dx = (1 / 2 * a) * (A_arr[x_idx+1, y_idx, z_idx, 2] - A_arr[x_idx-1, y_idx, z_idx, 2])

                    dAx_dy = (1 / 2 * a) * (A_arr[x_idx, y_idx+1, z_idx, 0] - A_arr[x_idx, y_idx-1, z_idx, 0])
                    dAz_dy = (1 / 2 * a) * (A_arr[x_idx, y_idx+1, z_idx, 2] - A_arr[x_idx, y_idx-1, z_idx, 2])

                    dAx_dz = (1/2*a)*(A_arr[x_idx, y_idx, z_idx + 1, 0] - A_arr[x_idx, y_idx, z_idx - 1, 0])
                    dAy_dz = (1/2*a)*(A_arr[x_idx, y_idx, z_idx + 1, 1] - A_arr[x_idx, y_idx, z_idx - 1, 1])

                    curl_A[x_idx,y_idx,z_idx,0] = dAz_dy - dAy_dz
                    curl_A[x_idx,y_idx,z_idx,1] = dAx_dz - dAz_dx
                    curl_A[x_idx,y_idx,z_idx,2] = dAy_dx - dAx_dy
        #Setting last element derivative to that of the first one
        
        for x_idx in range(N):
            for y_idx in range(N):
                for z_idx in range(N):
                    if x_idx == N-1:
                        curl_A[x_idx,y_idx,z_idx,:] = curl_A[0,y_idx,z_idx,:]
                    if y_idx == N-1:
                        curl_A[x_idx,y_idx,z_idx,:] = curl_A[x_idx,0,z_idx,:]
                    if z_idx == N-1:
                        curl_A[x_idx,y_idx,z_idx,:] = curl_A[x_idx,y_idx,0,:]

        #Could be elegant with nested for loops but meh
        curl_A[N-1,0,0]=curl_A[0,0,0]
        curl_A[N-1,N-1,0]=curl_A[0,0,0]
        curl_A[N-1,0,N-1]=curl_A[0,0,0]
        curl_A[N-1,N-1,N-1]=curl_A[0,0,0]
        curl_A[0,N-1,0]=curl_A[0,0,0]
        curl_A[0,N-1,N-1]=curl_A[0,0,0]
        curl_A[0,0,N-1]=curl_A[0,0,0]

    return curl_A    

def mag_charges_interp(B_arr,flux):
    for i in range(N-1):
        for j in range(N-1):
            for k in range(N-1):
                B_l = 0.25*(B_arr[i,j,k,0]+B_arr[i,j+1,k,0]+B_arr[i,j+1,k+1,0]+B_arr[i,j,k+1,0])
                B_r = 0.25*(B_arr[i+1,j,k,0]+B_arr[i+1,j+1,k,0]+B_arr[i+1,j+1,k+1,0]+B_arr[i+1,j,k+1,0])
                B_b = 0.25*(B_arr[i,j,k,1]+B_arr[i+1,j,k,1]+B_arr[i+1,j,k+1,1]+B_arr[i,j,k+1,1])
                B_f = 0.25*(B_arr[i,j+1,k,1]+B_arr[i+1,j+1,k,1]+B_arr[i+1,j+1,k+1,1]+B_arr[i,j+1,k+1,1])
                B_d = 0.25*(B_arr[i,j,k,2]+B_arr[i+1,j,k,2]+B_arr[i+1,j+1,k,2]+B_arr[i,j+1,k,2])
                B_u = 0.25*(B_arr[i,j,k+1,2]+B_arr[i+1,j,k+1,2]+B_arr[i+1,j+1,k+1,2]+B_arr[i,j+1,k+1,2])
                flux[i,j,k] = B_r-B_l+B_f-B_b+B_u-B_d
    return flux

def mag_charges(B_arr,flux):
    for i in range(1,N-1,2):
        for j in range(1,N-1,2):
            for k in range(1,N-1,2):
                B_l = B_arr[i-1,j,k,0]
                B_r = B_arr[i+1,j,k,0]
                B_b = B_arr[i,j-1,k,1]
                B_f = B_arr[i,j+1,k,1]
                B_d = B_arr[i,j,k-1,2]
                B_u = B_arr[i,j,k+1,2]
                flux[i,j,k] = B_r-B_l+B_f-B_b+B_u-B_d
    return flux

def B_fluxes(B_arr,div_B):
    for i in range(N-1):
        for j in range(N-1):
            for k in range(N-1):
                i_f=1;j_f=1;k_f=1
                if i==0: 
                    i_b=2 
                else: 
                    i_b = 1
                if j==0: 
                    j_b=2 
                else: 
                    j_b = 1
                if k==0: 
                    k_b=2 
                else: 
                    k_b = 1

                dB_x = (0.5/a)*(B_arr[i-i_b,j,k,0]-B_arr[i+i_f,j,k,0])
                dB_y = (0.5/a)*(B_arr[i,j-j_b,k,1]-B_arr[i,j+j_f,k,1])
                dB_z = (0.5/a)*(B_arr[i,j,k-k_b,2]-B_arr[i,j,k+k_f,2])
                div_B[i,j,k] = dB_x+dB_y+dB_z
    for x_idx in range(N):
        for y_idx in range(N):
            for z_idx in range(N):
                if x_idx == N-1:
                    div_B[x_idx,y_idx,z_idx] = div_B[0,y_idx,z_idx]
                if y_idx == N-1:
                    div_B[x_idx,y_idx,z_idx] = div_B[x_idx,0,z_idx]
                if z_idx == N-1:
                    div_B[x_idx,y_idx,z_idx] = div_B[x_idx,y_idx,0]

    for x_idx in [0,N-1]:
        for y_idx in [0,N-1]:
            for z_idx in [0,N-1]:
                div_B[x_idx,y_idx,z_idx] = div_B[0,0,0]

    return div_B

#@njit(parallel=True)
def B_spec(B_arr,B_mag_arr):
    ###FFT of the B array###
    #print(np.shape(B_arr),np.shape(B_arr[:,:,:,0]));exit()
    B_x_fft = np.fft.fftn(B_arr[:,:,:,0])
    B_y_fft = np.fft.fftn(B_arr[:,:,:,1])
    B_z_fft = np.fft.fftn(B_arr[:,:,:,2])
    fft_B_mag = np.fft.fftn(B_mag_arr)

    return B_x_fft, B_y_fft, B_z_fft, fft_B_mag
# spec_B = B_spec(B_arr,B_mag_arr)
# print(np.shape(spec_B))

def partitionfunc(n):
    ##Identify number of possible k space mag bins##
    lis = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                lis.append(x**2+y**2+z**2)
    combs = set(lis)
    return np.array(list(combs))

def uniquesums(n):
    ##Identify number of possible k space mag bins##
    lis = []
    for x in range(n):
        for y in range(n):
            for z in range(n):
                lis.append(x+y+z)
    combs = set(lis)
    return np.array(list(combs))

def partition_min_max(n,k,l, m):
    '''n is the integer to partition, k is the length of partitions,
    l is the min partition element size, m is the max partition element size '''
    if k < 1:
        raise StopIteration
    if k == 1:
        if n <= m and n>=l :
            yield (n,)
        raise StopIteration
    for i in range(l,m+1):
        for result in partition_min_max(n-i,k-1,i,m):
            yield result+(i,)

def partition(N, size):
    n = N + size - 1
    for splits in combinations(range(n), size - 1):
        yield [s1 - s0 - 1 for s0, s1 in zip((-1,) + splits, splits + (n,))]

@njit(parallel=True)
def spec_conv_red1(B_x_fft,B_y_fft,B_z_fft, k,stack_spec,B_k,combs,K_c):
    for idx_x in prange(len(k)):
        for idx_y in prange(len(k)):
            for idx_z in prange(len(k)):

                ##Find idx in cov arr##
                idx = np.where(combs == idx_x**2+idx_y**2+idx_z**2)[0][0]
                #print(idx_x,idx_y,idx_z)
                K_c[idx] = idx_x**2+idx_y**2+idx_z**2
                # if K_c <= 3*N**2/4:
                #     idx = idx_x+idx_y+idx_z
                #     print(idx)
                #     B_k[0] = B_x_fft[idx_x,idx_y,idx_z]
                #     B_k[1] = B_y_fft[idx_x, idx_y,idx_z]
                #     B_k[2] = B_z_fft[idx_x, idx_y,idx_z]
                #     stack_spec[idx] += np.dot(np.conj(B_k),B_k)
                # else:
                #     idx = abs(idx_x-N) + abs(idx_y-N) + abs(idx_z-N)
                #     print(idx)
                #     B_k[0] = B_x_fft[idx_x, idx_y, idx_z]
                #     B_k[1] = B_y_fft[idx_x, idx_y, idx_z]
                #     B_k[2] = B_z_fft[idx_x, idx_y, idx_z]
                #     stack_spec[idx] += np.dot(np.conj(B_k), B_k)
                #idx = idx_x+idx_y+idx_z
                B_k[0] = B_x_fft[idx_x, idx_y, idx_z]
                B_k[1] = B_y_fft[idx_x, idx_y,idx_z]
                B_k[2] = B_z_fft[idx_x, idx_y,idx_z]
                stack_spec[idx] += np.dot(np.conj(B_k), B_k)
    # print(len(stack_spec[int(3 * N / 2):]))
    # print(len(stack_spec[:int(3 * N / 2)]));exit()
    #stack_spec[int(3 * N / 2):] = np.zeros((int(3 * N / 2)), dtype=complex)

    return K_c, stack_spec

def spec_convolve_red2(B_x_fft,B_y_fft,B_z_fft, stack_spec,B_k,sums,K_c):
    for idx in prange(len(sums)):
        print(sums[idx])
        #partitions = list(partition_min_max(sums[idx],3,0,sums[idx]))
        partitions = partition(sums[idx],3)
        print(partitions)
        for part in partitions:
            # print(np.array(list(permutations(part))));exit()
            # part_perms = list(set(list(permutations(part))))
            # for a in part_perms:
            print(part)
            idx_x = part[0]
            idx_y = part[1]
            idx_z = part[2]

            B_k[0] = B_x_fft[idx_x, idx_y, idx_z]
            B_k[1] = B_y_fft[idx_x, idx_y, idx_z]
            B_k[2] = B_z_fft[idx_x, idx_y, idx_z]
            K_c[idx] = idx_x ** 2 + idx_y ** 2 + idx_z ** 2 ##Needs work##
            stack_spec[idx] += np.dot(np.conj(B_k), B_k)
    return K_c,stack_spec

@jit()
def K_c_mag(x,y,z):

    if x <= int(N/2):
        K_x = x
    else:
        K_x = x - N

    if y <= int(N/2):
        K_y = y
    else:
        K_y = y - N
    if z <= int(N / 2):
        K_z = z
    else:
        K_z = z - N

    return np.sqrt(K_x**2+K_y**2+K_z**2)

def K_c_bin_num(N):
    lis = []
    for x in range(N):
        for y in range(N):
            for z in range(N):
                lis.append(K_c_mag(x,y,z))
    return len(list(set(sorted(lis))))

#@njit(parallel=True)
#@jit()
def spec_convolve(B_x_fft,B_y_fft,B_z_fft, stack,B_k,K_c,stack_spec,sorted_list,bin_chunk):
    idx = 0
    for idx_x in prange(N):
        for idx_y in prange(N):
            for idx_z in prange(N):
                B_k[0] = B_x_fft[idx_x, idx_y, idx_z]
                B_k[1] = B_y_fft[idx_x, idx_y,idx_z]
                B_k[2] = B_z_fft[idx_x, idx_y,idx_z]
                stack[idx] = np.dot(np.conj(B_k), B_k)
                K_c[idx] = K_c_mag(idx_x,idx_y,idx_z)
                sorted_list[idx,0] = K_c_mag(idx_x,idx_y,idx_z)
                sorted_list[idx,1] = np.dot(np.conj(B_k), B_k)

                idx = idx + 1
    #sorted_list = np.array(list(sorted(zip(K_c,stack))))
    #print(sorted_list)
    sorted_list = sorted_list[np.argsort(sorted_list[:,0])]
    idx_sets = [np.argwhere(i==sorted_list[:,0]) for i in np.unique(sorted_list[:,0])]
    #print('indexing done')
    #print(idx_sets[1]);print(sorted_list[:,0][:8]);exit()
    #for k_idx in prange(len(list(set(sorted_list[:,0]))),bin_chunk):
    #for k_idx in prange(len(list(set(sorted_list[:, 0])))):
    #print(np.shape(idx_sets));exit()
    for k_idx,idx_set in enumerate(idx_sets):

        #k = list(set(sorted_list[:,0]))[k_idx:k_idx+bin_chunk]
        #k = list(set(sorted_list[:, 0]))[k_idx]
        #idxs = np.argwhere(sorted_list[:,0]==k)

        #bin = np.sum(sorted_list[:,1][sorted_list[:,0]==k])

        #bin = np.sum(sorted_list[:, 1][idxs])
        #print(len(sorted_list[:,1][sorted_list[:,0]==k]),bin)

        #print(k_idx)
        #print([sorted_list[:, 0][i] for i in idx_set])
        bin = np.sum([sorted_list[:,1][i] for i in idx_set])

        stack_spec[k_idx] = bin

    return np.array(list(sorted(zip(list(set(sorted_list[:,0])),stack_spec))))


#@njit(parallel=True)
def Higgs_direction(phi_arr, n,sigma):

    for x_idx in prange(N):
        for y_idx in prange(N):
            for z_idx in prange(N):
                phi = phi_arr[x_idx,y_idx,z_idx]
                mag = phi_mag(phi)
                phi_dag = phi.conj().T
                for i in [0,1,2]:

                    ##Diagnostics for mat multiplication
                    # print(phi, phi_dag)
                    # print(np.matmul(phi_dag, np.matmul(sigma[i], phi)))
                    # exit()
                    ## Checked and works

                    n[x_idx,y_idx,z_idx,i] = -1*np.matmul(phi_dag, np.matmul(sigma[i], phi)) / mag

                    # if np.isnan(n[x_idx,y_idx,z_idx,i])==True:
                    #     print(n[x_idx,y_idx,z_idx,i])
                    #     print(phi);print(phi_dag);print(mag);exit()

    return n


def fourier_trnsfm_plot(fft_B_mag):

    clsp_x = np.mean(fft_B_mag,axis=0)
    clsp_y = np.mean(clsp_x,axis=0)
    #print(np.shape(clsp_y))
    k = 2*pi*np.arange(1,N+1)/L
    plt.loglog(k,np.abs(clsp_y))
    plt.xlabel('|k|');plt.ylabel(r'$B_k$')
    plt.show()
    return

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
    K = np.arange(0, N)
    k = np.zeros((N), dtype=float)
    k[:int(N / 2)] = K[:int(N / 2)]
    k[int(N / 2):] = K[:int(N / 2)] - N
    ##Binning in K'##
    del_k = 1
    stack_spec = np.zeros((int(3 * N)), dtype=complex)
    k_c = np.arange(0, int(3 * N))  ###k mag list####

    N_mag_bins = partitionfunc(N)
    print('# of mag bins', len(N_mag_bins))

    E_M = spec_conv(B_x_fft,B_y_fft,B_z_fft,k,stack_spec,np.zeros((3),dtype=complex),N_mag_bins)
    print('Shape convolved spec arr', np.shape(E_M));exit()
    #Folding the spectrum array#
    stack_spec[:int(3 * N / 2)] = E_M[:int(3 * N / 2)] + np.flip(E_M[int(3 * N / 2):])
    ###stack_spec is the final summed over bins spetrum#####
    #plt.semilogy(k_c[:int(3 * N/2)], stack_spec[:int(3 * N/2)])
    #plt.show()

    write('stack_spec_%s'%i, Master_path, stack_spec)

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

# pool = Pool(N_cpu)
# pool.map(run_multiple_all, range(N_runs))
# #run_multiple_all()
# pool.close()
