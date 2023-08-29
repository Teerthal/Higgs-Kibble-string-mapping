import numpy as np
import random
import time
from numba import njit,prange, jit
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from numpy import ndarray
from multiprocessing import Pool
from smear_paras import *
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

@njit(parallel=True)
def derivatives(phi_arr_i, diff_arr, run_idx):
    arr = phi_arr_i
    for x_idx in prange(N-1):
        for y_idx in prange(N-1):
            for z_idx in prange(N-1):
                ##Derivatives using central differences

                dphi_0_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 0] - arr[x_idx-1, y_idx, z_idx, 0])
                dphi_1_x = (1 / 2 * a) * (arr[x_idx+1, y_idx, z_idx, 0] - arr[x_idx-1, y_idx, z_idx, 1])

                dphi_0_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 0] - arr[x_idx, y_idx-1, z_idx, 0])
                dphi_1_y = (1 / 2 * a) * (arr[x_idx, y_idx+1, z_idx, 0] - arr[x_idx, y_idx-1, z_idx, 1])

                dphi_0_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 0] - arr[x_idx, y_idx, z_idx - 1, 0])
                dphi_1_z = (1/2*a)*(arr[x_idx, y_idx, z_idx + 1, 0] - arr[x_idx, y_idx, z_idx - 1, 1])

                diff_arr[x_idx, y_idx, z_idx, 0] = [dphi_0_x, dphi_0_y, dphi_0_z]
                diff_arr[x_idx, y_idx, z_idx, 1] = [dphi_1_x, dphi_1_y, dphi_1_z]

    #Setting last element derivative to that of the first one
    diff_arr[-1,:,:,:,:] = diff_arr[0,:,:,:,:]
    diff_arr[:, -1, :, :, :] = diff_arr[:, 0, :, :, :]
    diff_arr[:, :, -1, :, :] = diff_arr[:, :, 0, :, :]

    return diff_arr
# @jit()
# def dot_pd(A,B):
#     sum = 0.
#     for i,j in np.column_stack((A,B)):
#         sum = sum + i*j
#     return sum

@njit(parallel=True)
def B_integ(diff_arr,stack_x,stack_y,stack_z,L):
    for x_idx in prange(L):
        for y_idx in prange(L):
            for i,j in zip([1,2,0],[2,0,1]):

                var = [np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,i]),diff_arr[x_idx,y_idx,z_idx,:,j])-\
                              np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,j]),diff_arr[x_idx,y_idx,z_idx,:,i]) for
                                z_idx in range(L)]

                stack_z[i - 1] = np.trapz(var)

            stack_y[y_idx] = stack_z
        stack_x[x_idx] = [np.trapz(stack_y[:,l]) for l in [0,1,2]]
    return [np.trapz(stack_x[:,l]) for l in [0,1,2]]


@njit(parallel=True)
def B_ana(diff_arr,phi_arr,stack_x,stack_y,stack_z,L,stack):
    idx = 0
    for x_idx in prange(L):
        for y_idx in prange(L):
            for z_idx in prange(L):
                for i,j in zip([1,2,0],[2,0,1]):

                    var = np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,i]),diff_arr[x_idx,y_idx,z_idx,:,j])-\
                                  np.dot(np.conj(diff_arr[x_idx,y_idx,z_idx,:,j]),diff_arr[x_idx,y_idx,z_idx,:,i])
                    var2 = np.dot(np.conj(phi_arr[x_idx,y_idx,z_idx,:]), diff_arr[x_idx,y_idx,z_idx,:,i])

                    stack[idx,0,i-1] = var
                    stack[idx,1,i-1] = var2
                idx = idx+1
    return stack

@njit(parallel=True)
def B_sur(diff_arr,phi_arr, stack_x,stack_y,stack_z,L):
    N = L
    for x_idx in prange(L):

        B_z_0x = [(np.dot(np.conj(phi_arr[x_idx,y_idx,0,:]),diff_arr[x_idx,y_idx,0,:,1])) for
                        y_idx in range(L)]
        B_z_0y = [-(np.dot(np.conj(phi_arr[x_idx,y_idx,0,:]),diff_arr[x_idx,y_idx,0,:,0])) for
                        y_idx in range(L)]
        B_z_Nx = [(np.dot(np.conj(phi_arr[x_idx,y_idx,N,:]),diff_arr[x_idx,y_idx,N,:,1])) for
                        y_idx in range(L)]
        B_z_Ny = [-(np.dot(np.conj(phi_arr[x_idx,y_idx,N,:]),diff_arr[x_idx,y_idx,N,:,0])) for
                        y_idx in range(L)]
        B_z_0x = np.trapz(B_z_0x)
        B_z_0y = np.trapz(B_z_0y)
        B_z_Nx = np.trapz(B_z_Nx)
        B_z_Ny = np.trapz(B_z_Ny)
        stack_x[x_idx,0] = B_z_0x
        stack_x[x_idx, 1] = B_z_0y
        stack_x[x_idx, 2] = 0.
        stack_x[x_idx, 3] = B_z_Nx
        stack_x[x_idx, 4] = B_z_Ny
        stack_x[x_idx, 5] = 0.
    B_z0_x = np.trapz(stack_x[:,0])
    B_z0_y= np.trapz(stack_x[:,1])
    B_z0_z= np.trapz(stack_x[:,2])
    B_zN_x= np.trapz(stack_x[:,3])
    B_zN_y= np.trapz(stack_x[:,4])
    B_zN_z= np.trapz(stack_x[:,5])

    int_x = B_z0_x + B_zN_x
    int_y = B_z0_y + B_zN_y
    int_z = B_z0_z + B_zN_z

    for y_idx in prange(L):

        B_x_0y = [(np.dot(np.conj(phi_arr[0,y_idx,z_idx,:]),diff_arr[0,y_idx,z_idx,:,2])) for
                        z_idx in range(L)]
        B_x_0z = [-(np.dot(np.conj(phi_arr[0,y_idx,z_idx,:]),diff_arr[0,y_idx,z_idx,:,1])) for
                        z_idx in range(L)]
        B_x_Ny = [(np.dot(np.conj(phi_arr[N,y_idx,z_idx,:]),diff_arr[N,y_idx,z_idx,:,2])) for
                        z_idx in range(L)]
        B_x_Nz = [-(np.dot(np.conj(phi_arr[N,y_idx,z_idx,:]),diff_arr[N,y_idx,z_idx,:,1])) for
                        z_idx in range(L)]
        B_x_0y = np.trapz(B_x_0y)
        B_x_0z = np.trapz(B_x_0z)
        B_x_Ny = np.trapz(B_x_Ny)
        B_x_Nz = np.trapz(B_x_Nz)
        stack_y[y_idx,1] = B_x_0y
        stack_y[y_idx, 2] = B_x_0z
        stack_y[y_idx, 0] = 0.
        stack_y[y_idx, 4] = B_x_Ny
        stack_y[y_idx, 5] = B_x_Nz
        stack_y[y_idx, 3] = 0.

    B_x0_x = np.trapz(stack_y[:,0])
    B_x0_y= np.trapz(stack_y[:,1])
    B_x0_z= np.trapz(stack_y[:,2])
    B_xN_x= np.trapz(stack_y[:,3])
    B_xN_y= np.trapz(stack_y[:,4])
    B_xN_z= np.trapz(stack_y[:,5])
    int_x = B_x0_x + B_xN_x + int_x
    int_y = B_x0_y + B_xN_y + int_y
    int_z = B_x0_z + B_xN_z + int_z


    for z_idx in prange(L):

        B_y_0x = [-(np.dot(np.conj(phi_arr[x_idx,0,z_idx,:]),diff_arr[x_idx,0,z_idx,:,2])) for
                        x_idx in range(L)]
        B_y_0z = [(np.dot(np.conj(phi_arr[x_idx,0,z_idx,:]),diff_arr[x_idx,0,z_idx,:,0])) for
                        x_idx in range(L)]
        B_y_Nx = [-(np.dot(np.conj(phi_arr[x_idx,N,z_idx,:]),diff_arr[x_idx,N,z_idx,:,2])) for
                        x_idx in range(L)]
        B_y_Nz = [(np.dot(np.conj(phi_arr[x_idx,N,z_idx,:]),diff_arr[x_idx,N,z_idx,:,0])) for
                        x_idx in range(L)]
        B_y_0x = np.trapz(B_y_0x)
        B_y_0z = np.trapz(B_y_0z)
        B_y_Nx = np.trapz(B_y_Nx)
        B_y_Nz = np.trapz(B_y_Nz)
        stack_z[z_idx,0] = B_y_0x
        stack_z[z_idx, 1] = 0.
        stack_z[z_idx, 2] = B_y_0z
        stack_z[z_idx, 3] = B_y_Nx
        stack_z[z_idx, 4] = 0.
        stack_z[z_idx, 5] = B_y_Nz

    B_y0_x = np.trapz(stack_z[:,0])
    B_y0_y= np.trapz(stack_z[:,1])
    B_y0_z= np.trapz(stack_z[:,2])
    B_yN_x= np.trapz(stack_z[:,3])
    B_yN_y= np.trapz(stack_z[:,4])
    B_yN_z= np.trapz(stack_z[:,5])
    int_x = B_y0_x + B_yN_x + int_x
    int_y = B_y0_y + B_yN_y + int_y
    int_z = B_y0_z + B_yN_z + int_z

    return [int_x,int_y, int_z]

