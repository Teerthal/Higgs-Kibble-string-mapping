import random
import time

from scipy.interpolate import splprep
import numpy as np
from scipy.interpolate import splev
from mag_charde_params import *
from itertools import combinations
from mag_lattice import Higgs_direction,read,write
from numba import njit,prange, jit, roc
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.pyplot import cm
import pickle
from multiprocessing import Pool

#N=4

@jit()
def triangle_combs(a):

    return [[a[0],a[1],a[2]],[a[0],a[2],a[3]],[a[0],a[3],a[1]],[a[1],a[3],a[2]]]
    #return list(combinations(a,3))
@jit()
def tri_ang(n1,n2,n3):
    n2n3 = n2[0] * n3[0] + n2[1] * n3[1] + n2[2] * n3[2]
    n1n2 = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
    n1n3 = n1[0] * n3[0] + n1[1] * n3[1] + n1[2] * n3[2]
    if n1n2**2 == 1 or n1n3**2==1:
        ang = 0
        #print('singularity warning in tri angle compute')
    else:
        #ang = np.arccos((np.dot(n2,n3)-np.dot(n1,n2)*np.dot(n1,n3))/np.sqrt( (1-(np.dot(n1,n2))**2)*(1-(np.dot(n1,n3))**2) ))
        ang = np.arccos((n2n3 - n1n2 * n1n3) / np.sqrt((1 - (n1n2) ** 2) * (1 - (n1n3) ** 2)))
    return ang

@jit()
def tri_area(n1,n2,n3):
    return  tri_ang(n1,n2,n3)+ tri_ang(n2,n3,n1) + tri_ang(n3,n1,n2) - np.pi


@jit()
def mtx_pd(A,B):
    pd_00 = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    pd_01 = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    pd_10 = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    pd_11 = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    mt_pd = np.array([[pd_00,pd_01],[pd_10,pd_11]])

    return mt_pd

@jit()
def mtA_vB(A,B):
    ##Returns product of 2x2 matrix A and 2-vector B##
    pd_0 = A[0][0]*B[0] + A[0][1]*B[1]
    pd_1 = A[1][0]*B[0] + A[1][1]*B[1]
    return np.array([pd_0,pd_1])

@jit()
def vA_dag_vB(A,B):
    ##Product 2-vector A^dag and 2-vector B##
    ##Tested and checked out##
    pd_0 = (np.real(A[0]) - np.imag(A[0]) * 1j) * B[0] + (np.real(A[1]) - np.imag(A[1]) * 1j) * B[1]#;print(pd_0)
    return pd_0


@jit()
def tetrahedra_2(lat1, tets,area_arr,N):

    #tets = []
    #area_arr = []
    idx = 0
    ar_idx = 0

    for x in prange(1,N-1,2):
        for y in prange(1,N-1,2):
            for z in prange(1,N-1,2):

                #Along z#
                #print(x,y,z)
                A = lat1[x,y,z]#;print(A)
                B = lat1[x,y,z-1]#;print(B)
                C = lat1[x-1,y-1,z-1]
                D = lat1[x+1,y-1,z-1]
                E = lat1[x+1,y+1,z-1]
                F = lat1[x-1,y+1,z-1]
                G = lat1[x-1,y-1,z+1]
                H= lat1[x+1,y-1,z+1]
                I = lat1[x+1,y+1,z+1]
                J = lat1[x-1,y+1,z+1]
                K = lat1[x,y,z+1]
                M = lat1[x+1,y,z]
                NN = lat1[x,y+1,z]
                O = lat1[x-1,y,z]
                L = lat1[x,y-1,z]


                tet1 = [A,B,C,D]
                tet2 = [A,B,D,E]
                tet3 = [A,B,E,F]
                tet4 = [A,B,F,C]
                tet5 = [A,C,L,D]
                tet6 = [A,D,L,H]
                tet7 = [A,H,L,G]
                tet8 = [A,G,L,C]
                tet9 = [A,D,M,E]
                tet10 = [A,E,M,I]
                tet11 = [A,I,M,H]
                tet12 = [A,H,M,D]
                tet13 = [A,E,NN,F]
                tet14 = [A,F, NN,J]
                tet15 = [A,J, NN,I]
                tet16 = [A,I, NN,E]
                tet17 = [A, F, O, C]
                tet18 = [A, C, O, G]
                tet19 = [A, G, O, J]
                tet20 = [A, J, O, F]
                tet21=[A,G,K,H]
                tet22=[A,H,K,I]
                tet23 =[A,I,K,J]
                tet24 = [A,J,K,G]


                for j in prange(4):tets[idx,j] = tet1[j]
                idx=idx+1
                for j in prange(4):tets[idx,j] = tet2[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet3[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet4[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet5[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet6[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet7[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet8[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet9[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet10[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet11[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet12[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet13[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet14[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet15[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet16[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet17[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet18[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet19[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet20[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet21[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet22[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet23[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet24[j]
                idx = idx+1

                sub_tets = [tet1,tet2,tet3,tet4,
                            tet5,tet6,tet7,tet8,
                            tet9,tet10,tet11,tet12,
                            tet13,tet14,tet15,tet16,
                            tet17,tet18,tet19,tet20,
                            tet21,tet22,tet23,tet24]



                for a in sub_tets:

                    triangles = [[a[0],a[1],a[2]],[a[0],a[2],a[3]],[a[0],a[3],a[1]],[a[1],a[3],a[2]]]
                    sum_area = 0

                    for triangle in triangles:

                        n1 = triangle[0];n2=triangle[1];n3=triangle[2]
                        area_sign = np.sign(np.dot(np.cross(n1,n2),n3))
                        Area = tri_area(n1,n2,n3)
                        sum_area = sum_area + area_sign*Area

                    area_arr[ar_idx] = sum_area

                    ar_idx=ar_idx+1

    return tets,area_arr

lat1 = np.zeros((N,N,N))



def run_tets2(i):
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
    n = np.zeros((N, N, N, 3), dtype=complex)

    phi_list = ['phi_arr_%s'%s for s in range(N_runs)]

    tets = np.zeros((int(((N-1)/2) ** 3 * 24), 4,3), dtype=complex)
    area_arr = np.zeros((int(((N-1)/2) ** 3 * 24)), dtype=complex)

    phi_1 = read('%s'%(phi_list[i]),Master_path)
    n_1 = Higgs_direction(phi_1,n,sigma)

    vol_charge = []
    for L_vol in range(3,N+1,2):
        tet_stack, area_stack = tetrahedra_2(n_1, tets,area_arr,L_vol)
        normed = np.real(area_stack / (4 * np.pi))
        N_mono = len(np.where(normed > 1 - err_mar)[0])
        N_amono = len(np.where(normed < -1 + err_mar)[0])
        #print(N_mono)
        #print(N_amono)
        tot_charge = N_mono-N_amono
        vol_charge.append([L_vol,tot_charge])
        #print(L_vol,tot_charge)
    vol_charge = np.array(vol_charge)
    #print(np.shape(vol_charge))

    write('vol_charge_%s'%i,Master_path,vol_charge)

    plt.plot(vol_charge[:,0]**2,vol_charge[:,1])
    plt.show()

    return



sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
n = np.zeros((N, N, N, 3), dtype=complex)

# pool = Pool(N_cpu)
# pool.map(run_tets2, range(N_runs))
# pool.close()

def quad(x,a):
    return a*x**2

def rms_cg_plot():
    cg_dat = np.array([read('vol_charge_%s' % s,Master_path) for s in range(N_runs)])
    cg_rms = np.sqrt(np.mean(cg_dat[:,:,1]**2,axis=0))
    x = cg_dat[0, :, 0]
    y = cg_rms
    print(x[0])
    popt, cov = np.polyfit(x[:-1], y[:-1], 1, cov=True)
    lin_fit = np.poly1d(popt)
    print(popt)
    print(np.sqrt(np.diag(cov)))

    print(lin_fit)
    xp = np.linspace(x[0],x[-1],500)
    fig,ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x,cg_rms,linestyle='',marker='.',c='k')#,label='Fit:%s'%(lin_fit))
    plt.plot(xp,lin_fit(xp),linestyle='--',c='r')
    plt.ylabel(r'$b_{rms}$',fontsize=20)
    plt.xlabel(r'$L$',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    plt.show()
    return

rms_cg_plot()