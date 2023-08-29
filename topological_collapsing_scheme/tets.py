import random
import time

from scipy.interpolate import splprep
import numpy as np
from scipy.interpolate import splev
from Parameters import *
from itertools import combinations
from Lattice import Higgs_direction,read,write
from numba import njit,prange, jit#, roc
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.pyplot import cm
import pickle
from multiprocessing import Pool
from scipy.optimize import curve_fit
import matplotlib

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
    if n1n2**2 >= 1.0 or n1n3**2 >=1.0:
        arg = (n2n3 - n1n2 * n1n3)
        # print(n1);print(n2);print(n3)
        # print(n2n3);print(n1n2);print(n1n3);print((np.dot(n1,n2))**2);print((np.dot(n1,n3))**2);exit()
        # print('singularity warning in tri angle compute')
        ang=0.0
    else:
        #ang = np.arccos((np.dot(n2,n3)-np.dot(n1,n2)*np.dot(n1,n3))/np.sqrt( (1-(np.dot(n1,n2))**2)*(1-(np.dot(n1,n3))**2) ))
        arg = (n2n3 - n1n2 * n1n3) / np.sqrt((1 - (n1n2) ** 2) * (1 - (n1n3) ** 2))
        if 1-0.00001<abs(arg)<1+0.00001:##Needed this cause of some weird places where arccos argument ended up outside the range
            arg=np.sign(arg)*1.0
        ang = np.arccos(arg)
    # if np.isnan(ang):
        # print(n1,n2,n3)
        # print(((1 - (n1n2) ** 2) * (1 - (n1n3) ** 2)))
        # print(n1n2,n1n3)
        # print(arg);
        # exit()
    return ang

@jit()
def tri_area(n1,n2,n3):
    ar=tri_ang(n1,n2,n3)+ tri_ang(n2,n3,n1) + tri_ang(n3,n1,n2) - np.pi
    return ar

def mono_scatter_plot(i):
    tet_stack = read('tet_stack_%s'%i, Master_path)
    area_stack = read('area_stack_%s'%i,Master_path)
    tet_locs = read('tet_locs_%s'%i,Master_path)

    normed = np.real(area_stack / (4 * np.pi))
    mono_idx = np.argwhere(np.logical_and(normed>= 1-err_mar, normed<=1 + err_mar))[:,0]
    antimon_idx = np.argwhere(np.logical_and(normed<= -1+err_mar, normed>=-1-err_mar))[:,0]
    print(np.shape(mono_idx),np.shape(antimon_idx))
    #print(np.shape(tet_stack[mono_idx]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.mean(tet_locs[mono_idx,:,0],axis=1),np.mean(tet_locs[mono_idx,:,1],axis=1),np.mean(tet_locs[mono_idx,:,2],axis=1), marker='o',c='blue',linewidths=1)
    ax.scatter(np.mean(tet_locs[antimon_idx, :, 0],axis=1), np.mean(tet_locs[antimon_idx, :, 1],axis=1), np.mean(tet_locs[antimon_idx, :, 2],axis=1), marker='x',c='red',linewidths=1)
    plt.show()
    return

@jit()
def mtx_pd(A,B):
    pd_00 = A[0][0]*B[0][0] + A[0][1]*B[1][0]
    pd_01 = A[0][0]*B[0][1] + A[0][1]*B[1][1]
    pd_10 = A[1][0] * B[0][0] + A[1][1] * B[1][0]
    pd_11 = A[1][0] * B[0][1] + A[1][1] * B[1][1]
    mt_pd = np.array([[pd_00,pd_01],[pd_10,pd_11]])

    # result = np.array([[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*B)] for X_row in A])
    # if np.allclose(np.matmul(A,B),mt_pd, rtol=1e-05, atol=1e-08) == False:
    #     print('!!matrix product error!!')
    #     print('A')
    #     print(A);print('B');print(B)
    #     print('matmul');print(np.matmul(A,B))
    #     print('alg1');
    #     print(mt_pd)
    #     print('alg2');print(result)
    #     exit()
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
def strings(n1,n2,n3,phi1,phi2,phi3, I, sigma):

    delta = 0
    #for [ve1,ve2],[p1,p2] in zip(zip([n1,n2,n3],[n2,n3,n1]),zip([phi1,phi2,phi3],[phi2,phi3,phi1])):
    ve1 = n1;ve2 = n2
    p1 = phi1;p2 = phi2
    a21cs = np.cross(ve1,ve2)#;print(a21cs)
    #print(a21cs)
    #print(np.sqrt(a21cs[0] * a21cs[0] + a21cs[1] * a21cs[1] + a21cs[2] * a21cs[2]))
    a21cs_mag = np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])
    if a21cs_mag!=0.0:
        a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])#;print(a21)
    else:
        a21 = a21cs
        # a21 = [0.0,0.0,0.0]
    arg=ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2]
    if 1 - 0.00001 < abs(arg) < 1 + 0.00001: arg=1.0*np.sign(arg)
    # theta21 = np.arccos(ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2])
    theta21 = np.arccos(arg)
    S21r = -1j*(sigma[0]*a21[0] + sigma[1]*a21[1] + sigma[2]*a21[2])*np.sin(theta21/2) + I*np.cos(theta21/2)
    # if np.isnan(S21r[0][0]): print(theta21, a21); print(ve1);print(ve2);print(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]);exit()
    #p2_r = np.matmul(S21r,p1)#;print(p2_r)
    p2_r = [S21r[0][0]*p1[0] + S21r[0][1]*p1[1], S21r[1][0]*p1[0]+S21r[1][1]*p1[1]]#;print(p2_r)
    #delta21 = -1*np.angle(np.dot(np.conj(p2).T, p2_r));
    #print(np.dot(np.conj(p2).T, p2_r))
    delta21 = -1*(np.angle((np.real(p2[0])-np.imag(p2[0])*1j)*p2_r[0] + (np.real(p2[1])-np.imag(p2[1])*1j)*p2_r[1]))
    #ph_21 = np.dot(phi2, np.conj(p2_r).T)
    #print(delta21/np.pi)
    # if theta21 == 0.0: delta21 = 0.
    if theta21 == 0.0 and delta21!=0: print('2',theta21,delta21/(2*np.pi))
    ####Checked##Works#####
    delta = delta + delta21
    #print(delta)
    ve1 = n2;ve2 = n3
    p1 = phi2;p2 = phi3
    a21cs = np.cross(ve1,ve2)
    # a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])
    a21cs_mag = np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])
    if a21cs_mag!=0.0:
        a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])#;print(a21)
    else:
        a21 = a21cs
        # a21 = [0.0,0.0,0.0]
    # theta21 = np.arccos(ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2])
    arg=ve1[0] * ve2[0] + ve1[1] * ve2[1] + ve1[2] * ve2[2]
    if 1- 0.00001 < abs(arg) < 1 + 0.00001: arg=1.0*np.sign(arg)
    theta21=np.arccos(arg)
    S32r = -1j*(sigma[0]*a21[0] + sigma[1]*a21[1] + sigma[2]*a21[2])*np.sin(theta21/2) + I*np.cos(theta21/2)
    # if np.isnan(S32r[0][0]): print(theta21,a21); print(ve1);print(ve2);print(a21cs);print(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]);exit()
    #p2_r = np.matmul(S21r,p1)#;print(p2_r)
    p2_r = [S32r[0][0]*p1[0] + S32r[0][1]*p1[1], S32r[1][0]*p1[0]+S32r[1][1]*p1[1]]
    #delta21 = -1*np.angle(np.dot(np.conj(p2).T, p2_r));print(delta21);
    #print(np.dot(np.conj(p2).T, p2_r))
    delta21 = -1*(np.angle((np.real(p2[0]) - np.imag(p2[0])*1j) * p2_r[0] + (np.real(p2[1]) - np.imag(p2[1])*1j) * p2_r[1]))
    #ph_32 = np.dot(phi3, np.conj(p2_r).T)
    # if theta21 == 0.0: delta21 = 0.
    if theta21 == 0.0 and delta21!=0: print('2',theta21,delta21/(2*np.pi))
    ####Checked##Works#####
    delta = delta + delta21#;print(delta21/np.pi)
    #print(delta/np.pi)
    ve1 = n3;ve2 = n1
    p1 = phi3;p2 = phi1
    a21cs = np.cross(ve1,ve2)
    # a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])#np.dot(a21cs,a21cs))#;print(a21)
    a21cs_mag = np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])
    if a21cs_mag!=0.0:
        a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])#;print(a21)
    else:
        a21 = a21cs
        # a21 = [0.0,0.0,0.0]
    # theta21 = np.arccos(ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2])#np.dot(ve1,ve2))#;print(ve1,ve2);print(theta21)
    arg=ve1[0] * ve2[0] + ve1[1] * ve2[1] + ve1[2] * ve2[2]
    if 1- 0.00001 < abs(arg) < 1 + 0.00001: arg=1.0*np.sign(arg)
    theta21=np.arccos(arg)
    S13r = -1j*(sigma[0]*a21[0] + sigma[1]*a21[1] + sigma[2]*a21[2])*np.sin(theta21/2) + I*np.cos(theta21/2)#;print(S21r)
    # if np.isnan(S13r[0][0]): print(theta21, a21);  print(ve1);print(ve2);print(a21cs);print(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]);exit()
    #p2_r = np.matmul(S21r,p1)#;print(p2_r)
    p2_r = [S13r[0][0]*p1[0] + S13r[0][1]*p1[1], S13r[1][0]*p1[0]+S13r[1][1]*p1[1]]
    #print(np.dot(np.conj(p2).T, p2_r))
    #delta21 = -1*np.angle(np.dot(np.conj(p2).T, p2_r));print(delta21)
    delta21 = -1*(np.angle((np.real(p2[0]) - np.imag(p2[0])*1j) * p2_r[0] + (np.real(p2[1]) - np.imag(p2[1])*1j) * p2_r[1]))
    #ph_13 = np.dot(phi1, np.conj(p2_r).T)
    #print(delta21/np.pi)
    # if theta21 == 0.0: delta21=0.
    if theta21 == 0.0 and delta21!=0: print(S13r);print('2',theta21,delta21/(2*np.pi))
    ####Checked##Works#####
    delta = delta + delta21
    #print(delta/np.pi)
    #h_123 = np.angle(np.matmul(np.matmul(np.matmul(S13r,np.matmul(S32r,S21r)),phi1),np.conjugate(phi1).T))
    #print(np.matmul(np.matmul(np.matmul(S13r, np.matmul(S32r, S21r)), phi1), np.conjugate(phi1).T))
    #print(np.vdot(phi1, np.matmul(np.matmul(S13r,np.matmul(S32r,S21r)),phi1)));exit()
    h_123 = np.angle(vA_dag_vB(phi1, mtA_vB(mtx_pd(S13r,mtx_pd(S32r,S21r)),phi1)))
    # arg=vA_dag_vB(phi1, mtA_vB(mtx_pd(S13r, mtx_pd(S32r, S21r)), phi1))
    # if np.isnan(h_123): print(phi1);print(S13r,S32r,S21r);print(arg);exit()
    # A_test = np.matmul(np.matmul(S13r,np.matmul(S32r,S21r)),phi1)
    # B_test = mtA_vB(mtx_pd(S13r,mtx_pd(S32r,S21r)),phi1)
    # if np.allclose(A_test, B_test, rtol=1e-05, atol=1e-08) == False:
    #     print(A_test)
    #     print(B_test)
    #     exit()

    # phi1_rt = mtA_vB(mtx_pd(S13r,mtx_pd(S32r, S21r)), phi1)
    # phase = np.angle(vA_dag_vB(phi1_rt, phi1))
    # print((phase - h_123)/np.pi)
    # print((delta - h_123)/np.pi);print('')

    # if np.isnan(delta + h_123):
    #     print(delta,h_123)

    return delta + h_123


def n(phi,sigma):
    phi_conj = np.conjugate(phi).T
    n = -1 * np.array([np.matmul(phi_conj, np.matmul(sigma[0], phi))
              , np.matmul(phi_conj,np.matmul(sigma[1], phi))
              , np.matmul(phi_conj, np.matmul(sigma[2], phi))])

    return n/np.sqrt(np.dot(np.conjugate(phi),phi))



#mono_scatter_plot()
#@njit(parallel=True)
#@jit(forceobj=True)
@jit()
def tetrahedra_2(lat1, phi_arr, tets,area_arr,tet_locs,del_arr,tri_loc_arr, ID2, sigma):

    #tets = []
    #area_arr = []
    idx = 0
    ar_idx = 0
    tr_idx = 0

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



                #print(A,B,C,D,E,F)

                Al = [x,y,z]
                Bl = [x,y,z-1]
                Cl = [x-1,y-1,z-1]
                Dl = [x+1,y-1,z-1]
                El = [x+1,y+1,z-1]
                Fl = [x-1,y+1,z-1]
                Gl = [x-1,y-1,z+1]
                Hl = [x+1,y-1,z+1]
                Il = [x+1,y+1,z+1]
                Jl = [x-1,y+1,z+1]
                Kl = [x,y,z+1]
                Ml = [x+1,y,z]
                Nl = [x,y+1,z]
                Ol = [x-1,y,z]
                Ll = [x,y-1,z]

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
                ##Swapped Ks in 21 through 24#Check location consitency##
                tet1l = [Al, Bl, Cl, Dl]
                tet2l = [Al, Bl, Dl, El]
                tet3l = [Al, Bl, El, Fl]
                tet4l = [Al, Bl, Fl, Cl]
                tet5l = [Al, Cl, Ll, Dl]
                tet6l = [Al, Dl, Ll,Hl]
                tet7l = [Al, Hl, Ll,Gl]
                tet8l = [Al, Gl, Ll,Cl]
                tet9l = [Al, Dl, Ml, El]
                tet10l = [Al, El, Ml, Il]
                tet11l = [Al, Il, Ml, Hl]
                tet12l = [Al, Hl, Ml, Dl]
                tet13l = [Al, El, Nl,  Fl]
                tet14l = [Al, Fl, Nl,Jl]
                tet15l = [Al, Jl, Nl,Il]
                tet16l = [Al, Il, Nl,El]
                tet17l = [Al, Fl, Ol, Cl]
                tet18l = [Al, Cl, Ol, Gl]
                tet19l = [Al, Gl, Ol, Jl]
                tet20l = [Al, Jl, Ol, Fl]
                tet21l = [Al, Gl, Kl, Hl]
                tet22l = [Al, Hl, Kl, Il]
                tet23l = [Al, Il, Kl, Jl]
                tet24l = [Al, Jl, Kl, Gl]

                for j in prange(4):tets[idx,j] = tet1[j];tet_locs[idx,j] = tet1l[j]
                idx=idx+1
                for j in prange(4):tets[idx,j] = tet2[j];tet_locs[idx,j] = tet2l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet3[j];tet_locs[idx,j] = tet3l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet4[j];tet_locs[idx,j] = tet4l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet5[j];tet_locs[idx,j] = tet5l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet6[j];tet_locs[idx,j] = tet6l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet7[j];tet_locs[idx,j] = tet7l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet8[j];tet_locs[idx,j] = tet8l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet9[j];tet_locs[idx,j] = tet9l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet10[j];tet_locs[idx,j] = tet10l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet11[j];tet_locs[idx,j] = tet11l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet12[j];tet_locs[idx,j] = tet12l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet13[j];tet_locs[idx,j] = tet13l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet14[j];tet_locs[idx,j] = tet14l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet15[j];tet_locs[idx,j] = tet15l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet16[j];tet_locs[idx,j] = tet16l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet17[j];tet_locs[idx,j] = tet17l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet18[j];tet_locs[idx,j] = tet18l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet19[j];tet_locs[idx,j] = tet19l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet20[j];tet_locs[idx,j] = tet20l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet21[j];tet_locs[idx,j] = tet21l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet22[j];tet_locs[idx,j] = tet22l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet23[j];tet_locs[idx,j] = tet23l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet24[j];tet_locs[idx,j] = tet24l[j]
                idx = idx+1

                sub_tets = [tet1,tet2,tet3,tet4,
                            tet5,tet6,tet7,tet8,
                            tet9,tet10,tet11,tet12,
                            tet13,tet14,tet15,tet16,
                            tet17,tet18,tet19,tet20,
                            tet21,tet22,tet23,tet24]

                sub_tet_locs = [tet1l,tet2l,tet3l,tet4l,
                            tet5l,tet6l,tet7l,tet8l,
                            tet9l,tet10l,tet11l,tet12l,
                            tet13l,tet14l,tet15l,tet16l,
                            tet17l,tet18l,tet19l,tet20l,
                            tet21l,tet22l,tet23l,tet24l]

                for a,a_locs in zip(sub_tets,sub_tet_locs):

                    triangles = [[a[0],a[1],a[2]],[a[0],a[2],a[3]],[a[0],a[3],a[1]],[a[1],a[3],a[2]]]
                    tri_locs = [[a_locs[0],a_locs[1],a_locs[2]],
                                [a_locs[0],a_locs[2],a_locs[3]],
                                [a_locs[0],a_locs[3],a_locs[1]],
                                [a_locs[1],a_locs[3],a_locs[2]]]

                    sum_area = 0

                    for triangle,tr_loc in zip(triangles,tri_locs):

                        n1 = triangle[0];n2=triangle[1];n3=triangle[2]
                        area_sign = np.sign(np.dot(np.cross(n1,n2),n3))
                        Area = tri_area(n1,n2,n3)
                        sum_area = sum_area + area_sign*Area

                        phi_1 = phi_arr[tr_loc[0][0],tr_loc[0][1],tr_loc[0][2]]
                        phi_2 = phi_arr[tr_loc[1][0],tr_loc[1][1],tr_loc[1][2]]
                        phi_3 = phi_arr[tr_loc[2][0],tr_loc[2][1],tr_loc[2][2]]

                        delta = strings(n1,n2,n3,
                                         phi_1,phi_2,phi_3,ID2,sigma)

                        del_arr[tr_idx] = delta
                        for o in prange(3):
                            tri_loc_arr[tr_idx,o] = tr_loc[o]

                        tr_idx = tr_idx + 1

                    area_arr[ar_idx] = sum_area

                    # if sum_area/(4*np.pi) > 1 - err_mar:
                    #     dels = np.array([del_arr[i]/np.pi for i in range(tr_idx - 4, tr_idx)])
                    #     str_id = np.argwhere(dels>= 2-err_mar)
                    #     asr_id = np.argwhere(dels<= -2+err_mar)
                    #     #print(ar_idx);print(tr_idx-4+str_id);print(tr_idx-4+asr_id)
                    #     # if len(str_id)==0:
                    #     #     print('monopole but no string')
                    #     #     print(dels)
                    #     #     print(sum_area)
                    #     #     print(list(range(tr_idx - 4, tr_idx)))
                    #     #     print(ar_idx)
                    #     #     #exit()
                    # if sum_area/(4*np.pi) < - 1 + err_mar:
                    #     dels = np.array([del_arr[i]/np.pi for i in range(tr_idx - 4, tr_idx)])
                    #     str_id = np.argwhere(dels>= 2-err_mar)
                    #     asr_id = np.argwhere(dels<= -2+err_mar)
                    #     #print(ar_idx);print(tr_idx-4+str_id);print(tr_idx-4+asr_id)
                    #     # if len(asr_id)==0:
                    #     #     print('anti monopole but no string')
                    #     #     print(dels)
                    #     #     print(sum_area)
                    #     #     print(list(range(tr_idx - 4, tr_idx)))
                    #     #     print(ar_idx)
                    #     #     #exit()

                    ar_idx=ar_idx+1

    return tets,area_arr,tet_locs,del_arr,tri_loc_arr





def run_tets2(i):
    lat1 = np.zeros((N, N, N))
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
    n = np.zeros((N, N, N, 3), dtype=complex)

    # phi_list = ['phi_arr_%s'%s for s in range(N_runs)]

    tets = np.zeros((int(((N-1)/2) ** 3 * 24), 4,3), dtype=complex)
    #tets = np.zeros((int((N - 1) ** 3 * 12), 4, 2), dtype=complex)
    area_arr = np.zeros((int(((N-1)/2) ** 3 * 24)), dtype=complex)
    tet_loc_arr = np.zeros((int(((N-1)/2) ** 3 * 24), 4,3),dtype=int)
    del_arr = np.zeros((int(((N-1)/2) ** 3 * 24 * 4)),dtype=complex)
    tri_loc_arr = np.zeros((int(((N-1)/2) ** 3 * 24 * 4), 3, 3))

    phi_1 = read('phi_arr_%s'%i,Master_path)
    n_1 = np.real(Higgs_direction(phi_1,n,sigma))
    # Hoft_vals = read('hoft_arr_%s'%i,Master_path)

    #print(np.shape(n_1))
    ID2 = np.array([[1, 0], [0, 1]],dtype=complex)
    sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)

    tet_stack, area_stack,tet_locs,deltas,tri_locs = tetrahedra_2(n_1, phi_1, tets,area_arr,tet_loc_arr,del_arr,tri_loc_arr,ID2,sigma)

    normed = np.real(area_stack / (4 * np.pi))
    if np.isnan(normed).any()==True:
        pb_idx = np.argwhere(np.isnan(normed) == True)[0][0]
        print(tet_stack[pb_idx])
        print(tet_locs[pb_idx][3])
        print(n_1[tet_locs[pb_idx][0][0],tet_locs[pb_idx][0][1],tet_locs[pb_idx][0][2]])
        print(n_1[tet_locs[pb_idx][1][0], tet_locs[pb_idx][1][1], tet_locs[pb_idx][1][2]])
        print(n_1[tet_locs[pb_idx][2][0], tet_locs[pb_idx][2][1], tet_locs[pb_idx][2][2]])
        print(n_1[tet_locs[pb_idx][3][0], tet_locs[pb_idx][3][1], tet_locs[pb_idx][3][2]])
        print(phi_1[tet_locs[pb_idx][3][0], tet_locs[pb_idx][3][1], tet_locs[pb_idx][3][2]])
        # print(Hoft_vals[tet_locs[pb_idx][3][0], tet_locs[pb_idx][3][1], tet_locs[pb_idx][3][2]])
        print('!!!Singularity or coordinate issues!!!')
        exit()

    #print('# of monopoles:', np.shape(np.where(np.logical_and(normed> 1-err_mar, normed<1 + err_mar))))
    #print('# of antimonopoles:', np.shape(np.where(np.logical_and(normed < -1+err_mar, normed>-1-err_mar))))
    print('# of monopoles:', np.shape(np.where(normed > 1 - err_mar)))
    print('# of antimonopoles:', np.shape(np.where(normed < -1 + err_mar)))
    print('zero sum', np.shape(np.where(np.logical_and(normed>= -1*err_mar, normed<=err_mar))))
    print('# of strings', np.shape(np.where(
        np.logical_and(np.abs(np.real(deltas/(np.pi)))<= 2+err_mar, np.abs(np.real(deltas/(np.pi)))>=2-err_mar)
    )))
    #plt.hist(np.abs(np.real(deltas/(np.pi))),bins=100);plt.show()#;exit()
    print('------!!!DISPARITIES!!!---------')
    print('');print('');
    print(np.where(np.logical_and(normed<1-err_mar,normed>err_mar))[0])
    print(np.where(np.logical_and(normed>-1+err_mar,normed<-1*err_mar)))
    print(np.where(np.logical_and(normed<-1-err_mar,normed>1+err_mar))[0])
    # print(len(np.where(np.logical_and(normed > 1 + err_mar, normed > - 1 - err_mar))[0]))
    # print(len(np.where(normed>err_mar)[0]))
    # print(len(np.where(normed<-1*err_mar)[0]))
    print('');print('');
    print('---------------------------------')
    print(np.shape(area_stack))
    write('tet_stack_%s'%i, Master_path, tet_stack)
    write('area_stack_%s'%i,Master_path, area_stack)
    write('tet_locs_%s' % i, Master_path, tet_locs)
    write('deltas_%s'%i, Master_path,deltas)
    write('tri_locs_%s'%i, Master_path, tri_locs)

    return [np.shape(np.where(normed > 1 - err_mar)), np.shape(np.where(normed < -1 + err_mar))]



sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
n = np.zeros((N, N, N, 3), dtype=complex)

pool = Pool(N_cpu)
#pool.map(run_tets2, range(N_runs))
pool.close()

#run_tets2(sigma,n)
#exit()

# @jit(forceobj=True)
def intersection(A,B):
    lis = []
    for i in B:
        mtch = np.argwhere(A==i)
        if len(mtch)!=0:
            lis.append(A[mtch[0,0]])

    return lis

# @jit(forceobj=True)
def ref_plk(a,b,c):
    if a==0.0:
        ref_x = float(N-1)
        return [ref_x, b, c]
    if b==0.0:
        ref_y = float(N-1)
        return [a, ref_y, c]
    if c==0.0:
        ref_z = float(N-1)
        return [a, b, ref_z]
    if a==float(N-1):
        ref_x = 0.
        return [ref_x, b, c]
    if b==float(N-1):
        ref_y = 0.
        return [a, ref_y, c]
    if c==float(N-1):
        ref_z = 0.
        return [a, b, ref_z]
    else:
        #print('not at the boundary')
        return [0,0,0]

# @jit(forceobj=True)
def matching(plk, astr_locs, antimon_idx, astring_idxs):
    xmtch = np.argwhere(astr_locs[:, 0] == plk[0])[:, 0]
    ymtch = np.argwhere(astr_locs[:, 1] == plk[1])[:, 0]
    zmtch = np.argwhere(astr_locs[:, 2] == plk[2])[:, 0]
    mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
    # mtch = intersection(xmtch,intersection(ymtch,zmtch))
    mtch = astring_idxs[mtch]
    #print(plk);print(mtch)
    # if len(mtch) > 1:
    #     #print('!!!more than 1 antistring match per plaqutte!!!')
    #     #exit()
    if len(mtch)==0:
        a=plk[0];b=plk[1];c=plk[2]
        refpk = ref_plk(a,b,c)
        #print('reflected plk')
        #print(refpk)
        if refpk==[0,0,0]:
            ##CODE 3 is no match found even across boundary##
            print('not at the boundary')
            out = [[3], []]
        else:
            xref = np.argwhere(astr_locs[:, 0] == refpk[0])[:, 0]
            yref = np.argwhere(astr_locs[:, 1] == refpk[1])[:, 0]
            zref = np.argwhere(astr_locs[:, 2] == refpk[2])[:, 0]
            ref_mtch = np.intersect1d(xref, np.intersect1d(yref, zref))
            # ref_mtch = intersection(xref,intersection(yref,zref))
            #print(ref_mtch)
            if len(ref_mtch)==0:
                print('antistring across boundary not found')
                out=[[3],[]]
            else:
                out = [[4],refpk,ref_mtch[0]]
        #print('-------')
    if len(mtch) == 1:
        mtch = mtch[0]
        amn_intp = int(mtch / 4)
        amono_mtch = np.intersect1d(antimon_idx, [amn_intp])
        # amono_mtch = intersection(antimon_idx,[amn_intp])
        if len(amono_mtch) == 1:
            #print(amono_mtch)
            ##CODE 1 means the output is the location of the antimonopole##
            code=1
            out = [[code],[amono_mtch],[mtch]]

        if len(amono_mtch) == 0:
            ##No antimonopole in adjacent cell##
            atet_idx = int(mtch / 4)
            ##CODE 2 implies output are the 4 plaquettes of the adjacent tet##
            code =2
            out = [[code], [atet_idx], [mtch]]

    return out

#@jit(forceobj=True)
def loop_matching(plk, astr_locs, astring_idxs):
    xmtch = np.argwhere(astr_locs[:, 0] == plk[0])[:, 0]
    ymtch = np.argwhere(astr_locs[:, 1] == plk[1])[:, 0]
    zmtch = np.argwhere(astr_locs[:, 2] == plk[2])[:, 0]
    #mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
    mtch = intersection(xmtch, intersection(ymtch, zmtch))
    mtch = astring_idxs[mtch]

    if len(mtch)==0:
        a=plk[0];b=plk[1];c=plk[2]
        refpk = ref_plk(a,b,c)
        #print('reflected plk')
        #print(refpk)
        if refpk==[0,0,0]:
            ##CODE 3 is no match found even across boundary##
            out = [[3], [plk]]
        else:
            xref = np.argwhere(astr_locs[:, 0] == refpk[0])[:, 0]
            yref = np.argwhere(astr_locs[:, 1] == refpk[1])[:, 0]
            zref = np.argwhere(astr_locs[:, 2] == refpk[2])[:, 0]
            #ref_mtch = np.intersect1d(xref, np.intersect1d(yref, zref))
            ref_mtch = intersection(xref, intersection(yref, zref))
            #print(ref_mtch)
            if len(ref_mtch)==0:
                out=[[3],[plk]]
            else:
                out = [[4],refpk,ref_mtch[0]]
        #print('-------')
    if len(mtch) == 1:
        mtch = mtch[0]
        ##No antimonopole in adjacent cell##
        atet_idx = int(mtch / 4)
        ##CODE 2 implies output are the 4 plaquettes of the adjacent tet##
        code =2
        out = [[code], [atet_idx], [mtch]]

    return out

#@jit(forceobj=True)
def tracing(i):
    area_stack = read('area_stack_%s' % i, Master_path)
    tet_locs = read('tet_locs_%s' % i, Master_path)
    deltas = read('deltas_%s' % i, Master_path)
    tri_locs = read('tri_locs_%s' % i, Master_path)
    tet_stk = read('tet_stack_%s' % i, Master_path)
    # Hoft_angs = read('hoft_arr_%s' % i, Master_path)

    string_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= 2 + err_mar, (np.real(deltas / (np.pi))) >= 2 - err_mar)
    )[:, 0]

    astring_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= -2 + err_mar, (np.real(deltas / (np.pi))) >= -2 - err_mar)
    )[:, 0]

    normed = np.real(area_stack / (4 * np.pi))
    mono_idx = np.argwhere(np.logical_and(normed >= 1 - err_mar, normed <= 1 + err_mar))[:, 0]
    antimon_idx = np.argwhere(np.logical_and(normed <= -1 + err_mar, normed >= -1 - err_mar))[:, 0]
    print('#Monopoles:%s'%(np.shape(mono_idx)))
    print('#Antimonopoles:%s'%(np.shape(antimon_idx)))
    pair_cnt = 0

    str_locs = np.mean(tri_locs[string_idxs], axis=1)
    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
    no_match_cnt = 0
    match_cnt = 0
    string_lists = []

    for mn_idx in mono_idx:
        pair_idxs = []
        #str_match = np.intersect1d(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        str_match = intersection(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        plk = np.mean(tri_locs[str_match[0]], axis=0)
        pair_idxs.append(np.mean(tet_locs[mn_idx],axis=0))
        mono_idx = np.delete(mono_idx, np.argwhere(mono_idx == mn_idx)[0, 0])
        if len(str_match)>2: print('huh');exit()
        #print(mn_idx)
        # if len(str_match)==0:
        #     print('!!No outgoing string from monopole!!')
        #     exit()
        # if len(str_match)>2:
        #     print('wut')
        if len(str_match) == 2:
            choice = random.choice(str_match)
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
            str_plq = np.mean(tri_locs[choice], axis=0)
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            pair_idxs.append(str_plq)
            if res[0][0] == 1:
                ##Antimonopole found##
                pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx,np.argwhere(antimon_idx==res[1][0])[0,0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                #print('Pair antimonopole found')
                match_cnt = match_cnt + 1
            while res[0][0] != 1:
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                if res[0][0] == 2:
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                (4 * res[1][0] + 3)]
                    #nxt_strs = np.intersect1d(string_idxs, nxt_plqs)
                    nxt_strs = intersection(string_idxs,nxt_plqs)
                    if len(nxt_strs)>2: print('huh');exit()
                    if len(nxt_strs) == 1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        tet_cen = np.mean(tet_locs[res[1][0]],axis=0)
                        pair_idxs.append(tet_cen)
                        pair_idxs.append(str_plq)
                        #print(str_plq)
                    if len(nxt_strs) == 0:
                        #print('no outgoing string in the next tet')
                        break

                    if len(nxt_strs) == 2:
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        tet_cen = np.mean(tet_locs[res[1][0]], axis=0)
                        pair_idxs.append(tet_cen)
                        pair_idxs.append(str_plq)
                    # continue

                if res[0][0] == 4:
                    str_plq = res[1]
                    pair_idxs.append(str_plq)

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break
                if res[0][0] == 1:
                    ##Antimonopole found##
                    pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    #print('Pair antimonopole found')
                    match_cnt = match_cnt + 1
                    break


        if len(str_match) == 1:
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs==str_match)[0,0])
            str_plq = np.mean(tri_locs[str_match[0]], axis=0)
            res = matching(str_plq, astr_locs, antimon_idx,astring_idxs)
            pair_idxs.append(str_plq)
            if res[0][0] == 1:
                ##Antimonopole found##
                pair_idxs.append(np.mean(tet_locs[res[1][0]][0],axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                #print('Pair antimonopole found')
                match_cnt = match_cnt+1
            while res[0][0] != 1:
                res = matching(str_plq, astr_locs, antimon_idx,astring_idxs)
                #print(res);print('')
                if res[0][0] == 4:
                    str_plq = res[1]
                    pair_idxs.append(str_plq)

                if res[0][0] == 2:
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4*res[1][0]),int(4*res[1][0]+1),(4*res[1][0]+2),(4*res[1][0]+3)]
                    #nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    nxt_strs = intersection(string_idxs, nxt_plqs)
                    if len(nxt_strs) > 2: print('huh');exit()
                    if len(nxt_strs)==1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        tet_cen = np.mean(tet_locs[res[1][0]], axis=0)
                        pair_idxs.append(tet_cen)
                        pair_idxs.append(str_plq)#;print(str_plq)
                    if len(nxt_strs) == 0:
                        #print('no outgoing string in the next tet')
                        break

                    if len(nxt_strs) == 2:
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        tet_cen = np.mean(tet_locs[res[1][0]], axis=0)
                        pair_idxs.append(tet_cen)
                        pair_idxs.append(str_plq)

                if res[0][0] == 1:
                    ##Antimonopole found##
                    pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    #print('Pair antimonopole found')
                    match_cnt = match_cnt + 1
                    break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break
        #exit()
        string_lists.append(pair_idxs)
    #print('')

    print('Remainder string/antistring locs:%s,%s'%(len(string_idxs),len(astring_idxs)))
    print('Remainder monopole/antimonopole locs:%s, %s'%(len(mono_idx),len(antimon_idx)))

    print('Antimonopole matches:%s'%match_cnt)
    write('string_lists_%s'%i,Master_path,string_lists)
    write('upd_str_idxs_%s' % i, Master_path, string_idxs)
    write('upd_astr_idxs_%s' % i, Master_path, astring_idxs)
    write('upd_mono_idxs_%s' % i, Master_path, mono_idx)
    write('upd_antimon_idxs_%s' % i, Master_path, antimon_idx)
    write('pair_list_%s'%i,Master_path,string_lists)

    return

def tracing_test(i):
    area_stack = read('area_stack_%s' % i, Master_path)
    tet_locs = read('tet_locs_%s' % i, Master_path)
    deltas = read('deltas_%s' % i, Master_path)
    tri_locs = read('tri_locs_%s' % i, Master_path)
    # tet_stk = read('tet_stack_%s' % i, Master_path)
    # Hoft_angs = read('hoft_arr_%s' % i, Master_path)
    tri_coords = np.mean(tri_locs, axis=1)
    tet_coords = np.mean(tet_locs,axis=1)
    string_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= 2 + err_mar, (np.real(deltas / (np.pi))) >= 2 - err_mar)
    )[:, 0]

    astring_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= -2 + err_mar, (np.real(deltas / (np.pi))) >= -2 - err_mar)
    )[:, 0]

    normed = np.real(area_stack / (4 * np.pi))
    mono_idx = np.argwhere(np.logical_and(normed >= 1 - err_mar, normed <= 1 + err_mar))[:, 0]
    antimon_idx = np.argwhere(np.logical_and(normed <= -1 + err_mar, normed >= -1 - err_mar))[:, 0]
    print('#Monopoles:%s'%(np.shape(mono_idx)))
    print('#Antimonopoles:%s'%(np.shape(antimon_idx)))
    pair_cnt = 0

    # str_locs = np.mean(tri_locs[string_idxs], axis=1)
    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
    # no_match_cnt = 0
    match_cnt = 0
    string_lists = []

    for mn_idx in mono_idx:
        pair_idxs = []
        str_match = np.intersect1d(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        # str_match = intersection(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        # plk = np.mean(tri_locs[str_match[0]], axis=0)
        # pair_idxs.append(np.mean(tet_locs[mn_idx],axis=0))
        pair_idxs.append(tet_coords[mn_idx])
        mono_idx = np.delete(mono_idx, np.argwhere(mono_idx == mn_idx)[0, 0])
        if len(str_match)>2: print('huh');exit()
        #print(mn_idx)
        # if len(str_match)==0:
        #     print('!!No outgoing string from monopole!!')
        #     exit()
        # if len(str_match)>2:
        #     print('wut')
        if len(str_match) == 2:
            choice = random.choice(str_match)
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
            # str_plq = np.mean(tri_locs[choice], axis=0)
            str_plq = tri_coords[choice]
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            pair_idxs.append(str_plq)
        if len(str_match) == 1:
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == str_match)[0, 0])
            # str_plq = np.mean(tri_locs[str_match[0]], axis=0)
            str_plq = tri_coords[str_match[0]]
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            pair_idxs.append(str_plq)
        if res[0][0] == 1:
            ##Antimonopole found##
            # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
            pair_idxs.append(tet_coords[res[1][0]][0])
            antimon_idx = np.delete(antimon_idx,np.argwhere(antimon_idx==res[1][0])[0,0])
            astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
            astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
            #print('Pair antimonopole found')
            match_cnt = match_cnt + 1
        while res[0][0] != 1:
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            if res[0][0] == 2:
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                            (4 * res[1][0] + 3)]
                nxt_strs = np.intersect1d(string_idxs, nxt_plqs)
                # nxt_strs = intersection(string_idxs,nxt_plqs)
                if len(nxt_strs)>2: print('huh');exit()
                if len(nxt_strs) == 1:
                    string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                    # str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                    # tet_cen = np.mean(tet_locs[res[1][0]],axis=0)
                    str_plq = tri_coords[nxt_strs[0]]
                    tet_cen = tet_coords[res[1][0]]
                    pair_idxs.append(tet_cen)
                    pair_idxs.append(str_plq)
                    #print(str_plq)
                if len(nxt_strs) == 0:
                    #print('no outgoing string in the next tet')
                    break

                if len(nxt_strs) == 2:
                    nxt_choice = random.choice(nxt_strs)
                    string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                    # str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                    # tet_cen = np.mean(tet_locs[res[1][0]], axis=0)
                    str_plq=tri_coords[nxt_choice]
                    tet_cen = tet_coords[res[1][0]]
                    pair_idxs.append(tet_cen)
                    pair_idxs.append(str_plq)
                # continue

            if res[0][0] == 4:
                str_plq = res[1]
                pair_idxs.append(str_plq)

            if res[0][0] == 3:
                print('----')
                print('no match found?')
                print(res)
                print(str_plq)
                print('----')
                break
            if res[0][0] == 1:
                ##Antimonopole found##
                pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                #print('Pair antimonopole found')
                match_cnt = match_cnt + 1
                break
        #exit()
        string_lists.append(pair_idxs)
    #print('')

    print('Remainder string/antistring locs:%s,%s'%(len(string_idxs),len(astring_idxs)))
    print('Remainder monopole/antimonopole locs:%s, %s'%(len(mono_idx),len(antimon_idx)))

    print('Antimonopole matches:%s'%match_cnt)
    write('string_lists_%s'%i,Master_path,string_lists)
    write('upd_str_idxs_%s' % i, Master_path, string_idxs)
    write('upd_astr_idxs_%s' % i, Master_path, astring_idxs)
    write('upd_mono_idxs_%s' % i, Master_path, mono_idx)
    write('upd_antimon_idxs_%s' % i, Master_path, antimon_idx)
    write('pair_list_%s'%i,Master_path,string_lists)

    return

# start = time.time()
# pool = Pool(N_cpu)
#pool.map(tracing, range(N_runs))
# pool.close()
# print('Time for tracing:%s'%(abs(start-time.time())))

# print('Pairing done')
def looping(i):
    area_stack = read('area_stack_%s' % i, Master_path)
    tet_locs = read('tet_locs_%s' % i, Master_path)
    deltas = read('deltas_%s' % i, Master_path)
    tri_locs = read('tri_locs_%s' % i, Master_path)
    tet_stk = read('tet_stack_%s' % i, Master_path)
    # Hoft_angs = read('hoft_arr_%s' % i, Master_path)
    string_idxs = read('upd_str_idxs_%s' % i, Master_path)
    astring_idxs = read('upd_astr_idxs_%s' % i, Master_path)
    mono_idx = read('upd_mono_idxs_%s' % i, Master_path)
    antimon_idx = read('upd_antimon_idxs_%s' % i, Master_path)
    str_locs = np.mean(tri_locs[string_idxs], axis=1)
    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
    string_list = list(read('string_lists_%s' % i, Master_path))
    loop_list = []

    for str in string_idxs:
        pair_idxs = []
        plk = np.mean(tri_locs[str], axis=0)
        res = loop_matching(plk,astr_locs,astring_idxs)
        pair_idxs.append(plk)
        while res[0][0] != 1:
            res = loop_matching(plk, astr_locs, astring_idxs)
            if res[0][0] == 2:
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                            (4 * res[1][0] + 3)]
                #nxt_strs = np.intersect1d(string_idxs, nxt_plqs)
                nxt_strs = intersection(string_idxs, nxt_plqs)
                astr_tet_mtch =res[1][0]
                if len(nxt_strs) == 1:
                    string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                    plk = np.mean(tri_locs[nxt_strs[0]], axis=0)
                    tet_cen = np.mean(tet_locs[res[1][0]], axis=0)
                    pair_idxs.append(tet_cen)
                    pair_idxs.append(plk)
                    # print(str_plq)
                if len(nxt_strs) == 0:
                    pair_idxs.append(plk)
                    #print('no outgoing string in the next tet')
                    # if astr_tet_mtch:
                    print('yes')
                    break
                # continue

            if res[0][0] == 4:
                ###Need for non periodic boundaries##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]], axis=0))
                # pair_idxs.append(np.mean(tri_locs[res[2][0]], axis=0))
                # pair_idxs.append(plk)
                ##----------------------------------------------------##
                plk = res[1]
                pair_idxs.append(plk)

            if res[0][0] == 3:
                ###Need for non periodic boundaries##
                # pair_idxs.append(res[1][0])

                xmtch = np.argwhere(np.mean(tri_locs,axis=1)[:,0] == res[1][0][0])[:, 0]
                ymtch = np.argwhere(np.mean(tri_locs,axis=1)[:,1] == res[1][0][1])[:, 0]
                zmtch = np.argwhere(np.mean(tri_locs,axis=1)[:,2] == res[1][0][2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))[0]
                # print(mtch)
                # print(int(mtch/4))
                last_plks=[int(mtch/4),int(mtch/4)+1,int(mtch/4)+2,int(mtch/4)+3]
                # print(intersection(astring_idxs,last_plks))
                last_mtch = intersection(astring_idxs,last_plks)
                if len(last_mtch)!=0:
                    last_plk=np.mean(tri_locs[last_mtch[0]], axis=0)
                    pair_idxs.append(last_plk)
                break
            #if res[0][0] == 1:
                ##Antimonopole found##
                #pair_idxs.append(np.mean(tet_locs[res[1][0]]q[0], axis=0))
                #antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                #print('Pair antimonopole found in loops??')
                #break
        string_list.append(pair_idxs)
        # i+=1;print(i)
        loop_list.append(pair_idxs)
    print('Loop dims',len(loop_list))
    # print('loop_lists_%s'%i)
    print('Remainder string/antistring locs:%s,%s'%(len(string_idxs),len(astring_idxs)))
    print('Remainder monopole/antimonopole locs:%s, %s'%(len(mono_idx),len(antimon_idx)))
    write('string_lists_%s' % i, Master_path,string_list)
    write('upd_str_idxs_%s' % i, Master_path, string_idxs)
    write('upd_astr_idxs_%s' % i, Master_path, astring_idxs)
    write('upd_mono_idxs_%s' % i, Master_path, mono_idx)
    write('upd_antimon_idxs_%s' % i, Master_path, antimon_idx)
    write('loop_lists_%s'%i,Master_path,loop_list)
    return len(loop_list)

# start=time.time()
# pool = Pool(N_cpu)
# #pool.map(looping, range(N_runs))
# pool.close()
# print('Time for loops:%s'%(abs(start-time.time())))


def tet_2_plot(i):
    #tet_stack = read('tet_stack_%s'%i, Master_path)
    area_stack = read('area_stack_%s'%i,Master_path)
    tet_locs = read('tet_locs_%s'%i,Master_path)
    deltas = read('deltas_%s'%i,Master_path)
    tri_locs = read('tri_locs_%s'%i, Master_path)
    string_list = read('string_lists_%s'%i,Master_path)
    # Hoft_angs = read('hoft_arr_%s' % i, Master_path)
    # loop_list = read('loop_lists_%s'%i,Master_path)

    # tet_161 = tet_locs[161]
    # print(tet_161)
    # print(deltas[int(4*161)])
    # print(deltas[int(4*161)+1])
    # print(deltas[int(4*161)+2])
    # print(deltas[int(4*161)+3])
    #
    # tet_2 = tet_locs[2]
    # print(tet_2)
    # print(deltas[int(2/4)])
    # print(deltas[int(2/4) + 1])
    # print(deltas[int(2/4) + 2])
    # print(deltas[int(2/4) + 3])
    #
    #exit()
    normed = np.real(area_stack / (4 * np.pi))
    #print('# of monopoles:', np.shape(np.where(np.logical_and(normed> 1-err_mar, normed<1 + err_mar))))
    #print('# of antimonopoles:', np.shape(np.where(np.logical_and(normed < -1+err_mar, normed>-1-err_mar))))
    print('# of monopoles:', np.shape(np.where(normed > 1 - err_mar)))
    print('# of antimonopoles:', np.shape(np.where(normed < -1 + err_mar)))
    string_idxs = np.where(
        np.logical_and((np.real(deltas/(np.pi)))<= 2+err_mar, (np.real(deltas/(np.pi)))>=2-err_mar)
    )
    astring_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= -2 + err_mar, (np.real(deltas / (np.pi))) >= -2 - err_mar)
    )[:, 0]

    print('# of strings', np.shape(string_list))
    # print('# of loops', np.shape(loop_list))
    # plt.hist(np.real(deltas/(np.pi)),bins=20)
    # plt.xlabel(r'$\delta/\pi$')
    # plt.show()

    mono_idx = np.argwhere(np.logical_and(normed>= 1-err_mar, normed<=1 + err_mar))[:,0]
    antimon_idx = np.argwhere(np.logical_and(normed<= -1+err_mar, normed>=-1-err_mar))[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)
    #fig.set_facecolor('black')
    #ax.set_facecolor('black')

    #ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    #ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    #ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    #ax.xaxis.pane.set_edgecolor('white')
    #ax.yaxis.pane.set_edgecolor('white')

    if N > 20:
        marker_size = .01
    else:
        marker_size = 10

    ax.scatter(np.mean(tet_locs[mono_idx,:,0],axis=1),np.mean(tet_locs[mono_idx,:,1],axis=1),np.mean(tet_locs[mono_idx,:,2],axis=1), marker='o',c='blue',linewidths=1,s=marker_size)
    ax.scatter(np.mean(tet_locs[antimon_idx, :, 0],axis=1), np.mean(tet_locs[antimon_idx, :, 1],axis=1), np.mean(tet_locs[antimon_idx, :, 2],axis=1), marker='o',c='red',linewidths=1,s=marker_size)
    # ax.scatter(np.mean(tri_locs[string_idxs,:,0],axis=2),np.mean(tri_locs[string_idxs,:,1],axis=2),np.mean(tri_locs[string_idxs,:,2],axis=2),marker='*',c='green',linewidth=1,s=marker_size)
    # ax.scatter(np.mean(tri_locs[astring_idxs, :, 0], axis=1), np.mean(tri_locs[astring_idxs, :, 1], axis=1),
    #            np.mean(tri_locs[astring_idxs, :, 2], axis=1), marker='^', c='red', linewidth=1, s=marker_size/2)
    # ax.scatter(np.mean(tet_locs[:, :, 0], axis=1), np.mean(tet_locs[:, :, 1], axis=1),
    #           np.mean(tet_locs[:, :, 2], axis=1), marker='D', c='grey', linewidths=1, s=marker_size,alpha=.5)
    # ax.scatter(np.mean(tri_locs[:, :, 0], axis=1), np.mean(tri_locs[:, :, 1], axis=1),
    #            np.mean(tri_locs[:, :, 2], axis=1), marker='o', c='grey', linewidth=1, s=marker_size,alpha=.2)

    colors = cm.brg(np.linspace(0,1,len(string_list)))
    # colors = cm.brg(np.linspace(0, 1, len(loop_list)))
    width = .6
    for str_id, string in enumerate(string_list):
    # print(loop_list[1])
    # for str_id, string in enumerate(loop_list):
        string = np.array(string)
        #print(np.argwhere(np.linalg.norm(np.diff(string,axis=0),axis=1)==4.))
        disc = np.argwhere(np.linalg.norm(np.diff(string,axis=0),axis=1)>3)

        #cnt_ls = []
        if len(disc)==0:
            ax.plot3D(string[:, 0], string[:, 1], string[:, 2], c=colors[str_id],linewidth=width)
        if len(disc)==1:
            ax.plot3D(string[:disc[0][0]+1, 0], string[:disc[0][0]+1, 1], string[:disc[0][0]+1, 2], c=colors[str_id],linewidth=width)
            ax.plot3D(string[disc[0][0]+1:, 0], string[disc[0][0]+1:, 1], string[disc[0][0]+1:, 2], c=colors[str_id],linewidth=width)
        if len(disc)>1:
            start = 0
            #print(disc)
            for idx in range(len(disc)):
                #print(disc[idx]);print(string[start:disc[idx][0]+1])
                ax.plot3D(string[start:disc[idx][0]+1, 0], string[start:disc[idx][0]+1, 1], string[start:disc[idx][0]+1, 2],
                          c=colors[str_id],linewidth=width)
                start = disc[idx][0]+1
            ax.plot3D(string[start:, 0], string[start:, 1], string[start:, 2],
                      c=colors[str_id],linewidth=width)
        #ax.plot3D(string[:, 0], string[:, 1], string[:, 2], c=colors[str_id])
        ##Smooting curve##
        # tck, u = splprep([string[:,0],string[:,1],string[:,2]], s=200,k=1)
        # x_knots, y_knots, z_knots = splev(tck[0], tck)
        # num_true_pts = len(string_list)
        # u_fine = np.linspace(0, 1, num_true_pts)
        # x_fine, y_fine, z_fine = splev(u_fine, tck)
        #ax.plot3D(x_fine,y_fine,z_fine)

    ###Remainder string idxs###
    string_idxs = read('upd_str_idxs_%s' % i, Master_path)
    astring_idxs = read('upd_astr_idxs_%s' % i, Master_path)
    mono_idx = read('upd_mono_idxs_%s' % i, Master_path)
    antimon_idx = read('upd_antimon_idxs_%s' % i, Master_path)

    # for str in string_idxs:
    #     plk = tri_locs[str]
    #     tl = np.mean(tet_locs[int(str/4)],axis=0)
    #     s1 = plk[1]-plk[0]
    #     s2 = plk[1] - plk[2]
    #     tang = np.cross(s1,s2)
    #     x1 = np.mean(plk,axis=0) + tang*0.07
    #     x2 = np.mean(plk,axis=0) - tang*0.07
    #     #ax.plot3D([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]])
    #     y1 = np.mean(plk, axis=0)
    #     ax.plot3D([tl[0],y1[0]],[tl[1],y1[1]],[tl[2],y1[2]],c='black',linewidth=.5)
    #
    # for astr in astring_idxs:
    #     plk = tri_locs[astr]
    #     tl = np.mean(tet_locs[int(astr/4)],axis=0)
    #     s1 = plk[1]-plk[0]
    #     s2 = plk[1] - plk[2]
    #     tang = np.cross(s1,s2)
    #     x1 = np.mean(plk,axis=0) + tang*0.07
    #     x2 = np.mean(plk,axis=0) - tang*0.07
    #     #ax.plot3D([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]])
    #     y1 = np.mean(plk, axis=0)
    #     ax.plot3D([tl[0],y1[0]],[tl[1],y1[1]],[tl[2],y1[2]],c='black',linewidth=.5)
    #plt.legend()
    # pickle.dump(fig,open('%s/string_int_box'%Master_path,'wb'))
    # plt.show()
    # figx=pickle.load(open('%s/string_int_box'%Master_path, 'rb'))
    # figx.show()

    #ax.zaxis.pane.set_edgecolor('white')
    ax = plt.gca()
    # for line in ax.xaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax.yaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax.zaxis.get_ticklines():
    #     line.set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([0,N])
    ax.set_ylim([0, N])
    ax.set_zlim([0, N])
    n_mn=np.shape(np.where(normed > 1 - err_mar))[1]
    # lp_ls = len([lp for lp in loop_list if len(lp)>1])
    # plt.title(r'Iteration:%i    # Mono:%i   # Loops:%i'%(i,n_mn,lp_ls) )
    plt.title(r'Iteration:%i    # Mono:%i  ' % (i, n_mn,))
    plt.tight_layout()
    plt.savefig('%s/box_%05i.png'%(Plot_path,i),dpi=600)
    # plt.show()

    # lengths = []
    # for ele in string_list:
    #
    #     length = len(ele)
    #     lengths.append(length,)
    # plt.hist(lengths,bins=10)

    # exp_ft = stats.expon.fit(lengths)
    # rX = np.linspace(0,max(lengths),100)
    # rY = stats.expon.pdf(rX,*exp_ft)
    # plt.plot(rX,rY*max(lengths)/max(rY))
    # plt.show()

    return plt

def tet_plot_jl(i,string_list,err_mar):
    
    #tet_stack = read('tet_stack_%s'%i, Master_path)
    area_stack = read('area_stack_%s'%i,Master_path)
    tet_locs = read('tet_locs_%s'%i,Master_path)
    deltas = read('deltas_%s'%i,Master_path)
    tri_locs = read('tri_locs_%s'%i, Master_path)
    # string_list = read('string_lists_%s'%i,Master_path)

    # Hoft_angs = read('hoft_arr_%s' % i, Master_path)
    # loop_list = read('loop_lists_%s'%i,Master_path)

    # tet_161 = tet_locs[161]
    # print(tet_161)
    # print(deltas[int(4*161)])
    # print(deltas[int(4*161)+1])
    # print(deltas[int(4*161)+2])
    # print(deltas[int(4*161)+3])
    #
    # tet_2 = tet_locs[2]
    # print(tet_2)
    # print(deltas[int(2/4)])
    # print(deltas[int(2/4) + 1])
    # print(deltas[int(2/4) + 2])
    # print(deltas[int(2/4) + 3])
    #
    #exit()
    normed = np.real(area_stack / (4 * np.pi))
    #print('# of monopoles:', np.shape(np.where(np.logical_and(normed> 1-err_mar, normed<1 + err_mar))))
    #print('# of antimonopoles:', np.shape(np.where(np.logical_and(normed < -1+err_mar, normed>-1-err_mar))))
    print('# of monopoles:', np.shape(np.where(normed > 1 - err_mar)))
    print('# of antimonopoles:', np.shape(np.where(normed < -1 + err_mar)))
    string_idxs = np.where(
        np.logical_and((np.real(deltas/(np.pi)))<= 2+err_mar, (np.real(deltas/(np.pi)))>=2-err_mar)
    )
    astring_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= -2 + err_mar, (np.real(deltas / (np.pi))) >= -2 - err_mar)
    )[:, 0]

    print('# of strings', np.shape(string_list))
    # print('# of loops', np.shape(loop_list))
    # plt.hist(np.real(deltas/(np.pi)),bins=20)
    # plt.xlabel(r'$\delta/\pi$')
    # plt.show()

    mono_idx = np.argwhere(np.logical_and(normed>= 1-err_mar, normed<=1 + err_mar))[:,0]
    antimon_idx = np.argwhere(np.logical_and(normed<= -1+err_mar, normed>=-1-err_mar))[:,0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)
    #fig.set_facecolor('black')
    #ax.set_facecolor('black')

    #ax.w_xaxis.set_pane_color((0, 0, 0, 1.0))
    #ax.w_yaxis.set_pane_color((0, 0, 0, 1.0))
    #ax.w_zaxis.set_pane_color((0, 0, 0, 1.0))
    #ax.xaxis.pane.set_edgecolor('white')
    #ax.yaxis.pane.set_edgecolor('white')

    if N > 20:
        marker_size = .01
    else:
        marker_size = 10

    ax.scatter(np.mean(tet_locs[mono_idx,:,0],axis=1),np.mean(tet_locs[mono_idx,:,1],axis=1),np.mean(tet_locs[mono_idx,:,2],axis=1), marker='o',c='blue',linewidths=1,s=marker_size)
    ax.scatter(np.mean(tet_locs[antimon_idx, :, 0],axis=1), np.mean(tet_locs[antimon_idx, :, 1],axis=1), np.mean(tet_locs[antimon_idx, :, 2],axis=1), marker='o',c='red',linewidths=1,s=marker_size)
    # ax.scatter(np.mean(tri_locs[string_idxs,:,0],axis=2),np.mean(tri_locs[string_idxs,:,1],axis=2),np.mean(tri_locs[string_idxs,:,2],axis=2),marker='*',c='green',linewidth=1,s=marker_size)
    # ax.scatter(np.mean(tri_locs[astring_idxs, :, 0], axis=1), np.mean(tri_locs[astring_idxs, :, 1], axis=1),
    #            np.mean(tri_locs[astring_idxs, :, 2], axis=1), marker='^', c='red', linewidth=1, s=marker_size/2)
    # ax.scatter(np.mean(tet_locs[:, :, 0], axis=1), np.mean(tet_locs[:, :, 1], axis=1),
    #           np.mean(tet_locs[:, :, 2], axis=1), marker='D', c='grey', linewidths=1, s=marker_size,alpha=.5)
    # ax.scatter(np.mean(tri_locs[:, :, 0], axis=1), np.mean(tri_locs[:, :, 1], axis=1),
    #            np.mean(tri_locs[:, :, 2], axis=1), marker='o', c='grey', linewidth=1, s=marker_size,alpha=.2)

    colors = cm.brg(np.linspace(0,1,len(string_list)))
    # colors = cm.brg(np.linspace(0, 1, len(loop_list)))
    width = .6
    for str_id, string in enumerate(string_list):
    # print(loop_list[1])
    # for str_id, string in enumerate(loop_list):
        string = np.array(string)
        #print(np.argwhere(np.linalg.norm(np.diff(string,axis=0),axis=1)==4.))
        disc = np.argwhere(np.linalg.norm(np.diff(string,axis=0),axis=1)>3)
        
        #cnt_ls = []
        if len(disc)==0:
            ax.plot3D(string[:, 0], string[:, 1], string[:, 2], c=colors[str_id],linewidth=width)
        if len(disc)==1:
            ax.plot3D(string[:disc[0][0]+1, 0], string[:disc[0][0]+1, 1], string[:disc[0][0]+1, 2], c=colors[str_id],linewidth=width)
            ax.plot3D(string[disc[0][0]+1:, 0], string[disc[0][0]+1:, 1], string[disc[0][0]+1:, 2], c=colors[str_id],linewidth=width)
        if len(disc)>1:
            start = 0
            #print(disc)
            for idx in range(len(disc)):
                #print(disc[idx]);print(string[start:disc[idx][0]+1])
                ax.plot3D(string[start:disc[idx][0]+1, 0], string[start:disc[idx][0]+1, 1], string[start:disc[idx][0]+1, 2],
                          c=colors[str_id],linewidth=width)
                start = disc[idx][0]+1
            ax.plot3D(string[start:, 0], string[start:, 1], string[start:, 2],
                      c=colors[str_id],linewidth=width)
        #ax.plot3D(string[:, 0], string[:, 1], string[:, 2], c=colors[str_id])
        ##Smooting curve##
        # tck, u = splprep([string[:,0],string[:,1],string[:,2]], s=200,k=1)
        # x_knots, y_knots, z_knots = splev(tck[0], tck)
        # num_true_pts = len(string_list)
        # u_fine = np.linspace(0, 1, num_true_pts)
        # x_fine, y_fine, z_fine = splev(u_fine, tck)
        #ax.plot3D(x_fine,y_fine,z_fine)

    ###Remainder string idxs###
    # string_idxs = read('upd_str_idxs_%s' % i, Master_path)
    # astring_idxs = read('upd_astr_idxs_%s' % i, Master_path)
    # mono_idx = read('upd_mono_idxs_%s' % i, Master_path)
    # antimon_idx = read('upd_antimon_idxs_%s' % i, Master_path)

    # for str in string_idxs:
    #     plk = tri_locs[str]
    #     tl = np.mean(tet_locs[int(str/4)],axis=0)
    #     s1 = plk[1]-plk[0]
    #     s2 = plk[1] - plk[2]
    #     tang = np.cross(s1,s2)
    #     x1 = np.mean(plk,axis=0) + tang*0.07
    #     x2 = np.mean(plk,axis=0) - tang*0.07
    #     #ax.plot3D([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]])
    #     y1 = np.mean(plk, axis=0)
    #     ax.plot3D([tl[0],y1[0]],[tl[1],y1[1]],[tl[2],y1[2]],c='black',linewidth=.5)
    #
    # for astr in astring_idxs:
    #     plk = tri_locs[astr]
    #     tl = np.mean(tet_locs[int(astr/4)],axis=0)
    #     s1 = plk[1]-plk[0]
    #     s2 = plk[1] - plk[2]
    #     tang = np.cross(s1,s2)
    #     x1 = np.mean(plk,axis=0) + tang*0.07
    #     x2 = np.mean(plk,axis=0) - tang*0.07
    #     #ax.plot3D([x1[0],x2[0]],[x1[1],x2[1]],[x1[2],x2[2]])
    #     y1 = np.mean(plk, axis=0)
    #     ax.plot3D([tl[0],y1[0]],[tl[1],y1[1]],[tl[2],y1[2]],c='black',linewidth=.5)
    #plt.legend()
    # pickle.dump(fig,open('%s/string_int_box'%Master_path,'wb'))
    # plt.show()
    # figx=pickle.load(open('%s/string_int_box'%Master_path, 'rb'))
    # figx.show()

    #ax.zaxis.pane.set_edgecolor('white')
    ax = plt.gca()
    # for line in ax.xaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax.yaxis.get_ticklines():
    #     line.set_visible(False)
    # for line in ax.zaxis.get_ticklines():
    #     line.set_visible(False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim([0,N])
    ax.set_ylim([0, N])
    ax.set_zlim([0, N])
    n_mn=np.shape(np.where(normed > 1 - err_mar))[1]
    # lp_ls = len([lp for lp in loop_list if len(lp)>1])
    # plt.title(r'Iteration:%i    # Mono:%i   # Loops:%i'%(i,n_mn,lp_ls) )
    plt.title(r'Iteration:%i    # Mono:%i  ' % (i, n_mn,))
    plt.tight_layout()
    plt.savefig('%s/box_%05i.png'%(Plot_path,i),dpi=600)
    # plt.show()

    # lengths = []
    # for ele in string_list:
    #
    #     length = len(ele)
    #     lengths.append(length,)
    # plt.hist(lengths,bins=10)

    # exp_ft = stats.expon.fit(lengths)
    # rX = np.linspace(0,max(lengths),100)
    # rY = stats.expon.pdf(rX,*exp_ft)
    # plt.plot(rX,rY*max(lengths)/max(rY))
    # plt.show()

    return plt

#tet_2_plot()
#exit()
def stat_plot():

    n_bins = 50

    len_lis = []
    max_lis = []
    for i in range(N_runs):
        string_lists = read('string_lists_%s'%i,Master_path)
        lens = [len(x) for x in string_lists]
        #n_l = np.histogram(lens,bins=20)

        if len(string_lists) != 0:
            max_lis.append(max(lens))
            len_lis.append(lens)
    cmpd_lis = np.hstack(len_lis)
    #print(np.shape(cmpd_lis))
    # idx = 0
    # counts_lis = []
    # for i in range(N_runs):
    #     string_lists = read('string_lists_%s'%i,Master_path)
    #     if len(string_lists) != 0:
    #         n_l = np.histogram(len_lis[idx],bins=np.linspace(0,max(max_lis),n_bins))
    #         #print(np.shape(n_l[0]))
    #         counts_lis.append(n_l[0])
    #         idx=idx+1
    # counts_lis = np.array(counts_lis)
    # print(np.shape(counts_lis))
    # bins_range = np.linspace(0,max(max_lis),n_bins)
    # bin_locs = [(bins_range[j]+bins_range[j+1])/2 for j in range(n_bins-1)]
    # plt.plot(bin_locs, np.mean(counts_lis,axis=0))
    # plt.plot(bin_locs, np.mean(counts_lis, axis=0)*bin_locs)
    #plt.show()
    n_bins = int(max(cmpd_lis)/2)
    plt.hist(cmpd_lis,bins=n_bins, density=True)

    bins = np.linspace(0, max(cmpd_lis), n_bins)
    n_l = np.histogram(cmpd_lis, bins=bins,density=True)
    bins_range = np.linspace(0,max(cmpd_lis),n_bins)
    bin_locs = [(bins_range[j]+bins_range[j+1])/2 for j in range(n_bins-1)]
    plt.plot(bin_locs,n_l[0]*bin_locs)

    strt_cut = 20

    exp_ft = stats.expon.fit(cmpd_lis)
    rX = np.linspace(1, max(cmpd_lis), 1000)
    rY = stats.expon.pdf(rX, *exp_ft)
    plt.plot(rX, rY)#* max(cmpd_lis) / max(rY))
    plt.show()

    return plt

def func(x, a, b):
    return a * np.exp(-b * x)
def lin_fn(x,a, b):
    return a*x+b
#stat_plot()
def stat_plot_2():

    len_lis = []
    max_lis = []
    loop_lis = []
    line_lis = []
    len_cutoff = 4
    for i in range(N_runs):
        string_lists = read('string_lists_%s'%i,Master_path)
        line_lists = read('pair_list_%s'%i, Master_path)
        loop_lists = read('loop_lists_%s' % i, Master_path)
        lens = [len(x) for x in string_lists]
        loop_lens = [len(x) for x in loop_lists]
        pair_lens = [len(x) for x in line_lists]
        #n_l = np.histogram(lens,bins=20)
        loop_lis.append(loop_lens)
        line_lis.append(pair_lens)
        len_lis.append(lens)
        # if len(string_lists) != 0:
        #     max_lis.append(max(lens))
        #     len_lis.append(lens)
    cmpd_lis = np.sort(np.hstack(len_lis))
    #cmpd_lis = cmpd_lis[np.argwhere(cmpd_lis==len_cutoff)[-1,0]:]
    #cmpd_lis = cmpd_lis[cmpd_lis>len_cutoff]
    cmpd_lis = cmpd_lis[cmpd_lis>len_cutoff]

    loop_lis = np.sort(np.hstack(loop_lis))
    line_lis = np.hstack(line_lis)
    #loop_cut = loop_lis[np.argwhere(loop_lis==len_cutoff)[-1,0]:]
    #loop_lis = loop_cut
    loop_lis = loop_lis[loop_lis>len_cutoff]


    n_bins = int(max(cmpd_lis)/2)
    bins = np.linspace(0, int(max(cmpd_lis)), n_bins)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2,2)

    ax1 = fig.add_subplot(gs[0,:])
    tot_his = np.histogram(cmpd_lis,bins=bins)
    bin_locs = np.array([(bins[j] + bins[j + 1]) / 2 for j in range(len(bins) - 1)])
    ax1.bar(bin_locs, tot_his[0], width=bins[1] - bins[0],alpha=.6)
    strt_cut = 30
    end_cut = 100
    tot_fit_lis = cmpd_lis[cmpd_lis>strt_cut]
    tot_fit_lis = tot_fit_lis[tot_fit_lis<end_cut]
    sbins = np.arange(strt_cut+1, end_cut + 1,2)

    n_l = np.histogram(tot_fit_lis,bins=sbins)
    sbin_locs = np.array([(sbins[j] + sbins[j + 1]) / 2 - strt_cut for j in range(len(sbins) - 1)])

    #ax1.bar(sbin_locs, n_l[0], width=sbins[1]-sbins[0])

    popt, pcov = curve_fit(func, sbin_locs, n_l[0])
    ax1.semilogy(sbin_locs + strt_cut, func(sbin_locs,*popt), label='a:%.1e b:%.1e'%(popt[0],popt[1]))
    #ax1.bar(sbin_locs + strt_cut, n_l[0], width=sbins[1] - sbins[0], alpha=.7)
    print(popt)
    # P = stats.expon.fit(n_l[0])
    # rX = sbin_locs
    # rP = stats.expon.pdf(rX, *P)
    # plt.semilogy(rX, rP)
    ax1.set_title('Combined')
    plt.legend()

    #####LOOP HIST#######
    ax2 = fig.add_subplot(gs[1,0])
    strt_cut = 50
    end_cut = 150

    ##Divide by volume to get density##

    full_lpb = np.linspace(len_cutoff, int(max(loop_lis)), int(max(loop_lis)/4))
    loop_bins = np.linspace(strt_cut+1, end_cut + 1,int(abs(strt_cut-end_cut)/4))
    loop_his = np.histogram(loop_lis, bins=loop_bins)
    sbin_locs = np.array([(loop_bins[j] + loop_bins[j + 1]) / 2 - strt_cut for j in range(len(loop_bins) - 1)])
    #ax2.bar(sbin_locs + strt_cut, loop_his[0], width=loop_bins[1] - loop_bins[0], alpha=.7)
    ax2.hist(loop_lis,bins=full_lpb,log=True,label='Loops',alpha=.4)

    popt, pcov = curve_fit(func, sbin_locs, loop_his[0])
    ax2.semilogy(sbin_locs + strt_cut, func(sbin_locs,*popt), label='a:%.1e b:%.1e'%(popt[0],popt[1]))
    ax2.set_title('Loops')
    plt.legend()


    ####PAIRS HIST####
    ax3 = fig.add_subplot(gs[1,-1])
    ax3.hist(line_lis,bins=bins,log=True,label='Pairs')

    strt_cut = 50
    end_cut = 250
    pair_bins = np.arange(strt_cut + 1, end_cut + 1, 2)
    pair_his = np.histogram(line_lis, bins=pair_bins)
    sbin_locs = np.array([(pair_bins[j] + pair_bins[j + 1]) / 2 - strt_cut for j in range(len(pair_bins) - 1)])
    #ax3.bar(sbin_locs + strt_cut, pair_his[0], width=sbins[1] - sbins[0], alpha=.7)

    popt, pcov = curve_fit(func, sbin_locs, pair_his[0])
    ax3.semilogy(sbin_locs + strt_cut, func(sbin_locs, *popt), label='a:%.1e b:%.1e'%(popt[0],popt[1]))
    ax3.set_title('Pairs')
    plt.legend()
    plt.show()
    return

#stat_plot_2()

def stat_plot_3():

    len_lis = []
    max_lis = []
    loop_lis = []
    line_lis = []
    len_cutoff = 4
    for i in range(N_runs):
        #string_lists = read('string_lists_%s'%i,Master_path)
        line_lists = read('pair_list_%s'%i, Master_path)
        loop_lists = read('loop_lists_%s' % i, Master_path)
        #lens = [len(x) for x in string_lists]
        loop_lens = [len(x) for x in loop_lists]
        pair_lens = [len(x) for x in line_lists]
        #n_l = np.histogram(lens,bins=20)
        loop_lis.append(loop_lens)
        line_lis.append(pair_lens)

    loop_lis = np.sort(np.hstack(loop_lis))
    line_lis = np.hstack(line_lis)
    loop_lis = loop_lis[loop_lis>len_cutoff]

    fig, ax = plt.subplots()

    ####PAIRS HIST####
    #ax3 = fig.add_subplot(gs[1,-1])
    bins = np.linspace(0, int(max(line_lis)), int(max(line_lis)/2))
    #plt.hist(line_lis,bins=bins,log=True)
    hist = np.histogram(line_lis, bins=bins)
    bin_locs = np.array([(bins[j] + bins[j + 1]) / 2 for j in range(len(bins) - 1)])
    plt.plot(bin_locs,np.log(hist[0]/bin_locs**3),linestyle='',marker='.',c='blue',alpha=.6)

    strt_cut = 100
    end_cut = 250
    pair_bins = np.arange(strt_cut + 1, end_cut + 1, 2)
    pair_his = np.histogram(line_lis, bins=pair_bins)
    sbin_locs = np.array([(pair_bins[j] + pair_bins[j + 1]) / 2 - strt_cut for j in range(len(pair_bins) - 1)])
    #ax3.bar(sbin_locs + strt_cut, pair_his[0], width=sbins[1] - sbins[0], alpha=.7)

    # popt, pcov = curve_fit(func, sbin_locs, pair_his[0])
    # xp = np.linspace(0,max(line_lis),1000)
    # plt.plot(xp + strt_cut, np.log(func(xp, *popt)/(xp + strt_cut)**3), linestyle='--',c='blue')
    # a = popt[0] / strt_cut ** 3 *np.exp(popt[1]*strt_cut)
    # print('pair paras')
    # print('a:%s'%a)
    # print('b:%s'%popt[1])
    # print('del a:%s'%(pcov[0]/strt_cut**3*np.exp(popt[1]*strt_cut)))
    # print('del b:%s'%pcov[1])

    end_fit = -10
    pos_ids = np.where(pair_his[0][:end_fit]>0)
    print('linear fit pairs')
    popt,cov = np.polyfit((sbin_locs + strt_cut)[pos_ids],np.log(pair_his[0][pos_ids]/((sbin_locs + strt_cut)**3)[pos_ids]),1,cov=True)
    lin_fit = np.poly1d(popt)
    xp = np.linspace(0,int(max(line_lis)),1000)
    plt.plot(xp, lin_fit(xp), linestyle='--',c='blue')
    print(np.exp(lin_fit))
    print('l_0:',-1/popt[0])
    print('A_0:',np.exp(popt[1]))
    print(np.sqrt(np.diag(cov)))
    print('sig l_0:', (1/popt[0]**2)*np.sqrt(np.diag(cov))[0])
    print('sig A_0:', np.exp(popt[1])*np.sqrt(np.diag(cov))[1])

    strt_cut = 75
    end_cut = 200
    full_lpb = np.linspace(len_cutoff, int(max(loop_lis)), int(max(loop_lis)/4))
    bin_locs = np.array([(full_lpb[j] + full_lpb[j + 1]) / 2 for j in range(len(full_lpb) - 1)])
    loop_bins = np.linspace(strt_cut+1, end_cut + 1,int(abs(strt_cut-end_cut)/4))
    loop_his = np.histogram(loop_lis, bins=loop_bins)
    full_bins = np.histogram(loop_lis, bins=full_lpb)
    sbin_locs = np.array([(loop_bins[j] + loop_bins[j + 1]) / 2 - strt_cut for j in range(len(loop_bins) - 1)])

    plt.plot(bin_locs, np.log(full_bins[0]/bin_locs**3), linestyle='', marker='.', color = 'red',alpha=.6)

    end_fit = -10
    pos_ids = np.where(loop_his[0][:end_fit]>0)
    print('linear fit loops')
    popt,cov = np.polyfit((sbin_locs + strt_cut)[pos_ids],np.log(loop_his[0][pos_ids]/((sbin_locs + strt_cut)**3)[pos_ids]),1,cov=True)
    lin_fit = np.poly1d(popt)
    xp = np.linspace(0,int(max(loop_lis)),1000)
    plt.plot(xp, lin_fit(xp), linestyle='--',c='red')
    print(np.exp(lin_fit))
    print('l_0:',-1/popt[0])
    print('A_0:',np.exp(popt[1]))
    print(np.sqrt(np.diag(cov)))
    print('sig l_0:', (1/popt[0]**2)*np.sqrt(np.diag(cov))[0])
    print('sig A_0:', np.exp(popt[1])*np.sqrt(np.diag(cov))[1])

    # popt, pcov = curve_fit(func, sbin_locs, loop_his[0])
    # a = popt[0]/strt_cut**3*np.exp(popt[1]*strt_cut)
    # print('loop paras')
    # print('a:%s'%a)
    # print('b:%s'%popt[1])
    # print('del a:%s'%(pcov[0]/strt_cut**3*np.exp(popt[1]*strt_cut)))
    # print('del b:%s'%pcov[1])
    #
    # xp = np.linspace(0,max(loop_lis),1000)
    # plt.plot(xp + strt_cut, np.log(func(xp, *popt)/(xp + strt_cut)**3), linestyle='--',c='red')
    # print(popt)
    # print(np.sqrt(np.diag(pcov)))


    plt.xlabel(r'$l$',fontsize=20)
    plt.ylabel(r'ln($n_l$)',fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)

    #plt.legend()
    plt.show()
    return

# stat_plot_3()


def mean_mono():
    N_mono = []
    for i in range(N_runs):
        area_stack = read('area_stack_%s' % i, Master_path)
        normed = np.real(area_stack / (4 * np.pi))
        mono_idx = np.argwhere(np.logical_and(normed>= 1-err_mar, normed<=1 + err_mar))[:,0]
        antimon_idx = np.argwhere(np.logical_and(normed<= -1+err_mar, normed>=-1-err_mar))[:,0]
        N_mono.append(len(mono_idx))
    print(np.mean(N_mono))
    return
#mean_mono()