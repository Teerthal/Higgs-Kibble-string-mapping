import numpy as np

from Parameters import *
from itertools import combinations
from Lattice import Higgs_direction,read,write
from numba import njit,prange, jit
import matplotlib.pyplot as plt

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
def tetrahedra(lat1, lat2, tets,area_arr,tet_locs):

    #tets = []
    #area_arr = []
    idx = 0
    ar_idx = 0
    for x in prange(N-1):
        for y in prange(N-1):
            for z in prange(N-1):

                #Along z#
                #print(x,y,z)
                A = lat1[x,y,z]#;print(A)
                B = lat1[x,y,z+1]#;print(B)
                C = lat2[x,y,z+1]
                D = lat2[x,y+1,z+1]
                E = lat2[x+1,y+1,z+1]
                F = lat2[x+1,y,z+1]
                #print(A,B,C,D,E,F)

                Al = [x,y,z]
                Bl = [x,y,z+1]
                Cl = [x,y,z+1]
                Dl = [x,y+1,z+1]
                El = [x+1,y+1,z+1]
                Fl = [x+1,y,z+1]

                tet1 = [A,B,C,D]
                tet2 = [A,B,D,E]
                tet3 = [A, B, E, F]
                tet4 = [A, B, F, C]

                tet1l = [Al, Bl, Cl, Dl]
                tet2l = [Al, Bl, Dl, El]
                tet3l = [Al, Bl, El, Fl]
                tet4l = [Al, Bl, Fl, Cl]

                #tets.extend([tet1,tet2,tet3,tet4])
                for j in prange(4):tets[idx,j] = tet1[j];tet_locs[idx,j] = tet1l[j]
                idx=idx+1
                for j in prange(4):tets[idx,j] = tet2[j];tet_locs[idx,j] = tet2l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet3[j];tet_locs[idx,j] = tet3l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet4[j];tet_locs[idx,j] = tet4l[j]
                idx = idx+1

                sub_tets = [tet1,tet2,tet3,tet4]
                for a in sub_tets:
                    #triangles = list(combinations(a,3))
                    triangles = triangle_combs(a)
                    sum_area = 0
                    for triangle in triangles:
                        #p12 = np.dot(triangle[0],triangle[1])
                        #p23 = np.dot(triangle[1],triangle[2])
                        #p31 = np.dot(triangle[2],triangle[0])
                        #Area = 1/(2*np.pi)*np.arccos((1+p12+p23+p31)/(np.sqrt(np.abs( 2*(1+p12)*(1+p23)*(1+p31) ))))
                        area_sign = np.sign(np.dot(triangle[0],np.cross(triangle[1],triangle[2])))

                        Area = tri_area(triangle[0],triangle[1],triangle[2])

                        sum_area = sum_area + area_sign*Area
                        #print(area_sign,Area)

                    #area_arr.append(sum_area)
                    area_arr[ar_idx] = sum_area
                    ar_idx=ar_idx+1
                    #exit()
                #Along x#
                A = lat1[x, y, z]
                B = lat1[x + 1, y, z]
                C = lat2[x + 1, y, z]
                D = lat2[x + 1, y + 1, z]
                E = lat2[x + 1, y + 1, z + 1]
                F = lat2[x + 1, y, z + 1]

                Al = [x, y, z]
                Bl = [x + 1, y, z]
                Cl = [x + 1, y, z]
                Dl = [x + 1, y + 1, z]
                El = [x + 1, y + 1, z + 1]
                Fl = [x + 1, y, z + 1]

                tet1 = [A, B, C, D]
                tet2 = [A, B, D, E]
                tet3 = [A, B, E, F]
                tet4 = [A, B, F, C]
                #tets.extend([tet1, tet2, tet3, tet4])

                tet1l = [Al, Bl, Cl, Dl]
                tet2l = [Al, Bl, Dl, El]
                tet3l = [Al, Bl, El, Fl]
                tet4l = [Al, Bl, Fl, Cl]

                #tets.extend([tet1,tet2,tet3,tet4])
                for j in prange(4):tets[idx,j] = tet1[j];tet_locs[idx,j] = tet1l[j]
                idx=idx+1
                for j in prange(4):tets[idx,j] = tet2[j];tet_locs[idx,j] = tet2l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet3[j];tet_locs[idx,j] = tet3l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet4[j];tet_locs[idx,j] = tet4l[j]
                idx = idx+1

                sub_tets = [tet1,tet2,tet3,tet4]
                for a in sub_tets:
                    triangles = triangle_combs(a)
                    sum_area = 0
                    for triangle in triangles:
                        #p12 = np.dot(triangle[0],triangle[1])
                        #p23 = np.dot(triangle[1],triangle[2])
                        #p31 = np.dot(triangle[2],triangle[0])

                        #Area = 1/(2*np.pi)*np.arccos((1+p12+p23+p31)/(np.sqrt(np.abs( 2*(1+p12)*(1+p23)*(1+p31) ))))
                        Area = tri_area(triangle[0], triangle[1], triangle[2])
                        area_sign = np.sign(np.dot(triangle[0], np.cross(triangle[1], triangle[2])))
                        sum_area = sum_area + area_sign * Area
                    #area_arr.append(sum_area)
                    area_arr[ar_idx] = sum_area
                    ar_idx = ar_idx+1

                #Aloong y#

                A = lat1[x,y,z]
                B = lat1[x,y+1,z]
                C = lat2[x,y+1,z]
                D = lat2[x,y+1,z+1]
                E = lat2[x+1,y+1,z+1]
                F = lat2[x+1,y+1,z]

                Al = [x,y,z]
                Bl = [x,y+1,z]
                Cl = [x,y+1,z]
                Dl = [x,y+1,z+1]
                El = [x+1,y+1,z+1]
                Fl = [x+1,y+1,z]

                tet1 = [A, B, C, D]
                tet2 = [A, B, D, E]
                tet3 = [A, B, E, F]
                tet4 = [A, B, F, C]

                tet1l = [Al, Bl, Cl, Dl]
                tet2l = [Al, Bl, Dl, El]
                tet3l = [Al, Bl, El, Fl]
                tet4l = [Al, Bl, Fl, Cl]

                #tets.extend([tet1,tet2,tet3,tet4])
                for j in prange(4):tets[idx,j] = tet1[j];tet_locs[idx,j] = tet1l[j]
                idx=idx+1
                for j in prange(4):tets[idx,j] = tet2[j];tet_locs[idx,j] = tet2l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet3[j];tet_locs[idx,j] = tet3l[j]
                idx = idx+1
                for j in prange(4):tets[idx,j] = tet4[j];tet_locs[idx,j] = tet4l[j]
                idx = idx+1

                sub_tets = [tet1,tet2,tet3,tet4]
                for a in sub_tets:
                    #triangles = list(combinations(a,3))
                    triangles = triangle_combs(a)
                    sum_area = 0
                    for triangle in triangles:

                        #p12 = np.dot(triangle[0],triangle[1])
                        #p23 = np.dot(triangle[1],triangle[2])
                        #p31 = np.dot(triangle[2],triangle[0])

                        #Area = 1/(2*np.pi)*np.arccos((1+p12+p23+p31)/(np.sqrt(np.abs( 2*(1+p12)*(1+p23)*(1+p31) ))))
                        Area = tri_area(triangle[0], triangle[1], triangle[2])
                        area_sign = np.sign(np.dot(triangle[0], np.cross(triangle[1], triangle[2])))
                        sum_area = sum_area + area_sign * Area
                    #area_arr.append(sum_area)
                    area_arr[ar_idx] = sum_area
                    ar_idx = ar_idx+1

    return tets,area_arr,tet_locs

lat1 = np.zeros((N,N,N))
lat2 = np.zeros((N,N,N))
#tet_stack,area_arr= tetrahedra(lat1,lat2)
#print(np.shape(tet_stack));print(np.shape(area_arr))

def run_tets(sigma,n):
    phi_list = ['phi_arr_%s'%s for s in range(N_runs)]

    for i in range(0,N_runs,2):
        tets = np.zeros((int((N - 1) ** 3 * 12), 4,3), dtype=complex)
        #tets = np.zeros((int((N - 1) ** 3 * 12), 4, 2), dtype=complex)
        area_arr = np.zeros((int((N - 1) ** 3 * 12)), dtype=complex)
        tet_loc_arr = np.zeros((int((N - 1) ** 3 * 12), 4,3))

        phi_1 = read('%s'%(phi_list[i]),Master_path)
        phi_2 = read('%s'%(phi_list[i+1]),Master_path)

        n_1 = Higgs_direction(phi_1,n,sigma)
        n_2 = Higgs_direction(phi_2, n, sigma)
        print(np.shape(n_1));print(np.shape(n_2))
        tet_stack, area_stack,tet_locs = tetrahedra(n_1,n_2,tets,area_arr,tet_loc_arr)
        #tet_stack, area_stack = tetrahedra(phi_1, phi_2,tets,area_arr)
        #print('# of monopoles:',np.shape(area_stack[area_stack/(4*np.pi)==1.0+0.0j]))
        #print('# of antimonopoles:',np.shape(area_stack[np.real(area_stack/(4*np.pi)) == -1.0+0.0j]))
        #print('zero sum', np.shape(area_stack[area_stack == 0.0 + 0.0j]))

        normed = np.real(area_stack / (4 * np.pi))
        print('# of monopoles:', np.shape(np.where(np.logical_and(normed>= 1-err_mar, normed<=1 + err_mar))))
        print('# of antimonopoles:', np.shape(np.where(np.logical_and(normed<= -1+err_mar, normed>=-1-err_mar))))
        print('zero sum', np.shape(np.where(np.logical_and(normed>= -1*err_mar, normed<=err_mar))))

        print(np.shape(area_stack))
        write('tet_stack_%s'%i, Master_path, tet_stack)
        write('area_stack_%s'%i,Master_path, area_stack)
        write('tet_locs_%s' % i, Master_path, tet_locs)
    return



#sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
#n = np.zeros((N, N, N, 3), dtype=complex)
#run_tets(sigma,n)

def mono_scatter_plot():
    for i in range(0,N_runs,2):
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

#@jit(forceobj=True)
#@jit()
def strings(n1,n2,n3,phi1,phi2,phi3, I, sigma):

    delta = 0
    #for [ve1,ve2],[p1,p2] in zip(zip([n1,n2,n3],[n2,n3,n1]),zip([phi1,phi2,phi3],[phi2,phi3,phi1])):
    ve1 = n1;ve2 = n2
    p1 = phi1;p2 = phi2
    a21cs = np.cross(ve1,ve2);print(a21cs)
    #print(a21cs)
    print(np.sqrt(a21cs[0] * a21cs[0] + a21cs[1] * a21cs[1] + a21cs[2] * a21cs[2]))
    a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]);print(a21)
    theta21 = np.arccos(ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2])
    print(theta21)
    S21r = -1j*(sigma[0]*a21[0] + sigma[1]*a21[1] + sigma[2]*a21[2])*np.sin(theta21/2) + I*np.cos(theta21/2)
    #p2_r = np.matmul(S21r,p1)#;print(p2_r)
    p2_r = [S21r[0][0]*p1[0] + S21r[0][1]*p1[1], S21r[1][0]*p1[0]+S21r[1][1]*p1[1]];print(p2_r)
    #delta21 = -1*np.angle(np.dot(np.conj(p2).T, p2_r));
    print(np.dot(np.conj(p2).T, p2_r))
    delta21 = np.angle((np.real(p2[0])-np.imag(p2[0])*1j)*p2_r[0] + (np.real(p2[1])-np.imag(p2[1])*1j)*p2_r[1])
    print(delta21/np.pi)
    ####Checked##Works#####
    delta = delta + delta21
    #print(delta)
    ve1 = n2;ve2 = n3
    p1 = phi2;p2 = phi3
    a21cs = np.cross(ve1,ve2)
    a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])
    theta21 = np.arccos(ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2])
    S32r = -1j*(sigma[0]*a21[0] + sigma[1]*a21[1] + sigma[2]*a21[2])*np.sin(theta21/2) + I*np.cos(theta21/2)
    #p2_r = np.matmul(S21r,p1)#;print(p2_r)
    p2_r = [S32r[0][0]*p1[0] + S32r[0][1]*p1[1], S32r[1][0]*p1[0]+S32r[1][1]*p1[1]]
    #delta21 = -1*np.angle(np.dot(np.conj(p2).T, p2_r));print(delta21);
    #print(np.dot(np.conj(p2).T, p2_r))
    delta21 = np.angle((np.real(p2[0]) - np.imag(p2[0])*1j) * p2_r[0] + (np.real(p2[1]) - np.imag(p2[1])*1j) * p2_r[1])
    ####Checked##Works#####
    delta = delta + delta21;print(delta21/np.pi)
    #print(delta/np.pi)
    ve1 = n3;ve2 = n1
    p1 = phi3;p2 = phi1
    a21cs = np.cross(ve1,ve2)
    a21 = a21cs/np.sqrt(a21cs[0]*a21cs[0]+a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2])#np.dot(a21cs,a21cs))#;print(a21)
    theta21 = np.arccos(ve1[0]*ve2[0]+ve1[1]*ve2[1]+ve1[2]*ve2[2])#np.dot(ve1,ve2))#;print(ve1,ve2);print(theta21)
    S13r = -1j*(sigma[0]*a21[0] + sigma[1]*a21[1] + sigma[2]*a21[2])*np.sin(theta21/2) + I*np.cos(theta21/2)#;print(S21r)
    #p2_r = np.matmul(S21r,p1)#;print(p2_r)
    p2_r = [S13r[0][0]*p1[0] + S13r[0][1]*p1[1], S13r[1][0]*p1[0]+S13r[1][1]*p1[1]]
    #print(np.dot(np.conj(p2).T, p2_r))
    #delta21 = -1*np.angle(np.dot(np.conj(p2).T, p2_r));print(delta21)
    delta21 = np.angle((np.real(p2[0]) - np.imag(p2[0])*1j) * p2_r[0] + (np.real(p2[1]) - np.imag(p2[1])*1j) * p2_r[1])
    print(delta21/np.pi)
    ####Checked##Works#####
    delta = delta + delta21
    print(delta)
    h_123 = np.matmul(S13r,np.matmul(S32r,S21r))
    print(h_123);exit()
    return delta

sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
I = np.array([[1,0],[0,1]])
#strings([0,0,1],[0.866025,0,-0.5],[-0.433013, 0.75, -0.5], [1,0],[np.cos(np.pi/3),np.sin(np.pi/3)],[np.cos(np.pi/3),np.sin(np.pi/3)*np.exp(2j*np.pi/3)], I, sigma)

def n(phi,sigma):
    phi_conj = np.conjugate(phi).T
    n = -1 * np.array([np.matmul(phi_conj, np.matmul(sigma[0], phi))
              , np.matmul(phi_conj,np.matmul(sigma[1], phi))
              , np.matmul(phi_conj, np.matmul(sigma[2], phi))])
    return n/np.sqrt(np.dot(np.conjugate(phi),phi))

phi_1 = [0,1]
alp = np.pi/5
bet = np.pi/4
phi_2 = [np.sin(np.pi/3)*np.exp(1j*alp),np.cos(np.pi/3)*np.exp(1j*bet)]
from Lattice import phi_mag
mag_1 = phi_mag(phi_1)
mag_2 = phi_mag(phi_2)
phi_dag_1 = np.conj(phi_1).T
phi_dag_2 = np.conj(phi_2).T
phi_3 = [np.sin(np.pi/4),np.cos(np.pi/4)]
#print(n(phi_2,sigma));exit()
#n_1 = -1*(np.matmul(phi_dag_1, np.matmul(sigma[0], phi_1))+np.matmul(phi_dag_1, np.matmul(sigma[1], phi_1))+np.matmul(phi_dag_1, np.matmul(sigma[2], phi_1))) / mag_1
#n_2 = -1*(np.matmul(phi_dag_2, np.matmul(sigma[0], phi_2))+np.matmul(phi_dag_2, np.matmul(sigma[1], phi_2))+np.matmul(phi_dag_2, np.matmul(sigma[2], phi_2))) / mag_2

n_1 = [0,0,1]
n_2 = [-np.sin(2*np.pi/4)*np.cos(alp-bet), np.sin(2*np.pi/4)*np.sin(alp-bet), np.cos(2*np.pi/4)]
#print(n_1);print(n_2)
#strings([0,0,-1],[-np.sin(2*np.pi/4)*np.cos(np.pi*(1/3-1/5)), -np.sin(2*np.pi/4)*np.sin(np.pi*(1/3-1/5)), -np.cos(2*np.pi/4)],[1, 0, 0], [0,1],[np.sin(np.pi/4)*np.exp(1j*np.pi/3),np.cos(np.pi/4)*np.exp(1j*np.pi/5)],[np.cos(np.pi/4),np.sin(np.pi/4)], I, sigma)
print(n(phi_1,sigma));print(n(phi_2,sigma));print(n(phi_3,sigma))
strings(n(phi_1,sigma),n(phi_2,sigma),n(phi_3,sigma), phi_1,phi_2,phi_3, I, sigma)
exit()



#mono_scatter_plot()
#@njit(parallel=True)
#@jit()
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

                #tets.extend([tet1,tet2,tet3,tet4])
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
                #print(idx)

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
                    #triangles = list(combinations(a,3))
                    #triangles = triangle_combs(a)
                    #tri_locs = triangle_combs(a_locs)
                    #print(a);print(a_locs);exit()
                    triangles = [[a[0],a[1],a[2]],[a[0],a[2],a[3]],[a[0],a[3],a[1]],[a[1],a[3],a[2]]]
                    tri_locs = [[a_locs[0],a_locs[1],a_locs[2]],
                                [a_locs[0],a_locs[2],a_locs[3]],
                                [a_locs[0],a_locs[3],a_locs[1]],
                                [a_locs[1],a_locs[3],a_locs[2]]]

                    sum_area = 0
                    #print(a)
                    for triangle,tr_loc in zip(triangles,tri_locs):
                        #p12 = np.dot(triangle[0],triangle[1])
                        #p23 = np.dot(triangle[1],triangle[2])
                        #p31 = np.dot(triangle[2],triangle[0])
                        #Area = 1/(2*np.pi)*np.arccos((1+p12+p23+p31)/(np.sqrt(np.abs( 2*(1+p12)*(1+p23)*(1+p31) ))))
                        #area_sign = np.sign(np.dot(triangle[0],np.cross(triangle[1],triangle[2])))
                        #n1 = [ele.real for ele in triangle[0]]
                        #n2 = [ele.real for ele in triangle[1]]
                        #n3 = [ele.real for ele in triangle[2]]
                        n1 = triangle[0];n2=triangle[1];n3=triangle[2]
                        #area_sign = np.sign(np.dot(np.cross(triangle[0],triangle[1]),triangle[2]))
                        #Area = tri_area(triangle[0],triangle[1],triangle[2])
                        area_sign = np.sign(np.dot(np.cross(n1,n2),n3))
                        Area = tri_area(n1,n2,n3)
                        sum_area = sum_area + area_sign*Area
                        #print(triangle);print(area_sign,Area)

                        phi_1 = phi_arr[tr_loc[0][0],tr_loc[0][1],tr_loc[0][2]]
                        phi_2 = phi_arr[tr_loc[1][0],tr_loc[1][1],tr_loc[1][2]]
                        phi_3 = phi_arr[tr_loc[2][0],tr_loc[2][1],tr_loc[2][2]]

                        #tri_loc = np.mean(tr_loc,axis=0)
                        # delta = strings(triangle[0],triangle[1],triangle[2],
                        #                  phi_1,phi_2,phi_3,ID2,sigma)
                        delta = strings(n1,n2,n3,
                                         phi_1,phi_2,phi_3,ID2,sigma)
                        #delta = 1
                        del_arr[tr_idx] = delta
                        for o in prange(3): tri_loc_arr[tr_idx,o] = tr_loc[o]
                        tr_idx = tr_idx + 1
                        #print(tr_idx,delta)
                    #area_arr.append(sum_area)

                    # if np.isnan(sum_area/(4*np.pi)) == True:
                    #     print(sum_area/(4*np.pi))#;exit()
                    #     print(a)
                    #     print(a_locs)
                    #     exit()

                    area_arr[ar_idx] = sum_area
                    ar_idx=ar_idx+1
                    #exit()
    return tets,area_arr,tet_locs,del_arr,tri_loc_arr

lat1 = np.zeros((N,N,N))



def run_tets2(sigma,n):
    phi_list = ['phi_arr_%s'%s for s in range(N_runs)]

    for i in range(0,N_runs):
        tets = np.zeros((int(((N-1)/2) ** 3 * 24), 4,3), dtype=complex)
        #tets = np.zeros((int((N - 1) ** 3 * 12), 4, 2), dtype=complex)
        area_arr = np.zeros((int(((N-1)/2) ** 3 * 24)), dtype=complex)
        tet_loc_arr = np.zeros((int(((N-1)/2) ** 3 * 24), 4,3),dtype=int)
        del_arr = np.zeros((int(((N-1)/2) ** 3 * 24 * 4)),dtype=complex)
        tri_loc_arr = np.zeros((int(((N-1)/2) ** 3 * 24 * 4), 3, 3))

        phi_1 = read('%s'%(phi_list[i]),Master_path)
        n_1 = Higgs_direction(phi_1,n,sigma)
        Hoft_vals = read('hoft_arr_%s'%i,Master_path)

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
            print(Hoft_vals[tet_locs[pb_idx][3][0], tet_locs[pb_idx][3][1], tet_locs[pb_idx][3][2]])
            exit()

        #print('# of monopoles:', np.shape(np.where(np.logical_and(normed> 1-err_mar, normed<1 + err_mar))))
        #print('# of antimonopoles:', np.shape(np.where(np.logical_and(normed < -1+err_mar, normed>-1-err_mar))))
        print('# of monopoles:', np.shape(np.where(normed > 1 - err_mar)))
        print('# of antimonopoles:', np.shape(np.where(normed < -1 + err_mar)))
        print('zero sum', np.shape(np.where(np.logical_and(normed>= -1*err_mar, normed<=err_mar))))
        print('# of strings', np.shape(np.where(np.logical_or(
            np.logical_and(np.abs(np.real(deltas/(np.pi)))<= 1+err_mar, np.abs(np.real(deltas/(np.pi)))>=1-err_mar),
            np.logical_and(np.abs(np.real(deltas/(np.pi)))<= 2+err_mar, np.abs(np.real(deltas/(np.pi)))>=2-err_mar)
        ))))
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

    return



sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
n = np.zeros((N, N, N, 3), dtype=complex)
run_tets2(sigma,n)
#exit()

def tet_2_plot():
    for i in range(0,N_runs,1):
        tet_stack = read('tet_stack_%s'%i, Master_path)
        area_stack = read('area_stack_%s'%i,Master_path)
        tet_locs = read('tet_locs_%s'%i,Master_path)
        deltas = read('deltas_%s'%i,Master_path)
        tri_locs = read('tri_locs_%s'%i, Master_path)


        normed = np.real(area_stack / (4 * np.pi))
        #print('# of monopoles:', np.shape(np.where(np.logical_and(normed> 1-err_mar, normed<1 + err_mar))))
        #print('# of antimonopoles:', np.shape(np.where(np.logical_and(normed < -1+err_mar, normed>-1-err_mar))))
        print('# of monopoles:', np.shape(np.where(normed > 1 - err_mar)))
        print('# of antimonopoles:', np.shape(np.where(normed < -1 + err_mar)))
        print('zero sum', np.shape(np.where(np.logical_and(normed>= -1*err_mar, normed<=err_mar))))
        # string_idxs = np.where(np.logical_or(
        #     np.logical_and(np.abs(np.real(deltas/(np.pi)))<= 1+err_mar, np.abs(np.real(deltas/(np.pi)))>=1-err_mar),
        #     np.logical_and(np.abs(np.real(deltas/(np.pi)))<= 2+err_mar, np.abs(np.real(deltas/(np.pi)))>=2-err_mar)
        # ))
        string_idxs = np.where(
            np.logical_and(np.abs(np.real(deltas/(np.pi)))<= 2+err_mar, np.abs(np.real(deltas/(np.pi)))>=2-err_mar)
        )

        print('# of strings', np.shape(string_idxs))
        plt.hist(np.abs(np.real(deltas/(np.pi))),bins=20);
        plt.xlabel(r'$\delta/\pi$')
        plt.show()

        normed = np.real(area_stack / (4 * np.pi))
        mono_idx = np.argwhere(np.logical_and(normed>= 1-err_mar, normed<=1 + err_mar))[:,0]
        antimon_idx = np.argwhere(np.logical_and(normed<= -1+err_mar, normed>=-1-err_mar))[:,0]
        print(np.shape(mono_idx),np.shape(antimon_idx))
        #print(np.shape(tet_stack[mono_idx]))

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if N > 20:
            marker_size = .01
        else:
            marker_size = 20
        ax.scatter(np.mean(tet_locs[mono_idx,:,0],axis=1),np.mean(tet_locs[mono_idx,:,1],axis=1),np.mean(tet_locs[mono_idx,:,2],axis=1), marker='o',c='blue',linewidths=1,s=marker_size)
        ax.scatter(np.mean(tet_locs[antimon_idx, :, 0],axis=1), np.mean(tet_locs[antimon_idx, :, 1],axis=1), np.mean(tet_locs[antimon_idx, :, 2],axis=1), marker='x',c='red',linewidths=1,s=marker_size)
        #ax.scatter(np.mean(tri_locs[string_idxs,:,0],axis=2),np.mean(tri_locs[string_idxs,:,1],axis=2),np.mean(tri_locs[string_idxs,:,2],axis=2),marker='*',c='green',linewidth=1,s=marker_size)
#       ax.scatter(np.mean(tet_locs[:, :, 0], axis=1), np.mean(tet_locs[:, :, 1], axis=1),
#                   np.mean(tet_locs[:, :, 2], axis=1), marker='^', c='grey', linewidths=1, s=marker_size,alpha=.5)
        plt.show()
    return

tet_2_plot()

def mono_scatter_plot():
    for i in range(0,N_runs,2):
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
        ax.scatter(np.mean(tet_locs[mono_idx,:,0],axis=1),np.mean(tet_locs[mono_idx,:,1],axis=1),np.mean(tet_locs[mono_idx,:,2],axis=1), marker='o',c='blue',linewidths=1,s=.01)
        ax.scatter(np.mean(tet_locs[antimon_idx, :, 0],axis=1), np.mean(tet_locs[antimon_idx, :, 1],axis=1), np.mean(tet_locs[antimon_idx, :, 2],axis=1), marker='x',c='red',linewidths=1,s=.01)
        plt.show()
    return

#mono_scatter_plot()