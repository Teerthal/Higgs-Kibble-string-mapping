import random

import numpy as np

from tets import *

@jit()
def n(phi,sigma):
    # phi_conj = np.conjugate(phi).T
    # n = -1 * np.array([np.matmul(phi_conj, np.matmul(sigma[0], phi))
    #           , np.matmul(phi_conj,np.matmul(sigma[1], phi))
    #           , np.matmul(phi_conj, np.matmul(sigma[2], phi))])
    mag = np.sqrt(np.dot(np.conj(phi),phi))
    phi_dag = phi.conj().T
    n = [-1 * np.matmul(phi_dag, np.matmul(sigma[i], phi)) / mag for i in [0,1,2]]
    # return n/np.sqrt(np.dot(np.conjugate(phi),phi))
    return np.real(np.array(n))

@jit()
def flip_n(n_vec):
    #takes a Higgs n vector and returns a phi such that n -> -n
    #or flips it
    # print(n(np.matmul(phi,sigma[1]),sigma))
    # print(n_vec[0]**2-n_vec[1]**2)
    alpha = np.arccos(n_vec[2]) / 2
    gamma = random.uniform(0, 2 * np.pi)
    # beta = 0.5 * (2 * gamma + np.arccos((n_vec[0] ** 2 - n_vec[1] ** 2)
    #                                     * (np.sin(2 * alpha)) ** (-2)))
    beta = np.arccos(n_vec[0] / np.sin(2 * alpha)) + gamma
    # beta = np.arcsin(-n_vec[1] / np.sin(2 * alpha)) + gamma
    new_phi = np.array([eta * np.cos(alpha) * np.exp(1j * beta), eta * np.sin(alpha) * np.exp(1j * gamma)])
    new_n = np.real(n(new_phi, sigma))

    n_vec = np.round(n_vec,5);new_n = np.round(new_n,5)

    if ((new_n[0]!=-n_vec[0]) and (new_n[1]!=-n_vec[1]) and (new_n[2]!=-n_vec[2])):
        print(n_vec);print(alpha,beta,gamma);print(new_n)
        print('n flipping not working');exit()
    if ((np.pi/2<alpha<0) or (2*np.pi<beta<0) or (2*np.pi<gamma<0)):
        print(alpha,beta,gamma)
        print('roh roh');exit()

    return new_phi

@jit()
def tet_charge(tet,phi_arr):
    # n1=np.real(n1);n2=np.real(n2);n3=np.real(n3);n4=np.real(n4)
    charge = 0
    # stk = np.array([[n1,n2,n3],[n2,n3,n4],[n3,n4,n1],[n4,n1,n2]])
    stk = triangle_combs(tet)
    # print(stk);print(len(stk))
    for a in stk:
        phis=[phi_arr[j[0],j[1],j[2]] for j in a]
        G,H,K=[np.real(n(o,sigma)) for o in phis]
        trar=tri_area(G, H, K)
        charge=charge+trar
    return np.real(np.round(charge/(np.pi*4),5))

def n_tnsfm_og(n_2,n_3,n_1):
    #takes a Higgs n vector and returns a phi such that n_4 -> -(n_2+n_3)/|(n_2+n_3)|
    #or flips it
    # print(n(np.matmul(phi,sigma[1]),sigma))
    # print(n_vec[0]**2-n_vec[1]**2)
    n_2=np.real(np.array(n_2));n_3=np.real(np.array(n_3));n_1=np.real(np.array(n_1))
    n2n3 = (0.5*(n_2+n_3)+0.1*n_1)
    # n2n3 = (n_2 + 0.1*n_3)
    # n2n3 = (0.5 * (n_2 + n_3)/
    #         np.sqrt((n_2+n_3)[0]**2+(n_2+n_3)[1]**2+(n_2+n_3)[2]**2) + 0.1 * n_1)
    # n_vec=n2n3/np.linalg.norm(n2n3)
    n_vec = n2n3 / np.sqrt(n2n3[0]**2+n2n3[1]**2+n2n3[2]**2)

    alpha = np.arccos(n_vec[2]) / 2
    gamma = random.uniform(0, 2 * np.pi)
    # beta = 0.5 * (2 * gamma + np.arccos((n_vec[0] ** 2 - n_vec[1] ** 2)
    #                                     * (np.sin(2 * alpha)) ** (-2)))
    beta = np.arccos(n_vec[0] / np.sin(2 * alpha)) + gamma
    # beta = np.arcsin(-n_vec[1] / np.sin(2 * alpha)) + gamma
    beta = np.arctan(-(n_vec[1]/n_vec[0]))+gamma
    # beta = np.arctan2(np.real(-n_vec[1]),np.real(n_vec[0])) + gamma
    new_phi = np.array([eta * np.cos(alpha) * np.exp(1j * beta), eta * np.sin(alpha) * np.exp(1j * gamma)])
    new_n = np.real(n(new_phi, sigma))
    n_vec = np.real(np.round(n_vec,5));new_n = np.real(np.round(new_n,5))
    # print('a', new_n);
    # print('b', n_vec)
    if ((new_n[0]!=-n_vec[0]) or (new_n[1]!=-n_vec[1]) or (new_n[2]!=-n_vec[2])):
        print(n_vec);
        print(alpha, beta, gamma);
        print(new_n)
        print('n flipping not working')

    # while ((new_n[0]!=-n_vec[0]) or (new_n[1]!=-n_vec[1]) or (new_n[2]!=-n_vec[2])):
    #     # print(n_vec);print(alpha,beta,gamma);print(new_n)
    #     alpha = np.arccos(n_vec[2]) / 2
    #     gamma = random.uniform(0, 2 * np.pi)
    #     # beta = 0.5 * (2 * gamma + np.arccos((n_vec[0] ** 2 - n_vec[1] ** 2)
    #     #                                     * (np.sin(2 * alpha)) ** (-2)))
    #     beta = np.arccos(n_vec[0] / np.sin(2 * alpha)) + gamma
    #     # beta = np.arcsin(-n_vec[1] / np.sin(2 * alpha)) + gamma
    #     beta = np.arctan(-(n_vec[1] / n_vec[0])) + gamma
    #     # beta = np.arctan2(np.real(-n_vec[1]),np.real(n_vec[0])) + gamma
    #     new_phi = np.array([eta * np.cos(alpha) * np.exp(1j * beta), eta * np.sin(alpha) * np.exp(1j * gamma)])
    #     new_n = np.real(n(new_phi, sigma))
    #     n_vec = np.real(np.round(n_vec, 5));
    #     new_n = np.real(np.round(new_n, 5))
    return new_phi

def n_tnsfm(n_2,n_3,n_1):
    #takes a Higgs n vector and returns a phi such that n_4 -> -(n_2+n_3)/|(n_2+n_3)|
    #or flips it
    # print(n(np.matmul(phi,sigma[1]),sigma))
    # print(n_vec[0]**2-n_vec[1]**2)
    n_2=np.real(np.array(n_2));n_3=np.real(np.array(n_3));n_1=np.real(np.array(n_1))
    n2n3 = (0.5*(n_2+n_3)+0.1*n_1)
    # n2n3 = (n_2 + 0.1*n_3)
    # n2n3 = (0.5 * (n_2 + n_3)/
    #         np.sqrt((n_2+n_3)[0]**2+(n_2+n_3)[1]**2+(n_2+n_3)[2]**2) + 0.1 * n_1)
    # n_vec=n2n3/np.linalg.norm(n2n3)
    n_vec = n2n3 / np.sqrt(n2n3[0]**2+n2n3[1]**2+n2n3[2]**2)

    alpha = np.arccos(n_vec[2]) / 2
    gamma = random.uniform(0, 2 * np.pi)
    # beta = 0.5 * (2 * gamma + np.arccos((n_vec[0] ** 2 - n_vec[1] ** 2)
    #                                     * (np.sin(2 * alpha)) ** (-2)))
    beta = np.arccos(n_vec[0] / np.sin(2 * alpha)) + gamma
    # beta = np.arcsin(-n_vec[1] / np.sin(2 * alpha)) + gamma
    beta = np.arctan(-(n_vec[1]/n_vec[0]))+gamma
    # beta = np.arctan2(np.real(-n_vec[1]),np.real(n_vec[0])) + gamma
    beta = np.arctan2(-(n_vec[1] ),n_vec[0]) + gamma
    new_phi = np.array([eta * np.cos(alpha) * np.exp(1j * beta), eta * np.sin(alpha) * np.exp(1j * gamma)])
    new_n = np.real(n(new_phi, sigma))
    n_vec = np.real(np.round(n_vec,5));new_n = np.real(np.round(new_n,5))
    # print('a', new_n);
    # print('b', n_vec)
    if ((new_n[0]!=-n_vec[0]) or (new_n[1]!=-n_vec[1]) or (new_n[2]!=-n_vec[2])):
        print(n_vec);
        print(alpha, beta, gamma);
        print(new_n)
        print('n flipping not working');exit()
        return [0]
    else:
        return new_phi

def deltas_tet(uncommon_ver,commons,ver_1,ver_5,phi_arr):
    phi_4 = [phi_arr[i[0], i[1], i[2]] for i in [uncommon_ver]][0]
    n_4 = n(phi_4, sigma)
    phi_3 = [phi_arr[i[0], i[1], i[2]] for i in [commons[0]]][0]
    n_3 = n(phi_3, sigma)
    phi_2 = [phi_arr[i[0], i[1], i[2]] for i in [commons[1]]][0]
    n_2 = n(phi_2, sigma)
    phi_1 = [phi_arr[i[0], i[1], i[2]] for i in [ver_1]][0]
    n_1 = n(phi_1, sigma)
    phi_5 = [phi_arr[i[0], i[1], i[2]] for i in [ver_5]][0]
    n_5 = n(phi_5, sigma)
    # print(phi_1,phi_4)
    # print(n_1);print(n_4);exit()
    del_234 = round(strings(n_2, n_3, n_4, phi_2, phi_3, phi_4, I, sigma), 5)
    del_123 = round(strings(n_1, n_2, n_3, phi_1, phi_2, phi_3, I, sigma), 5)
    del_143 = round(strings(n_1, n_4, n_3, phi_1, phi_4, phi_3, I, sigma), 5)
    del_124 = round(strings(n_1, n_2, n_4, phi_1, phi_2, phi_4, I, sigma), 5)

    return phi_1,phi_2,phi_3,phi_4,phi_5,n_1,n_2,n_3,n_4,n_5,del_124,del_143,del_123,del_234

def del_plk(triangle,phi_arr):
    plq_crds = triangle;
    plq_crds = [[int(c) for c in cds] for cds in plq_crds]
    phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
    n_vecs = [n(phi, sigma) for phi in phis];
    delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)
    print(delt / (2 * np.pi));
    return delt

# @jit()
def bnd_check(inp,arr):##checks if the points abc are hitting the boundary and return reflection if they are

    a,b,c=inp

    if (a==0.0) or (b==0.0) or (c==0.0) or (a==float(N-1)) or (b==float(N-1)) or (c==float(N-1)):
        out = [True, [ref_plk(a,b,c)]]

        ix = [0.0, float(N - 1)]
        ###Lattice edges###
        if (np.isin(ix, a).any() and np.isin(ix, b).any()):
            edges = []
            for xb in ix:
                for yb in ix:
                    edges.append([xb, yb, c])
            out = [11, edges]
        if (np.isin(ix, b).any() and np.isin(ix, c).any()):
            edges = []
            for yb in ix:
                for zb in ix:
                    edges.append([a, yb, zb])
            out = [12, edges]
        if (np.isin(ix, c).any() and np.isin(ix, a).any()):
            edges = []
            for zb in ix:
                for xb in ix:
                    edges.append([xb, b, zb])
            out = [13, edges]
        if (np.isin(ix, a).any() and np.isin(ix, b).any() and np.isin(ix, c).any()):
            ###Lattice vertices###
            vertices = []
            for xb in ix:
                for yb in ix:
                    for zb in ix:
                        vertices.append([xb, yb, zb])
            out = [10, vertices]

        # if out[0] in [True, 11, 10]:
        for j in out[1]:
            # print('-----------')
            # print(a, b, c);
            # print(out[0])
            # print(j)
            # print('-----------')
            arr[int(j[0]), int(j[1]), int(j[2])] = arr[int(a), int(b), int(c)]
        return arr

    else:
        # out = [False,0]
        return arr

def bnd_test(latt_arr):
    # Periodic Boundary Condition check

    for x_idx in prange(N):
        for y_idx in prange(N):
            for z_idx in prange(N):

                if z_idx == int(N-1):
                    if  (latt_arr[x_idx, y_idx, z_idx, 0] != latt_arr[x_idx, y_idx, 0, 0]) and (
                            latt_arr[x_idx, y_idx, z_idx, 1] != latt_arr[x_idx, y_idx, 0, 1]):
                        print('boundary issue')
                        print(x_idx,y_idx,z_idx)
                        exit()

                if y_idx == int(N-1):
                    if (latt_arr[x_idx, y_idx, z_idx, 0] != latt_arr[x_idx, 0, z_idx, 0]) and (
                    latt_arr[x_idx, y_idx, z_idx, 1] != latt_arr[x_idx, 0, z_idx, 1]):
                        print('boundary issue')
                        print(x_idx, y_idx, z_idx)
                        exit()

                if x_idx == int(N-1):
                    if (latt_arr[x_idx, y_idx, z_idx, 0] != latt_arr[0, y_idx, z_idx, 0]) and (
                    latt_arr[x_idx, y_idx, z_idx, 1] != latt_arr[0, y_idx, z_idx, 1]):
                        print('boundary issue')
                        print(x_idx, y_idx, z_idx)
                        exit()
    return

# @jit(forceobj=True)
# def collapse(i,phi_arr,area_stack,tet_locs,deltas,tri_locs):
def collapse(i):
    phi_arr = read('phi_arr_%s' % i, Master_path)
    area_stack = read('area_stack_%s' % i, Master_path)
    tet_locs = read('tet_locs_%s' % i, Master_path)
    deltas = read('deltas_%s' % i, Master_path)
    tri_locs = read('tri_locs_%s' % i, Master_path)

    string_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= 2 + err_mar, (np.real(deltas / (np.pi))) >= 2 - err_mar)
    )[:, 0]

    astring_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= -2 + err_mar, (np.real(deltas / (np.pi))) >= -2 - err_mar)
    )[:, 0]

    normed = np.real(area_stack / (4 * np.pi))
    mono_idx = np.argwhere(np.logical_and(normed >= 1 - err_mar, normed <= 1 + err_mar))[:, 0]
    antimon_idx = np.argwhere(np.logical_and(normed <= -1 + err_mar, normed >= -1 - err_mar))[:, 0]
    # print('#Monopoles:%s'%(np.shape(mono_idx)))
    # print('#Antimonopoles:%s'%(np.shape(antimon_idx)))
    # pair_cnt = 0

    str_locs = np.mean(tri_locs[string_idxs], axis=1)
    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
    tri_coords = np.mean(tri_locs,axis=1)
    # no_match_cnt = 0
    # match_cnt = 0
    # string_lists = []

    for mn_idx in mono_idx:
    # for mn_idx in [mono_idx[0]]:
        pair_idxs = []
        #str_match = np.intersect1d(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        str_match = intersection(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        plk = np.mean(tri_locs[str_match[0]], axis=0)
        # pair_idxs.append(np.mean(tet_locs[mn_idx],axis=0))
        mono_idx = np.delete(mono_idx, np.argwhere(mono_idx == mn_idx)[0, 0])
        mn_tet=tet_locs[mn_idx]
        # print(mn_idx,len(str_match))

        if len(str_match) == 2:  # 2 strings coming our of the monopole tet
            choice = random.choice(str_match)

            mn_tri = tri_locs[choice]  # plauette triangle coordinates through which string passes
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
            str_plq = np.mean(tri_locs[choice], axis=0)
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

            for tet_ver in mn_tet:  # Finding vertex in the tet that is not part on the plaquette
                xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
                ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
                zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch) == 0:
                    swap_vertex = tet_ver

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                # match_cnt = match_cnt + 1

                # Randomly choose one of the vertices on the plauqette that is common
                # between the monopole and antimonopole tets
                # amn_swap_ver = random.choice(tri_locs[str_match[0]])
                amn_swap_ver = random.choice(mn_tri)

                # phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])],\
                #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]\
                #     = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])],\
                #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]

                phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                    = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                phi_arr = bnd_check(amn_swap_ver, phi_arr)
                # print('--6--');
                # bnd_test(phi_arr)

            while res[0][0] != 1:  # There is no antimonopole so find the next tet that the string is going into
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                # print(res[0][0])
                if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                    # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    nxt_strs = intersection(string_idxs, nxt_plqs)

                    if len(nxt_strs) == 1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        next_tri = tri_locs[nxt_strs][0]
                    if len(nxt_strs) == 2:
                        # print('yes')
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                        ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                        zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                        mtch = intersection(xmtch, intersection(ymtch, zmtch))
                        if len(mtch) == 0:
                            uncommon_vertex = vertex

                    phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    phi_arr = bnd_check(uncommon_vertex, phi_arr)
                    # print('--7--');
                    # bnd_test(phi_arr)

                    mn_tri = next_tri
                    swap_vertex = uncommon_vertex

                    if len(nxt_strs) == 0:
                        print('no outgoing string in the next tet')
                        break

                if res[0][0] == 1:
                    ##Antimonopole found##
                    # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    # print('Pair antimonopole found')
                    # match_cnt = match_cnt + 1

                    # Randomly choose one of the vertices on the plauqette that is common
                    # between the monopole and antimonopole tets.
                    amn_swap_ver = random.choice(next_tri)

                    phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    phi_arr = bnd_check(amn_swap_ver, phi_arr)
                    # print('--8--');
                    # bnd_test(phi_arr)

                    # print('pair found 2')
                    break

                if res[0][0] == 4:  # Hitting the boundary
                    str_plq = res[1]
                    ##finding the tri_coords of the reflected plk
                    xmtch = np.argwhere(tri_coords[:, 0] == str_plq[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:, 1] == str_plq[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:, 2] == str_plq[2])[:, 0]
                    mtch = intersection(xmtch, intersection(ymtch, zmtch))
                    pair_idxs.append(str_plq)
                    ##This is strictly for the code 2 to work
                    next_tri = tri_locs[mtch][0]
                    ##assigning name so mn_tri switches over to the reflected one
                    mn_tri = tri_locs[mtch][0]

                    ##test##

                    res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                    # print(res[0][0])

                    if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                    (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                        # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                        nxt_strs = intersection(string_idxs, nxt_plqs)

                        if len(nxt_strs) == 1:
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                            next_tri = tri_locs[nxt_strs][0]

                        if len(nxt_strs) == 2:
                            nxt_choice = random.choice(nxt_strs)
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                            next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                        if len(nxt_strs) == 0:
                            print('no outgoing string in the next tet')
                            break

                        for vertex in mn_tri:
                            xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                            mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            if len(mtch) == 0:
                                uncommon_vertex = vertex

                        phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        phi_arr = bnd_check(uncommon_vertex, phi_arr)
                        mn_tri = next_tri
                        swap_vertex = uncommon_vertex
                        # print('--9--');
                        # bnd_test(phi_arr)

                        if len(nxt_strs) == 0:
                            # print('no outgoing string in the next tet')
                            break

                    if res[0][0] == 1:
                        ##Antimonopole found##
                        # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                        antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        # print('Pair antimonopole found')
                        # match_cnt = match_cnt + 1

                        # Randomly choose one of the vertices on the plauqette that is common
                        # between the monopole and antimonopole tets.
                        amn_swap_ver = random.choice(next_tri)

                        phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]
                        phi_arr = bnd_check(amn_swap_ver, phi_arr)
                        # print('--10--');
                        # bnd_test(phi_arr)

                        # print('pair found 2')
                        break

                    if res[0][0] == 3:
                        print('----')
                        print('no match found?')
                        print(res)
                        print(str_plq)
                        print('----')
                        break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break

        if len(str_match) == 1:#Only 1 string coming our of the monopole tet
            mn_tri = tri_locs[str_match[0]]  # plauette triangle coordinates through which string passes
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == str_match)[0, 0])
            str_plq = np.mean(tri_locs[str_match[0]], axis=0)
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            # print(res[0][0])

            for tet_ver in mn_tet:#Finding vertex in the tet that is not part on the plaquette
                xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
                ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
                zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch)==0:
                    swap_vertex=tet_ver

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                # match_cnt = match_cnt + 1

                #Randomly choose one of the vertices on the plauqette that is common
                #between the monopole and antimonopole tets
                amn_swap_ver = random.choice(tri_locs[str_match[0]])

                phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
                    = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]

                phi_arr=bnd_check(amn_swap_ver,phi_arr)
                # print('--1--');bnd_test(phi_arr)

            while res[0][0]!=1:#There is no antimonopole so find the next tet that the string is going into
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                # print(res[0][0])
                if res[0][0] == 2:#Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4*res[1][0]),int(4*res[1][0]+1),(4*res[1][0]+2),(4*res[1][0]+3)]#plaquettes in the adjacent tet
                    #nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    nxt_strs = intersection(string_idxs, nxt_plqs)

                    if len(nxt_strs)==1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        next_tri = tri_locs[nxt_strs][0]
                    if len(nxt_strs) == 2:
                        # print('yes')
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0]==next_tri[:,0])[:, 0]
                        ymtch = np.argwhere(vertex[1]==next_tri[:,1])[:, 0]
                        zmtch = np.argwhere(vertex[2]==next_tri[:,2])[:, 0]
                        mtch=intersection(xmtch,intersection(ymtch,zmtch))
                        if len(mtch)==0:
                            uncommon_vertex=vertex

                    phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])]\
                        =phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]


                    phi_arr = bnd_check(uncommon_vertex, phi_arr)
                    # print('***************')
                    # print(uncommon_vertex)
                    # print(rf_chk)
                    # print('--2--')
                    # bnd_test(phi_arr)
                    # print('***************')
                    mn_tri=next_tri
                    swap_vertex=uncommon_vertex

                    if len(nxt_strs) == 0:
                        print('no outgoing string in the next tet')
                        break

                if res[0][0] == 1:
                    ##Antimonopole found##
                    # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    # print('Pair antimonopole found')
                    # match_cnt = match_cnt + 1

                    # Randomly choose one of the vertices on the plauqette that is common
                    # between the monopole and antimonopole tets.
                    amn_swap_ver = random.choice(next_tri)

                    phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    phi_arr = bnd_check(amn_swap_ver, phi_arr)
                    # print('--3--');
                    # bnd_test(phi_arr)
                    # print('pair found 2')
                    break

                if res[0][0] == 4:#Hitting the boundary
                    str_plq = res[1]
                    ##finding the tri_coords of the reflected plk
                    xmtch = np.argwhere(tri_coords[:,0] == str_plq[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:,1] == str_plq[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:,2] == str_plq[2])[:, 0]
                    mtch = intersection(xmtch, intersection(ymtch, zmtch))
                    pair_idxs.append(str_plq)
                    ##This is strictly for the code 2 to work
                    next_tri=tri_locs[mtch][0]
                    ##assigning name so mn_tri switches over to the reflected one
                    mn_tri =tri_locs[mtch][0]

                    res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

                    if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                    (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                        nxt_strs = intersection(string_idxs, nxt_plqs)

                        if len(nxt_strs) == 1:
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                            next_tri = tri_locs[nxt_strs][0]

                        if len(nxt_strs) == 2:
                            nxt_choice = random.choice(nxt_strs)
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                            next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                        if len(nxt_strs) == 0:
                            print('no outgoing string in the next tet')
                            break

                        for vertex in mn_tri:
                            xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                            mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            if len(mtch) == 0:
                                uncommon_vertex = vertex

                        phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]\
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        phi_arr = bnd_check(uncommon_vertex, phi_arr)

                        mn_tri = next_tri
                        swap_vertex = uncommon_vertex
                        # print('--4--')
                        # bnd_test(phi_arr)

                        if len(nxt_strs) == 0:
                            # print('no outgoing string in the next tet')
                            break

                    if res[0][0] == 1:
                        ##Antimonopole found##
                        # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                        antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        # print('Pair antimonopole found')
                        # match_cnt = match_cnt + 1

                        # Randomly choose one of the vertices on the plauqette that is common
                        # between the monopole and antimonopole tets.
                        amn_swap_ver = random.choice(next_tri)

                        phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        phi_arr = bnd_check(amn_swap_ver, phi_arr)

                        # print('--5--');
                        # bnd_test(phi_arr)

                        # print('pair found 2')
                        break

                    if res[0][0] == 3:
                        print('----')
                        print('no match found?')
                        print(res)
                        print(str_plq)
                        print('----')
                        break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break

    # bnd_test(phi_arr)
    return phi_arr

def collapse_opt(i):
    phi_arr = read('phi_arr_%s' % i, Master_path)
    area_stack = read('area_stack_%s' % i, Master_path)
    tet_locs = read('tet_locs_%s' % i, Master_path)
    deltas = read('deltas_%s' % i, Master_path)
    tri_locs = read('tri_locs_%s' % i, Master_path)

    string_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= 2 + err_mar, (np.real(deltas / (np.pi))) >= 2 - err_mar)
    )[:, 0]

    astring_idxs = np.argwhere(
        np.logical_and((np.real(deltas / (np.pi))) <= -2 + err_mar, (np.real(deltas / (np.pi))) >= -2 - err_mar)
    )[:, 0]

    normed = np.real(area_stack / (4 * np.pi))
    mono_idx = np.argwhere(np.logical_and(normed >= 1 - err_mar, normed <= 1 + err_mar))[:, 0]
    antimon_idx = np.argwhere(np.logical_and(normed <= -1 + err_mar, normed >= -1 - err_mar))[:, 0]
    # print('#Monopoles:%s'%(np.shape(mono_idx)))
    # print('#Antimonopoles:%s'%(np.shape(antimon_idx)))
    # pair_cnt = 0

    # str_locs = np.mean(tri_locs[string_idxs], axis=1)
    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
    tri_coords = np.mean(tri_locs,axis=1)
    tet_coords = np.mean(tet_locs, axis=1)
    # no_match_cnt = 0
    # match_cnt = 0
    # string_lists = []

    for mn_idx in mono_idx:
    # for mn_idx in [mono_idx[0]]:
        pair_idxs = []
        str_match = np.intersect1d(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        # str_match = intersection(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        # plk = np.mean(tri_locs[str_match[0]], axis=0)
        # pair_idxs.append(np.mean(tet_locs[mn_idx],axis=0))
        mono_idx = np.delete(mono_idx, np.argwhere(mono_idx == mn_idx)[0, 0])
        mn_tet=tet_locs[mn_idx]
        # print(mn_idx,len(str_match))

        if len(str_match) == 2:  # 2 strings coming our of the monopole tet
            choice = random.choice(str_match)

            mn_tri = tri_locs[choice]  # plauette triangle coordinates through which string passes
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
            # str_plq = np.mean(tri_locs[choice], axis=0)
            str_plq = tri_coords[choice]
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

            for tet_ver in mn_tet:  # Finding vertex in the tet that is not part on the plaquette
                xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
                ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
                zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
                # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                if len(mtch) == 0:
                    swap_vertex = tet_ver

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                # match_cnt = match_cnt + 1

                # Randomly choose one of the vertices on the plauqette that is common
                # between the monopole and antimonopole tets
                # amn_swap_ver = random.choice(tri_locs[str_match[0]])
                amn_swap_ver = random.choice(mn_tri)

                # phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])],\
                #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]\
                #     = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])],\
                #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]

                phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                    = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                phi_arr = bnd_check(amn_swap_ver, phi_arr)
                # print('--6--');
                # bnd_test(phi_arr)

            while res[0][0] != 1:  # There is no antimonopole so find the next tet that the string is going into
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                # print(res[0][0])
                if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                    nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    # nxt_strs = intersection(string_idxs, nxt_plqs)

                    if len(nxt_strs) == 1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        # str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        str_plq = tri_coords[nxt_strs[0]]
                        next_tri = tri_locs[nxt_strs][0]
                    if len(nxt_strs) == 2:
                        # print('yes')
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        # str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        str_plq = tri_coords[nxt_choice]
                        next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                        ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                        zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                        # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                        mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                        if len(mtch) == 0:
                            uncommon_vertex = vertex

                    phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    phi_arr = bnd_check(uncommon_vertex, phi_arr)
                    # print('--7--');
                    # bnd_test(phi_arr)

                    mn_tri = next_tri
                    swap_vertex = uncommon_vertex

                    if len(nxt_strs) == 0:
                        print('no outgoing string in the next tet')
                        break

                if res[0][0] == 1:
                    ##Antimonopole found##
                    # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    # print('Pair antimonopole found')
                    # match_cnt = match_cnt + 1

                    # Randomly choose one of the vertices on the plauqette that is common
                    # between the monopole and antimonopole tets.
                    amn_swap_ver = random.choice(next_tri)

                    phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    phi_arr = bnd_check(amn_swap_ver, phi_arr)
                    # print('--8--');
                    # bnd_test(phi_arr)

                    # print('pair found 2')
                    break

                if res[0][0] == 4:  # Hitting the boundary
                    str_plq = res[1]
                    ##finding the tri_coords of the reflected plk
                    xmtch = np.argwhere(tri_coords[:, 0] == str_plq[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:, 1] == str_plq[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:, 2] == str_plq[2])[:, 0]
                    # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                    mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                    pair_idxs.append(str_plq)
                    ##This is strictly for the code 2 to work
                    next_tri = tri_locs[mtch][0]
                    ##assigning name so mn_tri switches over to the reflected one
                    mn_tri = tri_locs[mtch][0]

                    ##test##

                    res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                    # print(res[0][0])

                    if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                    (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                        nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                        # nxt_strs = intersection(string_idxs, nxt_plqs)

                        if len(nxt_strs) == 1:
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                            # str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                            str_plq = tri_coords[nxt_strs[0]]
                            next_tri = tri_locs[nxt_strs][0]

                        if len(nxt_strs) == 2:
                            nxt_choice = random.choice(nxt_strs)
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                            # str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                            str_plq = tri_coords[nxt_choice]
                            next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                        if len(nxt_strs) == 0:
                            print('no outgoing string in the next tet')
                            break

                        for vertex in mn_tri:
                            xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                            # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                            if len(mtch) == 0:
                                uncommon_vertex = vertex

                        phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        phi_arr = bnd_check(uncommon_vertex, phi_arr)
                        mn_tri = next_tri
                        swap_vertex = uncommon_vertex
                        # print('--9--');
                        # bnd_test(phi_arr)

                        if len(nxt_strs) == 0:
                            # print('no outgoing string in the next tet')
                            break

                    if res[0][0] == 1:
                        ##Antimonopole found##
                        # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                        antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        # print('Pair antimonopole found')
                        # match_cnt = match_cnt + 1

                        # Randomly choose one of the vertices on the plauqette that is common
                        # between the monopole and antimonopole tets.
                        amn_swap_ver = random.choice(next_tri)

                        phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]
                        phi_arr = bnd_check(amn_swap_ver, phi_arr)
                        # print('--10--');
                        # bnd_test(phi_arr)

                        # print('pair found 2')
                        break

                    if res[0][0] == 3:
                        print('----')
                        print('no match found?')
                        print(res)
                        print(str_plq)
                        print('----')
                        break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break

        if len(str_match) == 1:#Only 1 string coming our of the monopole tet
            mn_tri = tri_locs[str_match[0]]  # plauette triangle coordinates through which string passes
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == str_match)[0, 0])
            # str_plq = np.mean(tri_locs[str_match[0]], axis=0)
            str_plq = tri_coords[str_match[0]]
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            # print(res[0][0])

            for tet_ver in mn_tet:#Finding vertex in the tet that is not part on the plaquette
                xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
                ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
                zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
                # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                if len(mtch)==0:
                    swap_vertex=tet_ver

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                # match_cnt = match_cnt + 1

                #Randomly choose one of the vertices on the plauqette that is common
                #between the monopole and antimonopole tets
                amn_swap_ver = random.choice(tri_locs[str_match[0]])

                phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
                    = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]

                phi_arr=bnd_check(amn_swap_ver,phi_arr)
                # print('--1--');bnd_test(phi_arr)

            while res[0][0]!=1:#There is no antimonopole so find the next tet that the string is going into
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                # print(res[0][0])
                if res[0][0] == 2:#Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4*res[1][0]),int(4*res[1][0]+1),(4*res[1][0]+2),(4*res[1][0]+3)]#plaquettes in the adjacent tet
                    nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    # nxt_strs = intersection(string_idxs, nxt_plqs)

                    if len(nxt_strs)==1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        # str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        str_plq = tri_coords[nxt_strs[0]]
                        next_tri = tri_locs[nxt_strs][0]
                    if len(nxt_strs) == 2:
                        # print('yes')
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        # str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        str_plq = tri_coords[nxt_choice]
                        next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0]==next_tri[:,0])[:, 0]
                        ymtch = np.argwhere(vertex[1]==next_tri[:,1])[:, 0]
                        zmtch = np.argwhere(vertex[2]==next_tri[:,2])[:, 0]
                        # mtch=intersection(xmtch,intersection(ymtch,zmtch))
                        mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                        if len(mtch)==0:
                            uncommon_vertex=vertex

                    phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])]\
                        =phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]


                    phi_arr = bnd_check(uncommon_vertex, phi_arr)
                    # print('***************')
                    # print(uncommon_vertex)
                    # print(rf_chk)
                    # print('--2--')
                    # bnd_test(phi_arr)
                    # print('***************')
                    mn_tri=next_tri
                    swap_vertex=uncommon_vertex

                    if len(nxt_strs) == 0:
                        print('no outgoing string in the next tet')
                        break

                if res[0][0] == 1:
                    ##Antimonopole found##
                    # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    # print('Pair antimonopole found')
                    # match_cnt = match_cnt + 1

                    # Randomly choose one of the vertices on the plauqette that is common
                    # between the monopole and antimonopole tets.
                    amn_swap_ver = random.choice(next_tri)

                    phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    phi_arr = bnd_check(amn_swap_ver, phi_arr)
                    # print('--3--');
                    # bnd_test(phi_arr)
                    # print('pair found 2')
                    break

                if res[0][0] == 4:#Hitting the boundary
                    str_plq = res[1]
                    ##finding the tri_coords of the reflected plk
                    xmtch = np.argwhere(tri_coords[:,0] == str_plq[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:,1] == str_plq[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:,2] == str_plq[2])[:, 0]
                    # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                    mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                    pair_idxs.append(str_plq)
                    ##This is strictly for the code 2 to work
                    next_tri=tri_locs[mtch][0]
                    ##assigning name so mn_tri switches over to the reflected one
                    mn_tri =tri_locs[mtch][0]

                    res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

                    if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                    (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                        # nxt_strs = intersection(string_idxs, nxt_plqs)
                        nxt_strs = np.intersect1d(string_idxs,nxt_plqs)

                        if len(nxt_strs) == 1:
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                            # str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                            str_plq = tri_coords[nxt_strs[0]]
                            next_tri = tri_locs[nxt_strs][0]

                        if len(nxt_strs) == 2:
                            nxt_choice = random.choice(nxt_strs)
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                            # str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                            str_plq = tri_coords[nxt_choice]
                            next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                        if len(nxt_strs) == 0:
                            print('no outgoing string in the next tet')
                            break

                        for vertex in mn_tri:
                            xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                            # mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            mtch = np.intersect1d(xmtch, np.intersect1d(ymtch, zmtch))
                            if len(mtch) == 0:
                                uncommon_vertex = vertex

                        phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]\
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        phi_arr = bnd_check(uncommon_vertex, phi_arr)

                        mn_tri = next_tri
                        swap_vertex = uncommon_vertex
                        # print('--4--')
                        # bnd_test(phi_arr)

                        if len(nxt_strs) == 0:
                            # print('no outgoing string in the next tet')
                            break

                    if res[0][0] == 1:
                        ##Antimonopole found##
                        # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                        antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        # print('Pair antimonopole found')
                        # match_cnt = match_cnt + 1

                        # Randomly choose one of the vertices on the plauqette that is common
                        # between the monopole and antimonopole tets.
                        amn_swap_ver = random.choice(next_tri)

                        phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        phi_arr = bnd_check(amn_swap_ver, phi_arr)

                        # print('--5--');
                        # bnd_test(phi_arr)

                        # print('pair found 2')
                        break

                    if res[0][0] == 3:
                        print('----')
                        print('no match found?')
                        print(res)
                        print(str_plq)
                        print('----')
                        break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break

    # bnd_test(phi_arr)
    return phi_arr

def loop_collapser_junk(i):

    phi_arr = read('phi_arr_%s' % i, Master_path)

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
    loop_list = list(read('loop_lists_%s' % i, Master_path))
    tri_coords = np.mean(tri_locs, axis=1)
    print(len(loop_list))
    # print(loop_list)
    for lp in loop_list:
        if len(lp) > 1:

            plk = random.choice(lp)
            plk = lp[0]

            xmtch = np.argwhere(tri_coords[:, 0] == plk[0])[:, 0]
            ymtch = np.argwhere(tri_coords[:, 1] == plk[1])[:, 0]
            zmtch = np.argwhere(tri_coords[:, 2] == plk[2])[:, 0]
            mtch = intersection(xmtch, intersection(ymtch, zmtch))

            choice = random.choice(tri_locs[mtch[0]])

            cds = [int(c) for c in choice]
            phi = phi_arr[cds[0], cds[1], cds[2]]
            n_vec = np.real(n(phi, sigma))
            new_phi = flip_n(n_vec)

            phi_arr[cds[0], cds[1], cds[2]]=new_phi
            phi_arr=bnd_check(choice,phi_arr)
        bnd_test(phi_arr)

    return phi_arr

def cd_id_matcher(inp,list):
    #returns the idx of coordinate list where inp coordinate is
    xmtch = np.argwhere(list[:, 0] == inp[0])[:, 0]
    ymtch = np.argwhere(list[:, 1] == inp[1])[:, 0]
    zmtch = np.argwhere(list[:, 2] == inp[2])[:, 0]
    mtch = intersection(xmtch, intersection(ymtch, zmtch))
    return mtch[0]

def loop_collapser(i):

    phi_arr = read('phi_arr_%s' % i, Master_path)

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
    loop_list = list(read('loop_lists_%s' % i, Master_path))
    tri_coords = np.mean(tri_locs, axis=1)
    # print(np.shape(tri_locs),np.shape(tri_coords))
    # print(len(loop_list))
    # print(loop_list)
    bnds = [0.0, float(N - 1)]
    tet_coords = np.mean(tet_locs,axis=1)
    # print(np.shape(tet_locs))
    # print(np.shape(np.mean(tet_locs,axis=1)));exit()
    # [print(len(l)) for l in loop_list]
    loop_list = [lp for lp in loop_list if len(lp)>1]
    for lp in loop_list:
        # print('******',len(lp),'*******')

        ##prepping the loop list##
        # print(np.round(lp,2))
        lp.pop(-1)#popping the last duplicate
        # plk = random.choice(lp)
        # start_i = random.choice(range(len(lp) - 3))
        loop_len = len(lp)

        start_i = random.choice(range(loop_len))
        lp=lp*2 #double copying the list so periodicity works and if
                #start tri is at the end of the list then it next tri loops around to the start

        plk=lp[start_i]
        next_i=start_i+2
        # last_i=start_i+4
        # last_plk = lp[last_i]
        nxt_plk = lp[next_i]

        sfty=0
        while (np.isin(plk[0], bnds) or np.isin(plk[1], bnds) or np.isin(plk[2], bnds)
               or np.isin(nxt_plk[0], bnds) or np.isin(nxt_plk[1], bnds) or np.isin(nxt_plk[2], bnds)
               or (plk[0]-int(plk[0]))*4==0.0 or (plk[1]-int(plk[1]))*4==0.0 or (plk[2]-int(plk[2]))*4==0.0):
            sfty+=1
            if sfty>len(lp): break
            # plk = random.choice(lp)
            # start_i = random.choice(range(len(lp)-3))
            start_i = random.choice(range(loop_len))
            plk = lp[start_i]
            next_i = start_i+2
            nxt_plk = lp[next_i]
            # last_i = start_i + 4
            # last_plk = lp[last_i]
        if sfty > len(lp): print('breaking:No suitable start');continue

        # ##can't start at a tet##
        # while ((plk[0]-int(plk[0]))*4==0.0 or (plk[1]-int(plk[1]))*4==0.0 or (plk[2]-int(plk[2]))*4==0.0):
        #     # plk = random.choice(lp)
        #     # start_i = random.choice(range(len(lp)-3))
        #     start_i = random.choice(range(loop_len))
        #     plk = lp[start_i]
        #     next_i = start_i+2
        #     nxt_plk = lp[next_i]
        #     # last_i = start_i + 4
        #     # last_plk = lp[last_i]

        start_tet_loc = lp[start_i+1]
        next_tet_loc = lp[next_i+1]
        # print(plk);print(start_tet_loc);print(nxt_plk);print(next_tet_loc)
        start_tri = tri_locs[cd_id_matcher(plk,tri_coords)]
        next_tri = tri_locs[cd_id_matcher(nxt_plk,tri_coords)]
        # last_tri = tri_locs[cd_id_matcher(last_plk,tri_coords)]
        # print(start_tri,next_tri,last_tri);exit()
        start_tet = tet_locs[cd_id_matcher(start_tet_loc,tet_coords)]
        next_tet = tet_locs[cd_id_matcher(next_tet_loc,tet_coords)]

        # print(start_tri);print(next_tri);
        # print(start_tet);print(next_tet)

        #find the uncommon vertex '1' between start tet and next tri
        for ver in start_tet:
            xmtch = np.argwhere(ver[0] == next_tri[:, 0])[:, 0]
            ymtch = np.argwhere(ver[1] == next_tri[:, 1])[:, 0]
            zmtch = np.argwhere(ver[2] == next_tri[:, 2])[:, 0]
            mtch = intersection(xmtch, intersection(ymtch, zmtch))
            if len(mtch) == 0:
                ver_1 = ver

        for ver in next_tet:
            xmtch = np.argwhere(ver[0] == next_tri[:, 0])[:, 0]
            ymtch = np.argwhere(ver[1] == next_tri[:, 1])[:, 0]
            zmtch = np.argwhere(ver[2] == next_tri[:, 2])[:, 0]
            mtch = intersection(xmtch, intersection(ymtch, zmtch))
            if len(mtch) == 0:
                ver_5 = ver

        #find common edge ('2' and '3') between start triangle and next triangle and the
        #uncommon vertex '4'
        commons = []
        for ver in next_tri:
            xmtch = np.argwhere(ver[0] == start_tri[:, 0])[:, 0]
            ymtch = np.argwhere(ver[1] == start_tri[:, 1])[:, 0]
            zmtch = np.argwhere(ver[2] == start_tri[:, 2])[:, 0]
            mtch = intersection(xmtch, intersection(ymtch, zmtch))
            if len(mtch) == 0:
                uncommon_ver = ver
            else:
                commons.append(ver)

        uncommon_ver=[int(c) for c in uncommon_ver]
        commons = [[int(c) for c in j] for j in commons]

        phi_1, phi_2, phi_3, phi_4,phi_5,\
            n_1, n_2, n_3, n_4,n_5,\
            del_124_o, del_143_o, del_123_o, del_234_o=\
                deltas_tet(uncommon_ver,commons,ver_1,ver_5,phi_arr)
        del_124_o, del_143_o, del_123_o, del_234_o = \
            [round(d/(2*np.pi),5) for d in [del_124_o,del_143_o,del_123_o,del_234_o]]

        print('---------------------------')
        print(tet_charge(start_tet, phi_arr))
        print(del_234_o,del_124_o, del_143_o, del_123_o);

        print(np.sign(np.dot(n_1,np.cross(n_2,n_3))))
        print(np.sign(np.dot(n_5, np.cross(n_2, n_3))))
        sign_123 = int(np.sign(np.dot(n_1,np.cross(n_2,n_3))))
        sign_523 = int(np.sign(np.dot(n_5, np.cross(n_2, n_3))))

        sfty1=0
        while sign_523!=sign_123 and sfty1<=10:
            sfty1+=1
            start_i = random.choice(range(loop_len))
            lp = lp * 2  # double copying the list so periodicity works and if
            # start tri is at the end of the list then it next tri loops around to the start

            plk = lp[start_i]
            next_i = start_i + 2
            # last_i=start_i+4
            # last_plk = lp[last_i]
            nxt_plk = lp[next_i]

            sfty = 0
            while (np.isin(plk[0], bnds) or np.isin(plk[1], bnds) or np.isin(plk[2], bnds)
                   or np.isin(nxt_plk[0], bnds) or np.isin(nxt_plk[1], bnds) or np.isin(nxt_plk[2], bnds)
                   or (plk[0] - int(plk[0])) * 4 == 0.0 or (plk[1] - int(plk[1])) * 4 == 0.0 or (
                           plk[2] - int(plk[2])) * 4 == 0.0):
                sfty += 1
                if sfty > len(lp): break
                # plk = random.choice(lp)
                # start_i = random.choice(range(len(lp)-3))
                start_i = random.choice(range(loop_len))
                plk = lp[start_i]
                next_i = start_i + 2
                nxt_plk = lp[next_i]
                # last_i = start_i + 4
                # last_plk = lp[last_i]
            if sfty > len(lp): print('breaking:No suitable start');continue

            # ##can't start at a tet##
            # while ((plk[0]-int(plk[0]))*4==0.0 or (plk[1]-int(plk[1]))*4==0.0 or (plk[2]-int(plk[2]))*4==0.0):
            #     # plk = random.choice(lp)
            #     # start_i = random.choice(range(len(lp)-3))
            #     start_i = random.choice(range(loop_len))
            #     plk = lp[start_i]
            #     next_i = start_i+2
            #     nxt_plk = lp[next_i]
            #     # last_i = start_i + 4
            #     # last_plk = lp[last_i]

            start_tet_loc = lp[start_i + 1]
            next_tet_loc = lp[next_i + 1]
            # print(plk);print(start_tet_loc);print(nxt_plk);print(next_tet_loc)
            start_tri = tri_locs[cd_id_matcher(plk, tri_coords)]
            next_tri = tri_locs[cd_id_matcher(nxt_plk, tri_coords)]
            # last_tri = tri_locs[cd_id_matcher(last_plk,tri_coords)]
            # print(start_tri,next_tri,last_tri);exit()
            start_tet = tet_locs[cd_id_matcher(start_tet_loc, tet_coords)]
            next_tet = tet_locs[cd_id_matcher(next_tet_loc, tet_coords)]

            # print(start_tri);print(next_tri);
            # print(start_tet);print(next_tet)

            # find the uncommon vertex '1' between start tet and next tri
            for ver in start_tet:
                xmtch = np.argwhere(ver[0] == next_tri[:, 0])[:, 0]
                ymtch = np.argwhere(ver[1] == next_tri[:, 1])[:, 0]
                zmtch = np.argwhere(ver[2] == next_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch) == 0:
                    ver_1 = ver

            for ver in next_tet:
                xmtch = np.argwhere(ver[0] == next_tri[:, 0])[:, 0]
                ymtch = np.argwhere(ver[1] == next_tri[:, 1])[:, 0]
                zmtch = np.argwhere(ver[2] == next_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch) == 0:
                    ver_5 = ver

            # find common edge ('2' and '3') between start triangle and next triangle and the
            # uncommon vertex '4'
            commons = []
            for ver in next_tri:
                xmtch = np.argwhere(ver[0] == start_tri[:, 0])[:, 0]
                ymtch = np.argwhere(ver[1] == start_tri[:, 1])[:, 0]
                zmtch = np.argwhere(ver[2] == start_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch) == 0:
                    uncommon_ver = ver
                else:
                    commons.append(ver)

            uncommon_ver = [int(c) for c in uncommon_ver]
            commons = [[int(c) for c in j] for j in commons]

            phi_1, phi_2, phi_3, phi_4, phi_5, \
                n_1, n_2, n_3, n_4, n_5, \
                del_124_o, del_143_o, del_123_o, del_234_o = \
                deltas_tet(uncommon_ver, commons, ver_1, ver_5, phi_arr)
            del_124_o, del_143_o, del_123_o, del_234_o = \
                [round(d / (2 * np.pi), 5) for d in [del_124_o, del_143_o, del_123_o, del_234_o]]

            print('---------------------------')
            print(tet_charge(start_tet, phi_arr))
            print(del_234_o, del_124_o, del_143_o, del_123_o);

            print(np.sign(np.dot(n_1, np.cross(n_2, n_3))))
            print(np.sign(np.dot(n_5, np.cross(n_2, n_3))))
            sign_123 = int(np.sign(np.dot(n_1, np.cross(n_2, n_3))))
            sign_523 = np.sign(np.dot(n_5, np.cross(n_2, n_3)))

        if sfty1==10: print('skipping:cant find right n5 on the right side of n2-n3');continue
        # exit()

        phi_4_n=n_tnsfm(n_2,n_3,n_1)
        if len(phi_4_n)!=2: print('skipping:cant flip correctly');continue

        phi_1, phi_2, phi_3, phi_4,phi_5,\
            n_1, n_2, n_3, n_4,n_5,\
            del_124, del_143, del_123, del_234=\
            deltas_tet(uncommon_ver,commons,ver_1,ver_5,phi_arr)
        del_124, del_143, del_123, del_234 = \
            [round(d/(2*np.pi),5) for d in [del_124,del_143,del_123,del_234]]

        # print(del_234,del_124,del_143,del_123)

        sfty=0
        while ((abs(del_234)!=0.0) or (abs(del_124)!=del_124_o)
               or (abs(del_143)!=del_143_o) or (abs(del_123)!=del_123_o)) and sfty<=100:
            sfty+=1
            rndmn_phase=random.uniform(0,2*np.pi)
            phi_4_n=np.exp(1j*rndmn_phase)*phi_4_n
            phi_arr[uncommon_ver[0], uncommon_ver[1], uncommon_ver[2]] = phi_4_n
            # bnd_check(uncommon_ver, phi_arr)

            phi_1, phi_2, phi_3, phi_4,phi_5,\
                n_1, n_2, n_3, n_4,n_5,\
                del_124, del_143, del_123, del_234=\
                deltas_tet(uncommon_ver,commons,ver_1,ver_5,phi_arr)
            del_124, del_143, del_123, del_234 = \
                [round(d/(2*np.pi),5) for d in [del_124,del_143,del_123,del_234]]
        if sfty==100: print('skipping:strings are not in the right place in the choice');continue

        # rndmn_phase=random.uniform(0,2*np.pi)
        # phi_4_n=np.exp(1j*rndmn_phase)*phi_4_n
        # phi_arr[uncommon_ver[0], uncommon_ver[1], uncommon_ver[2]] = phi_4_n
        #
        # phi_1, phi_2, phi_3, phi_4,\
        #     n_1, n_2, n_3, n_4,\
        #     del_124, del_143, del_123, del_234=deltas_tet(uncommon_ver,commons,ver_1,phi_arr)
        # del_124, del_143, del_123, del_234 = \
        #     [round(d/(2*np.pi),5) for d in [del_124,del_143,del_123,del_234]]
        #
        # bnd_check(uncommon_ver, phi_arr)

        print(del_234, del_124, del_143, del_123)
        # print(n_1);print(n_2);print(n_3);print(n_4)
        new_start_tet_chg = tet_charge(start_tet, phi_arr)
        new_next_tet_chg=tet_charge(next_tet,phi_arr)
        # sfty=0
        # while abs(new_start_tet_chg)!=0 and sfty<1000:
        #     phi_4_n = n_tnsfm(n_2, n_3, n_1)
        #     phi_arr[uncommon_ver[0], uncommon_ver[1], uncommon_ver[2]] = phi_4_n
        #     new_start_tet_chg = tet_charge(start_tet, phi_arr)
        #
        #     sfty+=1

        print(new_start_tet_chg)
        print(new_next_tet_chg)

        bnd_check(uncommon_ver,phi_arr)
        # print(phi_arr[uncommon_ver[0],uncommon_ver[1],uncommon_ver[2]])

        # print(start_tet)
        # print(tet_charge(start_tet,phi_arr))
        # print(tet_charge(next_tet, phi_arr))

        print('---------------------------'); break

        # exit()
        # choice = random.choice(tri_locs[mtch[0]])
        #
        # cds = [int(c) for c in choice]
        # phi = phi_arr[cds[0], cds[1], cds[2]]
        # n_vec = np.real(n(phi, sigma))
        # new_phi = flip_n(n_vec)
        #
        # phi_arr[cds[0], cds[1], cds[2]]=new_phi
        # phi_arr=bnd_check(choice,phi_arr)
    # bnd_test(phi_arr)

    return phi_arr

def loop_collapser_test(i):

    phi_arr = read('phi_arr_%s' % i, Master_path)

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
    loop_list = list(read('loop_lists_%s' % i, Master_path))
    tri_coords = np.mean(tri_locs, axis=1)
    print(len(loop_list))
    # print(loop_list)

    bnds = [0.0,float(N-1)]

    for lp in loop_list:
        len_lp = len(lp)
        if len_lp > 1:
            # for plk in [lp[0],lp[-1]]:
            itr=0
            for plk in lp:

                itr+=1
                if (np.isin(plk[0], bnds) or np.isin(plk[1], bnds) or np.isin(plk[2], bnds)):
                    # plk=lp[itr]
                    itr += 1
                print(itr)
                if itr<len_lp:
                    # plk = random.choice(lp)
                    # plk = lp[0]
                    plk=lp[itr-1]
                    nxt_plk = lp[itr]

                    xmtch = np.argwhere(tri_coords[:, 0] == plk[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:, 1] == plk[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:, 2] == plk[2])[:, 0]
                    mtch = intersection(xmtch, intersection(ymtch, zmtch))

                    xmtch = np.argwhere(tri_coords[:, 0] == nxt_plk[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:, 1] == nxt_plk[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:, 2] == nxt_plk[2])[:, 0]
                    nxt_mtch = intersection(xmtch, intersection(ymtch, zmtch))

                    if len(mtch)!=0 and len(nxt_mtch)!=0:

                        plq_crds = tri_locs[mtch[0]];
                        plq_crds = [[int(c) for c in cds] for cds in plq_crds]

                        nxt_tri = tri_locs[nxt_mtch[0]]

                        for ver in tri_locs[mtch[0]]:  # Finding vertex in the tet that is not part on the plaquette
                            xmtch = np.argwhere(ver[0] == nxt_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(ver[1] == nxt_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(ver[2] == nxt_tri[:, 2])[:, 0]
                            mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            if len(mtch) == 0:
                                choice = ver

                        # choice = random.choice(tri_locs[mtch[0]])

                        cds = [int(c) for c in choice]
                        # phi = phi_arr[cds[0], cds[1], cds[2]]
                        # n_vec = np.real(n(phi, sigma))
                        # new_phi = flip_n(n_vec)

                        # plq_crds = tri_locs[mtch[0]];
                        # plq_crds = [[int(c) for c in cds] for cds in plq_crds]

                        phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
                        # print(phis)
                        n_vecs = [n(phi, sigma) for phi in phis];
                        delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)
                        # print(delt)
                        phi_arr[cds[0], cds[1], cds[2]]=np.exp(1j*random.uniform(0,2*np.pi))*phi_arr[cds[0], cds[1], cds[2]]
                        # print(phis)
                        phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
                        n_vecs = [n(phi, sigma) for phi in phis];
                        delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)

                        while abs(np.round(delt/(2*np.pi),5)) == 1.0:
                            phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
                            n_vecs = [n(phi, sigma) for phi in phis];
                            delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)
                            phi_arr[cds[0], cds[1], cds[2]] = np.exp(1j * random.uniform(0, 2 * np.pi)) * phi_arr[
                                cds[0], cds[1], cds[2]]
                        # print(delt);exit()

                        phi_arr=bnd_check(choice,phi_arr)
    bnd_test(phi_arr)
    return phi_arr

def loop_collapser_test2(i):
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

    return phi_arr

def collapse_test(i):
    #Working on a version that phase shifts the phi such that
    #it is ensured that no new strings are created when moving through a plaquette
    #Partially progressed on testing but too many issues that would need resolving through
    #iterations or resolving too many special conditional statements.
    phi_arr = read('phi_arr_%s' % i, Master_path)
    area_stack = read('area_stack_%s' % i, Master_path)
    tet_locs = read('tet_locs_%s' % i, Master_path)
    deltas = read('deltas_%s' % i, Master_path)
    tri_locs = read('tri_locs_%s' % i, Master_path)
    tet_stk = read('tet_stack_%s' % i, Master_path)

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
    tri_coords = np.mean(tri_locs,axis=1)
    no_match_cnt = 0
    match_cnt = 0
    string_lists = []

    for mn_idx in mono_idx:
    # for mn_idx in [mono_idx[0]]:
        pair_idxs = []
        #str_match = np.intersect1d(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        str_match = intersection(string_idxs,[int(4 * mn_idx),int(4 * mn_idx)+1,int(4 * mn_idx)+2,int(4 * mn_idx+3)])
        plk = np.mean(tri_locs[str_match[0]], axis=0)
        # pair_idxs.append(np.mean(tet_locs[mn_idx],axis=0))
        mono_idx = np.delete(mono_idx, np.argwhere(mono_idx == mn_idx)[0, 0])
        mn_tet=tet_locs[mn_idx]
        print(mn_idx,len(str_match))

        if len(str_match) == 2:  # 2 strings coming our of the monopole tet
            choice = random.choice(str_match)

            mn_tri = tri_locs[choice]  # plauette triangle coordinates through which string passes
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
            str_plq = np.mean(tri_locs[choice], axis=0)
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

            ##Removing string on plq through which we are moving the monopole through
            plq_crds = tri_locs[choice];
            plq_crds = [[int(c) for c in cds] for cds in plq_crds]
            phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
            n_vecs = [n(phi, sigma) for phi in phis];
            delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)

            # print(res[0][0])
            for tet_ver in mn_tet:  # Finding vertex in the tet that is not part on the plaquette
                xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
                ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
                zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch) == 0:
                    swap_vertex = tet_ver

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                match_cnt = match_cnt + 1

                # Randomly choose one of the vertices on the plauqette that is common
                # between the monopole and antimonopole tets
                # amn_swap_ver = random.choice(tri_locs[str_match[0]])
                amn_swap_ver = random.choice(mn_tri)

                # phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])],\
                #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]\
                #     = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])],\
                #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]

                phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                    = np.exp(1j * delt) * phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                ##Removing string on plq through which we are moving the monopole through
                plq_crds = mn_tri;
                plq_crds = [[int(c) for c in cds] for cds in plq_crds]
                phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
                n_vecs = [n(phi, sigma) for phi in phis];
                delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)
                print(delt);

                # rf_chk =bnd_check(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
                # if rf_chk[0]:
                #     phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])]=\
                #         phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                phi_arr = bnd_check(amn_swap_ver, phi_arr)
                print('--6--');
                bnd_test(phi_arr)

            while res[0][0] != 1:  # There is no antimonopole so find the next tet that the string is going into
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                print(res[0][0])
                if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                    # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    nxt_strs = intersection(string_idxs, nxt_plqs)

                    if len(nxt_strs) == 1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        next_tri = tri_locs[nxt_strs][0]
                    if len(nxt_strs) == 2:
                        # print('yes')
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                        ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                        zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                        mtch = intersection(xmtch, intersection(ymtch, zmtch))
                        if len(mtch) == 0:
                            uncommon_vertex = vertex
                    # Do the swap: swap ver: og uncommon vertex in the mono tet;
                    # uncommomn: the one to be swapped to
                    # phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])],\
                    #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]\
                    #     =phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])],\
                    #     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]

                    ##Removing string on plq through which we are moving the monopole through
                    plq_crds = next_tri;
                    plq_crds = [[int(c) for c in cds] for cds in plq_crds]
                    phis = [phi_arr[i[0], i[1], i[2]] for i in plq_crds]
                    n_vecs = [n(phi, sigma) for phi in phis];
                    delt = strings(n_vecs[0], n_vecs[1], n_vecs[2], phis[0], phis[1], phis[2], I, sigma)
                    print(delt / (2 * np.pi));
                    exit()

                    phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    # rf_chk = bnd_check(int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2]))
                    # if rf_chk[0]:
                    #     phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])] = \
                    #         phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]
                    phi_arr = bnd_check(uncommon_vertex, phi_arr)
                    # print(uncommon_vertex)
                    # print(rf_chk)
                    print('--7--');
                    bnd_test(phi_arr)

                    mn_tri = next_tri
                    swap_vertex = uncommon_vertex

                    if len(nxt_strs) == 0:
                        print('no outgoing string in the next tet')
                        break

                if res[0][0] == 1:
                    ##Antimonopole found##
                    # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    # print('Pair antimonopole found')
                    match_cnt = match_cnt + 1

                    # Randomly choose one of the vertices on the plauqette that is common
                    # between the monopole and antimonopole tets.
                    amn_swap_ver = random.choice(next_tri)

                    # phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])], \
                    #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
                    #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
                    #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                    phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    # rf_chk = bnd_check(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
                    # if rf_chk[0]:
                    #     phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])] = \
                    #         phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                    phi_arr = bnd_check(amn_swap_ver, phi_arr)
                    print('--8--');
                    bnd_test(phi_arr)

                    # print('pair found 2')
                    break

                if res[0][0] == 4:  # Hitting the boundary
                    str_plq = res[1]
                    ##finding the tri_coords of the reflected plk
                    xmtch = np.argwhere(tri_coords[:, 0] == str_plq[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:, 1] == str_plq[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:, 2] == str_plq[2])[:, 0]
                    mtch = intersection(xmtch, intersection(ymtch, zmtch))
                    pair_idxs.append(str_plq)
                    ##This is strictly for the code 2 to work
                    next_tri = tri_locs[mtch][0]
                    ##assigning name so mn_tri switches over to the reflected one
                    mn_tri = tri_locs[mtch][0]

                    ##test##

                    res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                    # print(res[0][0])

                    if res[0][
                        0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                    (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                        # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                        nxt_strs = intersection(string_idxs, nxt_plqs)

                        if len(nxt_strs) == 1:
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                            next_tri = tri_locs[nxt_strs][0]

                        if len(nxt_strs) == 2:
                            nxt_choice = random.choice(nxt_strs)
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                            next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                        if len(nxt_strs) == 0:
                            print('no outgoing string in the next tet')
                            break

                        for vertex in mn_tri:
                            xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                            mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            if len(mtch) == 0:
                                uncommon_vertex = vertex
                        # Do the swap: swap ver: og uncommon vertex in the mono tet;
                        # uncommomn: the one to be swapped to
                        # phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])], \
                        #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
                        #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
                        #     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]

                        phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        ##need to do this but don't know why this breaks things
                        # ref_bound_ver = ref_plk(int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2]))
                        # phi_arr[int(ref_bound_ver[0]), int(ref_bound_ver[1]), int(ref_bound_ver[2])]\
                        #     =phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]
                        phi_arr = bnd_check(uncommon_vertex, phi_arr)
                        mn_tri = next_tri
                        swap_vertex = uncommon_vertex
                        print('--9--');
                        bnd_test(phi_arr)

                        if len(nxt_strs) == 0:
                            # print('no outgoing string in the next tet')
                            break

                    if res[0][0] == 1:
                        ##Antimonopole found##
                        # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                        antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        # print('Pair antimonopole found')
                        match_cnt = match_cnt + 1

                        # Randomly choose one of the vertices on the plauqette that is common
                        # between the monopole and antimonopole tets.
                        amn_swap_ver = random.choice(next_tri)

                        # phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])], \
                        #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
                        #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
                        #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                        phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]
                        ##need to do this but don't know why this breaks things
                        # ref_bound_ver = ref_plk(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
                        # phi_arr[int(ref_bound_ver[0]), int(ref_bound_ver[1]), int(ref_bound_ver[2])]\
                        #     =phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                        phi_arr = bnd_check(amn_swap_ver, phi_arr)
                        print('--10--');
                        bnd_test(phi_arr)

                        # print('pair found 2')
                        break

                    if res[0][0] == 3:
                        print('----')
                        print('no match found?')
                        print(res)
                        print(str_plq)
                        print('----')
                        break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break

        if len(str_match) == 1:#Only 1 string coming our of the monopole tet
            mn_tri = tri_locs[str_match[0]]  # plauette triangle coordinates through which string passes
            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == str_match)[0, 0])
            str_plq = np.mean(tri_locs[str_match[0]], axis=0)
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

            # print(res[0][0])
            for tet_ver in mn_tet:#Finding vertex in the tet that is not part on the plaquette
                xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
                ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
                zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                if len(mtch)==0:
                    swap_vertex=tet_ver

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                match_cnt = match_cnt + 1

                #Randomly choose one of the vertices on the plauqette that is common
                #between the monopole and antimonopole tets
                amn_swap_ver = random.choice(tri_locs[str_match[0]])

                # phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])],\
                #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]\
                #     = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])],\
                #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]

                phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
                    = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]

                # rf_chk =bnd_check(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
                # if rf_chk[0]:
                #     phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])]=\
                #         phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                phi_arr=bnd_check(amn_swap_ver,phi_arr)
                print('--1--');bnd_test(phi_arr)

            while res[0][0]!=1:#There is no antimonopole so find the next tet that the string is going into
                res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                print(res[0][0])
                if res[0][0] == 2:#Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    nxt_plqs = [int(4*res[1][0]),int(4*res[1][0]+1),(4*res[1][0]+2),(4*res[1][0]+3)]#plaquettes in the adjacent tet
                    #nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                    nxt_strs = intersection(string_idxs, nxt_plqs)

                    print(len(nxt_strs));print(mn_tri);print(next_tri)
                    delt=del_plk(mn_tri,phi_arr);del_plk(next_tri,phi_arr)

                    if len(nxt_strs)==1:
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                        next_tri = tri_locs[nxt_strs][0]
                        swp_factor=1##Swap factor is 1 if only 1 outgoing string
                    if len(nxt_strs) == 2:
                        # print('yes')
                        nxt_choice = random.choice(nxt_strs)
                        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                        str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                        next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()
                        swp_factor = exp(-i*delt)  ##Swap factor is 1 if only 1 outgoing string

                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0]==next_tri[:,0])[:, 0]
                        ymtch = np.argwhere(vertex[1]==next_tri[:,1])[:, 0]
                        zmtch = np.argwhere(vertex[2]==next_tri[:,2])[:, 0]
                        mtch=intersection(xmtch,intersection(ymtch,zmtch))
                        if len(mtch)==0:
                            uncommon_vertex=vertex
                    #Do the swap: swap ver: og uncommon vertex in the mono tet;
                    # uncommomn: the one to be swapped to
                    # phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])],\
                    #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]\
                    #     =phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])],\
                    #     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]

                    phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])]\
                        =swp_factor*phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]

                    # rf_chk = bnd_check(int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2]))
                    # if rf_chk[0]:
                    #     phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])] = \
                    #         phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]

                    del_plk(mn_tri, phi_arr);del_plk(next_tri,phi_arr)

                    phi_arr = bnd_check(uncommon_vertex, phi_arr)

                    del_plk(mn_tri, phi_arr);del_plk(next_tri,phi_arr)
                    # print('***************')
                    # print(uncommon_vertex)
                    # print(rf_chk)
                    print('--2--')
                    bnd_test(phi_arr)
                    # print('***************')
                    mn_tri=next_tri
                    swap_vertex=uncommon_vertex

                    if len(nxt_strs) == 0:
                        print('no outgoing string in the next tet')
                        break

                if res[0][0] == 1:
                    ##Antimonopole found##
                    # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                    antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                    astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                    astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                    # print('Pair antimonopole found')
                    match_cnt = match_cnt + 1

                    # Randomly choose one of the vertices on the plauqette that is common
                    # between the monopole and antimonopole tets.
                    amn_swap_ver = random.choice(next_tri)

                    # phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])], \
                    #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
                    #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
                    #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                    phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
                        = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                    # rf_chk = bnd_check(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
                    # if rf_chk[0]:
                    #     phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])] = \
                    #         phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                    phi_arr = bnd_check(amn_swap_ver, phi_arr)
                    print('--3--');
                    bnd_test(phi_arr)

                    # print('pair found 2')
                    break

                if res[0][0] == 4:#Hitting the boundary
                    str_plq = res[1]
                    ##finding the tri_coords of the reflected plk
                    xmtch = np.argwhere(tri_coords[:,0] == str_plq[0])[:, 0]
                    ymtch = np.argwhere(tri_coords[:,1] == str_plq[1])[:, 0]
                    zmtch = np.argwhere(tri_coords[:,2] == str_plq[2])[:, 0]
                    mtch = intersection(xmtch, intersection(ymtch, zmtch))
                    pair_idxs.append(str_plq)
                    ##This is strictly for the code 2 to work
                    next_tri=tri_locs[mtch][0]
                    ##assigning name so mn_tri switches over to the reflected one
                    mn_tri =tri_locs[mtch][0]

                    ##test##

                    res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
                    # print(res[0][0])

                    if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                                    (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                        # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                        nxt_strs = intersection(string_idxs, nxt_plqs)

                        if len(nxt_strs) == 1:
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                            next_tri = tri_locs[nxt_strs][0]

                        if len(nxt_strs) == 2:
                            nxt_choice = random.choice(nxt_strs)
                            string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                            str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                            next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()

                        ##Removing string on plq through which we are moving the monopole through
                        del_plk(mn_tri,phi_arr)
                        del_plk(next_tri, phi_arr)

                        if len(nxt_strs) == 0:
                            print('no outgoing string in the next tet')
                            break

                        for vertex in mn_tri:
                            xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                            ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                            zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                            mtch = intersection(xmtch, intersection(ymtch, zmtch))
                            if len(mtch) == 0:
                                uncommon_vertex = vertex
                        # Do the swap: swap ver: og uncommon vertex in the mono tet;
                        # uncommomn: the one to be swapped to
                        # phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])], \
                        #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
                        #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
                        #     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]

                        phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]\
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        del_plk(mn_tri,phi_arr);del_plk(next_tri,phi_arr)

                        ##need to do this but don't know why this breaks things
                        # ref_bound_ver = ref_plk(int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2]))
                        # phi_arr[int(ref_bound_ver[0]), int(ref_bound_ver[1]), int(ref_bound_ver[2])]\
                        #     =phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]

                        phi_arr = bnd_check(uncommon_vertex, phi_arr)
                        del_plk(mn_tri,phi_arr);del_plk(next_tri,phi_arr)

                        mn_tri = next_tri
                        swap_vertex = uncommon_vertex
                        print('--4--')
                        bnd_test(phi_arr)

                        if len(nxt_strs) == 0:
                            # print('no outgoing string in the next tet')
                            break

                    if res[0][0] == 1:
                        ##Antimonopole found##
                        # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                        antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                        astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                        astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                        # print('Pair antimonopole found')
                        match_cnt = match_cnt + 1

                        # Randomly choose one of the vertices on the plauqette that is common
                        # between the monopole and antimonopole tets.
                        amn_swap_ver = random.choice(next_tri)

                        del_plk(next_tri, phi_arr)

                        # phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])], \
                        #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
                        #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
                        #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                        phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                            = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]

                        del_plk(next_tri, phi_arr)

                        ##need to do this but don't know why this breaks things
                        # ref_bound_ver = ref_plk(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
                        # phi_arr[int(ref_bound_ver[0]), int(ref_bound_ver[1]), int(ref_bound_ver[2])]\
                        #     =phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
                        phi_arr = bnd_check(amn_swap_ver, phi_arr)

                        print('--5--');
                        del_plk(next_tri, phi_arr)
                        print(next_tri)
                        bnd_test(phi_arr)

                        # print('pair found 2')
                        break

                    if res[0][0] == 3:
                        print('----')
                        print('no match found?')
                        print(res)
                        print(str_plq)
                        print('----')
                        break

                if res[0][0] == 3:
                    print('----')
                    print('no match found?')
                    print(res)
                    print(str_plq)
                    print('----')
                    break


        # if len(str_match) == 2:  # 2 strings coming our of the monopole tet
        #     choice = random.choice(str_match)
        #     mn_tri = tri_locs[choice]  # plauette triangle coordinates through which string passes
        #     string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
        #     str_plq = np.mean(tri_locs[str_match[0]], axis=0)
        #     res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
        #     # print(res[0][0])
        #     for tet_ver in mn_tet:  # Finding vertex in the tet that is not part on the plaquette
        #         xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
        #         ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
        #         zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
        #         mtch = intersection(xmtch, intersection(ymtch, zmtch))
        #         if len(mtch) == 0:
        #             swap_vertex = tet_ver
        #
        #     if res[0][0] == 1:
        #         ##Antimonopole found##
        #         # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
        #         antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
        #         astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
        #         astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
        #         # print('Pair antimonopole found')
        #         match_cnt = match_cnt + 1
        #
        #         # Randomly choose one of the vertices on the plauqette that is common
        #         # between the monopole and antimonopole tets
        #         amn_swap_ver = random.choice(tri_locs[str_match[0]])
        #
        #         # phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])],\
        #         #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]\
        #         #     = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])],\
        #         #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
        #         phi_arr[int(amn_swap_ver[0]),int(amn_swap_ver[1]), int(amn_swap_ver[2])]\
        #             = phi_arr[int(swap_vertex[0]),int(swap_vertex[1]), int(swap_vertex[2])]
        #         rf_chk =bnd_check(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
        #         if rf_chk[0]:
        #             phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])]=\
        #                 phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
        #     print('--6--');
        #     bnd_test(phi_arr)
        #     while res[0][0] != 1:  # There is no antimonopole so find the next tet that the string is going into
        #         res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
        #         # print(res[0][0])
        #
        #         if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
        #             astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
        #             astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
        #             nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
        #                         (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
        #             # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
        #             nxt_strs = intersection(string_idxs, nxt_plqs)
        #
        #             if len(nxt_strs) == 1:
        #                 string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
        #                 str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
        #                 next_tri = tri_locs[nxt_strs][0]
        #
        #             if len(nxt_strs) == 2:
        #                 # print('yes')
        #                 nxt_choice = random.choice(nxt_strs)
        #                 string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
        #                 str_plq = np.mean(tri_locs[nxt_choice], axis=0)
        #                 next_tri = tri_locs[nxt_choice]
        #             if len(nxt_strs) == 0:
        #                 print('no outgoing string in the next tet')
        #                 break
        #             for vertex in mn_tri:
        #                 xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
        #                 ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
        #                 zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
        #                 mtch = intersection(xmtch, intersection(ymtch, zmtch))
        #                 if len(mtch) == 0:
        #                     uncommon_vertex = vertex
        #             mn_tri = next_tri
        #             # Do the swap: swap ver: og uncommon vertex in the mono tet;
        #             # uncommomn: the one to be swapped to
        #             # phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])],\
        #             #     phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]\
        #             #     =phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])],\
        #             #     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]
        #             phi_arr[int(uncommon_vertex[0]),int(uncommon_vertex[1]),int(uncommon_vertex[2])]\
        #                 =phi_arr[int(swap_vertex[0]),int(swap_vertex[1]),int(swap_vertex[2])]
        #             rf_chk = bnd_check(int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2]))
        #             if rf_chk[0]:
        #                 phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])] = \
        #                     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]
        #             swap_vertex=uncommon_vertex
        #             print('--7--');
        #             bnd_test(phi_arr)
        #
        #             if len(nxt_strs) == 0:
        #                 # print('no outgoing string in the next tet')
        #                 break
        #
        #         if res[0][0] == 1:
        #             ##Antimonopole found##
        #             # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
        #             antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
        #             astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
        #             astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
        #             # print('Pair antimonopole found')
        #             match_cnt = match_cnt + 1
        #
        #             # Randomly choose one of the vertices on the plauqette that is common
        #             # between the monopole and antimonopole tets
        #             amn_swap_ver = random.choice(next_tri)
        #
        #             # phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])], \
        #             #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
        #             #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
        #             #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
        #             phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
        #                 = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]
        #             rf_chk = bnd_check(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
        #             if rf_chk[0]:
        #                 phi_arr[int(rf_chk[1][0]), int(rf_chk[1][1]), int(rf_chk[1][2])] = \
        #                     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
        #             # print('pair found 2')
        #             print('--8--');
        #             bnd_test(phi_arr)
        #             break
        #
        #         if res[0][0] == 4:  # Hitting the boundary
        #             str_plq = res[1]
        #
        #             ##finding the tri_coords of the reflected plk
        #             xmtch = np.argwhere(tri_coords[:,0] == str_plq[0])[:, 0]
        #             ymtch = np.argwhere(tri_coords[:,1] == str_plq[1])[:, 0]
        #             zmtch = np.argwhere(tri_coords[:,2] == str_plq[2])[:, 0]
        #             mtch = intersection(xmtch, intersection(ymtch, zmtch))
        #             pair_idxs.append(str_plq)
        #             ##This is strictly for the code 2 to work
        #             next_tri=tri_locs[mtch][0]
        #             ##assigning name so mn_tri switches over to the reflected one
        #             mn_tri =tri_locs[mtch][0]
        #
        #             ##test##
        #             res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
        #
        #
        #             if res[0][0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
        #                 astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
        #                 astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
        #                 nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
        #                             (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
        #                 # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
        #                 nxt_strs = intersection(string_idxs, nxt_plqs)
        #
        #                 if len(nxt_strs) == 1:
        #                     string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
        #                     str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
        #                     next_tri = tri_locs[nxt_strs][0]
        #                 if len(nxt_strs) == 2:
        #                     # print('yes')
        #                     nxt_choice = random.choice(nxt_strs)
        #                     string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
        #                     str_plq = np.mean(tri_locs[nxt_choice], axis=0)
        #                     next_tri = tri_locs[nxt_choice]  # ;print(next_tri);exit()
        #                 if len(nxt_strs) == 0:
        #                     print('no outgoing string in the next tet')
        #                     break
        #                 for vertex in mn_tri:
        #                     xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
        #                     ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
        #                     zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
        #                     mtch = intersection(xmtch, intersection(ymtch, zmtch))
        #                     if len(mtch) == 0:
        #                         uncommon_vertex = vertex
        #                 # Do the swap: swap ver: og uncommon vertex in the mono tet;
        #                 # uncommomn: the one to be swapped to
        #                 # phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])], \
        #                 #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
        #                 #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
        #                 #     phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]
        #                 phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
        #                     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]
        #
        #                 ##need to do this but don't know why this breaks things
        #                 ref_bound_ver = ref_plk(int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2]))
        #                 phi_arr[int(ref_bound_ver[0]), int(ref_bound_ver[1]), int(ref_bound_ver[2])]\
        #                     =phi_arr[int(uncommon_vertex[0]), int(uncommon_vertex[1]), int(uncommon_vertex[2])]
        #                 print('--9--');
        #                 bnd_test(phi_arr)
        #                 mn_tri = next_tri
        #                 swap_vertex = uncommon_vertex
        #
        #                 if len(nxt_strs) == 0:
        #                     # print('no outgoing string in the next tet')
        #                     break
        #
        #             if res[0][0] == 1:
        #                 ##Antimonopole found##
        #                 # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
        #                 antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
        #                 astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
        #                 astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
        #                 # print('Pair antimonopole found')
        #                 match_cnt = match_cnt + 1
        #
        #                 # Randomly choose one of the vertices on the plauqette that is common
        #                 # between the monopole and antimonopole tets.
        #                 amn_swap_ver = random.choice(next_tri)
        #
        #                 # phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])], \
        #                 #     phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])] \
        #                 #     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])], \
        #                 #     phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
        #                 phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
        #                     = phi_arr[int(swap_vertex[0]), int(swap_vertex[1]), int(swap_vertex[2])]
        #
        #                 ##need to do this but don't know why this breaks things
        #                 ref_bound_ver = ref_plk(int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2]))
        #                 phi_arr[int(ref_bound_ver[0]), int(ref_bound_ver[1]), int(ref_bound_ver[2])] \
        #                     = phi_arr[int(amn_swap_ver[0]), int(amn_swap_ver[1]), int(amn_swap_ver[2])]
        #                 print('--10--');
        #                 bnd_test(phi_arr)
        #                 break
        #
        #         if res[0][0] == 3:
        #             print('----')
        #             print('no match found?')
        #             print(res)
        #             print(str_plq)
        #             print('----')
        #             break


    bnd_test(phi_arr)
    return phi_arr

def collapser2():
    if len(str_match) == 2:  # Only 1 string coming our of the monopole tet
        choice = random.choice(str_match)
        mn_tri = tri_locs[choice]  # plauette triangle coordinates through which string passes
        string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == choice)[0, 0])
        str_plq = np.mean(tri_locs[str_match[0]], axis=0)
        res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)

        for tet_ver in mn_tet:  # Finding vertex in the tet that is not part on the plaquette
            xmtch = np.argwhere(tet_ver[0] == mn_tri[:, 0])[:, 0]
            ymtch = np.argwhere(tet_ver[1] == mn_tri[:, 1])[:, 0]
            zmtch = np.argwhere(tet_ver[2] == mn_tri[:, 2])[:, 0]
            mtch = intersection(xmtch, intersection(ymtch, zmtch))
            if len(mtch) == 0:
                swap_vertex = tet_ver

        if res[0][0] == 1:
            ##Antimonopole found##
            # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
            antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
            astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
            astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
            # print('Pair antimonopole found')
            match_cnt = match_cnt + 1

            # Randomly choose one of the vertices on the plauqette that is common
            # between the monopole and antimonopole tets
            amn_swap_ver = random.choice(tri_locs[str_match[0]])

            phi_arr[int(amn_swap_ver[0]),
            int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                = phi_arr[int(swap_vertex[0]),
            int(swap_vertex[1]), int(swap_vertex[2])]

        while res[0][0] != 1:  # There is no antimonopole so find the next tet that the string is going into
            res = matching(str_plq, astr_locs, antimon_idx, astring_idxs)
            # print(res[0][0])

            if res[0][
                0] == 2:  # Found the antistring but no antimonopole so find the string coming out of the adjacent tet
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                nxt_plqs = [int(4 * res[1][0]), int(4 * res[1][0] + 1), (4 * res[1][0] + 2),
                            (4 * res[1][0] + 3)]  # plaquettes in the adjacent tet
                # nxt_strs = np.intersect1d(string_idxs,nxt_plqs)
                nxt_strs = intersection(string_idxs, nxt_plqs)

                if len(nxt_strs) == 1:
                    string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_strs)[0, 0])
                    str_plq = np.mean(tri_locs[nxt_strs[0]], axis=0)
                    next_tri = tri_locs[nxt_strs][0]
                    for vertex in mn_tri:
                        xmtch = np.argwhere(vertex[0] == next_tri[:, 0])[:, 0]
                        ymtch = np.argwhere(vertex[1] == next_tri[:, 1])[:, 0]
                        zmtch = np.argwhere(vertex[2] == next_tri[:, 2])[:, 0]
                        mtch = intersection(xmtch, intersection(ymtch, zmtch))
                        if len(mtch) == 0:
                            uncommon_vertex = vertex
                    # print('start',mn_tet)
                    # print('next',next_tri)
                    # print('ucommon vertex',uncommon_vertex)
                    # print('mono tet',mn_tet)
                    # print('next tet', tet_locs[res[1][0]])
                    # print('swap ver',swap_vertex)
                    # Do the swap: swap ver: og uncommon vertex in the mono tet;
                    # uncommomn: the one to be swapped to
                    phi_arr[int(uncommon_vertex[0]),
                    int(uncommon_vertex[1]), int(uncommon_vertex[2])] \
                        = phi_arr[int(swap_vertex[0]),
                    int(swap_vertex[1]), int(swap_vertex[2])]
                if len(nxt_strs) == 0:
                    # print('no outgoing string in the next tet')
                    break
                if len(nxt_strs) == 2:
                    # print('yes')
                    nxt_choice = random.choice(nxt_strs)
                    string_idxs = np.delete(string_idxs, np.argwhere(string_idxs == nxt_choice)[0, 0])
                    str_plq = np.mean(tri_locs[nxt_choice], axis=0)
                    # tet_cen = np.mean(tet_locs[res[1][0]], axis=0)
                    # pair_idxs.append(tet_cen)
                    # pair_idxs.append(str_plq)

            if res[0][0] == 1:
                ##Antimonopole found##
                # pair_idxs.append(np.mean(tet_locs[res[1][0]][0], axis=0))
                antimon_idx = np.delete(antimon_idx, np.argwhere(antimon_idx == res[1][0])[0, 0])
                astring_idxs = np.delete(astring_idxs, np.argwhere(astring_idxs == res[2][0])[0, 0])
                astr_locs = np.mean(tri_locs[astring_idxs], axis=1)
                # print('Pair antimonopole found')
                match_cnt = match_cnt + 1

                # Randomly choose one of the vertices on the plauqette that is common
                # between the monopole and antimonopole tets
                amn_swap_ver = random.choice(next_tri)

                phi_arr[int(amn_swap_ver[0]),
                int(amn_swap_ver[1]), int(amn_swap_ver[2])] \
                    = phi_arr[int(swap_vertex[0]),
                int(swap_vertex[1]), int(swap_vertex[2])]
                # print('pair found 2')
                break

            if res[0][0] == 4:  # Hitting the boundary
                str_plq = res[1]
                xmtch = np.argwhere(tri_coords[:, 0] == str_plq[0])[:, 0]
                ymtch = np.argwhere(tri_coords[:, 1] == str_plq[1])[:, 0]
                zmtch = np.argwhere(tri_coords[:, 2] == str_plq[2])[:, 0]
                mtch = intersection(xmtch, intersection(ymtch, zmtch))
                pair_idxs.append(str_plq)
                next_tri = tri_locs[mtch][0]

            if res[0][0] == 3:
                print('----')
                print('no match found?')
                print(res)
                print(str_plq)
                print('----')
                break
    return
