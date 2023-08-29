import random

import matplotlib.pyplot as plt
import numpy as np

from Lattice import *
from tets import *
from collapser import *
import matplotlib.animation as animation
import plotly.graph_objects as go
# import pyvista as pv

def run_mag(i):
    print('Computing Derivatives')
    start = time.time()
    phi_arr_i = read("phi_arr_%s"%i,Master_path)
    write("diff_arr_%s" %i, Master_path, derivatives(phi_arr_i,diff_arr,i))
    print('Time Elapsed', np.abs(time.time() - start))

    start = time.time()
    diff_arr_i = read("diff_arr_%s" % i, Master_path)

    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print('');       print('');        print('')

    print('Computing B across lattice')
    start = time.time()

    stack = np.zeros((N, N, N, 3), dtype=complex)
    stack_mag = np.zeros((N, N, N), dtype=complex)
    B_vec = np.zeros(3, dtype=complex)

    B_arr_i, B_mag_arr_i = B_stack(diff_arr_i, stack, stack_mag, B_vec)
    write("B_arr_%s" % i, Master_path, np.real(B_arr_i))
    write("B_mag_arr_%s" % i, Master_path, B_mag_arr_i)

    print('Time Elapsed for B arr', np.abs(time.time() - start))
    print('');        print('');        print('')
    print('----------------------------------------------------------')
    print(np.shape(B_arr_i))

    # fig = go.Figure(data=go.Volume(x=B_mag_arr_i[:, :, :, 0].flatten(),
    #                                y=B_mag_arr_i[:, :, :, 1].flatten(), z=B_mag_arr_i[:, :, :, 2].flatten()
    #                                , value=B_mag_arr_i[:, :, :, 3].flatten(), opacity=0.075,
    #                                colorscale='Blues', surface_count=50,
    #                                colorbar={"title": "|B|"}))
    X,Y,Z=np.meshgrid(x_arr,y_arr,z_arr)

    B_mag_cut = 1e-3
    B_mag_arr_i[np.real(B_mag_arr_i) <B_mag_cut]=B_mag_cut
    cleaned_B=np.real(B_mag_arr_i)
    cntrs = np.log(cleaned_B).flatten()
    B_mag_arr_i_re = np.real(B_arr_i)
    cntrs = cleaned_B.flatten()
    fig = go.Figure(data=go.Volume(x=X.flatten(),
                                   y=Y.flatten(), z=Z.flatten(),
                                   value=cntrs, opacity=0.075,
                                   surface_count=20,
                                   colorbar={"title": "(|B|)"}))
    #Starting integration on a spherical shell around th monopoles
    xis = np.random.choice(X.flatten(),N)
    yis = np.random.choice(Y.flatten(), N)
    zis = np.random.choice(Z.flatten(), N)
    # fig=go.Figure()
    fig.add_trace(go.Streamtube(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        u=B_mag_arr_i_re[:, :, :, 0].flatten()/cleaned_B.flatten(),
        v=B_mag_arr_i_re[:, :, :, 1].flatten()/cleaned_B.flatten(),
        w=B_mag_arr_i_re[:, :, :, 2].flatten()/cleaned_B.flatten(),
        starts=dict(x=xis,y=yis,z=zis),
        sizeref=0.5,
        colorscale='amp',
        showscale=False,
        maxdisplayed=500))
    fig.write_html(Master_path + '/plots/3d_mag_%05i.html'%i)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    u = B_mag_arr_i_re[:, :, :, 0].flatten()
    v = B_mag_arr_i_re[:, :, :, 1].flatten()
    w = B_mag_arr_i_re[:, :, :, 2].flatten()
    ax.quiver3D(x, y, z, u, v, w)#, normalize=True)#,arrow_length_ratio=0.5,length=0.1)
    # plt.show()
    fig.savefig('%s/3d_mag_quiver.pdf' % Plot_path)

    # exit()
    #

    # nx = ny=nz=N
    # spacing = 0.1
    # origin = (-(nx - 1) * spacing / 2, -(ny - 1) * spacing / 2, -(nz - 1) * spacing / 2)
    # origin = (3,3,3)
    # mesh = pv.UniformGrid(dimensions=(nx, ny, nz),
    #                       spacing=(spacing, spacing, spacing), origin=origin)
    # # print(mesh.n_points)
    # vectors = np.empty((mesh.n_points,3))
    # vectors[:,0]=B_mag_arr_i_re[:,:,:,0].flatten()
    # vectors[:, 1] = B_mag_arr_i_re[:, :, :, 0].flatten()
    # vectors[:, 2] = B_mag_arr_i_re[:, :, :, 0].flatten()
    # mesh['vectors']=vectors
    # stream, src = mesh.streamlines(
    #     'vectors', return_source=True, terminal_speed=0.0, n_points=200, source_radius=0.1
    # )
    # p=pv.Plotter()
    # p.add_mesh(mesh.outline())
    # p.add_mesh(stream.tube(radius=0.01))
    # p.show()
    # # stream.tube(radius=0.0015).plot()
    # exit()

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
    print('----------------------------------------------------------');
    plt.clf();plt.plot(np.real(E_M[:,0]),np.real(E_M[:,1]));
    plt.savefig('%s/B_k_%05i.png'%(Plot_path,i),dpi=600)
    # exit()

    return

def run_multiple_all(i):

    print('')
    print('----------------------------------------------------------')
    print('Generating randomized lattices')
    print('')
    start = time.time()
    data = jit_populate(phi_arr, Hoft_arr)
    write("phi_arr_%s" % (i), Master_path, data[0])
    write("hoft_arr_%s" % (i), Master_path, data[1])

    ft_file = open('%s/ft_dump.bin'%Master_path, 'wb+')
    ft_data_re = np.asfortranarray(np.real(data[0])).reshape((int(N*N*N),2))
    ft_datr_im= np.asfortranarray(np.imag(data[0])).reshape((int(N*N*N),2))
    # print(ft_data)
    print(np.shape(ft_data_re))
    ft_data = np.concatenate((ft_data_re,ft_datr_im),axis=1)
    print(np.shape(ft_data))
    np.savetxt('%s/ft_dump.txt'%Master_path,ft_data)
    # ft_data.tofile(ft_file);exit()

    print('Time Elapsed', np.abs(time.time() - start))

    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Computing Topological quantities')
    start = time.time()
    run_tets2(i)
    print('Time for computing topological quantities:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Plotting monopoles')
    # mono_scatter_plot(i)
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Finding Pairs')
    start = time.time()
    tracing_test(i)
    print('Time for tracing:%s'%(abs(start-time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Finding Loops')
    start = time.time()
    # looping(i)
    print('Time for loops:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Plotting Network')
    tet_2_plot(i)
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Computing B and plotting')
    run_mag(i)
    print('----------------------------------------------------------')
    print('')

    return

def dumbbell_collapse_initial():
    start = time.time()

    # phi_arr = read('phi_arr_%s' % 0, Master_path)
    # area_stack = read('area_stack_%s' % 0, Master_path)
    # tet_locs = read('tet_locs_%s' % 0, Master_path)
    # deltas = read('deltas_%s' % 0, Master_path)
    # tri_locs = read('tri_locs_%s' % 0, Master_path)
    #
    # new_phi_arr = collapse(0,phi_arr,area_stack,tet_locs,deltas,tri_locs)

    new_phi_arr = collapse_opt(0)

    write("phi_arr_%s" % (1), Master_path, new_phi_arr)
    print('Time Elapsed', np.abs(time.time() - start))

    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Computing Topological quantities')
    start = time.time()
    no_mn, no_amn = run_tets2(1)
    print('Time for computing topological quantities:%s' % (abs(start - time.time())))
    print('----------------------------------------------------------')
    print('')

    print('----------------------------------------------------------')
    print('Plotting monopoles')
    # mono_scatter_plot(1)
    print('----------------------------------------------------------')
    print('')

    tracing_test(1)
    # looping(1)
    tet_2_plot(1)

    it = 1
    while no_mn[1] != 0:
        # for it in range(it,20):
        new_phi_arr = collapse_opt(it)
        os.system('rm %s/phi_arr_%s.npy' % (Master_path,it))
        os.system('rm %s/*_%s.npy' % (Master_path, it))
        it = it+1
        write("phi_arr_%s" % (it), Master_path, new_phi_arr)
        no_mn, no_amn = run_tets2(it)
        tracing_test(it)
        # no_loops = looping(it)
        tet_2_plot(it)
        if no_mn[1] != no_amn[1]:
            print('Big error monopole-antimonopole number discrepancy')
            break

        print('-------------')
        print('Iteration:', it)
        print(no_mn, no_amn)
        print('-------------')
    print('-------------')
    print('Total iterations for collapsing initial dumbbells:%i' % it)
    print('-------------')

    print('-------------')
    print('Initial loops left after collapsing initial dumbbells:%i' % it)
    print('-------------')
    print('----------------------------')
    print('Finished collapsing')
    print('----------------------------')

    return it

def lp_clp(it):
    new_phi_arr = loop_collapser(it)
    os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
    os.system('rm %s/*_%s.npy' % (Master_path, it))
    it = it+1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)

    tracing(it)
    looping(it)

    tet_2_plot(it)  # ;exit()

    new_phi_arr = collapse(it)
    os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
    os.system('rm %s/*_%s.npy' % (Master_path, it))
    it = it+1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)

    tracing(it)
    looping(it)
    tet_2_plot(it)

    while no_mn[1] != 0:
        # for it in range(it,20):
        new_phi_arr = collapse(it)
        os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
        os.system('rm %s/*_%s.npy' % (Master_path, it))
        it = it+1
        write("phi_arr_%s" % (it), Master_path, new_phi_arr)
        no_mn, no_amn = run_tets2(it)
        tracing(it)
        looping(it)
        tet_2_plot(it)
        if no_mn[1] != no_amn[1]:
            print('Big error monopole-antimonopole number discrepancy')
            break


    new_phi_arr = loop_collapser(it)
    os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
    os.system('rm %s/*_%s.npy' % (Master_path, it))
    it = it+1
    write("phi_arr_%s" % (it), Master_path, new_phi_arr)
    no_mn, no_amn = run_tets2(it)

    tracing(it)
    no_loops = looping(it)
    tet_2_plot(it)

    # for j in range(100):
    sfty=0
    while (no_loops!= 0 or no_mn[1]!=0) and sfty <= iter_cap:
        sfty=sfty+1
        new_phi_arr = collapse(it)
        os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
        os.system('rm %s/*_%s.npy' % (Master_path, it))
        it = it+1
        write("phi_arr_%s" % (it), Master_path, new_phi_arr)
        no_mn, no_amn = run_tets2(it)

        tracing(it)
        looping(it)
        tet_2_plot(it)

        while no_mn[1] != 0:
            # for it in range(it,20):
            new_phi_arr = collapse(it)
            os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
            os.system('rm %s/*_%s.npy' % (Master_path, it))
            it = it+1
            write("phi_arr_%s" % (it), Master_path, new_phi_arr)
            no_mn, no_amn = run_tets2(it)
            tracing(it)
            looping(it)
            tet_2_plot(it)
            if no_mn[1] != no_amn[1]:
                print('Big error monopole-antimonopole number discrepancy')
                break


        new_phi_arr = loop_collapser(it)
        os.system('rm %s/phi_arr_%s.npy' % (Master_path, it))
        os.system('rm %s/*_%s.npy' % (Master_path, it))
        it = it+1
        write("phi_arr_%s" % (it), Master_path, new_phi_arr)
        no_mn, no_amn = run_tets2(it)

        tracing(it)
        no_loops = looping(it)
        tet_2_plot(it)
        print('**************')
        print('#loop collapse iteration%i' % it)
        print('#Loops:%i' % (no_loops))
        print('**************')

    print('---------------------------')
    print('Total iterations:%i' % it)
    if it == iter_cap: print('Exceeded allowed iterations !Quitting!')
    print('---------------------------')
    tet_2_plot(it)

    print('----------------------------------------------------------')
    print('Computing B and plotting')
    run_mag(it)
    print('----------------------------------------------------------')
    print('')

    return it

import os as os
os.system('rm %s/*.png'%Plot_path)

run_multiple_all(0);exit()
# tet_2_plot(0)
# exit()
it = dumbbell_collapse_initial()
run_mag(it)

exit()

it = lp_clp(it)

# for j in range(it+1):
#     tet_2_plot(j)

# os.chdir(Plot_path)
# os.system('ffmpeg -framerate 1 -pattern_type glob -i '*.png' -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4')

# fig,ax = plt.subplots()
# log_anim = animation.FuncAnimation(fig,tet_2_plot,frames=it)
# log_anim.save('%s/animation.mp4'%(Plot_path), writer=animation.FFMpegWriter())



