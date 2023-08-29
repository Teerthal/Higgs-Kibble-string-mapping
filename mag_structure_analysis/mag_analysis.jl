using PyCall
include("topo_fns.jl")
using .topo_routines

include("gen_routines.jl")
using .base_routines

using PyPlot

using NPZ
using WriteVTK


nudge_factor = 0.5
i=0

Master_path ="/home/cuddlypuff/Dropbox (ASU)/dumbell_MHD/topological_collapse/mag_structure_analysis/test/"
# Master_path ="/home/teerthal/Dropbox (ASU)/dumbell_MHD/topological_collapse/mag_structure_analysis/test/"
# Master_path="/home/teerthal/Dropbox (ASU)/dumbell_MHD/topological_collapse/test/"

scriptdir = @__DIR__
pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
# prepper = pyimport("mag_analysis_routines")
# no_mn,no_amn = prepper.initialize_topo_nopair(i)
topo = pyimport("tets")
mag_py = pyimport("Lattice")

# err_marg = 1e-3
err_marg = 1e-1


@time string_list = pairing(i,Master_path,err_marg)

# scaled_strs = [(Int.(round.(i)) for i in string_list[k].*12) for k in range(1,size(string_list,1),step=1)]
# lines = [MeshCell(PolyData.Lines(),l) for l in scaled_strs[1]]

# topo.tet_plot_jl(i,string_list,err_marg)
# topo.tet_plot_jl(i,string_list,err_marg,xlims=[1,2],ylims=[0,1],zlims=[1,2])
# exit()

phi_arr = npzread(string(Master_path,"phi_arr_",i,".npy"))
N = size(phi_arr)[1]

topo.tet_plot_slice_jl(i,string_list,err_marg,N)

gw=0.65
lambda=1.0
vev=1.0
sw = sqrt(0.22)

dp_arr_0 = zeros(ComplexF64,(N,N,N,2,3))
B_arr_0 = zeros(Float64,(N,N,N,6))
# B_mag_arr_0 = zeros(Float64,(N,N,N))
A_arr_0 = zeros(ComplexF64,(N,N,N,3))
curl_A_0 = zeros(Float64,(N,N,N,3))

# dphi_arr = pbc_cen_diff(phi_arr,dp_arr_0,N)
# B_arr = B_cal_arr(dphi_arr,B_arr_0,N)
# A_arr = A_cal_arr(dphi_arr,phi_arr,A_arr_0,N).*(2*sw*vev/gw)
# curl_A = curl_A_arr(A_arr, curl_A_0,N)

diff_phi_arr = mag_py.derivatives(phi_arr, dp_arr_0)
A_arr = mag_py.A_stack(diff_phi_arr, phi_arr, A_arr_0,interp=false)
curl_A = mag_py.curl_A(A_arr,curl_A_0,interp=false)
B_arr = curl_A
# B_arr,B_mag_arr =mag_py.B_stack(diff_phi_arr, zeros(Float64,(N,N,N,3)), zeros(Float64,(N,N,N)),zeros(Float64,(3)),interp=false)
coord_stack = zeros(Float64,(3,N,N,N))
for i in range(1,N,step=1)
    for j in range(1,N,step=1)
        for k in range(1,N,step=1)
            coord_stack[:,i,j,k]=[i,j,k]
        end
    end
end

# pygui(true)
# using3D()
# subplot(111, projection="3d")
# # plt=PyPlot.PyObject(PyPlot.axes3D)
# # fig = figure()
# # ax = fig.gca(projection="3d")
# # ax = fig.gca()
# x = coord_stack[1,:,:,:].-1
# y = coord_stack[2,:,:,:].-1
# z = coord_stack[3,:,:,:].-1
# # u = B_arr[:,:,:,4]
# # v = B_arr[:,:,:,5]
# # w = B_arr[:,:,:,6]
# u = B_arr[:,:,:,1]
# v = B_arr[:,:,:,2]
# w = B_arr[:,:,:,3]

# quiver(x,y,z, u,v,w,normalize=true,length = 0.1)
# xlim([0,1])
# ylim([6,7.2])
# zlim([0,1])
# show()

for n in range(1,N,step=1)
    gcf()
    # pygui(true)
    x = coord_stack[1,n,1:2:end,1:2:end].-1
    y = coord_stack[2,n,1:2:end,1:2:end].-1
    z = coord_stack[3,n,1:2:end,1:2:end].-1
    mag = sqrt((B_arr[n,1:2:end,1:2:end,1]).^2+(B_arr[n,1:2:end,1:2:end,2]).^2+(B_arr[n,1:2:end,1:2:end,3]).^2)
    u = B_arr[n,1:2:end,1:2:end,1]#/mag
    v = B_arr[n,1:2:end,1:2:end,2]#/mag
    w = B_arr[n,1:2:end,1:2:end,3]#/mag
    # println(size(y),size(v))
    im=contourf(transpose(y),transpose(z),transpose(log(mag)),20)
    streamplot(transpose(y),transpose(z),transpose(v),transpose(w),linewidth=0.65,
    arrowsize=0.5,arrowstyle="->",color="k")
    quiver(transpose(y),transpose(z),transpose(v),transpose(w))
    # show()
    xlim([0,N-1])
    ylim([0,N-1])
    title(string("x:",n))
    colorbar(im)
    savefig(string("test/plots/mag_slice_",n,".png"),dpi=600)
    close()
    
end


# points = vcat(vec(Int.(x)),vec(Int.(y)),vec(Int.(z)))
# y = vec(Int.(y))
# z = vec(Int.(z))
# println(size(points))
# vtk_grid(string("raw_",i),Int.(coord_stack.*12),lines) do vtk
#     # vtk["B"] = (u,v,w)
#     # vtk["E"] = E
#     # vtk["B"] = sqrt.((B_x.^2).+(B_y).^2 .+(B_z))
# end

# vtk_grid(string("raw_",i),x,y,z) do vtk
#     vtk["B"] = (u,v,w)
#     # vtk["strings"] = (string_list[1][:][1],string_list[1][:][2],string_list[1][:][3])
#     # vtk["E"] = E
#     # vtk["B"] = sqrt.((B_x.^2).+(B_y).^2 .+(B_z))
# end