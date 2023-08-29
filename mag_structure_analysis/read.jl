using PyCall
include("topo_fns.jl")
using .topo_routines

include("collapse_swap.jl")
using .swap_collapse

include("collapse_main.jl")
using .main_collapse

nudge_factor = 0.5
i=0

Master_path ="/home/cuddlypuff/Dropbox (ASU)/dumbell_MHD/topological_collapse/test/"
# Master_path="/home/teerthal/Dropbox (ASU)/dumbell_MHD/topological_collapse/test/"

scriptdir = @__DIR__
pushfirst!(PyVector(pyimport("sys")."path"), scriptdir)
prepper = pyimport("run_dumbbell_collapse")
no_mn,no_amn = prepper.initialize_topo(i)
topo = pyimport("tets")

# prepper.run_mag(i)

# @time pairing(i,Master_path)
# exit()

# @time dumbell_collapser_opt(i,Master_path)
# @time dumbell_collapser_nudge(i,Master_path,nudge_factor)
# @time dumbell_collapser(i,Master_path,nudge_factor)

# i=i+1

# no_mn,no_amn=topo.run_tets2(i)
# exit()


while no_mn[2]!=0
    # @time dumbell_collapser_opt(i,Master_path)
    @time dumbell_collapser(i,Master_path,nudge_factor)

    rm(string(Master_path,"phi_arr_",i,".npy"))
    rm(string(Master_path,"tet_stack_",i,".npy"))
    rm(string(Master_path,"area_stack_",i,".npy"))
    rm(string(Master_path,"tet_locs_",i,".npy"))
    rm(string(Master_path,"deltas_",i,".npy"))
    rm(string(Master_path,"tri_locs_",i,".npy"))

    global i=i+1
    no_mn,no_amn=topo.run_tets2(i)

    # @time pairing(i,Master_path)
    # exit()
    topo.tracing_test(i)
    topo.tet_2_plot(i)

    println(string("------",i,"------"))
    println(no_mn[2])
    println(string("-------------"))
    if no_mn[2]!=no_amn[2]
        println("periodicity breaking")
        exit()
    end

    if no_mn[2]==0
        break
    end

end

println(string("**********************"))
println(string("Total # iterations:",i))
println(string("**********************"))

prepper.run_mag(i)