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
no_mn,no_amn = prepper.initialize_topo_nopair(i)
topo = pyimport("tets")
prepper.run_mag(i)

# err_marg = 1e-3
err_marg = 1e-1
# prepper.run_mag(i)

# tet_fluxes(i,Master_path)
# exit()

@time string_list = pairing(i,Master_path,err_marg)

# @time dumbell_collapser(i,Master_path,nudge_factor,err_marg)
# exit()
# topo.tet_plot_jl(i,string_list,err_marg)
# exit()

# @time dumbell_collapser_opt(i,Master_path)
# @time dumbell_collapser_nudge(i,Master_path,nudge_factor)

# i=i+1

# no_mn,no_amn=topo.run_tets2(i)
# @time string_list = pairing(i,Master_path,err_marg)
# topo.tet_plot_jl(i,string_list,err_marg)
# exit()


while no_mn[2]!=0
    @time dumbell_collapser_opt(i,Master_path)
    # @time dumbell_collapser(i,Master_path,nudge_factor,err_marg)
    # @time dumbell_collapser_min(i,Master_path,nudge_factor)
    # exit()
    junk_files = filter(x->endswith(x,string(i,".npy")),readdir(Master_path))
    for f in junk_files
        rm(string(Master_path,f))
    end

    global i=i+1
    no_mn,no_amn=topo.run_tets2(i)

    @time string_list = pairing(i,Master_path,err_marg)

    # exit()
    # topo.tracing_test(i)
    # topo.tet_2_plot(i)
    topo.tet_plot_jl(i,string_list,err_marg)

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