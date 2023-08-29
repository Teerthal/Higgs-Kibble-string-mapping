module swap_collapse

using NPZ
using Statistics
using StatsBase
using Random
using LinearAlgebra

include("topo_fns.jl")
using .topo_routines

export dumbell_collapser_opt

function dumbell_collapser_opt(i,Master_path)

    phi_arr = npzread(string(Master_path,"phi_arr_",i,".npy"))
    area_stack = real(npzread(string(Master_path,"area_stack_",i,".npy")))
    tet_locs = npzread(string(Master_path,"tet_locs_",i,".npy"))
    deltas = real(npzread(string(Master_path,"deltas_",i,".npy")))
    tri_locs = npzread(string(Master_path,"tri_locs_",i,".npy"))

    N = size(phi_arr)[1]

    err_marg = 1e-1 #margin of error when determinig presence of strings based on phase

    #idxs of where strings and astrings are
    string_idxs = intersect(findall(deltas .<=(2.0*pi+err_marg)), 
    findall(deltas .>=(2.0*pi-err_marg)))
    astring_idxs = intersect(findall(deltas .<=(-2.0*pi+err_marg)),
    findall(deltas .>=(-2.0*pi-err_marg)))

    #safety to kill if #strings new astrings
    if size(string_idxs)!=size(astring_idxs)
        println("major issue with #strings new #astrings") 
        exit() 
    end

    println("# strings:",size(string_idxs))

    normed_areas = area_stack./(4*pi)
    println("area stack size:",size(normed_areas))

    mn_idxs = intersect(findall(normed_areas .<=(1.0+err_marg)),
    findall(normed_areas .>=(1.0-err_marg)))
    amn_idxs = intersect(findall(normed_areas .<=(-1.0+err_marg)),
    findall(normed_areas .>=(-1.0-err_marg)))

    #safety to kill if #monopoles neq #amonopoles
    if size(mn_idxs)!=size(amn_idxs)
        println("major issue with #monopoles neq #amonopoles") 
        exit() 
    end

    println("# monopoles,#antimonopoles:",size(mn_idxs),size(amn_idxs))
    
    # tri_locs = round.(tri_locs;digits=5)
    tri_locs = floor.(Int,tri_locs)
    astr_locs = mean(tri_locs[astring_idxs,:,:],dims=2)[:,1,:]
    astr_locs=round.(astr_locs;digits=5)
    tri_coords = mean(tri_locs,dims=2)[:,1,:]
    tri_coords=round.(tri_coords;digits=5)
    tet_coords = mean(tet_locs,dims=2)[:,1,:]
    tet_coords=round.(tet_coords;digits=5)
    

    for mn_idx in mn_idxs
        # println("---------------")
        # println(mn_idx)
        nxt_plks=[4*(mn_idx-1)+1,4*(mn_idx-1)+2,4*(mn_idx-1)+3,4*(mn_idx-1)+4]
        str_match = string_idxs[findall(in(nxt_plks),string_idxs)]
        # deleteat!(mn_idxs,findfirst(mn_idxs .==mn_idx))
        mn_tet=tet_locs[mn_idx,:,:]

        if size(str_match)[1]>2 println("huh??"); exit() end
        if size(str_match)[1]==2
            choice = sample(str_match)
            str_plk = tri_coords[choice,:]
            deleteat!(string_idxs,findall(string_idxs .==choice))
            mn_tri = tri_locs[choice,:,:]
        end
        if size(str_match)[1]==1
            str_plk = tri_coords[str_match[1],:]
            deleteat!(string_idxs,findall(string_idxs .==str_match[1]))
            mn_tri=tri_locs[str_match[1],:,:]
        end
        if size(str_match)[1]==0
            println(str_match)
            println("Nice")
            exit()
        end

        res = matching(str_plk,astr_locs,amn_idxs,astring_idxs,N)
        swap_vertex,cmns = find_swap(mn_tri,mn_tet)

        if res[1][1]==1
            deleteat!(amn_idxs,findall(amn_idxs .==res[2][1]))
            deleteat!(astring_idxs,findall(astring_idxs .== res[3][1]))
            astr_locs=tri_coords[astring_idxs,:]
            
            amn_swap_ver = sample_vertex(mn_tri)
            phi_arr[amn_swap_ver[1]+1,amn_swap_ver[2]+1,amn_swap_ver[3]+1,:] =
            phi_arr[swap_vertex[1]+1,swap_vertex[2]+1,swap_vertex[3]+1,:]
            phi_arr = bnd_enforce(amn_swap_ver,phi_arr,N)
        end
        next_tri=[]
        while res[1][1]!=1
            res = matching(str_plk,astr_locs,amn_idxs,astring_idxs,N)
            # println(res[1][1])
            # println(next_tri)
            if res[1][1]==2
                deleteat!(astring_idxs,findall(astring_idxs .==res[3][1]))
                astr_locs=tri_coords[astring_idxs,:]
                nxt_plks = [4*(res[2][1]-1)+1,4*(res[2][1]-1)+2,
                4*(res[2][1]-1)+3,4*(res[2][1]-1)+4]
                nxt_strs = string_idxs[findall(in(nxt_plks),string_idxs)]

                if size(nxt_strs)[1]>2 println("Huh??");exit()end
                if size(nxt_strs)[1]==0 println("nono");exit();break end

                if size(nxt_strs)[1]==2
                    nxt_choice = sample(nxt_strs)
                    str_plk=tri_coords[nxt_choice,:]
                    deleteat!(string_idxs,findall(string_idxs .==nxt_choice))
                    next_tri = tri_locs[nxt_choice,:,:]
                end

                if size(nxt_strs)[1]==1
                    str_plk=tri_coords[nxt_strs[1],:]
                    deleteat!(string_idxs,findall(string_idxs .==nxt_strs))
                    next_tri=tri_locs[nxt_strs[1],:,:]
                end
                uncommon_vertex,cmns = find_swap(next_tri,mn_tri)

                phi_arr[uncommon_vertex[1]+1,uncommon_vertex[2]+1,uncommon_vertex[3]+1,:] =
                phi_arr[swap_vertex[1]+1,swap_vertex[2]+1,swap_vertex[3]+1,:]
                phi_arr = bnd_enforce(uncommon_vertex,phi_arr,N)
                mn_tri = next_tri
                swap_vertex = uncommon_vertex
                # println("2",next_tri)
            end

            if res[1][1]==4
                str_plk = res[2]
                xmtch = findall(tri_coords[:,1] .==str_plk[1])
                ymtch = findall(tri_coords[:,2] .==str_plk[2])
                zmtch = findall(tri_coords[:,3] .==str_plk[3])
                mtch = zmtch[findall(in(ymtch[findall(in(xmtch),ymtch)]),zmtch)]
                next_tri=tri_locs[mtch[1],:,:]
                mn_tri = next_tri

                res = matching(str_plk,astr_locs,amn_idxs,astring_idxs,N)
                # println(res[1][1])
                if res[1][1]==2
                    deleteat!(astring_idxs,findall(astring_idxs .==res[3][1]))
                    astr_locs=tri_coords[astring_idxs,:]
                    nxt_plks = [4*(res[2][1]-1)+1,4*(res[2][1]-1)+2,
                    4*(res[2][1]-1)+3,4*(res[2][1]-1)+4]
                    nxt_strs = string_idxs[findall(in(nxt_plks),string_idxs)]

                    if size(nxt_strs)[1]>2 println("Huh??");exit()end
                    if size(nxt_strs)[1]==0 println("???");exit();break end

                    if size(nxt_strs)[1]==2
                        nxt_choice = sample(nxt_strs)
                        str_plk=tri_coords[nxt_choice,:]
                        deleteat!(string_idxs,findall(string_idxs .==nxt_choice))
                        next_tri = tri_locs[nxt_choice,:,:]
                    end

                    if size(nxt_strs)[1]==1
                        str_plk=tri_coords[nxt_strs[1],:]
                        deleteat!(string_idxs,findall(string_idxs .==nxt_strs))
                        next_tri=tri_locs[nxt_strs[1],:,:]
                    end
                    uncommon_vertex,cmns = find_swap(next_tri,mn_tri)
                    phi_arr[uncommon_vertex[1]+1,uncommon_vertex[2]+1,uncommon_vertex[3]+1,:] =
                    phi_arr[swap_vertex[1]+1,swap_vertex[2]+1,swap_vertex[3]+1,:]
                    phi_arr = bnd_enforce(uncommon_vertex,phi_arr,N)
                    mn_tri = next_tri
                    swap_vertex = uncommon_vertex
                end

                if res[1][1]==1
                    deleteat!(amn_idxs,findall(amn_idxs .==res[2][1]))
                    deleteat!(astring_idxs,findall(astring_idxs .== res[3][1]))
                    astr_locs=tri_coords[astring_idxs,:]

                    amn_swap_ver = sample_vertex(next_tri)
                    phi_arr[amn_swap_ver[1]+1,amn_swap_ver[2]+1,amn_swap_ver[3]+1,:] =
                    phi_arr[swap_vertex[1]+1,swap_vertex[2]+1,swap_vertex[3]+1,:]
                    phi_arr = bnd_enforce(amn_swap_ver,phi_arr,N)
                    break
                end

            end

            if res[1][1]==1
                deleteat!(amn_idxs,findall(amn_idxs .==res[2][1]))
                deleteat!(astring_idxs,findall(astring_idxs .== res[3][1]))
                astr_locs=tri_coords[astring_idxs,:]

                amn_swap_ver = sample_vertex(next_tri)
                phi_arr[amn_swap_ver[1]+1,amn_swap_ver[2]+1,amn_swap_ver[3]+1,:] =
                phi_arr[swap_vertex[1]+1,swap_vertex[2]+1,swap_vertex[3]+1,:]
                phi_arr = bnd_enforce(amn_swap_ver,phi_arr,N)
                break
            end
            
        end
        # bnd_test(phi_arr,N)
        # println("---------------")
    end
    
    npzwrite(string(Master_path,"phi_arr_",i+1,".npy"),phi_arr)

end

end