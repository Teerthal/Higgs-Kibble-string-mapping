module topo_routines

using NPZ
using Statistics
using StatsBase
using Random
using LinearAlgebra

export pairing,ref_plk,matching,make_jagged,
find_swap,find_ucommons,sample_vertex,
bnd_enforce,bnd_test,n, n_nudge,n_flip,
n_nudge_tri,tri_ang,tri_area,tet_charge,
mtA_vB,vA_dag_vB,mtx_pd,delta_cal,del_plk,n_annihilate

function ref_plk(a,b,c,N)
    if a==0.0
        ref_x = round(N-1.;digits=5)
        out = [ref_x,b,c]
    elseif b==0.0
        ref_y = round(N-1.;digits=5)
        out = [a,ref_y,c]
    elseif c==0.0
        ref_z = round(N-1.;digits=5)
        out = [a,b,ref_z]
    elseif a==round(N-1.;digits=5)
        ref_x = round(0.0;digits=5)
        out = [ref_x,b,c]
    elseif b==round(N-1.;digits=5)
        ref_y = round(0.0;digits=5)
        out = [a,ref_y,c]
    elseif c==round(N-1.;digits=5)
        ref_z = round(0.0;digits=5)
        out = [a,b,ref_z]
    else
        out=[0,0,0]
    end

    return out
end

function matching(plk,astr_locs,amon_idxs,astring_idxs,N)
    xmtch = findall(astr_locs[:,1] .==plk[1])
    ymtch = findall(astr_locs[:,2] .==plk[2])
    zmtch = findall(astr_locs[:,3] .==plk[3])
    # mtch = intersect(xmtch,intersect(ymtch,zmtch))
    mtch = zmtch[findall(in(ymtch[findall(in(xmtch),ymtch)]),zmtch)]
    mtch = astring_idxs[mtch]
    if size(mtch)[1]==0
        a,b,c=plk
        refpk = ref_plk(a,b,c,N)
        if refpk==[0,0,0]
            println("Not at the boundary. Quitting")
            exit()
        else
            xref = findall(astr_locs[:,1] .==refpk[1])
            yref = findall(astr_locs[:,2] .==refpk[2])
            zref = findall(astr_locs[:,3] .==refpk[3])
            # ref_mtch = intersect(xref,intersect(yref,zref))
            ref_mtch = zref[findall(in(yref[findall(in(xref),yref)]),zref)]
            if size(ref_mtch)[1]==0
                println("Antistring across boundary no found. Quitting")
                exit()
            else
                out = [[4],refpk,ref_mtch[1]]
            end
        end
    elseif size(mtch)[1]==1
        mtch = mtch[1]
        amn_intp = ceil(Int,mtch/4)
        amono_mtch = intersect(amon_idxs,amn_intp)
        if size(amono_mtch)[1]==1
            code = 1
            out = [[code],[amono_mtch],[mtch]]
        else
            code = 2
            out = [[code],[amn_intp],[mtch]]
        end
    end

    return out
end

function make_jagged(::Type{T}, col_lengths) where T
    [zeros(T, (i,3)) for i in col_lengths]
end

function tet_flux(B_arr, tet_idx, tri_locs,tet_coords)
    tet_cen = tet_coords[tet_idx,:]
    flux = 0.0
    for tri_idx in range(4*(tet_idx-1)+1,4*(tet_idx-1)+4,step=1)
        
        # Validated that the plaquette vertice arangements are 
        # correct such that the surface area elements are correct
        # and point in the right direction w.r.t center of the 
        # tetrahedra

        #define 3 vectors on plaquette w.r.t tet center
        a = tri_locs[tri_idx,1,:]
        b = tri_locs[tri_idx,2,:]
        c = tri_locs[tri_idx,3,:]

        B_a = B_arr[a[1]+1,a[2]+1,a[3]+1,:]
        B_b = B_arr[b[1]+1,b[2]+1,b[3]+1,:]
        B_c = B_arr[c[1]+1,c[2]+1,c[3]+1,:]
        B_plk = mean([B_a,B_b,B_c],dims = 1)[1]

        a = tri_locs[tri_idx,1,:]-tet_cen
        b = tri_locs[tri_idx,2,:]-tet_cen
        c = tri_locs[tri_idx,3,:]-tet_cen
        #calculate unit surface area
        area = (cross(a,b)+cross(b,c)+cross(c,a)) .*0.5
        area = area#/norm(area)
        #B.dS on plaquette
        plk_flx = dot(B_plk,area)
        flux=flux+plk_flx
    end
    
    return flux
end

function tet_flux2(B_arr, tet_idx, tet_coords,tet_locs)
    tet_cen = tet_coords[tet_idx,:]
    flux = 0.0
    for i in range(1,4,step=1)
        a = tet_locs[tet_idx,i,:]
        B_a = B_arr[a[1]+1,a[2]+1,a[3]+1,:]
        println(B_a)
        a = a-tet_cen

        #B.dS on plaquette
        flx = dot(B_a,a)
        flux=flux+flx*pi
    end
    return flux/pi
end

function tet_div_flux(div_B, tet_idx,tet_locs)

    flux = 0.0
    for i in range(1,4,step=1)
        ver = tet_locs[tet_idx,i,:]
        flux = flux+ div_B[ver[1]+1,ver[2]+1,ver[3]+1]
    end
    return flux
end

export tet_fluxes
function tet_fluxes(i,Master_path)

    tet_locs = npzread(string(Master_path,"tet_locs_",i,".npy"))
    tri_locs = npzread(string(Master_path,"tri_locs_",i,".npy"))
    B_arr = npzread(string(Master_path,"B_arr_",i,".npy"))
    divB_arr = npzread(string(Master_path,"divB_arr_",i,".npy"))

    tri_locs = floor.(Int,tri_locs)
    tet_locs = floor.(Int,tet_locs)
    tri_coords = mean(tri_locs,dims=2)[:,1,:]
    tri_coords=round.(tri_coords;digits=5)
    tet_coords = mean(tet_locs,dims=2)[:,1,:]
    tet_coords=round.(tet_coords;digits=5)

    no_tets = size(tet_locs)[1]
    no_tris = size(tri_locs)[1]
    for tet_idx in range(1,no_tets,step=1)
        flx = tet_flux(B_arr, tet_idx, tri_locs,tet_coords)
    end
end

function pairing(i,Master_path,err_marg)

    phi_arr = npzread(string(Master_path,"phi_arr_",i,".npy"))
    area_stack = real(npzread(string(Master_path,"area_stack_",i,".npy")))
    tet_locs = npzread(string(Master_path,"tet_locs_",i,".npy"))
    deltas = real(npzread(string(Master_path,"deltas_",i,".npy")))
    tri_locs = npzread(string(Master_path,"tri_locs_",i,".npy"))
    
    # B_arr = npzread(string(Master_path,"B_arr_",i,".npy"))
    # divB_arr = npzread(string(Master_path,"divB_arr_",i,".npy"))

    N = size(phi_arr)[1]

    # err_marg = 1e-1 #margin of error when determinig presence of strings based on phase

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
    # println(mn_idxs[1])
    # println(size(mn_idxs)[1])
    # println(mn_idxs[size(mn_idxs)[1]])
    pair_list = []
    match_cnt = 0
    # using ProfileView
    for mn_num in eachindex(mn_idxs)
        mn_idx = mn_idxs[mn_num]
        # println(tet_flux(B_arr,mn_idx, tri_locs,tet_coords))
        # println(tet_locs[mn_idx,:,:])
        # println(tet_flux2(B_arr, mn_idx, tet_coords,tet_locs));exit()
        # println(tet_div_flux(divB_arr, mn_idx,tet_locs))#;exit()
        # println(mn_num)
        pairs = []
        # str_match = intersect(string_idxs,
        # [4*(mn_idx-1)+1,4*(mn_idx-1)+2,4*(mn_idx-1)+3,4*(mn_idx-1)+4])
        nxt_plks=[4*(mn_idx-1)+1,4*(mn_idx-1)+2,4*(mn_idx-1)+3,4*(mn_idx-1)+4]
        str_match = string_idxs[findall(in(nxt_plks),string_idxs)]
        push!(pairs,tet_coords[mn_idx,:])
        # deleteat!(mn_idxs,findfirst(mn_idxs .==mn_idx))

        if size(str_match)[1]>2 println("huh??"); exit() end
        if size(str_match)[1]==2
            choice = sample(str_match)
            str_plk = tri_coords[choice,:]
            deleteat!(string_idxs,findall(string_idxs .==choice))
        end
        if size(str_match)[1]==1
            str_plk = tri_coords[str_match[1],:]
            deleteat!(string_idxs,findall(string_idxs .==str_match[1]))
        end
        if size(str_match)[1]==0
            println(str_match)
            println("Nice")
            exit()
        end

        res = matching(str_plk,astr_locs,amn_idxs,astring_idxs,N)
        push!(pairs, str_plk)
        if res[1][1]==1
            push!(pairs,tet_coords[res[2][1][1],:])
            deleteat!(amn_idxs,findall(amn_idxs .==res[2][1]))
            deleteat!(astring_idxs,findall(astring_idxs .== res[3][1]))
            astr_locs=tri_coords[astring_idxs,:]
            # global match_cnt= match_cnt+1
            match_cnt = match_cnt+1
        end

        while res[1][1]!=1
            res = matching(str_plk,astr_locs,amn_idxs,astring_idxs,N)
            if res[1][1]==2
                deleteat!(astring_idxs,findall(astring_idxs .==res[3][1]))
                astr_locs=tri_coords[astring_idxs,:]
                nxt_plks = [4*(res[2][1]-1)+1,4*(res[2][1]-1)+2,
                4*(res[2][1]-1)+3,4*(res[2][1]-1)+4]
                # nxt_strs = intersect(string_idxs,nxt_plks)
                nxt_strs = string_idxs[findall(in(nxt_plks),string_idxs)]
                # println("--------")
                # @time println(intersect(string_idxs,nxt_plks))
                # @time println(string_idxs[findall(in(nxt_plks),string_idxs)])
                # println("--------")
                if size(nxt_strs)[1]>2 println("Huh??");exit()end
                if size(nxt_strs)[1]==0 break end

                if size(nxt_strs)[1]==2
                    nxt_choice = sample(nxt_strs)
                    str_plk=tri_coords[nxt_choice,:]
                    deleteat!(string_idxs,findall(string_idxs .==nxt_choice))
                end

                if size(nxt_strs)[1]==1
                    str_plk=tri_coords[nxt_strs[1],:]
                    deleteat!(string_idxs,findall(string_idxs .==nxt_strs))
                end
                tet_cen = tet_coords[res[2][1][1],:]
                push!(pairs, tet_cen)
                push!(pairs, str_plk)
                
            end

            if res[1][1]==4
                str_plk = res[2]
                push!(pairs, str_plk)
            end

            if res[1][1]==1
                push!(pairs,tet_coords[res[2][1][1],:])
                deleteat!(amn_idxs,findall(amn_idxs .==res[2][1]))
                deleteat!(astring_idxs,findall(astring_idxs .== res[3][1]))
                astr_locs=tri_coords[astring_idxs,:]
                # global match_cnt= match_cnt+1
                match_cnt = match_cnt+1
                break
            end
            
        end

        push!(pair_list,pairs)
    end
    println(match_cnt)
    # lens = []
    # for str_i in pair_list
    #     push!(lens, size(str_i,1))
    # end
    # str_strings = make_jagged(Float64,lens)
    # for (str_i,strg) in enumerate(pair_list)
    #     for pair_i in eachindex(strg)
    #         println(pair_list[str_i][pair_i])
    #         for j in range(1,3,step=1)
    #             str_strings[str_i,pair_i,j]=pair_list[[str_i][pair_i]][j]
    #         end
    #     end
    #     prinln(str_strings[str_i])
    #     exit()
    # end
    # exit()
    # npzwrite(string(Master_path,"string_lists_",i,".npy"),pair_list)
    return pair_list
end

function find_swap(a,b)
    #finds the vertex in b that is not common to a
    swap_vertex=[]
    cmn_vers = []
    for tet_ver in eachrow(b)
        xmtch = findall(a[:,1] .==tet_ver[1])
        ymtch = findall(a[:,2] .==tet_ver[2])
        zmtch = findall(a[:,3] .==tet_ver[3])
        # mtch = intersect(xmtch,intersect(ymtch,zmtch))
        mtch = zmtch[findall(in(ymtch[findall(in(xmtch),ymtch)]),zmtch)]
        if size(mtch)[1]==0
            swap_vertex=tet_ver
            # push!(swap_vertex,tet_ver)
        else
            push!(cmn_vers,tet_ver)
        end
    end
    return swap_vertex,cmn_vers
end

function find_ucommons(a,b)
    uncmns = []
    for aa in eachrow(a)
        if (aa[1]!=b[1]||aa[2]!=b[2]||aa[3]!=b[3])
            push!(uncmns,aa)
        end
    end
    return uncmns
end

function sample_vertex(a)
    #returns one of a. 
    #input is typically an array of size (n,3)
    #n=3 for a set of coordinates for triangle vertices
    idx = sample(range(1,size(a,1),step=1))
    return a[idx,:]
end

function bnd_enforce(inp,arr,N)
    a,b,c=inp
    ix = [0,N-1]
    out = []
    # println

    if ((a==ix[1]||a==ix[2])&&(b==ix[1]||b==ix[2])&&(c==ix[1]||c==ix[2]))
    for xb in ix
        for yb in ix
            for zb in ix
                push!(out,[xb,yb,zb])
            end
        end
    end
    elseif ((a==ix[1]||a==ix[2])&&(b==ix[1]||b==ix[2]))
        for xb in ix
            for yb in ix
                push!(out,[xb,yb,c])
            end
        end
    elseif ((b==ix[1]||b==ix[2])&&(c==ix[1]||c==ix[2]))
        for yb in ix
            for zb in ix
                push!(out,[a,yb,zb])
            end
        end
    elseif ((a==ix[1]||a==ix[2])&&(c==ix[1]||c==ix[2]))
        for xb in ix
            for zb in ix
                push!(out,[xb,b,zb])
            end
        end
    elseif (a==0||b==0||c==0||a==N-1||b==N-1||c==N-1)
        push!(out,floor.(Int, ref_plk(a,b,c,N)))
    end
    
    if size(out,1)!=0
        for j in range(1,size(out,1),step=1)
            ca,cb,cc=out[j].+1
            arr[ca,cb,cc,:]=arr[a+1,b+1,c+1,:]
        end
    end
    return arr
end

function bnd_test(arr,N)
    for x_idx in range(1,N,step=1)
        for y_idx in range(1,N,step=1)
            for z_idx in range(1,N,step=1)
                if z_idx==N
                    if (arr[x_idx,y_idx,z_idx,:]!=arr[x_idx,y_idx,1,:])
                        println("periodicity lost")
                        println(x_idx," ",y_idx," ",z_idx)
                        println(arr[x_idx,y_idx,z_idx,:])
                        println(arr[x_idx,y_idx,1,:])
                        exit()
                    end
                end
                if y_idx==N
                    if (arr[x_idx,y_idx,z_idx,:]!=arr[x_idx,1,z_idx,:])
                        println("periodicity lost")
                        println(x_idx," ",y_idx," ",z_idx)
                        println(arr[x_idx,y_idx,z_idx,:])
                        println(arr[x_idx,1,z_idx,:])
                        exit()
                    end
                end

                if x_idx==N
                    if (arr[x_idx,y_idx,z_idx,:]!=arr[1,y_idx,z_idx,:])
                        println("periodicity lost")
                        println(x_idx," ",y_idx," ",z_idx)
                        println(arr[x_idx,y_idx,z_idx,:])
                        println(arr[1,y_idx,z_idx,:])
                        exit()
                    end
                end

            end
        end
    end
end

function n(phi)
    n1 = conj(phi[1])*phi[2] + phi[1]*conj(phi[2])
    n2 = -1im*conj(phi[1])*phi[2] + 1im*phi[1]*conj(phi[2])
    n3 = conj(phi[1])*phi[1] - phi[2]*conj(phi[2])
    return [-real(n1),-real(n2),-real(n3)]
end

function n_nudge(n1,n5,n2,nudge_factor)
    # new_n = n2+ n1.*nudge_factor
    # new_n = -(n2+ n1.*nudge_factor)
    # new_n = (n2+n1).*-1.0
    new_n = -(n5+n1)-n2 .*0.9
    new_n = -(n5+n1)#-n2 .*0.9
    new_n = -n2
    # new_n=new_n/sqrt(new_n[1]^2+new_n[2]^2+new_n[3]^2)
    # new_n = new_n - 0.9*n2
    new_n=new_n/sqrt(new_n[1]^2+new_n[2]^2+new_n[3]^2)
    alpha = acos(-new_n[3])/2.0
    gamma = rand(Float64)*2.0*pi
    # beta = acos(-new_n[1]/sin(2.0*alpha))+gamma
    beta = atan(new_n[2],-new_n[1])+gamma
    new_phi = [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
    return new_phi
end

function n_annihilate(ver_1,ver_2,ver_3,ver_4,ver_5,mn_tri,phi_arr,eps)
    n1 = n(phi_arr[ver_1[1]+1,ver_1[2]+1,ver_1[3]+1,:])
    n2 = n(phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:])
    n3 = n(phi_arr[ver_3[1]+1,ver_3[2]+1,ver_3[3]+1,:])
    n4 = n(phi_arr[ver_4[1]+1,ver_4[2]+1,ver_4[3]+1,:])
    n5 = n(phi_arr[ver_5[1]+1,ver_5[2]+1,ver_5[3]+1,:])

    tri_123 = [ver_1[1] ver_1[2] ver_1[3];
            ver_2[1] ver_2[2] ver_2[3];
            ver_3[1] ver_3[2] ver_3[3]]
    tri_124 = [ver_1[1] ver_1[2] ver_1[3];
            ver_2[1] ver_2[2] ver_2[3];
            ver_4[1] ver_4[2] ver_4[3]]
    tri_235 = [ver_2[1] ver_2[2] ver_2[3];
            ver_3[1] ver_3[2] ver_3[3];
            ver_5[1] ver_5[2] ver_5[3]]
    tri_245 = [ver_2[1] ver_2[2] ver_2[3];
            ver_4[1] ver_4[2] ver_4[3];
            ver_5[1] ver_5[2] ver_5[3]]
    # println(tri_245)
    # tri_124 = [ver_1;ver_2;ver_4]
    # tri_235 = [ver_2;ver_3;ver_5]
    # tri_245 = [ver_2;ver_4;ver_5];println(tri_245)
    unr_tris = [tri_123,tri_124,tri_235,tri_245]
    phase_123_i = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
    phase_124_i = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
    phase_234_i = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
    phase_245_i = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
    phase_235_i = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
    unr_phases_i = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
    # unr_phases_i = floor.([0.43,1.0,0.39,1.0])
    # flags = findall(unr_phases_i .== 1.0);println(flags)
    # unr_str_phases_i = [unr_phases_i[f] for f in flags];println(unr_str_phases_i)
    
    new_phi = n_nudge(n1,n5,n2,eps)
    # new_phi = n_flip(n3+n4-n2,n2,eps)
    phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = new_phi
    phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = new_phi.*exp(-1im*phase_234_i)
    phase_123_new = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
    phase_124_new = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
    phase_234_new = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
    phase_245_new = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
    phase_235_new = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
    unr_phases_new = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
    while phase_234_new==1.0 && (unr_phases_new!=unr_phases_i)
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        new_phi.*exp(1im*rand(Float64)*2.0*pi)
        phase_123_new = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
        phase_124_new = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
        phase_234_new = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
        phase_245_new = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
        phase_235_new = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
        unr_phases_new = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
    end
    return phi_arr
end

function adj_tets(ver,tet_locs,t_m,t_a,N)

    a,b,c=ver
    ix = [0.0,N-1.0]
    refs = []
    # push!(refs,ver)
    # println(ver)
    
    if ((a==ix[1]||a==ix[2])&&(b==ix[1]||b==ix[2])&&(c==ix[1]||c==ix[2]))
    for xb in ix
        for yb in ix
            for zb in ix
                push!(refs,[xb,yb,zb])
            end
        end
    end
    elseif ((a==ix[1]||a==ix[2])&&(b==ix[1]||b==ix[2]))
        for xb in ix
            for yb in ix
                push!(refs,[xb,yb,c])
            end
        end
    elseif ((b==ix[1]||b==ix[2])&&(c==ix[1]||c==ix[2]))
        for yb in ix
            for zb in ix
                push!(refs,[a,yb,zb])
            end
        end
    elseif ((a==ix[1]||a==ix[2])&&(c==ix[1]||c==ix[2]))
        for xb in ix
            for zb in ix
                push!(refs,[xb,b,zb])
            end
        end
    elseif (a==0||b==0||c==0||a==N-1||b==N-1||c==N-1)
        push!(refs,floor.(Int, ref_plk(a,b,c,N)))
    else
        push!(refs,ver)
    end
    # println(refs)
    out = []
    for p in refs
        # println(p)
        xmtch = findall(tet_locs[:,:,1] .==p[1])
        ymtch = findall(tet_locs[:,:,2] .==p[2])
        zmtch = findall(tet_locs[:,:,3] .==p[3])
        mtch = zmtch[findall(in(ymtch[findall(in(xmtch),ymtch)]),zmtch)]
        # println(mtch)
        # println(size(mtch))
        for m in mtch
            if m[1]!=t_m && m[1]!=t_a
                push!(out,m[1])
            end
        end
    end
    # println(size(out))
    # println(out)
    # exit()
    return out
end

function gen_phi()
    alpha = 0.5*acos(sample([-1.0,1.0])*rand(Float64))
    beta = rand(Float64)*2.0*pi
    gamma = rand(Float64)*2.0*pi
    return [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
end

export n_annihilate_2

function n_annihilate_2(ver_1,ver_2,ver_3,ver_4,ver_5,mn_tri,phi_arr,
    eps,tet_locs,err,t_m,t_a,N)
    phi_og = phi_arr
    adjs = adj_tets(ver_2,tet_locs,t_m,t_a,N)
    adj_cgs_i = floor.([tet_charge(tet_locs[a,:,:],phi_arr,err) for a in adjs])
    # n1 = n(phi_arr[ver_1[1]+1,ver_1[2]+1,ver_1[3]+1,:])
    # n2 = n(phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:])
    # n3 = n(phi_arr[ver_3[1]+1,ver_3[2]+1,ver_3[3]+1,:])
    # n4 = n(phi_arr[ver_4[1]+1,ver_4[2]+1,ver_4[3]+1,:])
    # n5 = n(phi_arr[ver_5[1]+1,ver_5[2]+1,ver_5[3]+1,:])

    tri_123 = [ver_1[1] ver_1[2] ver_1[3];
            ver_2[1] ver_2[2] ver_2[3];
            ver_3[1] ver_3[2] ver_3[3]]
    tri_124 = [ver_1[1] ver_1[2] ver_1[3];
            ver_2[1] ver_2[2] ver_2[3];
            ver_4[1] ver_4[2] ver_4[3]]
    tri_235 = [ver_2[1] ver_2[2] ver_2[3];
            ver_3[1] ver_3[2] ver_3[3];
            ver_5[1] ver_5[2] ver_5[3]]
    tri_245 = [ver_2[1] ver_2[2] ver_2[3];
            ver_4[1] ver_4[2] ver_4[3];
            ver_5[1] ver_5[2] ver_5[3]]
    # println(tri_245)
    # tri_124 = [ver_1;ver_2;ver_4]
    # tri_235 = [ver_2;ver_3;ver_5]
    # tri_245 = [ver_2;ver_4;ver_5];println(tri_245)
    unr_tris = [tri_123,tri_124,tri_235,tri_245]
    phase_123_i = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
    phase_124_i = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
    phase_234_i = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
    phase_245_i = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
    phase_235_i = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
    unr_phases_i = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
    # unr_phases_i = floor.([0.43,1.0,0.39,1.0])
    # flags = findall(unr_phases_i .== 1.0);println(flags)
    # unr_str_phases_i = [unr_phases_i[f] for f in flags];println(unr_str_phases_i)
    
    # new_phi = n_nudge(n1,n5,n2,eps)
    # new_phi = n_flip(n3+n4-n2,n2,eps)
    # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = new_phi
    phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = gen_phi()#.*exp(-1im*phase_234_i)
    bnd_enforce(ver_2,phi_arr,N)
    adj_cgs_f = floor.([tet_charge(tet_locs[a,:,:],phi_arr,err) for a in adjs])
    charge_m_f = floor(abs(tet_charge(tet_locs[t_m,:,:],phi_arr,err)))
    charge_a_f = floor(abs(tet_charge(tet_locs[t_a,:,:],phi_arr,err)))
    tries=0
    # while ((phase_234_new==1.0) && (unr_phases_new!=unr_phases_i) && (adj_cgs_f!=adj_cgs_i))
    while ((adj_cgs_f!=adj_cgs_i) || (charge_m_f!=0.0) || (charge_a_f!=0.0))
        # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        # new_phi.*exp(1im*rand(Float64)*2.0*pi)
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = gen_phi()
        bnd_enforce(ver_2,phi_arr,N)
        adj_cgs_f = floor.([tet_charge(tet_locs[a,:,:],phi_arr,err) for a in adjs])
        charge_m_f = floor(abs(tet_charge(tet_locs[t_m,:,:],phi_arr,err)))
        charge_a_f = floor(abs(tet_charge(tet_locs[t_a,:,:],phi_arr,err)))
        # println(adj_cgs_i==adj_cgs_f)
        tries = tries+1
        if tries > 10000
            println("failed")
            # phi_arr=phi_og
            # break
            return phi_og
        end
    end

    phase_123_new = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
    phase_124_new = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
    phase_234_new = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
    phase_245_new = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
    phase_235_new = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
    unr_phases_new = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])

    while ((phase_234_new==1.0) || (unr_phases_new!=unr_phases_i))
        # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        # new_phi.*exp(1im*rand(Float64)*2.0*pi)
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:]*exp(1im*rand(Float64)*2.0*pi)
        bnd_enforce(ver_2,phi_arr,N)
        phase_123_new = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
        phase_124_new = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
        phase_234_new = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
        phase_245_new = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
        phase_235_new = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
        unr_phases_new = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
        # println(adj_cgs_i==adj_cgs_f)
        tries = tries+1
        if tries > 20000
            println("failed")
            phi_arr=phi_og
            return phi_og
            # break
        end
    end

    println(tries)
    # println(adj_cgs_f)
    # exit()
    return phi_arr
end

export nhilate_swap

function nhilate_swap(ver_1,ver_2,ver_3,ver_4,ver_5,mn_tri,phi_arr,
    eps,tet_locs,err,t_m,t_a,N)
    adjs = adj_tets(ver_2,tet_locs,t_m,t_a,N)
    adj_cgs_i = [tet_charge(tet_locs[a,:,:],phi_arr,err) for a in adjs]
    # n1 = n(phi_arr[ver_1[1]+1,ver_1[2]+1,ver_1[3]+1,:])
    # n2 = n(phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:])
    # n3 = n(phi_arr[ver_3[1]+1,ver_3[2]+1,ver_3[3]+1,:])
    # n4 = n(phi_arr[ver_4[1]+1,ver_4[2]+1,ver_4[3]+1,:])
    # n5 = n(phi_arr[ver_5[1]+1,ver_5[2]+1,ver_5[3]+1,:])
    phi_1 = phi_arr[ver_1[1]+1,ver_1[2]+1,ver_1[3]+1,:]
    phi_2 = phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:]
    phi_3 = phi_arr[ver_3[1]+1,ver_3[2]+1,ver_3[3]+1,:]
    phi_4 = phi_arr[ver_4[1]+1,ver_4[2]+1,ver_4[3]+1,:]
    phi_5 = phi_arr[ver_5[1]+1,ver_5[2]+1,ver_5[3]+1,:]

    tri_123 = [ver_1[1] ver_1[2] ver_1[3];
            ver_2[1] ver_2[2] ver_2[3];
            ver_3[1] ver_3[2] ver_3[3]]
    tri_124 = [ver_1[1] ver_1[2] ver_1[3];
            ver_2[1] ver_2[2] ver_2[3];
            ver_4[1] ver_4[2] ver_4[3]]
    tri_235 = [ver_2[1] ver_2[2] ver_2[3];
            ver_3[1] ver_3[2] ver_3[3];
            ver_5[1] ver_5[2] ver_5[3]]
    tri_245 = [ver_2[1] ver_2[2] ver_2[3];
            ver_4[1] ver_4[2] ver_4[3];
            ver_5[1] ver_5[2] ver_5[3]]
    # println(tri_245)
    # tri_124 = [ver_1;ver_2;ver_4]
    # tri_235 = [ver_2;ver_3;ver_5]
    # tri_245 = [ver_2;ver_4;ver_5];println(tri_245)
    unr_tris = [tri_123,tri_124,tri_235,tri_245]
    phase_123_i = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
    phase_124_i = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
    phase_234_i = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
    phase_245_i = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
    phase_235_i = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
    unr_phases_i = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
    # unr_phases_i = floor.([0.43,1.0,0.39,1.0])
    # flags = findall(unr_phases_i .== 1.0);println(flags)
    # unr_str_phases_i = [unr_phases_i[f] for f in flags];println(unr_str_phases_i)
    println(abs(tet_charge(tet_locs[t_m,:,:],phi_arr,err)))
    println(abs(tet_charge(tet_locs[t_a,:,:],phi_arr,err)))
    # new_phi = n_nudge(n1,n5,n2,eps)
    # new_phi = n_flip(n1,n2,eps)
    # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = new_phi
    phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = phi_4
    # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = gen_phi()#.*exp(-1im*phase_234_i)
    bnd_enforce(ver_2,phi_arr,N)
    adj_cgs_f = [tet_charge(tet_locs[a,:,:],phi_arr,err) for a in adjs]
    charge_m_f = abs(tet_charge(tet_locs[t_m,:,:],phi_arr,err))
    charge_a_f = abs(tet_charge(tet_locs[t_a,:,:],phi_arr,err))
    tries=0
    println(adj_cgs_f!=adj_cgs_i)
    println(charge_m_f)
    println(charge_a_f);exit()
    # while ((phase_234_new==1.0) && (unr_phases_new!=unr_phases_i) && (adj_cgs_f!=adj_cgs_i))
    while ((adj_cgs_f!=adj_cgs_i) || (charge_m_f!=0.0) || (charge_a_f!=0.0))
        # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        # new_phi.*exp(1im*rand(Float64)*2.0*pi)
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = gen_phi()
        bnd_enforce(ver_2,phi_arr,N)
        adj_cgs_f = [tet_charge(tet_locs[a,:,:],phi_arr,err) for a in adjs]
        charge_m_f = abs(tet_charge(tet_locs[t_m,:,:],phi_arr,err))
        charge_a_f = abs(tet_charge(tet_locs[t_a,:,:],phi_arr,err))
        # println(adj_cgs_i==adj_cgs_f)
        tries = tries+1
    end

    phase_123_new = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
    phase_124_new = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
    phase_234_new = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
    phase_245_new = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
    phase_235_new = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
    unr_phases_new = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])

    while ((phase_234_new==1.0) || (unr_phases_new!=unr_phases_i))
        # phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        # new_phi.*exp(1im*rand(Float64)*2.0*pi)
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:] = 
        phi_arr[ver_2[1]+1,ver_2[2]+1,ver_2[3]+1,:]*exp(1im*rand(Float64)*2.0*pi)
        bnd_enforce(ver_2,phi_arr,N)
        phase_123_new = abs(round(del_plk(tri_123,phi_arr)/(2*pi),digits=1))
        phase_124_new = abs(round(del_plk(tri_124,phi_arr)/(2*pi),digits=1))
        phase_234_new = abs(round(del_plk(mn_tri,phi_arr)/(2*pi),digits=1))
        phase_245_new = abs(round(del_plk(tri_245,phi_arr)/(2*pi),digits=1))
        phase_235_new = abs(round(del_plk(tri_235,phi_arr)/(2*pi),digits=1))
        unr_phases_new = floor.([phase_123_i,phase_124_i,phase_235_i,phase_245_i])
        # println(adj_cgs_i==adj_cgs_f)
        tries = tries+1
    end

    # println(adj_cgs_i==adj_cgs_f)
    println(tries)
    # println(adj_cgs_f)
    # exit()
    return phi_arr
end

function n_flip(n1,n2,nudge_factor)
    # new_n = n2+ n1.*nudge_factor
    new_n = n1
    new_n=new_n/sqrt(new_n[1]^2+new_n[2]^2+new_n[3]^2)
    alpha = acos(-new_n[3])/2.0
    gamma = rand(Float64)*2.0*pi
    # beta = acos(-new_n[1]/sin(2.0*alpha))+gamma
    beta = atan(new_n[2],-new_n[1])+gamma
    new_phi = [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
    return new_phi
end

function n_nudge_tri(n3,n4,n5,nudge_factor)
    new_n = -(n5+ (n4+n3).*nudge_factor)
    new_n=new_n/sqrt(new_n[1]^2+new_n[2]^2+new_n[3]^2)
    alpha = acos(-new_n[3])/2.0
    gamma = rand(Float64)*2.0*pi
    # beta = acos(-new_n[1]/sin(2.0*alpha))+gamma
    beta = atan(new_n[2],-new_n[1])+gamma
    new_phi = [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
    return new_phi
end


function tri_ang(n1,n2,n3)
    n2n3 = n2[1] * n3[1] + n2[2] * n3[2] + n2[3] * n3[3]
    n1n2 = n1[1] * n2[1] + n1[2] * n2[2] + n1[3] * n2[3]
    n1n3 = n1[1] * n3[1] + n1[2] * n3[2] + n1[3] * n3[3]

    if n1n2^2 >= 1.0 || n1n3^2 >= 1.0
        arg = (n2n3 - n1n2 * n1n3)
        ang = 0.0
    else
        arg = (n2n3 - n1n2 * n1n3) / sqrt((1.0 - (n1n2)^2) * (1.0 - (n1n3)^2))
        if 1.0-0.00001<abs(arg)<1+0.00001
            arg = sign(arg)*1.0
        end
        ang = acos(arg)
    end
    return ang
end

function tri_area(n1,n2,n3)
    ar= tri_ang(n1,n2,n3)+tri_ang(n2,n3,n1)+tri_ang(n3,n1,n2) - pi
    return ar
end

function tet_charge(a,phi_arr,err_mar)
    charge=0.0
    triangles = [[a[1,:],a[2,:],a[3,:]],
    [a[1,:],a[3,:],a[4,:]],
    [a[1,:],a[4,:],a[2,:]],
    [a[2,:],a[4,:],a[3,:]]]
    for tri in triangles
        phis = [phi_arr[j[1]+1,j[2]+1,j[3]+1,:] for j in tri]
        G,H,K = [n(o) for o in phis]
        trar = tri_area(G,H,K)
        charge = charge+trar
    end
    # err_mar = 1e-3
    if ((1.0-err_mar)<=abs(real(charge/(4*pi)))<=(1.0+err_mar))
        charge = sign(real(charge/(4*pi)))
    else
        charge = real(charge/(4*pi))
        # charge = 0.0
    end
    # return real(round(charge/(4*pi),digits=3))
    return charge
end

function mtA_vB(A,B)
    pd_0 = A[1,1] .*B[1] + A[1,2] .*B[2]
    pd_1 = A[2,1] .*B[1] + A[2,2] .*B[2]
    return [pd_0,pd_1]
end

function vA_dag_vB(A,B)
    pd_0 = conj(A[1]) .*B[1] + conj(A[2]) .*B[2]
    return pd_0
end

function mtx_pd(A,B)
    pd_00 = A[1,1]*B[1,1] + A[1,2]*B[2,1]
    pd_01 = A[1,1]*B[1,2] + A[1,2]*B[2,2]
    pd_10 = A[2,1] * B[1,1] + A[2,2] * B[2,1]
    pd_11 = A[2,1] * B[1,2] + A[2,2] * B[2,2]
    mt_pd = [pd_00 pd_01 ; pd_10 pd_11]
    return mt_pd
end

function delta_cal(n1,n2,n3,phi1,phi2,phi3)

    sigma = [[0 1;1 0],[0 -1im; 1im 0],[1 0; 0 -1]]
    I = [1 0; 0 1]

    delta = 0
    
    ve1 = n1;ve2 = n2
    p1 = phi1;p2 = phi2
    a21cs = cross(ve1,ve2)
    a21cs_mag = sqrt(a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]+a21cs[3]*a21cs[3])
    if a21cs_mag!=0.0
        a21 = a21cs/sqrt(a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]+a21cs[3]*a21cs[3])#;print(a21)
    else
        a21 = a21cs
    end
    arg=ve1[1]*ve2[1]+ve1[2]*ve2[2]+ve1[3]*ve2[3]
    if (1 - 0.00001 < abs(arg) < 1 + 0.00001) arg=1.0*sign(arg) end
    theta21 = acos(arg)
    S21r = -1im .*(sigma[1] .*a21[1] + sigma[2] .*a21[2] + sigma[3] .*a21[3]) .*sin(theta21/2) + I .*cos(theta21/2)
    p2_r = [S21r[1,1]*p1[1] + S21r[1,2]*p1[2], S21r[2,1]*p1[1]+S21r[2,2]*p1[2]]#;print(p2_r)
    delta21 = -1*(angle((real(p2[1])-imag(p2[1])*1im)*p2_r[1] + (real(p2[2])-imag(p2[2])*1im)*p2_r[2]))
    # if (theta21 == 0.0 && delta21!=0) print('2',theta21,delta21/(2*np.pi)) end
    delta = delta + delta21
    ve1 = n2;ve2 = n3
    p1 = phi2;p2 = phi3
    a21cs = cross(ve1,ve2)
    a21cs_mag = sqrt(a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]+a21cs[3]*a21cs[3])
    if a21cs_mag!=0.0
        a21 = a21cs/sqrt(a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]+a21cs[3]*a21cs[3])
    else
        a21 = a21cs
    end
    arg=ve1[1] * ve2[1] + ve1[2] * ve2[2] + ve1[3] * ve2[3]
    if (1- 0.00001 < abs(arg) < 1 + 0.00001) arg=1.0*sign(arg) end
    theta21=acos(arg)
    S32r = -1im .*(sigma[1] .*a21[1] + sigma[2] .*a21[2] + sigma[3] .*a21[3]) .*sin(theta21/2) + I .*cos(theta21/2)
    p2_r = [S32r[1,1]*p1[1] + S32r[1,2]*p1[2], S32r[2,1]*p1[1]+S32r[2,2]*p1[2]]
    delta21 = -1*(angle((real(p2[1]) - imag(p2[1])*1im) * p2_r[1] + (real(p2[2]) - imag(p2[2])*1im) * p2_r[2]))
    # if (theta21 == 0.0 && delta21)!=0 print('2',theta21,delta21/(2*np.pi)) end
    delta = delta + delta21
    ve1 = n3;ve2 = n1
    p1 = phi3;p2 = phi1
    a21cs = cross(ve1,ve2)
    a21cs_mag = sqrt(a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]+a21cs[3]*a21cs[3])
    if a21cs_mag!=0.0
        a21 = a21cs/sqrt(a21cs[1]*a21cs[1]+a21cs[2]*a21cs[2]+a21cs[3]*a21cs[3])
    else
        a21 = a21cs
    end
    arg=ve1[1] * ve2[1] + ve1[2] * ve2[2] + ve1[3] * ve2[3]
    if (1- 0.00001 < abs(arg) < 1 + 0.00001) arg=1.0*sign(arg) end
    theta21=acos(arg)
    S13r = -1im .*(sigma[1] .*a21[1] + sigma[2] .*a21[2] + sigma[3] .*a21[3]) .*sin(theta21/2) + I .*cos(theta21/2)#;print(S21r)
    p2_r = [S13r[1,1]*p1[1] + S13r[1,2]*p1[2], S13r[2,1]*p1[1]+S13r[2,2]*p1[2]]
    delta21 = -1*(angle((real(p2[1]) - imag(p2[1])*1im) * p2_r[1] + (real(p2[2]) - imag(p2[2])*1im) * p2_r[2]))
    # if (theta21 == 0.0 && delta21!=0) print(S13r);print('2',theta21,delta21/(2*np.pi)) end
    delta = delta + delta21
    h_123 = angle(vA_dag_vB(phi1, mtA_vB(mtx_pd(S13r,mtx_pd(S32r,S21r)),phi1)))

    return delta + h_123
end

function del_plk(tri,phi_arr)
    phis = [phi_arr[i[1]+1,i[2]+1,i[3]+1,:] for i in eachrow(tri)]
    nvecs = [n(phi) for phi in phis]
    del = delta_cal(nvecs[1],nvecs[2],nvecs[3],phis[1],phis[2],phis[3])
    return del
end

end