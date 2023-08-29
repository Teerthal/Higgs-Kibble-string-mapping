module base_routines

using NPZ
using Statistics
using StatsBase
using Random
using LinearAlgebra
using FFTW

# include("parameters.jl")
# using .paras

function gen_phi()
    alpha = 0.5*acos(sample([-1.0,1.0])*rand(Float64))
    beta = rand(Float64)*2.0*pi
    gamma = rand(Float64)*2.0*pi
    return [cos(alpha)*exp(1im*beta),sin(alpha)*exp(1im*gamma)]
end

export random_gen

function random_gen(phi_arr,N)

    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)

                phi_arr[i,j,k,:]=gen_phi()
                
                if i==N
                    phi_arr[i,j,k,:]=phi_arr[1,j,k,:]
                end
                if j==N
                    phi_arr[i,j,k,:]=phi_arr[i,1,k,:]
                end
                if k==N
                    phi_arr[i,j,k,:]=phi_arr[i,j,1,:]
                end
            end
        end
    end

    return phi_arr
end

export pbc_cen_diff
function pbc_cen_diff(phi_arr,dphi_arr,N,a=1)
    
    i_f=1;j_f=1;k_f=1
    for i in range(1,N-1,step=1)
        for j in range(1,N-1,step=1)
            for k in range(1,N-1,step=1)
                if i==1 i_b=-N+2 else i_b = 1 end
                if j==1 j_b=-N+2 else j_b = 1 end
                if k==1 k_b=-N+2 else k_b = 1 end
                dphi_arr[i,j,k,:,1] = 0.5*a*(phi_arr[i-i_b,j,k,:]-phi_arr[i+i_f,j,k,:])
                dphi_arr[i,j,k,:,2] = 0.5*a*(phi_arr[i,j-j_b,k,:]-phi_arr[i,j+j_f,k,:])
                dphi_arr[i,j,k,:,3] = 0.5*a*(phi_arr[i,j,k-k_b,:]-phi_arr[i,j,k+k_f,:])
            end
        end
    end

    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)
                if i == N
                    dphi_arr[i,j,k,:,:] = dphi_arr[1,j,k,:,:]
                end
                if j == N
                    dphi_arr[i,j,k,:,:] = dphi_arr[i,1,k,:,:]
                end
                if k == N
                    dphi_arr[i,j,k,:,:] = dphi_arr[i,j,1,:,:]
                end
            end
        end
    end

    for i in [1,N]
        for j in [1,N]
            for k in [1,N]
                dphi_arr[i,j,k,:,:] = dphi_arr[1,1,1,:,:]
            end
        end
    end
    return dphi_arr

end 

export B_cal_arr
function B_cal_arr(dphi_arr,B_arr,N)
    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)            
                B_arr[i,j,k,1]=i
                B_arr[i,j,k,2]=j
                B_arr[i,j,k,3]=k
                B_x = 1im*((conj(dphi_arr[i,j,k,1,2])*dphi_arr[i,j,k,1,3]+
                    conj(dphi_arr[i,j,k,2,2])*dphi_arr[i,j,k,2,3])-
                    (conj(dphi_arr[i,j,k,1,3])*dphi_arr[i,j,k,1,2]+
                    conj(dphi_arr[i,j,k,2,3])*dphi_arr[i,j,k,2,2]))
                B_y = 1im*((conj(dphi_arr[i,j,k,1,3])*dphi_arr[i,j,k,1,1]+
                    conj(dphi_arr[i,j,k,2,3])*dphi_arr[i,j,k,2,1])-
                    (conj(dphi_arr[i,j,k,1,1])*dphi_arr[i,j,k,1,3]+
                    conj(dphi_arr[i,j,k,2,1])*dphi_arr[i,j,k,2,3]))
                B_z = 1im*((conj(dphi_arr[i,j,k,1,1])*dphi_arr[i,j,k,1,2]+
                    conj(dphi_arr[i,j,k,2,1])*dphi_arr[i,j,k,2,2])-
                    (conj(dphi_arr[i,j,k,1,2])*dphi_arr[i,j,k,1,1]+
                    conj(dphi_arr[i,j,k,2,2])*dphi_arr[i,j,k,2,1]))

                B_arr[i,j,k,4] = B_x
                B_arr[i,j,k,5] = B_y
                B_arr[i,j,k,6] = B_z
            end
        end
    end
    return real(B_arr)
end

export A_cal_arr
function A_cal_arr(dphi_arr,phi_arr,A_arr,N)
    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)
                A_x = 1im*(conj(phi_arr[i,j,k,1])*dphi_arr[i,j,k,1,1]+conj(phi_arr[i,j,k,2])*dphi_arr[i,j,k,2,1])
                A_y = 1im*(conj(phi_arr[i,j,k,1])*dphi_arr[i,j,k,1,2]+conj(phi_arr[i,j,k,2])*dphi_arr[i,j,k,2,2])
                A_z = 1im*(conj(phi_arr[i,j,k,1])*dphi_arr[i,j,k,1,3]+conj(phi_arr[i,j,k,2])*dphi_arr[i,j,k,2,3])
                A_arr[i,j,k,1] = A_x
                A_arr[i,j,k,2] = A_y
                A_arr[i,j,k,3] = A_z
            end
        end
    end
    return real(A_arr)
end

export curl_A_arr
function curl_A_arr(A_arr, curl_A,N,a=1)
    for i in range(1,N-1,step=1)
        for j in range(1,N-1,step=1)
            for k in range(1,N-1,step=1)            
                i_f=1;j_f=1;k_f=1
                if i==1 i_b=-N+2 else i_b = 1 end
                if j==1 j_b=-N+2 else j_b = 1 end
                if k==1 k_b=-N+2 else k_b = 1 end
                dAy_dx = 0.5*a*(A_arr[i-i_b,j,k,2]-A_arr[i+i_f,j,k,2])
                dAz_dx = 0.5*a*(A_arr[i-i_b,j,k,3]-A_arr[i+i_f,j,k,3])
                dAx_dy = 0.5*a*(A_arr[i,j-j_b,k,1]-A_arr[i,j+j_f,k,1])
                dAz_dy = 0.5*a*(A_arr[i,j-j_b,k,3]-A_arr[i,j+j_f,k,3])
                dAx_dz = 0.5*a*(A_arr[i,j,k-k_b,1]-A_arr[i,j,k+k_f,1])
                dAy_dz = 0.5*a*(A_arr[i,j,k-k_b,2]-A_arr[i,j,k+k_f,2])
                
                curl_A[i,j,k,1] = dAz_dy - dAy_dz
                curl_A[i,j,k,2] = dAx_dz - dAz_dx
                curl_A[i,j,k,3] = dAy_dx - dAx_dy

                # println(curl_A[i,j,k,:]);exit()
            end
        end
    end
    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)
                if i == N
                    curl_A[i,j,k,:] = curl_A[1,j,k,:]
                end
                if j == N
                    curl_A[i,j,k,:] = curl_A[i,1,k,:]
                end
                if k == N
                    curl_A[i,j,k,:] = curl_A[i,j,1,:]
                end
            end
        end
    end

    for i in [1,N]
        for j in [1,N]
            for k in [1,N]
                curl_A[i,j,k,:] = curl_A[1,1,1,:]
            end
        end
    end
    return curl_A
end


export B_div
function B_div(B_arr, div_B,N,a=1)
    for i in range(1,N-1,step=1)
        for j in range(1,N-1,step=1)
            for k in range(1,N-1,step=1)            
                i_f=1;j_f=1;k_f=1
                if i==1 i_b=-N+2 else i_b = 1 end
                if j==1 j_b=-N+2 else j_b = 1 end
                if k==1 k_b=-N+2 else k_b = 1 end
                dB_x = 0.5*a*(B_arr[i-i_b,j,k,4]-B_arr[i+i_f,j,k,4])
                dB_y = 0.5*a*(B_arr[i,j-j_b,k,5]-B_arr[i,j+j_f,k,5])
                dB_z = 0.5*a*(B_arr[i,j,k-k_b,6]-B_arr[i,j,k+k_f,6])
                div_B[i,j,k] = dB_x+dB_y+dB_z
            end
        end
    end
    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)
                if i == N
                    div_B[i,j,k] = div_B[1,j,k]
                end
                if j == N
                    div_B[i,j,k] = div_B[i,1,k]
                end
                if k == N
                    div_B[i,j,k] = div_B[i,j,1]
                end
            end
        end
    end

    for i in [1,N]
        for j in [1,N]
            for k in [1,N]
                div_B[i,j,k] = div_B[1,1,1]
            end
        end
    end
    return div_B
end

export interp_charges
function interp_charges(B_arr,flux,N)
    #This calculates surface integrals over all cells 
    #by averaging the Bs on the faces
    for i in range(1,N-1,step=1)
        for j in range(1,N-1,step=1)
            for k in range(1,N-1,step=1)
                B_l = 0.25*(B_arr[i,j,k,4]+B_arr[i,j+1,k,4]+B_arr[i,j+1,k+1,4]+B_arr[i,j,k+1,4])
                B_r = 0.25*(B_arr[i+1,j,k,4]+B_arr[i+1,j+1,k,4]+B_arr[i+1,j+1,k+1,4]+B_arr[i+1,j,k+1,4])
                B_b = 0.25*(B_arr[i,j,k,5]+B_arr[i+1,j,k,5]+B_arr[i+1,j,k+1,5]+B_arr[i,j,k+1,5])
                B_f = 0.25*(B_arr[i,j+1,k,5]+B_arr[i+1,j+1,k,5]+B_arr[i+1,j+1,k+1,5]+B_arr[i,j+1,k+1,5])
                B_d = 0.25*(B_arr[i,j,k,6]+B_arr[i+1,j,k,6]+B_arr[i+1,j+1,k,6]+B_arr[i,j+1,k,6])
                B_u = 0.25*(B_arr[i,j,k+1,6]+B_arr[i+1,j,k+1,6]+B_arr[i+1,j+1,k+1,6]+B_arr[i,j+1,k+1,6])
                flux[i,j,k] = B_r-B_l+B_f-B_b+B_u-B_d
            end
        end
    end
    return flux
end

export charges
function charges(B_arr,flux,N)
    #This computes charges by taking the actual central points
    #on the faces of 3x3x3 cells.
    for i in range(2,N-1,step=2)
        for j in range(2,N-1,step=2)
            for k in range(2,N-1,step=2)
                B_l = (B_arr[i-1,j,k,4])
                B_r = (B_arr[i+1,j,k,4])
                B_b = (B_arr[i,j-1,k,5])
                B_f = (B_arr[i,j+1,k,5])
                B_d = (B_arr[i,j,k-1,6])
                B_u = (B_arr[i,j,k+1,6])
                flux[i,j,k] = B_r-B_l+B_f-B_b+B_u-B_d
            end
        end
    end
    return flux
end

export div_charges
function div_charges(div_B,charges_div,N)
    #This computes charges by taking the actual central points
    #on the faces of 3x3x3 cells.
    # for i in range(1,N-1,step=1)
    #     for j in range(1,N-1,step=1)
    #         for k in range(1,N-1,step=1)
    #             B_l = (div_B[i,j,k]+div_B[i,j+1,k]+div_B[i,j+1,k+1]+div_B[i,j,k+1])
    #             B_r = (div_B[i+1,j,k]+div_B[i+1,j+1,k]+div_B[i+1,j+1,k+1]+div_B[i+1,j,k+1])
    #             B_b = (div_B[i,j,k]+div_B[i+1,j,k]+div_B[i+1,j,k+1]+div_B[i,j,k+1])
    #             B_f = (div_B[i,j+1,k]+div_B[i+1,j+1,k]+div_B[i+1,j+1,k+1]+div_B[i,j+1,k+1])
    #             B_d = (div_B[i,j,k]+div_B[i+1,j,k]+div_B[i+1,j+1,k]+div_B[i,j+1,k])
    #             B_u = (div_B[i,j,k+1]+div_B[i+1,j,k+1]+div_B[i+1,j+1,k+1]+div_B[i,j+1,k+1])
    #             charges_div[i,j,k] = B_r+B_l+B_f+B_b+B_u+B_d
    #         end
    #     end
    # end

    #coarse approach wherein charges are computed over 3x3 cells
    for i in range(2,N-1,step=4)
        for j in range(2,N-1,step=4)
            for k in range(2,N-1,step=4)
                B_l = 0
                for m in [j-1,j,j+1]
                    for n in [k-1,k,k+1]
                        B_l = B_l+div_B[i-1,m,n]
                    end
                end
                B_r = 0
                for m in [j-1,j,j+1]
                    for n in [k-1,k,k+1]
                        B_r = B_r+div_B[i+1,m,n]
                    end
                end
                B_c = 0
                for m in [j-1,j,j+1]
                    for n in [k-1,k,k+1]
                        B_c = B_c+div_B[i,m,n]
                    end
                end
                charges_div[i,j,k] = B_r+B_l+B_c
                # println(B_r)
                # println(B_l)
                # println(B_c)
                # println(charges_div[i,j,k])
            end
        end
    end

    return charges_div
end

function K_c_mag(x,y,z,N)
    x=x-1
    y=y-1
    z=z-1
    if x <= floor(Int,N/2)
        K_x = x
    else
        K_x = x - N
    end

    if y <= floor(Int,N/2)
        K_y = y
    else
        K_y = y - N
    end

    if z <= floor(Int,N/2)
        K_z = z
    else
        K_z = z - N
    end

    return sqrt(K_x^2+K_y^2+K_z^2)
    # return sqrt((x-1)^2+(y-1)^2+(z-1)^2)
end

export Kc_bin_nums
function Kc_bin_nums(N)
    lis = []
    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)
                push!(lis,K_c_mag(i,j,k,N))
            end
        end
    end
    return size(unique!(sort!(lis)),1)
end


export B_spectrum
function B_spectrum(B_arr,N,nbins)
    
    Bx_fft = fft(B_arr[:,:,:,4])
    By_fft = fft(B_arr[:,:,:,5])
    Bz_fft = fft(B_arr[:,:,:,6])
    
    N_bins = Kc_bin_nums(N)
    spec_stack = zeros((N_bins,2))
    sorted = zeros((N^3,2))
    idx = 1
    for i in range(1,N,step=1)
        for j in range(1,N,step=1)
            for k in range(1,N,step=1)
                Bkx = Bx_fft[i,j,k]
                Bky = By_fft[i,j,k]
                Bkz = Bz_fft[i,j,k]
                sorted[idx,1]=K_c_mag(i,j,k,N)
                sorted[idx,2]=real(conj(Bkx)*Bkx+conj(Bky)*Bky+conj(Bkz)*Bkz)
                idx=idx+1
            end
        end
    end

    sorted = sortslices(sorted,dims=1)
    # for s in eachrow(sorted) println(s) end
    unique_ks = unique!(sorted[:,1])
    # println(maximum(unique_ks))
    # println(sqrt(3)*N/2)
    for (k_idx, k) in enumerate(unique_ks)
        idxs = findall(sorted[:,1].==k)
        spec_stack[k_idx,1] = k
        spec_stack[k_idx,2] = sum(sorted[idxs,2])
        # println(k," ",sum(sorted[idxs,2]))
    end
    # println(size(spec_stack,1))
    # println(size(spec_stack,1)/nbins)
    # println(floor.(Int,size(spec_stack,1)/nbins))
    bins = range(1,size(spec_stack,1),step=floor.(Int,size(spec_stack,1)/nbins))
    nbins =size(bins,1)
    ΔK = maximum(spec_stack[:,1])/nbins    
    binned_spec = zeros((nbins-1,2))
    start =1
    for (bidx,b) in enumerate(bins[1:nbins-1])
        stop = bins[bidx+1]
        # println(spec_stack[start:b,1])
        Kc = mean(spec_stack[start:b,1])
        # println(Kc);exit()
        E_m = sum(spec_stack[start:b,2])/(2*ΔK)
        binned_spec[bidx,1]=Kc
        binned_spec[bidx,2]=E_m
        start = b
    end
    
    return spec_stack,binned_spec
end

end