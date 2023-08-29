import numpy as np
from scipy.optimize import  curve_fit

from Lattice import *

def k_cub(x,a,b):
    return a*x**3+b

def aveg_spec():
    stack = np.array([read('stack_spec_%s'%i, Master_path)[:,1] for i in range(N_runs)])

    mean_spec = np.mean(stack, axis=0)

    #k_c = np.arange(0, int(3 * N))  ###k mag list####
    ###stack_spec is the final summed over bins spetrum#####

    #k_x = k_c[:int(3 * N/2)]
    #E_m = mean_spec[:int(3 * N/2)]

    k_x = read('stack_spec_0',Master_path)[:,0]

    plt.loglog(k_x, mean_spec)
    #scaling = mean_spec[int(3 * N/2)]/k_c[int(3 * N/2)]**3
    #plt.loglog(k_x,k_x**3*scaling,label=r'$k^3$')

    #polfit = np.polyfit(k_x,E_m,3)
    #label = ['a:%.2e'%(polfit[0]),'b:%.2e'%(polfit[1]),'c:%.2e'%(polfit[2]),'d:%.2e'%(polfit[3])]
    #plt.plot(k_x,np.poly1d(polfit)(k_x),linestyle='--', label=label)

    #popt, pcov = curve_fit(k_cub, k_x, E_m)
    #plt.loglog(k_x, k_cub(k_x, *popt), label = 'fit: a=%.2e,    b=%.2e'%tuple(popt),linestyle=':')

    plt.xlabel(r'$K_c$')
    plt.ylabel(r'$E_M$')
    plt.legend();plt.show()

    return

aveg_spec()