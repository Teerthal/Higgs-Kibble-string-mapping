import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dirs = [
"/media/teerthal/Repo/Kibble/smear/surf/run_1",
"/media/teerthal/Repo/Kibble/smear/surf/run_2",
"/media/teerthal/Repo/Kibble/smear/surf/run_3",
"/media/teerthal/Repo/Kibble/smear/surf/run_4",
"/media/teerthal/Repo/Kibble/smear/surf/run_5",
"/media/teerthal/Repo/Kibble/smear/surf/run_6",
"/media/teerthal/Repo/Kibble/smear/surf/run_7",
"/media/teerthal/Repo/Kibble/smear/surf/run_8",
"/media/teerthal/Repo/Kibble/smear/surf/run_9",
"/media/teerthal/Repo/Kibble/smear/surf/run_11"
]

dirs = [
"/media/teerthal/Repo/Kibble/smear/surf/run_10"
]

# dirs = [
# "/media/cuddlypuff/HDPH-UT/Kibble/smear/run_1",
# "/media/cuddlypuff/HDPH-UT/Kibble/smear/run_2",
# "/media/cuddlypuff/HDPH-UT/Kibble/smear/run_3",
# "/media/cuddlypuff/HDPH-UT/Kibble/smear/run_4",
# "/media/cuddlypuff/HDPH-UT/Kibble/smear/run_5"
# ]

def read(filename,path):
    data = np.load("%s/%s"%(path,filename),allow_pickle=True)
    return data

def quad_fn(x,a, c):
    return a*x**(-2)+c
def lin_fn(x,a, b):
    return a*x+b

list = []
stack = []
for dir in dirs:
    for filename in os.listdir(dir):
        if filename.startswith("int") and filename.endswith(".npy"):

            dat = read(filename,dir)

            if len(dat) == 249:
                list.append(os.path.join(dir, filename))
                stack.append(dat)
                #print(np.shape(dat))
stack = np.array(stack)

N = 501
def smear_plot_2():
    mag_stck = []
    slopes=[]
    fig,ax = plt.subplots()
    start = 100
    lens = np.arange(3,N,2)
    vols = np.array(lens**3)
    mix = np.sqrt(0.22)
    g = 0.65
    for i in stack:

        B_mag = np.real([np.sqrt(np.dot(np.conj(x),x)) for x in i])*2*mix/g
        #print(np.shape(B_mag));exit()
        mag_stck.append(B_mag/vols)
        #plt.plot(np.log(lens), np.log(B_mag/vols),linestyle='',marker='.',markersize=.5,c='blue',markeredgecolor='none')
        # plt.plot((lens), (B_mag / vols), linestyle='', marker='.', markersize=.5, c='blue',
        #          markeredgecolor='none')

    ###Enter the dimensionless constants sin(\theta_w) and g###

    B_mean = np.mean(mag_stck, axis=0)
    B_std = np.std(mag_stck, axis=0)
    #plt.errorbar(np.log(lens), np.log(B_mean), yerr=np.log(B_std))
    plt.fill_between(np.log(lens), np.log(B_mean-B_std), np.log(B_mean+B_std),alpha=.7,color='tab:blue',ec='none')
    plt.plot(np.log(lens), np.log(B_mean),linestyle='',marker='.',color='k',markersize=5)
    #plt.bar(np.log(lens),np.log(B_mean-B_std), fill=0)

    # plt.errorbar(lens, (B_mean), yerr=(B_std),ecolor='k')
    # #plt.fill_between((lens), (B_mean-B_std), (B_mean+B_std),alpha=1,color='grey')
    # plt.plot((lens),(B_mean),linestyle='',marker='.',color='k',markersize=5)


    popt,pcov = curve_fit(lin_fn, np.log(lens[start:]),np.log(B_mean[start:]), sigma=B_std[start:])
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    xp = np.log(np.linspace(2, N, 1000))
    plt.plot(xp,lin_fn(xp,*popt), linestyle='--',c='r')
    #plt.loglog(np.exp(xp), np.exp(lin_fn(xp, *popt)), linestyle='--', c='r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel(r'ln($\lambda$)',fontsize=20)
    plt.ylabel(r'ln($B_\lambda$)',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.show()
    return
smear_plot_2()