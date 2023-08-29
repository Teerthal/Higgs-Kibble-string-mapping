import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dirs = [
"/media/teerthal/Repo/Kibble/smear/run_1",
"/media/teerthal/Repo/Kibble/smear/run_2",
"/media/teerthal/Repo/Kibble/smear/run_3",
"/media/teerthal/Repo/Kibble/smear/run_4",
"/media/teerthal/Repo/Kibble/smear/run_5"
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

list = []
stack = []
for dir in dirs:
    for filename in os.listdir(dir):
        if filename.endswith(".npy"):
            list.append(os.path.join(dir,filename))
            stack.append(read(filename,dir))

stack = np.array(stack)
print(np.shape(stack))
N=301

def quad_fn(x,a, c):
    return a*x**(-2)+c
def lin_fn(x,a, b):
    return a*x+b
def smear_plot():
    mag_stck = []
    slopes=[]
    strt = int(150)
    end = int(290)
    fig,ax = plt.subplots()
    vols = np.arange(2,N)**3
    for i in stack:

        B_mag = np.real([np.sqrt(np.dot(np.conj(x),x)) for x in i])

        mag_stck.append(B_mag[2:]/vols)
        plt.plot(np.log(np.arange(2,N)), np.log(np.real(B_mag[2:])/vols),linestyle='',marker='.',markersize=.5,c='blue',markeredgecolor='none')
    print(np.shape(mag_stck))
    mag_stck = np.array(mag_stck[2:])
    B_mag = np.real(np.mean(mag_stck,axis=0))
    #B_mag = np.sqrt(np.mean(np.real(mag_stck)**2,axis=0))
    B_err = np.std(np.real(mag_stck),axis=0)#/np.sqrt(len(stack))
    print(max(B_err))
    B_err = np.abs(np.array([max(mag_stck[:,x]) for x in range(N-2)])-
                   np.array([min(mag_stck[:,x]) for x in range(N-2)]))/2
    print('max',max(B_err))
    #B_err = np.std(np.abs(mag_stck), axis=0)
    #B_err = np.array([np.sqrt(np.mean(np.abs(mag_stck[:,y]))) for y in range(N)])
    vols = np.arange(strt, end) ** 3

    plt.plot(np.log(np.arange(2,N)), np.log(B_mag),linestyle='',marker='.',c='black', markeredgecolor='none',markersize=5)
    err_idxs = [0,1,2,3,4,5,6,7,8,10,12,15,18,22,26,30,36,45,52,60,70,80,100,125,155,190,215,250,-1]
    print(err_idxs)


    plt.errorbar(np.log(np.arange(2,N)[err_idxs]), np.log(B_mag[err_idxs]),yerr=(B_err[err_idxs]),linestyle='',c='k',ecolor='k', fmt='none',capsize=2,marker='',capthick=.5,lw=.5)


    #plt.fill_between(np.arange(2,N), B_mag + 2*B_err, B_mag - 2*B_err,alpha=.5)
    x = 1/(np.arange(strt,end)**(2))

    x = np.arange(strt+2,end)
    y = np.real(B_mag[strt+2:end])
    popt,cov = np.polyfit(np.log(x),np.log(y),1,cov=True)
    lin_fit = np.poly1d(popt)
    xp = np.log(np.linspace(2,N,1000))

    print(popt)
    print(np.sqrt(np.diag(cov)))
    print(lin_fit)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.loglog(np.exp(xp), np.exp(lin_fit(xp)),linestyle='--',c='blue')  # ,label='slope:%.2f intercept:%.1f'%(popt[0],popt[1]))


    x = np.arange(strt+2,end)
    y = np.real(B_mag[strt+2:end])
    print(np.shape(B_err), np.shape(y))
    popt,pcov = curve_fit(lin_fn, np.log(x),np.log(y), sigma=B_err[strt+2:end])
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    xp = np.log(np.linspace(2, N, 1000))
    plt.plot(xp,lin_fn(xp,*popt), linestyle='--',c='r')
    plt.plot(xp,lin_fn(xp,-2,1),linestyle='--',c='g')
    #plt.plot(np.arange(N),np.absolute(stack[:,0]))
    #plt.plot(np.arange(N), np.absolute(stack[:, 1]))
    #plt.plot(np.arange(N), np.absolute(stack[:, 2]))
    plt.xlabel(r'$\lambda$',fontsize=20)
    plt.ylabel(r'$B_\lambda$',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    #plt.legend()
    #slopes.append(popt[0])
    # print(np.shape(slopes))
    # print(np.mean(slopes))
    # print(np.std(slopes))
    plt.show()
    return
#smear_plot()

def smear_plot_2():
    mag_stck = []
    slopes=[]
    strt = int(250)
    end = int(299)
    fig,ax = plt.subplots()
    vols = np.arange(2,N)**3
    for i in stack:

        B_mag = np.real([np.sqrt(np.dot(np.conj(x),x)) for x in i])

        mag_stck.append(B_mag[2:]/vols)
        plt.plot(np.log(np.arange(2,N)), np.log(np.real(B_mag[2:])/vols),linestyle='',marker='.',markersize=.5,c='blue',markeredgecolor='none')

        x = np.arange(strt + 2, end)
        y = np.real(B_mag[strt + 2:end])

        # x = np.arange(2, N)
        # y = np.real(B_mag[2:])

        popt, cov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
        lin_fit = np.poly1d(popt)
        xp = np.log(np.linspace(2, N, 1000))
        slopes.append(popt[0])

    print(np.mean(slopes))
    print(np.std(slopes))

    mag_stck = np.array(mag_stck[2:])

    B_mag = np.real(np.mean(mag_stck,axis=0))
    B_err = np.std(np.real(mag_stck),axis=0)#/np.sqrt(len(stack))
    print('max',max(B_err))
    vols = np.arange(strt, end) ** 3

    plt.plot(np.log(np.arange(2,N)), np.log(B_mag),linestyle='',marker='.',c='black', markeredgecolor='none',markersize=5)
    err_idxs = [0,1,2,3,4,5,6,7,8,10,12,15,18,22,26,30,36,45,52,60,70,80,100,125,155,190,215,250,-1]
    print(err_idxs)


    #plt.errorbar(np.log(np.arange(2,N)[err_idxs]), np.log(B_mag[err_idxs]),yerr=(B_err[err_idxs]),linestyle='',c='k',ecolor='k', fmt='none',capsize=2,marker='',capthick=.5,lw=.5)


    #plt.fill_between(np.arange(2,N), B_mag + 2*B_err, B_mag - 2*B_err,alpha=.5)
    x = 1/(np.arange(strt,end)**(2))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.loglog(np.exp(xp), np.exp(lin_fit(xp)),linestyle='--',c='blue')  # ,label='slope:%.2f intercept:%.1f'%(popt[0],popt[1]))


    x = np.arange(strt+2,end)
    y = np.real(B_mag[strt+2:end])

    popt,pcov = curve_fit(lin_fn, np.log(x),np.log(y), sigma=B_err[strt+2:end])
    popt, pcov = curve_fit(quad_fn, x, y)


    # tot_stck = np.hstack(mag_stck)
    # print(len(mag_stck)*(N-2))
    # len_stck = np.hstack([np.arange(2,N)]*len(mag_stck))
    # popt, pcov = curve_fit(lin_fn, np.log(len_stck), np.log(tot_stck))


    print(popt)
    print(np.sqrt(np.diag(pcov)))
    xp = np.log(np.linspace(2, N, 1000))
    plt.plot(xp,np.log(quad_fn(xp,*popt)), linestyle='--',c='r')
    plt.plot(xp,lin_fn(xp,-2,1),linestyle='--',c='g')
    #plt.plot(np.arange(N),np.absolute(stack[:,0]))
    #plt.plot(np.arange(N), np.absolute(stack[:, 1]))
    #plt.plot(np.arange(N), np.absolute(stack[:, 2]))
    plt.xlabel(r'ln($\lambda$)',fontsize=20)
    plt.ylabel(r'ln($B_\lambda$)',fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    #plt.legend()
    #slopes.append(popt[0])
    # print(np.shape(slopes))
    # print(np.mean(slopes))
    # print(np.std(slopes))
    plt.show()
    return
smear_plot_2()