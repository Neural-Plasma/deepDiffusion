import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from os.path import join as pjoin
import ini


run_arr = [2,3,4,5,6,7,8]

w = 10.

dx_arr = []
abs_error_arr = []
for i in run_arr:
    path = pjoin('..','run%d'%i)
    # print(str(pjoin(path,'input%d'%i+'.ini')),i)
    params = ini.parse(open(pjoin(path,'input%d'%i+'.ini')).read())
    # params = ini.parse(open('input.ini').read())
    dx = float(params['grid']['dx'])
    dx_arr.append(dx)
    abs_error = np.load(pjoin(path,"abs_error.npz"))['abs_error']
    abs_error_arr.append(abs_error)

dx_arr = np.array(dx_arr)
abs_error_arr = np.array(abs_error_arr)

# dx = np.array([0.04, 0.045, 0.05, 0.06, 0.07, 0.1, 0.2, 0.3])
nx = (w/dx_arr)
# nx = nx.astype(int)

# error_abs = np.array([0.003516051970030699 , 0.008874910808513214, 0.007759362705849371, 0.014018598198026144, 0.010008256009356661, 0.010523814033550905, 0.049702872015442046, 0.11597987922352004])

figsize = np.array([77,77/1.618])
dpi = 300
ppi = np.sqrt(1920**2+1200**2)/24

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=10)

fig, axs = plt.subplots(1,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
axs.semilogy(nx,abs_error_arr,'.-', lw=2)
# axs.semilogy(batch_size,val_loss_bsize,'.-', lw=2)
# axs.axvline(x = 5,linestyle ="--",color ='white')
axs.set_xlabel('Grid resolution')
axs.set_ylabel('Absolute Error')
# axs.legend(['training loss', 'validation loss'])
plt.savefig(pjoin('.','error_abs.png'),dpi=dpi)

#plt.show()
