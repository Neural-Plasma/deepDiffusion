import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from os.path import join as pjoin

w = 10.
h = 10.
dx = 0.1
dy = 0.1
nx, ny = int(w/dx), int(h/dy)
x = np.linspace(0,w,nx)
y = np.linspace(0,h,ny)

X,Y = np.meshgrid(x,y)

##### Source ########
s = np.exp(-((X-w/2)**2+(Y-h/2)**2))



figsize = np.array([150,150/1.618])
dpi = 300
ppi = np.sqrt(1920**2+1200**2)/24

mp.rc('text', usetex=True)
mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
mp.rc('axes', titlesize=14)
mp.rc('axes', labelsize=14)
mp.rc('xtick', labelsize=14)
mp.rc('ytick', labelsize=14)
mp.rc('legend', fontsize=14)
fig, axs = plt.subplots(1,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
im = axs.contourf(X,Y,s, 100,cmap=plt.get_cmap('hot'))
axs.axvline(x = 5,linestyle ="--",color ='white')

# axs.set_xlabel('Epochs')
# axs.set_ylabel('Loss')

fig.colorbar(im,ax=axs)
plt.savefig(pjoin('.','source_plot.png'),dpi=dpi)
plt.show()
