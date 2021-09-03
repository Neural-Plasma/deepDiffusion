#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib as mp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import os
from os.path import join as pjoin
# import plotly.graph_objects as go
#========= Configuration ===========
parser = argparse.ArgumentParser(description='Grid Data Animator')
parser.add_argument('-a', default=True, type=bool, help='Show Animation (True/False)')
parser.add_argument('-s', default=False, type=bool, help='Save Animation (True/False)')
parser.add_argument('-d', default=False, type=bool, help='3D Animation (True/False)')
args = parser.parse_args()

show_anim = args.a
save_anim = args.s
Vis3D = args.d
interval = 100 #in mseconds

path =pjoin('..','data')

file_name = "data"#"rhoNeutral" #"P"


#========== Figure Directory Setup =============
figPath  = pjoin(path,"figures")  # DO NOT CHANGE THE PATH
if os.path.exists(figPath):
    print("figure directory exists. Existing figures will be replaced.")
else:
    os.mkdir(figPath)

h5 = h5py.File(pjoin(path,file_name+'.hdf5'),'r')

lx, ly = h5.attrs["lx"], h5.attrs["ly"]
nx, ny = h5.attrs["nx"], h5.attrs["ny"]
nsteps = h5.attrs["nsteps"] - 1

x = np.linspace(0, lx, nx, dtype='float64')
y = np.linspace(0, ly, ny, dtype='float64')
X, Y = np.meshgrid(x, y)

maxP = np.max(h5["%d"%nsteps]);
minP = np.min(h5["%d"%nsteps]);

# yy, zz = np.meshgrid(range(2), range(2))
# xx = yy*0
PY, PZ = np.meshgrid(np.linspace(0, ly, ny, dtype='float64'), np.linspace(minP, maxP, ny, dtype='float64'))
PX = (x[int(nx/2)-10])*np.ones(PY.shape)

# dataset index
# data_num = np.arange(start=0, stop=Nt, step=dp, dtype=int)



if (show_anim == True):
    def animate(i):
        #======Potential Data=========
        data = h5["%d"%i]
        # data = np.transpose(data)

        ax1.cla()
        if Vis3D == True:
            if i>=50:
                # ax1.plot_surface(PX,PY,PZ,alpha=1,cmap=cm.viridis)
                ax1.text(0.85*lx, 0.5*ly,  0.85*maxP,'ML', color = 'black',fontsize = 14)
            else:
                ax1.text(0.85*lx, 0.5*ly,  0.85*maxP,'FD', color = 'black',fontsize = 14)
            img1 = ax1.plot_surface(X,Y,data, rstride=2, cstride=2, cmap=cm.hot)
            ax1.text(0.05*lx, 0.5*ly,  0.85*maxP,'FD', color = 'black',fontsize = 14)

            ax1.set_zlim([minP, maxP])
        else:
            img1 = ax1.contourf(X,Y,data)
        ax1.set_title('(TimeSteps = %d'%i+')')
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.set_xlim([0, lx])
        ax1.set_ylim([0, ly])
        cax.cla()
        fig.colorbar(img1, cax=cax)


##### FIG SIZE CALC ############
figsize = np.array([150,150/1.618]) #Figure size in mm
dpi = 300                         #Print resolution
ppi = np.sqrt(1920**2+1200**2)/24 #Screen resolution

mp.rc('text', usetex=False)
mp.rc('font', family='sans-serif', size=10, serif='Computer Modern Roman')
mp.rc('axes', titlesize=10)
mp.rc('axes', labelsize=10)
mp.rc('xtick', labelsize=10)
mp.rc('ytick', labelsize=10)
mp.rc('legend', fontsize=10)

if (show_anim == True):
    fig,ax1 = plt.subplots(figsize=figsize/25.4,constrained_layout=False,dpi=ppi)
    div = make_axes_locatable(ax1)
    cax = div.append_axes('right', '4%', '4%')
    data = h5["0"]
    # data = np.transpose(data)
    if Vis3D == True:
        fig = plt.figure(figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
        ax1 = plt.axes(projection ="3d")
        img1 = ax1.plot_surface(X,Y,data, rstride=2, cstride=2, cmap=cm.hot)
    else:
        img1 = ax1.contourf(X,Y,data)
    cbar = fig.colorbar(img1,cax=cax)
    ani = animation.FuncAnimation(fig,animate,frames=nsteps,interval=interval,blit=False)
    # ani.save('phase_space.gif',writer='imagemagick')
    plt.show()
    if(save_anim == True):
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=(1/interval), metadata=dict(artist='Me'), bitrate=1800)
        except RuntimeError:
            print("ffmpeg not available trying ImageMagickWriter")
            writer = animation.ImageMagickWriter(fps=(1/interval))
        print("Saving movie to "+figPath+"/. Please wait .....")
        ani.save(pjoin(figPath,'animation_deepDiff.mp4'))
