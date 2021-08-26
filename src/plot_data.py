import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import matplotlib as mp
import config

def model_history(history):
    plt.figure()
    plt.semilogy(history.history['loss'][1:], '.-', lw=2)
    plt.semilogy(history.history['val_loss'][1:], '.-', lw=2)
    plt.xlabel('epochs')
    plt.ylabel('Validation loss')
    plt.legend(['training loss', 'validation loss'])
    plt.show()

def plot_solution_2D(X,Y,uall):
    figsize = np.array([150,150/1.618])
    dpi = 300
    ppi = np.sqrt(1920**2+1200**2)/24

    mp.rc('text', usetex=config.use_latex)
    mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
    mp.rc('axes', titlesize=14)
    mp.rc('axes', labelsize=14)
    mp.rc('xtick', labelsize=14)
    mp.rc('ytick', labelsize=14)
    mp.rc('legend', fontsize=14)
    fig1, axs = plt.subplots(2,2,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

    im0 = axs[0,0].contourf(X,Y,uall[1,:,:], 100,cmap=plt.get_cmap('hot'),vmin=np.min(uall), vmax=np.max(uall))
    im1 = axs[0,1].contourf(X,Y,uall[2,:,:], 100,cmap=plt.get_cmap('hot'),vmin=np.min(uall), vmax=np.max(uall))
    im2 = axs[1,0].contourf(X,Y,uall[3,:,:], 100,cmap=plt.get_cmap('hot'),vmin=np.min(uall), vmax=np.max(uall))
    im3 = axs[1,1].contourf(X,Y,uall[4,:,:], 100,cmap=plt.get_cmap('hot'),vmin=np.min(uall), vmax=np.max(uall))
    # fig1.colorbar(im0,ax=axs[0,0])
    # fig1.colorbar(im1,ax=axs[0,1])
    # fig1.colorbar(im2,ax=axs[1,0])
    # fig1.colorbar(im3,ax=axs[1,1])
    axs[0,0].set_title('t = 25')
    axs[0,1].set_title('t = 50')
    axs[1,0].set_title('t = 75')
    axs[1,1].set_title('t = 100')
    fig1.colorbar(im3, ax=axs.ravel().tolist())
    plt.savefig(pjoin('data','2d_sol.png'),dpi=dpi)
    # plt.show()

def plot_solution_1D(x1D,u1D):
    figsize = np.array([150,150/1.618])
    dpi = 300
    ppi = np.sqrt(1920**2+1200**2)/24

    mp.rc('text', usetex=config.use_latex)
    mp.rc('font', family='sans-serif', size=14, serif='Computer Modern Roman')
    mp.rc('axes', titlesize=14)
    mp.rc('axes', labelsize=14)
    mp.rc('xtick', labelsize=14)
    mp.rc('ytick', labelsize=14)
    mp.rc('legend', fontsize=14)
    fig1, axs = plt.subplots(2,2,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)

    axs[0,0].plot(x1D,u1D[1,:],lw=2)
    axs[0,1].plot(x1D,u1D[2,:],lw=2)
    axs[1,0].plot(x1D,u1D[3,:],lw=2)
    axs[1,1].plot(x1D,u1D[4,:],lw=2)
    axs[0,0].set_title('t = 25')
    axs[0,1].set_title('t = 50')
    axs[1,0].set_title('t = 75')
    axs[1,1].set_title('t = 100')
    plt.savefig(pjoin('data','1d_sol.png'),dpi=dpi)
