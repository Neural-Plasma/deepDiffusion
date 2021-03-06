import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
import matplotlib as mp
import config

def model_history(history):
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
    fig0, axs = plt.subplots(1,1,figsize=figsize/25.4,constrained_layout=True,dpi=ppi)
    axs.semilogy(history.history['loss'][1:], '.-', lw=2)
    axs.semilogy(history.history['val_loss'][1:], '.-', lw=2)
    axs.set_xlabel('Epochs')
    axs.set_ylabel('L1 loss')
    axs.legend(['training loss', 'validation loss'])
    plt.savefig(pjoin('data','train_history.png'),dpi=dpi)
    plt.show()

def plot_solution_2D(X,Y,uall):
    figsize = np.array([180,180/1.618])
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

    if config.add_labels:
        axs[0,0].text(0.05*config.lx, 0.85*config.ly, 'FD', color = 'white',fontsize = 14)
        axs[0,0].text(0.85*config.lx, 0.85*config.ly, 'FD', color = 'white', fontsize = 14)
        axs[0,1].text(0.05*config.lx, 0.85*config.ly, 'FD', color = 'white',fontsize = 14)
        axs[0,1].text(0.85*config.lx, 0.85*config.ly, 'ML', color = 'white', fontsize = 14)
        axs[1,0].text(0.05*config.lx, 0.85*config.ly, 'FD', color = 'white',fontsize = 14)
        axs[1,0].text(0.85*config.lx, 0.85*config.ly, 'ML', color = 'white', fontsize = 14)
        axs[1,1].text(0.05*config.lx, 0.85*config.ly, 'FD', color = 'white',fontsize = 14)
        axs[1,1].text(0.85*config.lx, 0.85*config.ly, 'ML', color = 'white', fontsize = 14)
        axs[0,0].axvline(x = 0.5*config.lx,linestyle ="--",color ='white')
        axs[0,1].axvline(x = 0.5*config.lx,linestyle ="--",color ='white')
        axs[1,0].axvline(x = 0.5*config.lx,linestyle ="--",color ='white')
        axs[1,1].axvline(x = 0.5*config.lx,linestyle ="--",color ='white')
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
    figsize = np.array([180,180/1.618])
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

    max1 = np.max(u1D[1,:])
    max2 = np.max(u1D[2,:])
    max3 = np.max(u1D[3,:])
    max4 = np.max(u1D[4,:])
    if config.add_labels:
        axs[0,0].text(0.05*config.lx, 0.85*max1, 'FD', color = 'black',fontsize = 14)
        axs[0,0].text(0.85*config.lx, 0.85*max1, 'FD', color = 'black', fontsize = 14)
        axs[0,1].text(0.05*config.lx, 0.85*max2, 'FD', color = 'black',fontsize = 14)
        axs[0,1].text(0.85*config.lx, 0.85*max2, 'ML', color = 'black', fontsize = 14)
        axs[1,0].text(0.05*config.lx, 0.85*max3, 'FD', color = 'black',fontsize = 14)
        axs[1,0].text(0.85*config.lx, 0.85*max3, 'ML', color = 'black', fontsize = 14)
        axs[1,1].text(0.05*config.lx, 0.85*max4, 'FD', color = 'black',fontsize = 14)
        axs[1,1].text(0.85*config.lx, 0.85*max4, 'ML', color = 'black', fontsize = 14)
        axs[0,0].axvline(x = 0.5*config.lx,linestyle ="--",color ='black')
        axs[0,1].axvline(x = 0.5*config.lx,linestyle ="--",color ='black')
        axs[1,0].axvline(x = 0.5*config.lx,linestyle ="--",color ='black')
        axs[1,1].axvline(x = 0.5*config.lx,linestyle ="--",color ='black')
    axs[0,0].set_title('t = 25')
    axs[0,1].set_title('t = 50')
    axs[1,0].set_title('t = 75')
    axs[1,1].set_title('t = 100')
    plt.savefig(pjoin('data','1d_sol.png'),dpi=dpi)
