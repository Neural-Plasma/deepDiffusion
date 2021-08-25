import matplotlib.pyplot as plt

def model_history(history):
    plt.figure()
    plt.semilogy(history.history['loss'][1:], '.-', lw=2)
    plt.semilogy(history.history['val_loss'][1:], '.-', lw=2)
    plt.xlabel('epochs')
    plt.ylabel('Validation loss')
    plt.legend(['training loss', 'validation loss'])
    plt.show()

def plot_solution_2D(X,Y,uall):
    fig1, axs = plt.subplots(2,2)
    im0 = axs[0,0].contourf(X,Y,uall[0,:,:], 100,cmap=plt.get_cmap('hot'))
    im1 = axs[0,1].contourf(X,Y,uall[1,:,:], 100,cmap=plt.get_cmap('hot'))
    im2 = axs[1,0].contourf(X,Y,uall[2,:,:], 100,cmap=plt.get_cmap('hot'))
    im3 = axs[1,1].contourf(X,Y,uall[3,:,:], 100,cmap=plt.get_cmap('hot'))
    fig1.colorbar(im0,ax=axs[0,0])
    fig1.colorbar(im1,ax=axs[0,1])
    fig1.colorbar(im2,ax=axs[1,0])
    fig1.colorbar(im3,ax=axs[1,1])
    # fig1.colorbar(im, ax=axs.ravel().tolist())
    # plt.show()

def plot_solution_1D(x1D,u1D):
    fig2, axs = plt.subplots(2,2)
    axs[0,0].plot(x1D,u1D[0,:],lw=2)
    axs[0,1].plot(x1D,u1D[1,:],lw=2)
    axs[1,0].plot(x1D,u1D[2,:],lw=2)
    axs[1,1].plot(x1D,u1D[3,:],lw=2)
    
