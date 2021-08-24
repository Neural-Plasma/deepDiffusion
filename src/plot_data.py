import matplotlib.pyplot as plt

def model_history(history):
    plt.figure()
    plt.semilogy(history.history['loss'][1:], '.-', lw=2)
    plt.semilogy(history.history['val_loss'][1:], '.-', lw=2)
    plt.xlabel('epochs')
    plt.ylabel('Validation loss')
    plt.legend(['training loss', 'validation loss'])
    plt.show()

def plot_solution(X,Y,uall):
    fig1, axs = plt.subplots(2,2)
    im = axs[0,0].contourf(X,Y,uall[0,:,:], 100,cmap=plt.get_cmap('hot'))
    im = axs[0,1].contourf(X,Y,uall[1,:,:], 100,cmap=plt.get_cmap('hot'))
    im = axs[1,0].contourf(X,Y,uall[2,:,:], 100,cmap=plt.get_cmap('hot'))
    im = axs[1,1].contourf(X,Y,uall[3,:,:], 100,cmap=plt.get_cmap('hot'))
    fig1.colorbar(im, ax=axs.ravel().tolist())
    plt.show()
