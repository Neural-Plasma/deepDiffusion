import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from os.path import join as pjoin
import ini


run_arr1 = [15,16,1,17,18]

batch_size_arr = []
train_loss_bsize = []
val_loss_bsize = []

for i in run_arr1:
    path = pjoin('..','run%d'%i)
    params = ini.parse(open(pjoin(path,'input%d'%i+'.ini')).read())
    batch_size = float(params['dnn']['batch_size'])
    batch_size_arr.append(batch_size)
    train_loss = np.load(pjoin(path,"train_history.npz"))['loss']
    train_loss_bsize.append(train_loss)
    val_loss = np.load(pjoin(path,"train_history.npz"))['val_loss']
    val_loss_bsize.append(val_loss)


batch_size_arr = np.array(batch_size_arr)
train_loss_bsize = np.array(train_loss_bsize)
val_loss_bsize = np.array(val_loss_bsize)

#################################################
run_arr2 = [10,11,1,12,13,14]

nlayer_arr = []
train_loss_nl = []
val_loss_nl = []

for i in run_arr2:
    path = pjoin('..','run%d'%i)
    params = ini.parse(open(pjoin(path,'input%d'%i+'.ini')).read())
    nlayer = float(params['dnn']['nlayer'])
    nlayer_arr.append(nlayer)
    train_loss = np.load(pjoin(path,"train_history.npz"))['loss']
    train_loss_nl.append(train_loss)
    val_loss = np.load(pjoin(path,"train_history.npz"))['val_loss']
    val_loss_nl.append(val_loss)


nlayer_arr = np.array(nlayer_arr)
train_loss_nl = np.array(train_loss_nl)
val_loss_nl = np.array(val_loss_nl)

#################################################
run_arr3 = [19,20,21,22,1,23,24,25,26,27]

nn_arr = []
train_loss_nn = []
val_loss_nn = []

for i in run_arr3:
    path = pjoin('..','run%d'%i)
    params = ini.parse(open(pjoin(path,'input%d'%i+'.ini')).read())
    nn = float(params['dnn']['nn'])
    nn_arr.append(nn)
    train_loss = np.load(pjoin(path,"train_history.npz"))['loss']
    train_loss_nn.append(train_loss)
    val_loss = np.load(pjoin(path,"train_history.npz"))['val_loss']
    val_loss_nn.append(val_loss)

nn_arr = np.array(nn_arr)
train_loss_nn = np.array(train_loss_nn)
val_loss_nn = np.array(val_loss_nn)




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
axs.semilogy(batch_size_arr,train_loss_bsize,'.-', lw=2)
axs.semilogy(batch_size_arr,val_loss_bsize,'.-', lw=2)
# axs.axvline(x = 5,linestyle ="--",color ='white')
axs.set_xlabel('batch size')
axs.set_ylabel('L1 loss')
axs.legend(['training loss', 'validation loss'])
plt.savefig(pjoin('.','batch_size_loss.png'),dpi=dpi)

plt.cla()

axs.semilogy(nlayer_arr,train_loss_nl,'.-', lw=2)
axs.semilogy(nlayer_arr,val_loss_nl,'.-', lw=2)
# axs.axvline(x = 5,linestyle ="--",color ='white')
axs.set_xlabel('number of layers')
axs.set_ylabel('L1 loss')
axs.legend(['training loss', 'validation loss'],loc="upper left")
plt.savefig(pjoin('.','nlayer_loss.png'),dpi=dpi)

plt.cla()

axs.semilogy(nn_arr,train_loss_nn,'.-', lw=2)
axs.semilogy(nn_arr,val_loss_nn,'.-', lw=2)
# axs.axvline(x = 5,linestyle ="--",color ='white')
axs.set_xlabel('number of nuerons per layer')
axs.set_ylabel('L1 loss')
axs.legend(['training loss', 'validation loss'])
plt.savefig(pjoin('.','nn_loss.png'),dpi=dpi)
plt.show()
