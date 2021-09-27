#!/usr/env/ python
import numpy as np
from os.path import join as pjoin
import os.path
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras


from diffusion import diffusion_left, diffusion_right, train_data, test_model, test_bench
import config
from dnn import dnn_model,train_dnn
from plot_data import model_history, plot_solution_2D, plot_solution_1D

parser = argparse.ArgumentParser(description='DNN Based Diffusion Eqn Solver')
parser.add_argument('-te','--test', action='store_true', help='Add this if you want to test the model')
parser.add_argument('-tr','--train', action='store_true', help='Add this if you want to train the model')
parser.add_argument('-be','--bench', action='store_true', help='Add this if you want to benchmark the model')
parser.add_argument('-wl','--wflow', action='store_true', help='workflow mode')

args        = parser.parse_args()
test_mode = args.test
train_mode = args.train
bench_mode = args.bench
wflow_mode = args.wflow


if wflow_mode:
    config.plot_fig = False
    config.nn = 10
    config.epochs = 2
    config.patience = 5
    config.batch_size=64
    config.nlayer = 2

savedir = pjoin("data",'dnn_model')

##### Initialization ########
# u = np.zeros((config.nx, config.ny))
u0L, uL, sLeft = \
[np.zeros((config.nbx, config.ny)) for _ in range(3)]

u0R, uR, sRight = \
[np.zeros((2*config.nbx, 2*config.ny)) for _ in range(3)]
X1,Y1 = np.meshgrid(config.x,config.y)
X2,Y2 = np.meshgrid(config.xd,config.yd)


##### Source ########
s1 = np.exp(-((X1-config.lx/2)**2+(Y1-config.ly/2)**2))
s2 = np.exp(-((X2-config.lx/2)**2+(Y2-config.ly/2)**2))
sLeft = s1[:config.nbx,:]
sRight = s2[2*config.nbx:,:]

# fig = plt.figure()
# ax1 = plt.axes(projection ="3d")
# ax1.plot_surface(X1[:config.nbx,:],Y1[:config.nbx,:],sLeft, rstride=2, cstride=2, cmap=cm.hot)
# ax1.plot_surface(X2[2*config.nbx-1:,:],Y2[2*config.nbx-1:,:],sRight, rstride=2, cstride=2, cmap=cm.hot)
# plt.show()
# exit()
################# Define and compile model ###############################
deep_diffusion = dnn_model(config.nn)
#################################################################

if train_mode:
    print('Running train mode')
    inputs_array,outputs_array,u1,u2 = train_data(sLeft,sRight,u0L,u0R)
    # print(u1.shape,u2.shape)
    # fig = plt.figure()
    # ax1 = plt.axes(projection ="3d")
    # ax1.plot_surface(X1[:config.nbx,:],Y1[:config.nbx,:],u1[0,:,:], rstride=2, cstride=2, cmap=cm.hot)
    # ax1.plot_surface(X2[2*config.nbx:,:],Y2[2*config.nbx:,:],u2[0,:,:], rstride=2, cstride=2, cmap=cm.hot)
    # plt.show()
    # exit()

    # exit()
    deep_diffusion,history = train_dnn(deep_diffusion,inputs_array,outputs_array,savedir)
    if config.plot_fig:
        model_history(history)
    keras.backend.clear_session()


if test_mode:
    print('Running test mode')
    print('Model data exists. loading model...')
    deep_diffusion = keras.models.load_model(pjoin(savedir))
    u1,u2 = test_model(sLeft,sRight,u0L,u0R, deep_diffusion)
    print(X1[:config.nbx,:].shape,Y1[:config.nbx,:].shape,u1[0,:,:].shape,X2[2*config.nbx:,:].shape,u2[0,:,:].shape)
    # exit()
    keras.backend.clear_session()
    if config.plot_fig:
        plot_solution_2D(X1[:,:config.nbx],Y1[:,:config.nbx],u1,X2[:,2*config.nbx:],Y2[:,2*config.nbx:],u2)
        # plot_solution_1D(config.x,uall[:,config.slice,:])
        plt.show()
    # if config.vtkData:
    #     from vtk_data import vtkwrite
    #     print('Writing VTK files for Paraview visualization ...')
    #     vtkwrite('data')

if bench_mode:
    print('Running benchmark mode')
    uall_bench = test_bench(sLeft,sRight,u0L,u0R,u, deep_diffusion)

    deep_diffusion = keras.models.load_model(pjoin(savedir))
    uall_pred = test_model(sLeft,sRight,u0L,u0R,u, deep_diffusion)
    keras.backend.clear_session()
    print(np.max(abs(uall_bench-uall_pred)))
    if config.plot_fig:
        plot_solution_2D(X,Y,(uall_bench-uall_pred))
        plot_solution_1D(config.x,(uall_bench[:,config.slice,:]-uall_pred[:,config.slice,:]))
        plt.show()
