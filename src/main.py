#!/usr/env/ python
import numpy as np
from os.path import join as pjoin
import os.path
import argparse
import matplotlib.pyplot as plt
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
u = np.zeros((config.nx, config.ny))
u0L, u0R, uL, uR, sLeft, sRight = \
[np.zeros((config.nbx, config.ny)) for _ in range(6)]
X,Y = np.meshgrid(config.x,config.y)

##### Source ########
s = np.exp(-((X-config.lx/2)**2+(Y-config.ly/2)**2))
sLeft = s[:config.nbx,:]
sRight = s[config.nbx:,:]



################# Define and compile model ###############################
deep_diffusion = dnn_model(config.nn)
#################################################################

if train_mode:
    print('Running train mode')
    inputs_array,outputs_array = train_data(sLeft,sRight,u0L,u0R,u)
    deep_diffusion,history = train_dnn(deep_diffusion,inputs_array,outputs_array,savedir)
    if config.plot_fig:
        model_history(history)
    keras.backend.clear_session()


if test_mode:
    print('Running test mode')
    print('Model data exists. loading model...')
    deep_diffusion = keras.models.load_model(pjoin(savedir))
    uall = test_model(sLeft,sRight,u0L,u0R,u, deep_diffusion)
    keras.backend.clear_session()
    if config.plot_fig:
        plot_solution_2D(X,Y,uall)
        plot_solution_1D(config.x,uall[:,config.slice,:])
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
