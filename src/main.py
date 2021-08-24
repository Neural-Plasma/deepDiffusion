import numpy as np
from os.path import join as pjoin
import os.path
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras


from diffusion import diffusion_left, diffusion_right, train_data, test_model
import config
from dnn import dnn_model,train_dnn
from plot_data import model_history, plot_solution

parser = argparse.ArgumentParser(description='DNN Based Diffusion Eqn Solver')
parser.add_argument('-te','--test', action='store_true', help='Add this if you want to test the model')
parser.add_argument('-tr','--train', action='store_true', help='Add this if you want to test the model')

args        = parser.parse_args()
test_mode = args.test
train_mode = args.train

savedir = pjoin("data",'dnn_model')

##### Initialization ########
u = np.zeros((config.nx, config.ny))
u0L, u0R, uL, uR, sLeft, sRight = \
[np.zeros((config.nbx, config.ny)) for _ in range(6)]
X,Y = np.meshgrid(config.x,config.y)

##### Source ########
s = np.exp(-((X-config.w/2)**2+(Y-config.h/2)**2))
sLeft = s[:config.nbx,:]
sRight = s[config.nbx:,:]



################# Define and compile model ###############################
deep_diffusion = dnn_model(50)
#################################################################

if train_mode:
    print('Running train mode')
    inputs_array,outputs_array = train_data(sLeft,sRight,u0L,u0R,u)
    deep_diffusion,history = train_dnn(deep_diffusion,inputs_array,outputs_array,savedir)
    model_history(history)
    keras.backend.clear_session()


if test_mode:
    print('Running test mode')
    print('Model data exists. loading model...')
    deep_diffusion = keras.models.load_model(pjoin(savedir))
    uall = test_model(sLeft,sRight,u0L,u0R,u, deep_diffusion)
    keras.backend.clear_session()

    plot_solution(X,Y,uall)
