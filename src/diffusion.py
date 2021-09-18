import numpy as np
import config
import h5py
import diagn

def diffusion_left(u0, u0Rs, sLeft):
    u = np.zeros(u0.shape)
    for i in range(1, config.nbx-1):
        for j in range(1, config.ny-1):
            uxx = (u0[i+1,j] - 2*u0[i,j] + u0[i-1,j]) / config.dx2
            uyy = (u0[i,j+1] - 2*u0[i,j] + u0[i,j-1]) / config.dy2
            u[i,j] = u0[i,j] + config.dt * config.D * (uxx + uyy) + config.dt * sLeft[i,j]

    i = config.nbx-1
    for j in range(1, config.ny-1):
        uxx = (u0Rs[j] - 2*u0[i,j] + u0[i-1,j]) / config.dx2
        uyy = (u0[i,j+1] - 2*u0[i,j] + u0[i,j-1]) / config.dy2
        u[i,j] = u0[i,j] + config.dt * config.D * (uxx + uyy) + config.dt * sLeft[i,j]

    return u

def diffusion_right(u0, u0Ls, sRight):
    u = np.zeros(u0.shape)
    i = 0
    for j in range(1, config.ny-1):
        uxx = (u0[i+1,j] - 2*u0[i,j] + u0Ls[j]) / config.dx2
        uyy = (u0[i,j+1] - 2*u0[i,j] + u0[i,j-1]) / config.dy2
        u[i,j] = u0[i,j] + config.dt * config.D * (uxx + uyy) + config.dt * sRight[i,j]

    for i in range(1, config.nbx-2):
        for j in range(1, config.ny-1):
            uxx = (u0[i+1,j] - 2*u0[i,j] + u0[i-1,j]) / config.dx2
            uyy = (u0[i,j+1] - 2*u0[i,j] + u0[i,j-1]) / config.dy2
            u[i,j] = u0[i,j] + config.dt * config.D * (uxx + uyy) + config.dt * sRight[i,j]

    return u

def train_data(sLeft,sRight,u0L,u0R):
    inputs = []
    outputs = []
    u1 = []
    u2 = []
    for m in range(config.nsteps):
        u0R_interp = []
        for l in range(0, len(u0R[0,:]), 2):
            u0R_interp.append(np.mean([u0R[0,l],u0R[0,l+1]]))
        u0R_interp = np.array(u0R_interp)
        u0L_interp = np.zeros(u0R[0,:].shape)
        for l in range(0, len(u0R[0,:]), 2):
            # print('l1',l)
            u0L_interp[l] = u0L[0,int(l/2)]
        for l in range(1, len(u0R[0,:])-1, 2):
            # print(l)
            u0L_interp[l] = (np.mean([u0L_interp[l],u0L_interp[l+1]]))
        print(u0R_interp)
        print(u0L_interp)
        uL = diffusion_left(u0L, u0R_interp, sLeft)
        uR = diffusion_right(u0R, u0L_interp, sRight)
        # for j in range(1, config.ny-1):
        #     inputs.append([u0L[config.nbx-1,j],sRight[0,j]])
        # for i in range(1, config.nbx-1):
        #     for j in range(1, config.ny-1):
        #         inputs.append([u0R[i,j],sRight[i,j]])
        # for i in range(0, config.nbx-1):
        #     for j in range(1, config.ny-1):
        #         outputs.append([uR[i,j]])
        u0L = uL.copy()
        u0R = uR.copy()
        u1.append(uL)
        u2.append(uR)
    u1 = np.asarray(u1)
    u2 = np.asarray(u2)
        # u[1:config.nbx,:]  = uL[1:,:]
        # u[config.nbx:-1,:] = uR[:-1,:]
    print("Preparing data")
    inputs_array = np.asarray(inputs)
    outputs_array = np.asarray(outputs)
    return inputs_array,outputs_array,u1,u2

def test_model(sLeft,sRight,u0L,u0R,u, deep_diffusion):
    uall = []
    for m in range(config.nsteps):
        uL = diffusion_left(u0L, u0R[0,:], sLeft)
        if m>=config.dnn_start:
            inputs = []
            for j in range(1, config.ny-1):
                inputs.append([u0L[config.nbx-1,j],sRight[0,j]])
            for i in range(1, config.nbx-1):
                for j in range(1, config.ny-1):
                    inputs.append([u0R[i,j],sRight[i,j]])
            inputs_array = np.asarray(inputs)
            #deep learning model
            uRPredict = deep_diffusion.predict(inputs_array)
            uRPredict = uRPredict.reshape((config.nbx-1,config.ny-2))
            uR = np.zeros((config.nbx, config.ny))
            uR[:config.nbx-1,1:config.ny-1] = uRPredict[:,:]

            # keras.backend.clear_session()
        else:
            uR = diffusion_right(u0R, u0L[config.nbx-1,:], sRight)

        u0L = uL.copy()
        u0R = uR.copy()
        u[1:config.nbx,:]  = uL[1:,:]
        u[config.nbx:-1,:] = uR[:-1,:]
        if config.dumpData:
            diagn.dataxy(config.f,m,u.copy().T)
        if (m%25==0):
            uall.append(u.copy().T)
    return np.array(uall)

def test_bench(sLeft,sRight,u0L,u0R,u, deep_diffusion):
    uall = []
    for m in range(config.nsteps):
        uL = diffusion_left(u0L, u0R[0,:], sLeft)
        uR = diffusion_right(u0R, u0L[config.nbx-1,:], sRight)

        u0L = uL.copy()
        u0R = uR.copy()
        u[1:config.nbx,:]  = uL[1:,:]
        u[config.nbx:-1,:] = uR[:-1,:]
        if (m%25==0):
            uall.append(u.copy().T)
    return np.array(uall)
