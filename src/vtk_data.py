import numpy as np
from pyevtk.hl import gridToVTK
from os.path import join as pjoin
import h5py
import os

def vtkwrite(path):
    file_name = "data"#"rhoNeutral" #"P"
    if os.path.exists(pjoin(path,'vtkdata')) == False:
        os.mkdir(pjoin(path,'vtkdata'))
    h5 = h5py.File(pjoin(path,file_name+'.hdf5'),'r')

    lx, ly = h5.attrs["lx"], h5.attrs["ly"]
    nx, ny = h5.attrs["nx"], h5.attrs["ny"]
    nsteps = h5.attrs["nsteps"]

    x = np.linspace(0, lx, nx, dtype='float64')
    y = np.linspace(0, ly, ny, dtype='float64')




    # data_num = np.arange(start=0, stop=Nt, step=dp, dtype=int)

    for i in range(nsteps):
        data = h5["/%d"%i]
        data = np.array(data)
        datavtk = data.reshape(nx,ny)
        # pointsToVTK(pjoin(path,'vtkdata','points_%d'%i), datax, datay, dataz)
        gridToVTK(pjoin(path,'vtkdata','points_%d'%i), x, y, pointData = {file_name : datavtk})
