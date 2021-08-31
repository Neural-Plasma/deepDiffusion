import numpy as np
from os.path import join as pjoin
import config
def dataxy(f,m,u):

    f["/%d"%int(m)] = u

    # f["/%d"%int(t)+"/energy"] = KE

    # dsetE.resize(dsetE.shape[0]+1, axis=0)
    # dsetE[-1:] = KE
    #
    # dsetQ.resize(dsetQ.shape[0]+1, axis=0)
    # dsetQ[-1:] = Qcollect
    #
    # with open(pjoin(path,'energy.txt'),"ab") as f:
    #     np.savetxt(f, np.column_stack([t, KE]))

    return 0

def attributes(f,lx,ly,nx,ny,nsteps):
    f.attrs["lx"] = lx
    f.attrs["ly"] = ly
    # f.attrs["dx"] = config.dx
    # f.attrs["dy"] = config.dy
    f.attrs["nx"] = nx
    f.attrs["ny"] = ny
    f.attrs["nsteps"] = nsteps
    return 0

def dustDiagn(f,fduration):

    f["/fall_duration"] = fduration
    # f["/%d"%int(t)+"/energy"] = KE
    return 0
