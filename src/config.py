import numpy as np
import ini

params = ini.parse(open('input.ini').read())
####### Parameters #######
# box size, mm
w = float(params['grid']['w'])
h = float(params['grid']['h'])
# intervals in x-, y- directions, mm
dx = float(params['grid']['dx'])
dy = float(params['grid']['dy'])
# Thermal diffusivity of steel, mm2.s-1
D = float(params['par']['D'])

# Number of timesteps
nsteps = int(params['time']['nsteps'])
dnn_start = int(params['time']['dnn_start'])

nn = int(params['dnn']['nn'])
epochs = int(params['dnn']['epochs'])
patience = int(params['dnn']['patience'])
batch_size=int(params['dnn']['batch_size'])
nlayer = int(params['dnn']['nlayer'])

plot_fig=bool(params['figures']['plot_fig'])
use_latex=bool(params['figures']['use_latex'])
add_labels=bool(params['figures']['add_labels'])

nx, ny = int(w/dx), int(h/dy)
nbx = int(nx/2)

slice = int(ny/2)

dx2, dy2 = dx*dx, dy*dy

x = np.linspace(0,w,nx)
y = np.linspace(0,h,ny)

dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
