import numpy as np
####### Parameters #######
# plate size, mm
w = 10.
h = 10.
# intervals in x-, y- directions, mm
dx = 0.1
dy = 0.1
# Thermal diffusivity of steel, mm2.s-1
D = 1.

nx, ny = int(w/dx), int(h/dy)

nbx = int(nx/2)

slice = int(ny/2)

dx2, dy2 = dx*dx, dy*dy

x = np.linspace(0,w,nx)
y = np.linspace(0,h,ny)

dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

# Number of timesteps
nsteps = 101

dnn_start = 50
