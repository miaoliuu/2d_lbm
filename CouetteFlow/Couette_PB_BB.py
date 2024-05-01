"""
Timm's Book Chapter 5.3.3.6

Couette Flow

    - Periodic Boundary

    - reset and moving wall (Half-way BB)

    - incompressible equilibrium

    - SRT collision

    - periodic streaming

    -------------------------
    | P   B   B   B   B   P |
    | P                   P |
    | P                   P |
    | P   B   B   B   B   P |
    -------------------------
      y
      |
      o -- x
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# D2Q9 Velocity Set
#   y        4  3  2
#   |         \ | /
#   o-- x    5- 0 -1
#             / | \
#            6  7  8
# ---------------------------------------------------------
D, Q = 2, 9
cx = np.array([  0,  1,  1,  0, -1, -1, -1,  0,  1])
cy = np.array([  0,  0,  1,  1,  1,  0, -1, -1, -1])
w = np.array([
    4.0/9.0,
    1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0,
    1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0])
Cs = 1.0 / np.sqrt(3.0)
oppo = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])
# =========================================================


# =========================================================
# Simulation parameters
# ---------------------------------------------------------
# Geometry
nx, ny = 3, 9
# simulation parameters
t_max = 2000
uMax = 0.1
tau = np.sqrt(3.0/16.0) + 0.5
omega = 1.0 / tau
niu = Cs**2 * (tau - 0.5)
rho0 = 1.0
# analytial solution of couette flow
yPosition = np.arange(ny) + 0.5
ana_vel = uMax * yPosition / ny
# show
print('tau=', tau, 'omega =', omega, 'niu =', niu, 'Re =', uMax*ny/niu)
# =========================================================


# =========================================================
# allocate memory and initial condition
# ---------------------------------------------------------
feq = np.zeros((Q, nx, ny))
# // ---- incompressible feq ----
for i in range(Q):
    feq[i, :, :] = w[i]

fstar = feq.copy()      # local collision
fprop = feq.copy()  # after streaming

rho = fprop.sum(0)
ux = fprop[[1,2,8], :, :].sum(0) - fprop[[4,5,6], :, :].sum(0)
uy = fprop[[2,3,4], :, :].sum(0) - fprop[[6,7,8], :, :].sum(0)

# convergence parameters
tol = 1.0e-12       # tolerance to steady state convergence
teval = 100         # time step to evaluate convergence
ux_old = np.ones((nx,ny))
# =========================================================


# =========================================================
# time loop
# ---------------------------------------------------------
for iT in range(1, t_max):
    # convergance check
    if iT%teval == 1:
        conv = np.abs(ux.mean() / (ux_old.mean() + 1.0e-32) - 1.0)
        print('iT =', iT, 'conv =', conv)
        if conv < tol:
            break
        else:
            ux_old = ux
    
    # incompressible linear equilibrium
    for i in range(Q):
        feq[i,:,:] = w[i] * (rho + 3.0 * (cx[i] * ux + cy[i] * uy))
    
    # collision
    fstar = (1.0 - omega) * fprop + omega * feq

    # periodic streaming
    for i in range(Q):
        fprop[i,:,:] = np.roll(fstar[i,:,:], (cx[i], cy[i]), axis=(0,1))
    
    # boundary half-way BB
    # bottom rest wall
    for i in [2,3,4]:
        fprop[i,:,0] = fstar[oppo[i],:,0]
    # top moving wall
    for i in [6,7,8]:
        fprop[i,:,-1] = fstar[oppo[i],:,-1] - (2.0 * w[oppo[i]] * rho0 / Cs**2) * (cx[oppo[i]]*uMax + cy[oppo[i]]*0.0)

    # macro
    rho = fprop.sum(0)
    ux = fprop[[1,2,8], :, :].sum(0) - fprop[[4,5,6], :, :].sum(0)
    uy = fprop[[2,3,4], :, :].sum(0) - fprop[[6,7,8], :, :].sum(0)
# =========================================================


# =========================================================
# Postprocessing
# ---------------------------------------------------------
# L2 error
error = np.zeros(nx)
for x in range(nx):
    error[x] = np.sqrt(np.sum((ux[x,:] - ana_vel)**2)) / np.sqrt(np.sum(ana_vel**2))
L2_error = error.sum() / nx
print('L2 = ', L2_error)

# visualization
plt.plot(yPosition, ana_vel, 'k-', label='Ana')
plt.plot(yPosition, ux[1], 'o', markerfacecolor='none', markersize=10, label='LBM')
plt.grid()
plt.legend()
plt.show()
# =========================================================
