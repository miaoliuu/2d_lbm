"""
Couette Flow with pressure variants

    - Periodic Boundary with pressure variants

    - reset and moving wall (Half-way BB)

    - incompressible equilibrium

    - SRT collision

    - periodic streaming

     --------------------
    i|  B   B   B   B  |o
    i|                 |o
    i|                 |o
    i|  B   B   B   B  |o
     --------------------
      y
      |
      o -- x

Analystical Solution

    Ux = U*(y/ly) + (1/(2*mu)) * (dP/dx) * (y**2 - ly*y)

In Fluent case

    lx = 1.5 [m]

    ly = 1.0 [m]

    rho = 1 [kg/m3]

    mu = 1 [Pa*s]
    
    U = 3 [m/s]

    dP/dx = -12 [Pa/m]

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

# Equilibrium function

def feq_ma2(iPop_, rho_, ux_, uy_):
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_]*ux_ + cy[iPop_]*uy_
    res = w[iPop_] * rho_ * (1.0 + 3.0*ci_u + 4.5*ci_u**2 - 1.5*uSqr)
    return res

def feq_ma2_incomp(iPop_, rho_, ux_, uy_):
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_]*ux_ + cy[iPop_]*uy_
    res = w[iPop_] * (rho_ + 3.0*ci_u + 4.5*ci_u**2 - 1.5*uSqr)
    return res

# 0-order and 1-order Moment

def moment(lat_):
    m0 = lat_.sum(0)
    m1x = lat_[[1,2,8], :, :].sum(0) - lat_[[4,5,6], :, :].sum(0)
    m1y = lat_[[2,3,4], :, :].sum(0) - lat_[[6,7,8], :, :].sum(0)
    return m0, m1x, m1y
# =========================================================


# =========================================================
# Simulation parameters
# ---------------------------------------------------------
# Geometry
lx, ly = 1.5, 1
resolution = 9
# GPBC wet-node, half-way BB link-wise
nx, ny = int(lx*resolution) + 1, int(ly*resolution)
# simulation parameters
t_max = int(1000 * resolution)
rho0 = 1.0
tau = np.sqrt(3.0/16.0) + 0.5
omega = 1.0 / tau
niu = Cs**2 * (tau - 0.5)
# covert factors
cf_l = 1.5 / (nx-1) # in Fluent lx = 1.5 [m]
cf_rho = 1.0 / rho0 # in Fluent set rho = 1 [kg/m3]
cf_niu = 1.0 / niu # in Fluent set miu = 1 [Pa*s], so niu = miu/rho == 1 [m2/s]
cf_u = cf_niu / cf_l
cf_dp = cf_rho * cf_u**2
cf_dp_dx = cf_dp / cf_l
print('cf_l =', cf_l)
print('cf_rho =', cf_rho)
print('cf_niu =', cf_niu)
print('cf_u =', cf_u)
print('cf_dp =', cf_dp)
print('cf_dp_dx =', cf_dp_dx)
# exported parameters
uTop = 3.0 / cf_u
dp_dx = -12.0 / cf_dp_dx # in Fluent set dp/dx = -12 [Pa/m]
drho_dx = dp_dx / Cs**2
rho_in = rho0 - (nx-1) * drho_dx / 2
rho_out = rho0 + (nx-1) * drho_dx / 2
# analytial solution of couette flow
yPosition = np.arange(ny) + 0.5
ana_vel = uTop*(yPosition/ny) + (1/(2*rho0*niu)) * dp_dx * (yPosition**2 - ny*yPosition)
# show
print('tau=', tau, 'omega =', omega, 'niu =', niu)
print('rh0_in =', rho_in, 'rho_out =', rho_out, 'd_rho =', drho_dx)
print('uTop =', uTop)
# =========================================================


# =========================================================
# allocate memory and initial condition
# ---------------------------------------------------------
# choice equilibrium function
compressible = False
# initial
feq = np.zeros((Q, nx, ny))
if compressible:
    for i in range(Q):
        feq[i, :, :] = feq_ma2(i, 1, 0, 0)
else:
    for i in range(Q):
        feq[i, :, :] = feq_ma2_incomp(i, 1, 0, 0)
fstar = feq.copy()  # local collision
fprop = feq.copy()  # after streaming
# macro
rho, ux, uy = moment(fprop)
# macro at virtual node
rho[0, :] = rho[-2, :] + (rho_in - rho[-2, :].mean())
ux[0, :] = ux[-2, :]
uy[0, :] = uy[-2, :]
rho[-1, :] = rho[1, :] + (rho_out - rho[1, :].mean())
ux[-1, :] = ux[1, :]
uy[-1, :] = uy[1, :]
if compressible:
    ux /= rho
    uy /= rho
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
        
    # equilibrium
    if compressible:
        for i in range(Q):
            feq[i,:,:] = feq_ma2(i, rho, ux, uy)
    else:
        for i in range(Q):
            feq[i,:,:] = feq_ma2_incomp(i, rho, ux, uy)
        
    # collision
    fstar = (1.0 - omega) * fprop + omega * feq
    
    # GPBC collision
    if compressible:
        for i in range(Q):
            # inlet
            fstar[i,0,:] = feq_ma2(i, rho[0,:], ux[0,:], uy[0,:]) + fstar[i,-2,:] - feq[i,-2,:]
            # outlet
            fstar[i,-1,:] = feq_ma2(i, rho[-1,:], ux[-1,:], uy[-1,:]) + fstar[i,1,:] - feq[i,1,:]
    else:
        for i in range(Q):
            # inlet
            fstar[i,0,:] = feq_ma2_incomp(i, rho[0,:], ux[0,:], uy[0,:]) + fstar[i,-2,:] - feq[i,-2,:]
            # outlet
            fstar[i,-1,:] = feq_ma2_incomp(i, rho[-1,:], ux[-1,:], uy[-1,:]) + fstar[i,1,:] - feq[i,1,:]

    # periodic streaming
    for i in range(Q):
        fprop[i,:,:] = np.roll(fstar[i,:,:], (cx[i], cy[i]), axis=(0,1))


    # boundary half-way BB
    # bottom rest wall
    for i in [2,3,4]:
        fprop[i,:,0] = fstar[oppo[i],:,0]
    # top moving wall
    for i in [6,7,8]:
        fprop[i,:,-1] = fstar[oppo[i],:,-1] - (2.0 * w[oppo[i]] * rho0 / Cs**2) * (cx[oppo[i]]*uTop + cy[oppo[i]]*0.0)
    
    # macro
    rho, ux, uy = moment(fprop)
    # macro at virtual node
    rho[0, :] = rho[-2, :] + (rho_in - rho[-2, :].mean())
    ux[0, :] = ux[-2, :]
    uy[0, :] = uy[-2, :]
    rho[-1, :] = rho[1, :] + (rho_out - rho[1, :].mean())
    ux[-1, :] = ux[1, :]
    uy[-1, :] = uy[1, :]
    if compressible:
        ux /= rho
        uy /= rho
# =========================================================


# =========================================================
# Postprocessing
# ---------------------------------------------------------
# L2 error
if compressible:
    error = np.zeros(nx)
    for x in range(nx):
        error[x] = np.sqrt(np.sum(((ux*rho/rho0)[x,:] - ana_vel)**2)) / np.sqrt(np.sum(ana_vel**2))
    L2_error = error.sum() / nx
    print('L2 = ', L2_error)
else:
    error = np.zeros(nx)
    for x in range(nx):
        error[x] = np.sqrt(np.sum((ux[x,:] - ana_vel)**2)) / np.sqrt(np.sum(ana_vel**2))
    L2_error = error.sum() / nx
    print('L2 = ', L2_error)

# visualization
if compressible:
    plt.plot(yPosition, ana_vel, 'k-', label='Ana')
    plt.plot(yPosition, (ux*rho/rho0)[1], 'o', markerfacecolor='none', markersize=10, label='LBM')
    plt.grid()
    plt.legend()
    plt.show()
else:
    plt.plot(yPosition, ana_vel, 'k-', label='Ana')
    plt.plot(yPosition, ux[1], 'o', markerfacecolor='none', markersize=10, label='LBM')
    plt.grid()
    plt.legend()
    plt.show()
# visualization exchange yPosition to y-axis
if compressible:
    plt.plot(ana_vel, yPosition, 'k-', label='Ana')
    plt.plot((ux*rho/rho0)[1], yPosition, 'o', markerfacecolor='none', markersize=10, label='LBM')
    plt.grid()
    plt.legend()
    plt.show()
else:
    plt.plot(ana_vel, yPosition, 'k-', label='Ana')
    plt.plot(ux[1], yPosition, 'o', markerfacecolor='none', markersize=10, label='LBM')
    plt.grid()
    plt.legend()
    plt.show()
# =========================================================
