"""
2007. Kim. GPBC

Corrugated channel flow

Poiseuille Flow

    - Periodic Boundary with pressure variants (GPBC by Kim)

    - reset wall and obstical (full-way BB, has assistant nodes)

    - compressible and incompressible equilibrium

    - SRT collision

    - periodic streaming

    - virtual node (i, o) at the physical boundary line (like wet-node)

        o -- y
        |       B   B   B   B   B   B   B   B
        x       -----------------------------
                i   F   F   F   F   F   F   o
                |                           |
                i                           o
                |                           |
                i   F   F   F   F   F   F   o
                -----------------------------
                B   B   B   B   B   B   B   B

"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# D2Q9 Velocity Set
#   o-- y    6  5  4
#   |         \ | /
#   x        7- 0 -3
#             / | \
#            8  1  2
# ---------------------------------------------------------
D, Q = 2, 9
cx = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
cy = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
w = np.array(
    [
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 9.0,
        1.0 / 36.0,
    ]
)
Cs = 1.0 / np.sqrt(3.0)
oppo = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])

# Equilibrium function


def feq_ma2(iPop_, rho_, ux_, uy_):
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_] * ux_ + cy[iPop_] * uy_
    res = w[iPop_] * rho_ * (1.0 + 3.0 * ci_u + 4.5 * ci_u**2 - 1.5 * uSqr)
    return res


def feq_ma2_incomp(iPop_, rho_, ux_, uy_):
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_] * ux_ + cy[iPop_] * uy_
    res = w[iPop_] * (rho_ + 3.0 * ci_u + 4.5 * ci_u**2 - 1.5 * uSqr)
    return res


# 0-order and 1-order Moment


def moment(lat_):
    m0 = lat_.sum(0)
    m1x = lat_[[1, 2, 8], :, :].sum(0) - lat_[[4, 5, 6], :, :].sum(0)
    m1y = lat_[[2, 3, 4], :, :].sum(0) - lat_[[6, 7, 8], :, :].sum(0)
    return m0, m1x, m1y


# =========================================================

# %%
# =========================================================
# Geometry
# ---------------------------------------------------------
# length and resolution
lx, ly = 1, 3
resolution = 60
# wet-node
# nx, ny = lx * resolution + 1, ly * resolution + 1
nx, ny = lx * resolution + 2, ly * resolution + 1
# xPos = np.linspace(0, lx, nx)
xPos = np.linspace(-lx / (nx - 2) / 2, lx + lx / (nx - 2) / 2, nx)
yPos = np.linspace(0, ly, ny)
meshPos = np.array(np.meshgrid(xPos, yPos, indexing="ij"))
mask = np.zeros((nx, ny)).astype(bool)
# wall mask
mask[0, :] = True
mask[-1, :] = True
# obstical mask
for i in range(nx):
    for j in range(ny):
        if (
            meshPos[0, i, j] > 0
            and meshPos[0, i, j] < 0.5
            and meshPos[1, i, j] >= 0.25
            and meshPos[1, i, j] < 0.5
        ):
            mask[i, j::resolution] = True
        if (
            meshPos[0, i, j] > 0.5
            and meshPos[0, i, j] < 1
            and meshPos[1, i, j] >= 2 / 3
            and meshPos[1, i, j] < 11 / 12
        ):
            mask[i, j::resolution] = True


# show geometry
def show_geometry(mask_):
    plt.figure("geometry")
    plt.pcolormesh(mask, cmap=plt.cm.gray)
    plt.axis("image")
    # plt.imshow(mask, cmap=plt.cm.gray, origin="upper")
    # plt.gca().xaxis.set_ticks_position("top")
    plt.show()


# show_geometry(mask)
# =========================================================

# %%
# =========================================================
# simulation parameters
# ---------------------------------------------------------
# control parameters
t_max = 10000 * ly
Re = 20.0
rho0 = 1.0  # reference [global mean] density
tau = 0.6
omega = 1.0 / tau
niu = Cs**2 * (tau - 0.5)
# covert factors
cf_l = 0.18 / (ny - 1)  # in Fluent ly = 0.18 [m]
cf_rho = 1.0 / rho0  # in Fluent set rho = 1 [kg/m3]
cf_niu = 0.001 / niu  # in Fluent set niu = 0.001 [m2/s]
cf_dp = cf_rho * (cf_niu / cf_l) ** 2
cf_dp_dx = cf_dp / cf_l
print("cf_rho =", cf_rho)
print("cf_l =", cf_l)
print("cf_niu =", cf_niu)
print("cf_dp =", cf_dp)
print("cf_dp_dx =", cf_dp_dx)
# exported parameters
dp_dx = -79.0 / cf_dp_dx  # in Fluent set dp/dx = -79 [Pa/m]
drho_dx = dp_dx / Cs**2
rho_in = rho0 - (ny - 1) * drho_dx / 2
rho_out = rho0 + (ny - 1) * drho_dx / 2
# show
print("tau=", tau, "omega =", omega, "niu =", niu)
print("rh0_in =", rho_in, "rho_out =", rho_out, "d_rho =", drho_dx)
# print("ux_mean=", Re * niu / (nx - 1))s
print("uy_mean=", Re * niu / (nx - 2))
# =========================================================

# %%
# =========================================================
# allocate memory and initial condition
# ---------------------------------------------------------
# choice equilibrium function
compressible = True


# implement of equillibrium
def impl_feq(iPop_, rho_, ux_, uy_):
    """feq in different implemention"""
    if compressible:
        return feq_ma2(iPop_, rho_, ux_, uy_)
    else:
        return feq_ma2_incomp(iPop_, rho_, ux_, uy_)


# implement of update macro
def impl_macro(lat_):
    """Update macro in different implemention"""
    rho, ux, uy = moment(fprop)
    if compressible:
        ux /= rho
        uy /= rho
    return rho, ux, uy


# initial
feq = np.zeros((Q, nx, ny))
for i in range(Q):
    feq[i, :, :] = impl_feq(i, 1, 0, 0)
# if compressible:
#     for i in range(Q):
#         feq[i, :, :] = feq_ma2(i, 1, 0, 0)
# else:
#     for i in range(Q):
#         feq[i, :, :] = feq_ma2_incomp(i, 1, 0, 0)
fstar = feq.copy()  # local collision
fprop = feq.copy()  # after streaming
# macro
rho, ux, uy = impl_macro(fprop)
# rho, ux, uy = moment(fprop)
# if compressible:
#     ux /= rho
#     uy /= rho
# macro at virtual node
rho[:, 0] = rho[:, -2] + (rho_in - rho[:, -2].mean())
ux[:, 0] = ux[:, -2]
uy[:, 0] = uy[:, -2]
rho[:, -1] = rho[:, 1] + (rho_out - rho[:, 1].mean())
ux[:, -1] = ux[:, 1]
uy[:, -1] = uy[:, 1]
# macro at wall and obstical
ux[mask] = 0.0
uy[mask] = 0.0
# convergence parameters
tol = 1.0e-12  # tolerance to steady state convergence
teval = 100  # time step to evaluate convergence
uy_old = np.ones((nx, ny))
# =========================================================

# %%
# =========================================================
# time loop
# ---------------------------------------------------------
for iT in range(1, t_max):
    # convergance check
    if iT % teval == 1:
        conv = np.abs(uy.mean() / (uy_old.mean() + 1.0e-32) - 1.0)
        print("iT =", iT, "conv =", conv)
        if conv < tol:
            break
        else:
            uy_old = uy

    # equilibrium
    for i in range(Q):
        feq[i, :, :] = impl_feq(i, rho, ux, uy)
    # if compressible:
    #     for i in range(Q):
    #         feq[i, :, :] = feq_ma2(i, rho, ux, uy)
    # else:
    #     for i in range(Q):
    #         feq[i, :, :] = feq_ma2_incomp(i, rho, ux, uy)

    # collision
    fstar = (1.0 - omega) * fprop + omega * feq

    # GPBC collision
    for i in range(Q):
        # inlet
        fstar[i, :, 0] = (
            impl_feq(i, rho[:, 0], ux[:, 0], uy[:, 0]) + fstar[i, :, -2] - feq[i, :, -2]
        )
        # outlet
        fstar[i, :, -1] = (
            impl_feq(i, rho[:, -1], ux[:, -1], uy[:, -1])
            + fstar[i, :, 1]
            - feq[i, :, 1]
        )
    # if compressible:
    #     for i in range(Q):
    #         # inlet
    #         fstar[i, :, 0] = (
    #             feq_ma2(i, rho[:, 0], ux[:, 0], uy[:, 0])
    #             + fstar[i, :, -2]
    #             - feq[i, :, -2]
    #         )
    #         # outlet
    #         fstar[i, :, -1] = (
    #             feq_ma2(i, rho[:, -1], ux[:, -1], uy[:, -1])
    #             + fstar[i, :, 1]
    #             - feq[i, :, 1]
    #         )
    # else:
    #     for i in range(Q):
    #         # inlet
    #         fstar[i, :, 0] = (
    #             feq_ma2_incomp(i, rho[:, 0], ux[:, 0], uy[:, 0])
    #             + fstar[i, :, -2]
    #             - feq[i, :, -2]
    #         )
    #         # outlet
    #         fstar[i, :, -1] = (
    #             feq_ma2_incomp(i, rho[:, -1], ux[:, -1], uy[:, -1])
    #             + fstar[i, :, 1]
    #             - feq[i, :, 1]
    #         )

    # simple BB collision on rest wall
    for i in range(Q):
        fstar[i, mask] = fprop[oppo[i], mask]

    # periodic streaming
    for i in range(Q):
        fprop[i, :, :] = np.roll(fstar[i, :, :], (cx[i], cy[i]), axis=(0, 1))

    # macro
    rho, ux, uy = impl_macro(fprop)
    # if compressible:
    #     ux /= rho
    #     uy /= rho
    # rho, ux, uy = moment(fprop)
    # macro at virtual node
    rho[:, 0] = rho[:, -2] + (rho_in - rho[:, -2].mean())
    ux[:, 0] = ux[:, -2]
    uy[:, 0] = uy[:, -2]
    rho[:, -1] = rho[:, 1] + (rho_out - rho[:, 1].mean())
    ux[:, -1] = ux[:, 1]
    uy[:, -1] = uy[:, 1]
    # macro at wall and obstical
    ux[mask] = 0.0
    uy[mask] = 0.0

# =========================================================

# %%
# =========================================================
# Post-process
# ---------------------------------------------------------
if compressible:
    vel_x = ux * rho / rho0
    vel_y = uy * rho / rho0
else:
    vel_x = ux
    vel_y = uy

output_file = "output_data.npz"
np.savez(
    output_file,
    compressible=compressible,
    niu=niu,
    xPos=xPos,
    yPos=yPos,
    mask=mask,
    rho=rho,
    vel_x=vel_x,
    vel_y=vel_y,
)
# =========================================================
