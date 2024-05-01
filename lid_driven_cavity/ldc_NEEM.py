"""
Lad Driven Cavity Flow

    - compressible and incompressible equilibrium

    - SRT collision

    - periodic streaming

    - Left: Dirichlet boundary of velocity (reset wall, NEEM)

    - Right: Dirichlet boundary of velocity (moving wall, NEEM)

    - Upper: Dirichlet boundary of velocity (reset wall, NEEM)

    - Lower: Dirichlet boundary of density (rest wall, NEEM)

    o -- y
    |
    x       B - B - B - B - B
            |   F   F   F   |
            B               B   |
            |               |   V vel
            B               B
            |   F   F   F   |
            B - B - B - B - B

"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

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


def feq_ma2(iPop_, rho_, ux_, uy_):
    """Equilibrium function of compressible flow"""
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_] * ux_ + cy[iPop_] * uy_
    res = w[iPop_] * rho_ * (1.0 + 3.0 * ci_u + 4.5 * ci_u**2 - 1.5 * uSqr)
    return res


# def feq_ma2(iPop_, rho_, ux_, uy_):
#     """Equilibrium function of compressible flow"""
#     res = np.zeros_like(rho_)
#     lbm.feq_ma2(res, rho_, ux_, uy_, cx[iPop_], cy[iPop_], w[iPop_], Cs*Cs, 1)
#     return res


def feq_ma2_incomp(iPop_, rho_, ux_, uy_):
    """Equilibrium function of incompressible flow"""
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_] * ux_ + cy[iPop_] * uy_
    res = w[iPop_] * (rho_ + 3.0 * ci_u + 4.5 * ci_u**2 - 1.5 * uSqr)
    return res


def moment(lat_):
    """0-order and 1-order Moment"""
    m0 = lat_.sum(0)
    m1x = lat_[[1, 2, 8], :, :].sum(0) - lat_[[4, 5, 6], :, :].sum(0)
    m1y = lat_[[2, 3, 4], :, :].sum(0) - lat_[[6, 7, 8], :, :].sum(0)
    return m0, m1x, m1y


def update_macro(lat_):
    """Update macro of compressible flow"""
    rho, ux, uy = moment(lat_)
    ux /= rho
    uy /= rho
    return rho, ux, uy


def update_macro_incomp(lat_):
    """Update macro of incompressible flow"""
    return moment(lat_)


# choice equilibrium function
compressible = True

# implement of equillibrium and update_macro
if compressible:
    impl_feq = feq_ma2
    impl_macro = update_macro
else:
    impl_feq = feq_ma2_incomp
    impl_macro = update_macro_incomp
# =========================================================


# %%
# =========================================================
# Post-process
# ---------------------------------------------------------
def show_vel_field(rho_, ux_, uy_):
    plt.figure("Density")
    plt.title(f"rho_max = {rho_.max():f}\nrho_min = {rho_.min():f}")
    plt.contourf(rho_.T, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    # plt.colorbar(orientation="horizontal")
    plt.colorbar()

    plt.figure("velocity X")
    plt.title(f"vel_x_max = {ux_.max():f}\nvel_x_min = {ux_.min():f}")
    plt.contourf(ux_.T, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    # plt.colorbar(orientation="horizontal")
    plt.colorbar()

    plt.figure("velocity Y")
    plt.title(f"vel_y_max = {uy_.max():f}\nvel_y_min = {uy_.min():f}")
    plt.contourf(uy_.T, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    # plt.colorbar(orientation="horizontal")
    plt.colorbar()

    plt.figure("velocity magnitude")
    vel_mag = np.sqrt(ux_**2 + uy_**2)
    plt.title(f"vel_mag_max = {vel_mag.max():f}\nvel_mag_min = {vel_mag.min():f}")
    plt.contourf(vel_mag.T, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    # plt.colorbar(orientation="horizontal")
    plt.colorbar()

    plt.show()


def write_case(
    ofile_name_,
    lx_,
    ly_,
    resolution_,
    nx_,
    ny_,
    xPos_,
    yPos_,
    reynolds_,
    tau_,
    omega_,
    niu_,
):
    """Data of geometry, mesh and control parameters"""
    np.savez(
        ofile_name_,
        lx=lx_,
        ly=ly_,
        resolution=resolution_,
        nx=nx_,
        ny=ny_,
        xPos=xPos_,
        yPos=yPos_,
        Re=reynolds_,
        tau=tau_,
        omega=omega_,
        niu=niu_,
    )


def write_data(ofile_name_, compressible_, iT_, rho0_, u0_, rho_, ux_, uy_):
    """Macro data"""
    if compressible_:
        vel_x = ux_ * rho_ / rho0_
        vel_y = uy_ * rho_ / rho0_
    else:
        vel_x = ux_
        vel_y = uy_

    np.savez(
        ofile_name_,
        compressible=compressible_,
        iT=iT_,
        rho0=rho0_,
        u0=u0_,
        rho=rho,
        vel_x=vel_x,
        vel_y=vel_y,
    )


# =========================================================

# %%
# =========================================================
# Geometry
# ---------------------------------------------------------
# length and resolution
lx, ly = 1, 1
resolution = 50
# wet-node
nx, ny = lx * resolution + 1, ly * resolution + 1
xPos = np.linspace(0, lx, nx)
yPos = np.linspace(0, ly, ny)
# =========================================================

# %%
# =========================================================
# simulation parameters
# ---------------------------------------------------------
# control parameters
t_max = 10000 * nx * ny
uMax = 0.1
rho0 = 1.0  # reference density
Re = 50.0
# boundaries macro
ux_right = uMax
uy_right = 0.0
rho_out = rho0
ux_wall = 0.0
uy_wall = 0.0
# exported parameters
niu = ux_right * (ny - 1) / Re
tau = niu / Cs**2 + 0.5
omega = 1.0 / tau
# show
print(f"tau = {tau}, omega = {omega}, niu = {niu}")
print(f"moving wall velocity = {ux_right}")
print(f"Mach Number = {ux_right/Cs}")
print(f"Reynolds Number = {Re}")
# write
case_out_file_name = f"tmp/ldc_case_Re{Re}.npz"
write_case(
    case_out_file_name, lx, ly, resolution, nx, ny, xPos, yPos, Re, tau, omega, niu
)
# =========================================================

# %%
# =========================================================
# allocate memory and initial condition
# ---------------------------------------------------------
# initial
feq = np.zeros((Q, nx, ny))
for i in range(Q):
    feq[i, :, :] = impl_feq(i, 1, 0, 0)
fstar = feq.copy()  # local collision
fprop = feq.copy()  # after streaming
# macro
rho, ux, uy = impl_macro(fprop)
# velocity at moving wall
ux[:, -1] = ux_right
uy[:, -1] = uy_right
# checkpiont parameters
tol = 1.0e-6
teval = 100  # time step for checkpoint
vel_mag_old = np.ones((nx, ny))
# =========================================================

t_elapsed = 0.0
t1_elapsed = 0.0
t2_elapsed = 0.0
t3_elapsed = 0.0
t4_elapsed = 0.0
t5_elapsed = 0.0
t6_elapsed = 0.0
t6_1_elapsed = 0.0
t6_2_elapsed = 0.0

t_start = perf_counter()

# %%
# =========================================================
# time loop
# ---------------------------------------------------------
for iT in range(1, t_max):
    t1_start = perf_counter()
    # check point
    if iT % teval == 1:
        vel_mag = np.sqrt(ux**2 + uy**2)
        conv = np.abs(vel_mag.mean() / (vel_mag_old.mean() + 1.0e-32) - 1.0)
        print("iT =", iT, "conv =", conv)
        if conv < tol:
            data_out_file_name = f"tmp/ldc_Re{Re}_{iT}.npz"
            write_data(data_out_file_name, compressible, iT, rho0, uMax, rho, ux, uy)
            break
        else:
            vel_mag_old = vel_mag
    t1_end = perf_counter()
    t1_elapsed = t1_elapsed + t1_end - t1_start

    t2_start = perf_counter()
    # equilibrium
    for i in range(Q):
        feq[i, :, :] = impl_feq(i, rho, ux, uy)
    t2_end = perf_counter()
    t2_elapsed = t2_elapsed + t2_end - t2_start

    t3_start = perf_counter()
    # collision
    fstar = (1.0 - omega) * fprop + omega * feq
    t3_end = perf_counter()
    t3_elapsed = t3_elapsed + t3_end - t3_start

    t4_start = perf_counter()
    # periodic streaming
    for i in range(Q):
        fprop[i, :, :] = np.roll(fstar[i, :, :], (cx[i], cy[i]), axis=(0, 1))
    t4_end = perf_counter()
    t4_elapsed = t4_elapsed + t4_end - t4_start

    t5_start = perf_counter()
    # macro
    rho, ux, uy = impl_macro(fprop)
    t5_end = perf_counter()
    t5_elapsed = t5_elapsed + t5_end - t5_start

    t6_start = perf_counter()
    # boundaries (NEEM)
    # upper  : reset wall (NEEM)
    t6_1_start = perf_counter()
    rho[0, 1:-1] = rho[1, 1:-1]
    ux[0, 1:-1] = ux_wall
    uy[0, 1:-1] = uy_wall
    t6_1_end = perf_counter()
    t6_1_elapsed = t6_1_elapsed + t6_1_end - t6_1_start

    t6_2_start = perf_counter()
    for i in range(Q):
        fprop[i, 0, 1:-1] = (
            impl_feq(i, rho[0, 1:-1], ux[0, 1:-1], uy[0, 1:-1])
            + fprop[i, 1, 1:-1]
            - impl_feq(i, rho[1, 1:-1], ux[1, 1:-1], uy[1, 1:-1])
        )
    t6_2_end = perf_counter()
    t6_2_elapsed = t6_2_elapsed + t6_2_end - t6_2_start

    # lower with corner : moving wall (NEEM)
    rho[-1, 1:-1] = rho[-2, 1:-1]
    ux[-1, 1:-1] = ux_wall
    uy[-1, 1:-1] = uy_wall
    for i in range(Q):
        fprop[i, -1, 1:-1] = (
            impl_feq(i, rho[-1, 1:-1], ux[-1, 1:-1], uy[-1, 1:-1])
            + fprop[i, -2, 1:-1]
            - impl_feq(i, rho[-2, 1:-1], ux[-2, 1:-1], uy[-2, 1:-1])
        )
    # left with corner : reset wall (NEEM)
    rho[:, 0] = rho[:, 1]
    ux[:, 0] = ux_wall
    uy[:, 0] = uy_wall
    for i in range(Q):
        fprop[i, :, 0] = (
            impl_feq(i, rho[:, 0], ux[:, 0], uy[:, 0])
            + fprop[i, :, 1]
            - impl_feq(i, rho[:, 1], ux[:, 1], uy[:, 1])
        )
    # right with corner : moving wall (NEEM)
    rho[:, -1] = rho[:, -2]
    ux[:, -1] = ux_right
    uy[:, -1] = uy_right
    for i in range(Q):
        fprop[i, :, -1] = (
            impl_feq(i, rho[:, -1], ux[:, -1], uy[:, -1])
            + fprop[i, :, -2]
            - impl_feq(i, rho[:, -2], ux[:, -2], uy[:, -2])
        )
    t6_end = perf_counter()
    t6_elapsed = t6_elapsed + t6_end - t6_start
# =========================================================
t_end = perf_counter()
t_elapsed = t_elapsed + t_end - t_start

print(f"Total elapsed time {t_elapsed} in seconds.")
print("Each process elapsed time:")
print(f"       check - {t1_elapsed}")
print(f" equilibrium - {t2_elapsed}")
print(f"   collision - {t3_elapsed}")
print(f"   streaming - {t4_elapsed}")
print(f"       macro - {t5_elapsed}")
print(f"          bc - {t6_elapsed}")
print(f"    bc macro - {t6_1_elapsed}")
print(f"    bc micro - {t6_2_elapsed}")


# %%
# show_vel_field(rho, ux, uy)
