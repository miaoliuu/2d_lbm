"""
Poiseuille Flow

    - compressible and incompressible equilibrium

    - SRT collision

    - periodic streaming

    - Inlet: Dirichlet boundary of velocity (velocity inlet, NEEM)

    - Outlet: Dirichlet boundary of density (pressure outlet, NEEM)

    - Upper: Dirichlet boundary of velocity (reset wall, NEEM)

    - Lower: Dirichlet boundary of velocity (reset wall, NEEM)

    o -- y
    |
    x       B - B - B - B - B - B - B - B
            i   F   F   F   F   F   F   o
            |                           |
            i                           o
            |                           |
            i                           o
            |                           |
            i                           o
            |                           |
            i   F   F   F   F   F   F   o
            B - B - B - B - B - B - B - B

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


def feq_ma2(iPop_, rho_, ux_, uy_):
    """Equilibrium function of compressible flow"""
    uSqr = ux_**2 + uy_**2
    ci_u = cx[iPop_] * ux_ + cy[iPop_] * uy_
    res = w[iPop_] * rho_ * (1.0 + 3.0 * ci_u + 4.5 * ci_u**2 - 1.5 * uSqr)
    return res


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
compressible = False

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
def get_l2_error(uy_out_, ana_vel_):
    """Calculate l2 error"""
    l2_error = np.sqrt(((uy_out_ - ana_vel_) ** 2).sum() / (ana_vel_**2).sum())
    return l2_error


def show_compared_velocity(xPos_, uy_out_, ana_vel_):
    """Compare Velocity"""
    plt.figure("Velocity Compare")
    plt.plot(xPos_, ana_vel_, "k-", label="Ana")
    plt.plot(xPos_, uy_out_, "ro", label="Sim")
    plt.grid()
    plt.legend()
    plt.show()


def show_vel_field(rho_, ux_, uy_):
    plt.figure("Density")
    plt.title(f"rho_max = {rho_.max():f}\nrho_min = {rho_.min():f}")
    plt.contourf(rho_, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    plt.colorbar(orientation="horizontal")

    plt.figure("velocity X")
    plt.title(f"vel_x_max = {ux_.max():f}\nvel_x_min = {ux_.min():f}")
    plt.contourf(ux_, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    plt.colorbar(orientation="horizontal")

    plt.figure("velocity Y")
    plt.title(f"vel_y_max = {uy_.max():f}\nvel_y_min = {uy_.min():f}")
    plt.contourf(uy_, levels=21, cmap=plt.cm.jet)
    plt.axis("image")
    plt.colorbar(orientation="horizontal")

    plt.show()


def write_data(compressible_, iT_, rho0_, u0_, rho_, ux_, uy_):
    """Macro data"""
    output_file = f"tmp/poiseuille_{iT_}.npz"
    np.savez(
        output_file,
        compressible=compressible_,
        iT=iT_,
        rho0=rho0_,
        u0=u0_,
        rho=rho_,
        ux=ux_,
        uy=uy_,
    )


# =========================================================

# %%
# =========================================================
# Geometry
# ---------------------------------------------------------
# length and resolution
lx, ly = 1, 3
resolution = 19
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
t_max = 10000 * lx * ly
uMax = 0.1
rho0 = 1.0  # reference density
tau = 0.8
omega = 1.0 / tau
niu = Cs**2 * (tau - 0.5)
# boundaries macro
ux_in = 0.0
uy_in = 2 / 3 * uMax
rho_out = rho0
ux_wall = 0.0
uy_wall = 0.0
# exported parameters
Re = uy_in * (nx - 1) / niu
# show
print(f"tau = {tau}, omega = {omega}, niu = {niu}")
print(f"inlet velocity = {uy_in}")
print(f"Mach Number = {uy_in/Cs}")
print(f"Reynolds Number = {Re}")
# =========================================================

# %%
# =========================================================
# simulation parameters
# ---------------------------------------------------------
# analytial solution
ana_vel = -4.0 * uMax / (lx**2) * (xPos - 0.0) * (xPos - lx)
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
# velocity at inlet nodes
ux[:, 0] = ux_in
uy[:, 0] = uy_in
# checkpiont parameters
tol = 1.0e-12
teval = 100  # time step for checkpoint
uy_old = np.ones((nx, ny))
# =========================================================

# %%
# =========================================================
# time loop
# ---------------------------------------------------------
for iT in range(1, t_max):
    # check point
    if iT % teval == 1:
        conv = np.abs(uy.mean() / (uy_old.mean() + 1.0e-32) - 1.0)
        print("iT =", iT, "conv =", conv)
        if conv < tol:
            # write_data(compressible, iT, rho0, uMax, rho, ux, uy)
            break
        else:
            uy_old = uy.copy()

    # equilibrium
    for i in range(Q):
        feq[i, :, :] = impl_feq(i, rho, ux, uy)

    # collision
    fstar = (1.0 - omega) * fprop + omega * feq

    # periodic streaming
    for i in range(Q):
        fprop[i, :, :] = np.roll(fstar[i, :, :], (cx[i], cy[i]), axis=(0, 1))

    # macro
    rho, ux, uy = impl_macro(fprop)

    # boundaries (NEEM)
    # upper : reset wall
    rho[0, 1:-1] = rho[1, 1:-1]
    ux[0, 1:-1] = ux_wall
    uy[0, 1:-1] = uy_wall
    for i in range(Q):
        fprop[i, 0, 1:-1] = (
            impl_feq(i, rho[0, 1:-1], ux[0, 1:-1], uy[0, 1:-1])
            + fprop[i, 1, 1:-1]
            - impl_feq(i, rho[1, 1:-1], ux[1, 1:-1], uy[1, 1:-1])
        )
    # lower : reset wall
    rho[-1, 1:-1] = rho[-2, 1:-1]
    ux[-1, 1:-1] = ux_wall
    uy[-1, 1:-1] = uy_wall
    for i in range(Q):
        fprop[i, -1, 1:-1] = (
            impl_feq(i, rho[-1, 1:-1], ux[-1, 1:-1], uy[-1, 1:-1])
            + fprop[i, -2, 1:-1]
            - impl_feq(i, rho[-2, 1:-1], ux[-2, 1:-1], uy[-2, 1:-1])
        )
    # inlet with corner : Dirichlet velocity (NEEM)
    rho[:, 0] = rho[:, 1]
    ux[:, 0] = 0.0
    uy[:, 0] = uy_in
    for i in range(Q):
        fprop[i, :, 0] = (
            impl_feq(i, rho[:, 0], ux[:, 0], uy[:, 0])
            + fprop[i, :, 1]
            - impl_feq(i, rho[:, 1], ux[:, 1], uy[:, 1])
        )
    # outlet with corner : Dirichlet density (NEEM)
    rho[:, -1] = rho_out
    ux[:, -1] = ux[:, -2]
    uy[:, -1] = uy[:, -2]
    for i in range(Q):
        fprop[i, :, -1] = (
            impl_feq(i, rho[:, -1], ux[:, -1], uy[:, -1])
            + fprop[i, :, -2]
            - impl_feq(i, rho[:, -2], ux[:, -2], uy[:, -2])
        )
# =========================================================

# %%
print(f"L2 Error = {get_l2_error(uy[:, -1], ana_vel)}")
show_compared_velocity(xPos, uy[:, -1], ana_vel)
show_vel_field(rho, ux, uy)
