"""
Obstical flow

    - compressible and incompressible equilibrium

    - SRT collision

    - periodic streaming

    - Inlet: Dirichlet boundary of velocity (velocity inlet, NEEM)

    - Outlet: Dirichlet boundary of density (outflow, NEEM)

    - Upper: Dirichlet boundary of velocity (reset wall, NEEM)

    - Lower: Dirichlet boundary of velocity (rest wall, NEEM)

    - Obstical: Dirichlet boundary velocity (Simple BB, BB at wet-node)

    o -- y
    |
    x       B - B - B - B - B - B - B - B
            i   F   F   F   F   F   F   o
            |                           |
            i                           o
            |           s               |
            i         s s s             o
            |           s               |
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
    """Update macro of incompressible flow"""
    rho, ux, uy = moment(lat_)
    ux /= rho
    uy /= rho
    return rho, ux, uy


def update_macro_incomp(lat_):
    """Update macro of incompressible flow"""
    return moment(lat_)


# choice equilibrium function
compressible = False

# implement of equillibrium
if compressible:
    impl_feq = feq_ma2
    impl_macro = update_macro
else:
    impl_feq = feq_ma2_incomp
    impl_macro = update_macro_incomp
# =========================================================


# %%
# =========================================================
# Write to disk
# ---------------------------------------------------------
def write_geometry(xPos_, yPos_, mask_):
    """Geometry data"""
    output_file = f"tmp/ob_geometry.npz"
    np.savez(
        output_file,
        xPos=xPos_,
        yPos=yPos_,
        mask=mask_,
    )


def write_data(compressible_, iT_, rho0_, u0_, rho_, ux_, uy_):
    """Macro data"""
    if compressible_:
        vel_x = ux_ * rho_ / rho0_
        vel_y = uy_ * rho_ / rho0_
    else:
        vel_x = ux_
        vel_y = uy_

    output_file = f"tmp/ob_{iT_}.npz"
    np.savez(
        output_file,
        compressible=compressible_,
        iT=iT_,
        rho0=rho0_,
        u0=u0_,
        rho=rho,
        vel_x=vel_x,
        vel_y=vel_y,
    )


def visualization(ax_, iT_, vel_mag_):
    ax_.clear()
    ax_.pcolormesh(vel_mag_)
    ax_.set_aspect("equal")
    ax_.set_title(f"t = {iT_}")
    # // ---- show ----
    plt.pause(0.001)


# =========================================================

# %%
# =========================================================
# Geometry
# ---------------------------------------------------------
# length and resolution
diameter = 1  # diameter of pillar
lx, ly = 10, 20
center = np.array([0.5 * lx, 0.25 * ly])
resolution = 20
# wet-node
nx, ny = lx * resolution + 1, ly * resolution + 1
xPos = np.linspace(0, lx, nx)
yPos = np.linspace(0, ly, ny)
meshPos = np.array(np.meshgrid(xPos, yPos, indexing="ij"))
mask = np.zeros((nx, ny), dtype=bool)


def get_circle_mask(diameter_, center_, meshPos_):
    """Get circle mask array

    Parameters
    ----------
    diameter_ : float
        Diameter of circle.
    center_ : NDArray
        Center of circle.
    meshPos_ : NDArray
        Mesh of flow domain.

    Returns
    -------
    NDArray[float]
        Mask array of flow domain.
    """
    mask = ((meshPos_ - center_.reshape(D, 1, 1)) ** 2).sum(0) < diameter_**2 / 4
    return mask


# obstical mask
mask = get_circle_mask(diameter, center, meshPos)


def show_geometry(mask_):
    """show geometry"""
    plt.figure("geometry")
    plt.pcolormesh(mask_, cmap=plt.cm.gray)
    plt.axis("image")
    # plt.imshow(mask_, cmap=plt.cm.gray, origin="upper")
    # plt.gca().xaxis.set_ticks_position("top")
    plt.show()


# show geometry
# show_geometry(mask)
write_geometry(xPos, yPos, mask)
# =========================================================

# %%
# =========================================================
# simulation parameters
# ---------------------------------------------------------
# control parameters
t_max = 10000 * lx * ly
Re = 40.0
rho0 = 1.0  # reference [global mean] density
tau = 0.6
omega = 1.0 / tau
niu = Cs**2 * (tau - 0.5)
# exported parameters
ux_in = 0.0
uy_in = Re * niu / (diameter * resolution)
ux_wall = 0.0
uy_wall = 0.0
# show
print(f"tau = {tau}, omega = {omega}, niu = {niu}")
print(f"u_in = {uy_in}, Mach Number = {uy_in/Cs}")
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
# macro at inlet nodes
ux[:, 0] = ux_in
uy[:, 0] = uy_in
# macro at obstical
ux[mask] = ux_wall
uy[mask] = uy_wall
# checkpiont parameters
teval = 100  # time step for checkpoint
fig, ax = plt.subplots()
# =========================================================

# %%
# =========================================================
# time loop
# ---------------------------------------------------------
for iT in range(1, t_max):
    # check point
    if iT % teval == 1:
        print(f"iT = {iT}, Average Density = {rho.mean()}")
        vel_mag = np.sqrt(ux**2 + uy**2)
        visualization(ax, iT, vel_mag)

    if iT % 10000 == 1:
        write_data(compressible, iT, rho0, uy_in, rho, ux, uy)

    # equilibrium
    for i in range(Q):
        feq[i, :, :] = impl_feq(i, rho, ux, uy)

    # collision
    fstar = (1.0 - omega) * fprop + omega * feq

    # simple BB collision on obstical
    for i in range(Q):
        fstar[i, mask] = fprop[oppo[i], mask]

    # periodic streaming
    for i in range(Q):
        fprop[i, :, :] = np.roll(fstar[i, :, :], (cx[i], cy[i]), axis=(0, 1))

    # macro
    rho, ux, uy = impl_macro(fprop)
    # macro at obstical and wall
    ux[mask] = ux_wall
    uy[mask] = uy_wall

    # boundaries (NEEM)
    # upper : Neumann boundary
    rho[0, 1:-1] = rho[1, 1:-1]
    ux[0, 1:-1] = ux_wall
    uy[0, 1:-1] = uy_wall
    for i in range(Q):
        fprop[i, 0, 1:-1] = (
            impl_feq(i, rho[0, 1:-1], ux[0, 1:-1], uy[0, 1:-1])
            + fprop[i, 1, 1:-1]
            - impl_feq(i, rho[1, 1:-1], ux[1, 1:-1], uy[1, 1:-1])
        )
    # lower : Neumann boundary
    rho[-1, 1:-1] = rho[-2, 1:-1]
    ux[-1, 1:-1] = ux_wall
    uy[-1, 1:-1] = uy_wall
    for i in range(Q):
        fprop[i, -1, 1:-1] = (
            impl_feq(i, rho[-1, 1:-1], ux[-1, 1:-1], uy[-1, 1:-1])
            + fprop[i, -2, 1:-1]
            - impl_feq(i, rho[-2, 1:-1], ux[-2, 1:-1], uy[-2, 1:-1])
        )
    # inlet : Dirichlet velocity (NEEM)
    rho[:, 0] = rho[:, 1]
    ux[:, 0] = ux_in
    uy[:, 0] = uy_in
    for i in range(Q):
        fprop[i, :, 0] = (
            impl_feq(i, rho[:, 0], ux[:, 0], uy[:, 0])
            + fprop[i, :, 1]
            - impl_feq(i, rho[:, 1], ux[:, 1], uy[:, 1])
        )
    # outlet : Neumann outflow (NEEM)
    rho[:, -1] = rho0
    ux[:, -1] = ux[:, -2]
    uy[:, -1] = uy[:, -2]
    for i in range(Q):
        fprop[i, :, -1] = (
            impl_feq(i, rho[:, -1], ux[:, -1], uy[:, -1])
            + fprop[i, :, -2]
            - impl_feq(i, rho[:, -2], ux[:, -2], uy[:, -2])
        )
# =========================================================
