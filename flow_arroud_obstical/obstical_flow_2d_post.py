"""
Post-process for Corrugated channel flow
"""

# %%
import glob
import numpy as np
import matplotlib.pyplot as plt

# %%
# =========================================================
# load geometry data
# ---------------------------------------------------------
input_file = f"tmp/ob_geometry.npz"
with np.load(input_file) as data:
    print(f"name list of variables: {data.files}")
    xPos = data["xPos"]
    yPos = data["yPos"]
    mask = data["mask"]
# =========================================================

# %%
# =========================================================
# load transient data
# ---------------------------------------------------------
iT = 50001
input_file = f"tmp/ob_{iT}.npz"
with np.load(input_file) as data:
    print(f"name list of variables: {data.files}")
    compressible = data["compressible"]
    iT = data["iT"]
    rho0 = data["rho0"]
    u0 = data["u0"]
    rho = data["rho"]
    vel_x = data["vel_x"]
    vel_y = data["vel_y"]
# =========================================================

# %%
# =========================================================
# 2D visualization
# ---------------------------------------------------------
plt.figure("geometry")
plt.pcolormesh(mask, cmap=plt.cm.gray)
plt.axis("image")
# plt.imshow(mask_, cmap=plt.cm.gray, origin="upper")
# plt.gca().xaxis.set_ticks_position("top")

plt.figure("Density")
plt.title(f"rho_max = {rho.max():f}\nrho_min = {rho.min():f}")
plt.contourf(rho, levels=21, cmap=plt.cm.jet)
plt.axis("image")
plt.colorbar(orientation="horizontal")

plt.figure("velocity X")
plt.title(f"vel_x_max = {vel_x.max():f}\nvel_x_min = {vel_x.min():f}")
plt.contourf(vel_x, levels=21, cmap=plt.cm.jet)
plt.axis("image")
plt.colorbar(orientation="horizontal")

plt.figure("velocity Y")
plt.title(f"vel_y_max = {vel_y.max():f}\nvel_y_min = {vel_y.min():f}")
plt.contourf(vel_y, levels=21, cmap=plt.cm.jet)
plt.axis("image")
plt.colorbar(orientation="horizontal")

vel_mag = np.sqrt(vel_x**2 + vel_y**2)
plt.figure("Velocity Magnitude")
plt.title(f"max = {vel_mag.max():f}\nmin = {vel_mag.min():f}")
plt.contourf(vel_mag, levels=21, cmap=plt.cm.jet)
plt.axis("image")
plt.colorbar(orientation="horizontal")

plt.figure("Streamline")
plt.title("Streamline")
plt.streamplot(
    yPos[50:200],
    xPos[81:122],
    vel_y[81:122, 50:200],
    vel_x[81:122, 50:200],
    density=2.5,
)
plt.axis("image")
# =========================================================

# %%
plt.show()
