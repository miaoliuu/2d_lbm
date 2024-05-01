"""
Post-process for Corrugated channel flow
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
# =========================================================
# load data
# ---------------------------------------------------------
input_file = "output_data.npz"
with np.load(input_file) as data:
    print(f"name list of variables: {data.files}")
    compressible = data["compressible"]
    niu = data["niu"]
    xPos = data["xPos"]
    yPos = data["yPos"]
    mask = data["mask"]
    rho = data["rho"]
    vel_x = data["vel_x"]
    vel_y = data["vel_y"]
# =========================================================

# %%
# =========================================================
# 2D visualization
# ---------------------------------------------------------
plt.figure("velocity Y")
plt.title(f"vel_y_max = {vel_y.max():f}\nvel_y_min = {vel_y.min():f}")
plt.contourf(vel_y, levels=21, cmap=plt.cm.jet)
plt.axis("image")
plt.colorbar(orientation="horizontal")
# =========================================================

# %%
# =========================================================
# plot line velocity
# ---------------------------------------------------------
Reynolds_num = vel_y[:, [64, 65]].mean(1).sum() / niu
plt.figure("velocity Y in line")
plt.title(f"Re = {Reynolds_num:f}")
plt.plot(vel_y[:, [64, 65]].mean(1), "ko", label="y = 64.5")
plt.xlabel("xPosition")
plt.ylabel("vel_y")
plt.legend()
plt.grid()
# =========================================================

# %%
# =========================================================
# write line velocity
# ---------------------------------------------------------
# vel_in_line = np.array((xPos, vel_y[:, [64, 65]].mean(1))).T
vel_in_line = np.array((xPos[1:-1], vel_y[1:-1, [64, 65]].mean(1))).T
# np.savetxt("corrugated_vel_y.csv", vel_in_line)
# =========================================================

# %%
# =========================================================
# velocity compare
# ---------------------------------------------------------
# LBM
lbm_data = vel_in_line
# lbm_data = np.loadtxt("corrugated_vel_y.csv")
# lbm_data[:,0] /= np.abs(lbm_data[:,0].max())
lbm_data[:, 1] /= np.abs(lbm_data[:, 1].max())
# lbm_data[:,1] /= np.abs(lbm_data[:,1].mean())
# Fluent
fluent_data = np.loadtxt("fluent_ux_at_0.65.csv")
fluent_data[:, 0] /= np.abs(fluent_data[:, 0].max())
fluent_data[:, 1] /= np.abs(fluent_data[:, 1].max())
# fluent_data[:,1] /= np.abs(fluent_data[:,1].mean())
# plot
plt.figure("velocity compare")
plt.plot(lbm_data[:, 0], lbm_data[:, 1], "ro", label="LBM")
plt.plot(fluent_data[:, 0], fluent_data[:, 1], "k-", label="FVM")
plt.xlabel("xPosition")
plt.ylabel("Normalized Velocity")
plt.legend()
plt.grid()
# =========================================================

# %%
plt.show()
