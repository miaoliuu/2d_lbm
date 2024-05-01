"""
Post-process for Lad Driven Cavity Flow
"""

# %%
import glob
import numpy as np
import matplotlib.pyplot as plt

# %%
# =========================================================
# load case data
# ---------------------------------------------------------
Re = 50.0
iT = 6901
case_ifile_name = f"tmp/ldc_case_Re{Re}.npz"
with np.load(case_ifile_name) as data:
    print(f"name list of variables: {data.files}")
    lx = data["lx"]
    ly = data["ly"]
    resolution = data["resolution"]
    nx = data["nx"]
    ny = data["ny"]
    xPos = data["xPos"]
    yPos = data["yPos"]
    Re = data["Re"]
    tau = data["tau"]
    omega = data["omega"]
    niu = data["niu"]
# =========================================================

# %%
# =========================================================
# load transient data
# ---------------------------------------------------------
data_ifile_name = f"tmp/ldc_Re{Re}_{iT}.npz"
with np.load(data_ifile_name) as data:
    print(f"name list of variables: {data.files}")
    compressible = data["compressible"]
    iT = data["iT"]
    rho0 = data["rho0"]
    u0 = data["u0"]
    rho = data["rho"]
    vel_x = data["vel_x"]
    vel_y = data["vel_y"]

points = nx * ny
# =========================================================

# %%
# =========================================================
# Convert to paraview .vtr
# ---------------------------------------------------------
ofile_name = f"tmp/ldc_Re{Re}_{iT}.vtr"
with open(ofile_name, mode="w") as ofile:
    # // ---- xml Header ----
    ofile.write('<?xml version="1.0"?>\n')
    # // ---- VTK file type Header ----
    ofile.write(
        '<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">\n'
    )
    # // ---- RectilinearGrid Header ----
    ofile.write(f'<RectilinearGrid WholeExtent="{0} {nx-1} {0} {ny-1} {0} {0}">\n\n')

    # // ---- Piece Header ----
    ofile.write(f'<Piece Extent="{0} {nx-1} {0} {ny-1} {0} {0}">\n')

    # // ---- Coordinates Header ----
    ofile.write("<Coordinates>\n")
    # // ---- * DataArray of Coordinates * ----
    ofile.write('<DataArray type="Float64" Name="X" format="ascii">\n')
    for i in range(nx):
        ofile.write("{:g}{:s}".format(xPos[i], (" " if (i + 1) % 10 else "\n")))
    ofile.write("\n")
    ofile.write("</DataArray>\n")
    ofile.write('<DataArray type="Float64" Name="Y" format="ascii">\n')
    for i in range(ny):
        ofile.write("{:g}{:s}".format(yPos[i], (" " if (i + 1) % 10 else "\n")))
    ofile.write("\n")
    ofile.write("</DataArray>\n")
    ofile.write('<DataArray type="Float64" Name="Z" format="ascii">\n')
    # // Todo
    ofile.write("</DataArray>\n")
    # // ---- Points Footer ----
    ofile.write("</Coordinates>\n")

    # // ---- PointData Header ----
    ofile.write("<PointData>\n")
    # // ---- * DataArray of point variables * ----
    # // ---- * density * ----
    ofile.write(
        '<DataArray type="Float64" Name="density" NumberOfComponents="1" format="ascii">\n'
    )
    for j in range(ny):
        for i in range(nx):
            ofile.write("{:g}{:s}".format(rho[i, j], (" " if (i + 1) % 10 else "\n")))
        ofile.write("\n")
    ofile.write("</DataArray>\n")
    # // ---- * vel_x * ----
    ofile.write(
        '<DataArray type="Float64" Name="vel_x" NumberOfComponents="1" format="ascii">\n'
    )
    for j in range(ny):
        for i in range(nx):
            ofile.write("{:g}{:s}".format(vel_x[i, j], (" " if (i + 1) % 10 else "\n")))
        ofile.write("\n")
    ofile.write("</DataArray>\n")
    # // ---- * vel_y * ----
    ofile.write(
        '<DataArray type="Float64" Name="vel_y" NumberOfComponents="1" format="ascii">\n'
    )
    for j in range(ny):
        for i in range(nx):
            ofile.write("{:g}{:s}".format(vel_y[i, j], (" " if (i + 1) % 10 else "\n")))
        ofile.write("\n")
    ofile.write("</DataArray>\n")
    # // ---- PointData Footer ----
    ofile.write("</PointData>\n")

    # // ---- CellData Header ----
    ofile.write("<CellData>\n")
    # // ---- * DataArray of cell variables * ----
    # ToDo
    # // ---- CellData Footer ----
    ofile.write("</CellData>\n")

    # // ---- Piece Footer ----
    ofile.write("</Piece>\n\n")

    # // ---- RectilinearGrid Footer ----
    ofile.write("</RectilinearGrid>\n")
    # // ---- VTK file type Footer ----
    ofile.write("</VTKFile>\n")
# =========================================================

# %%
# =========================================================
# 2D visualization
# ---------------------------------------------------------
plt.figure("Density")
plt.title(f"rho_max = {rho.max():f}\nrho_min = {rho.min():f}")
plt.contourf(rho.T, levels=21, cmap=plt.cm.jet)
plt.axis("image")
# plt.colorbar(orientation="horizontal")
plt.colorbar()

plt.figure("velocity X")
plt.title(f"vel_x_max = {vel_x.max():f}\nvel_x_min = {vel_x.min():f}")
plt.contourf(vel_x.T, levels=21, cmap=plt.cm.jet)
plt.axis("image")
# plt.colorbar(orientation="horizontal")
plt.colorbar()

plt.figure("velocity Y")
plt.title(f"vel_y_max = {vel_y.max():f}\nvel_y_min = {vel_y.min():f}")
plt.contourf(vel_y.T, levels=21, cmap=plt.cm.jet)
plt.axis("image")
# plt.colorbar(orientation="horizontal")
plt.colorbar()

vel_mag = np.sqrt(vel_x**2 + vel_y**2)
plt.figure("Velocity Magnitude")
plt.title(f"max = {vel_mag.max():f}\nmin = {vel_mag.min():f}")
plt.contourf(vel_mag.T, levels=21, cmap=plt.cm.jet)
plt.axis("image")
# plt.colorbar(orientation="horizontal")
plt.colorbar()

plt.figure("Streamline")
plt.title(f"Re = {Re}")
plt.streamplot(
    xPos,
    yPos,
    vel_x.T,
    vel_y.T,
    density=2.5,
)
plt.axis("image")

plt.show()
# =========================================================
