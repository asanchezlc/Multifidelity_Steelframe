import os
import numpy as np

import helpers.outils as outils
import json

"""
File Duties:

Generates random forces time series for each node and direction for time history analysis.
Saves the time series in "src/sap2000/functions" folder

Remark:
    - After running this file, the generated files can be used for assigning time history loads
"""
# Filenames
filename_channels = "geometry_OMA_Setup_1.txt"  # used in MOVA too
filename_json = 'steelframe_rigid_X_cantilever_Y.json'
config_path = os.path.join('src', 'data')
output_path = os.path.join('src', 'sap2000', 'functions')

# Acquisition configuration
fs = 207  # [Hz] Sampling frequency
T = 15*60  # [sec] Period of the time series (15 minutes)
af = 1  # Amplitute of the force (SAME AS FOR LUMPED MODEL!)
rng = np.random.RandomState(12345)  # Set the seed
n_points_per_floor = 4  # 4 corners
n_directions = 3  # X, Y, Z

# Load data
os.makedirs(output_path, exist_ok=True)
directions = ['X', 'Y', 'Z', 'RX', 'RY', 'RZ']
directions = directions[:n_directions]
nodes, lines, planes, color_planes = outils.MOVA_read_geometry(os.path.join(config_path, filename_channels))
with open(os.path.join(config_path, filename_json), "r") as f:
    CONFIG = json.load(f)
n_floors = CONFIG["n_floors"]
Lc = CONFIG["base"]["Lc"]

# Derived parameters
dt = 1/fs  # [sec] time resolution
df = 1/T  # [Hz] frequency resolution
N = int(T/dt)  # number of data points
t = np.arange(0, T+dt, dt)  # time instants array
_ndof = n_points_per_floor * n_directions * n_floors  # number of DOF (4 corners, 3 dofs)
x_centre = np.mean([nodes[i][0] for i in nodes])
y_centre = np.mean([nodes[i][1] for i in nodes])

# Generate random forces for time history analysis
forces = np.array([rng.randn(N+1)*af for r in range(_ndof)]).T

count = 0
for node, coords in nodes.items():
    x, y, z = coords[0], coords[1], coords[2]
    if z == 0:
        continue  # skip base nodes
    for i, dir in enumerate(directions):
        f = forces[:, count]
        f_matrix = np.array([t, f]).T
        filename_f = f"x={str(round(x,3)).replace('.', '-')}_y={str(round(y,3)).replace('.', '-')}_z={str(round(z,3)).replace('.', '-')}_F{dir}.txt"
        np.savetxt(os.path.join(output_path, filename_f), f_matrix, delimiter="\t")
        count += 1
