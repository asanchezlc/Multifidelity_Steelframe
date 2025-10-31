
import json
import os

from scipy import signal
from scipy import linalg as LA
import numpy as np

import helpers.outils as outils

# Filenames
filename_channels = "geometry_OMA_Setup_1.txt"  # used in MOVA too
K_filename = 'steelframe_rigid_X_cantilever_Y_K.txt'
M_filename = 'steelframe_rigid_X_cantilever_Y_M.txt'
output_name = K_filename.split('_K')[0] + '_state_space.txt'
filename_json = 'steelframe_rigid_X_cantilever_Y.json'

# Acquisition configuration
xi = 0.02  # damping ratio for all modes (2%)
fs = 207  # [Hz] Sampling freqiency
T = 15*60  # [sec] Period of the time series (15 minutes)
SNR = 10  # Signal-to-Noise ratio (noise to add to the output)
rng = np.random.RandomState(12345)  # Set the seed

# Data loading
path_channels = os.path.join('src', 'data')
path_matrices = os.path.join('src', 'output')
output_path = os.path.join('src', 'output')
nodes, lines, planes, color_planes = outils.MOVA_read_geometry(os.path.join(path_channels, filename_channels))
channels_oma_geometry = outils.read_sensors_from_MOVA_file(os.path.join(path_channels, filename_channels))
K = np.loadtxt(os.path.join(path_matrices, K_filename), delimiter="\t")
M = np.loadtxt(os.path.join(path_matrices, M_filename), delimiter="\t")
_ndof = np.shape(K)[0]
with open(os.path.join(path_channels, filename_json), "r") as f:
    CONFIG = json.load(f)
n_floors = CONFIG["n_floors"]
Lc = CONFIG["base"]["Lc"]


# Derived parameters
dt = 1/fs  # [sec] time resolution
df = 1/T  # [Hz] frequency resolution
N = int(T/dt)  # number of data points
fmax = fs/2  # Nyquist frequency
t = np.arange(0, T+dt, dt)  # time instants array

# Modal properties
lam, FI = LA.eigh(K, b=M)  # Solving eigen value problem
fn = np.sqrt(lam)/(2*np.pi)  # Natural frequencies
FI_1 = np.array([FI[:, k]/max(abs(FI[:, k])) for k in range(_ndof)]).T  # Unity displacement normalised mode shapes
FI_1 = FI_1[:, np.argsort(fn)]  # Ordering from smallest to largest
fn = np.sort(fn)

# Modal matrices
# K_M = FI_M.T @ K @ FI_M # Modal stiffness
M_M = FI_1.T @ M @ FI_1  # Modal mass
C_M = np.diag(np.array([2*M_M[i, i]*xi*fn[i]*(2*np.pi) for i in range(_ndof)]))  # Modal damping
C = LA.inv(FI_1.T) @ C_M @ LA.inv(FI_1)  # Damping matrix
# n = _ndof*2 # order of the system

# =========================================================================
# STATE-SPACE FORMULATION
a1 = np.zeros((_ndof, _ndof))  # Zeros (ndof x ndof)
a2 = np.eye(_ndof)  # Identity (ndof x ndof)
A1 = np.hstack((a1, a2))  # horizontal stacking (ndof x 2*ndof)
a3 = -LA.inv(M) @ K  # M^-1 @ K (ndof x ndof)
a4 = -LA.inv(M) @ C  # M^-1 @ C (ndof x ndof)
A2 = np.hstack((a3, a4))  # horizontal stacking(ndof x 2*ndof)
# vertical stacking of A1 e A2
Ac = np.vstack((A1, A2))  # State Matrix A (2*ndof x 2*ndof))

b2 = -LA.inv(M)
# Input Influence Matrix B (2*ndof x n°input=ndof)
Bc = np.vstack((a1, b2))

# N.B. number of rows = n°output*ndof
# n°output may be 1, 2 o 3 (displacements, velocities, accelerations)
# the Cc matrix has to be defined accordingly
c1 = np.hstack((a2, a1))  # displacements row
c2 = np.hstack((a1, a2))  # velocities row
c3 = np.hstack((a3, a4))  # accelerations row
# Output Influence Matrix C (n°output*ndof x 2*ndof)
Cc = np.vstack((c1, c2, c3))

# Direct Transmission Matrix D (n°output*ndof x n°input=ndof)
Dc = np.vstack((a1, a1, b2))

# =========================================================================
# Using SciPy's LTI to solve the system

# Defining the system
sys = signal.lti(Ac, Bc, Cc, Dc)
af = 1  # Amplitute of the force

# Assembling the forcing vectors (N x ndof) (random white noise!)
# N.B. N=number of data points; ndof=number of DOF
u = np.array([rng.randn(N+1)*af for r in range(_ndof)]).T

# Solving the system
tout, yout, xout = signal.lsim(sys, U=u, T=t)

# d = yout[:,0:_ndof] # displacement
# v = yout[:,_ndof:2*_ndof] # velocity
a = yout[:, 2*_ndof:3*_ndof]  # acceleration
# =========================================================================

# =========================================================================
# EXPORTING DATA AS ACCELEROMETERS

# EXPORT AS ACCELEROMETERS
channels_oma = dict()
for i, ch in enumerate(channels_oma_geometry):
    node = channels_oma_geometry[ch]['node']
    coordinates = nodes[node]
    dir = channels_oma_geometry[ch]['dir']
    channels_oma[ch] = {'coord': {'x': coordinates[0], 'y': coordinates[1], 'z': coordinates[2]},
                        'dir': dir}

accelerations_oma = np.zeros((a.shape[0], len(channels_oma)))

# Noise
# ar = af/(10**(SNR/10))
ar = af / (10 ** (SNR / 20))  # Noise amplitude

# Position of the centre of the structure (where the original DOFS are defined)
x_centre = np.mean([nodes[i][0] for i in nodes])
y_centre = np.mean([nodes[i][1] for i in nodes])

# Save as accelerometers
for i, (ch_name, ch) in enumerate(channels_oma.items()):
    x = ch['coord']['x'] - x_centre
    y = ch['coord']['y'] - y_centre
    z = ch['coord']['z']
    dirx, diry, _ = ch['dir']   # solo X-Y

    k = int(round(z / Lc)) - 1

    ax = a[:, k]
    ay = a[:, n_floors + k]
    at = a[:, 2*n_floors + k]

    ax_pt = ax - at * y         # a_x en (x,y)
    ay_pt = ay + at * x         # a_y en (x,y)

    acc = dirx * ax_pt + diry * ay_pt  # proyection along sensor direction
    acc_noise = acc + ar * rng.standard_normal(size=acc.shape)  # adding noise

    accelerations_oma[:, i] = acc_noise

# Saving file
np.savetxt(os.path.join(output_path, output_name), accelerations_oma, delimiter="\t")
