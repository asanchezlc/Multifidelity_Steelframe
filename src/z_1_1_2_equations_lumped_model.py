
import sympy as sym

import helpers.outils as outils

"""
Rigid diaphragm (in-plane) 3-DOF story stiffness matrix:
DOFs: [Ux, Uy, Thetaz]
Four identical corner columns; lateral stiffness of each column obtained
from an Euler-Bernoulli beam with rotational springs at both ends.

Beam with infinite flexural stiffness in its plane (no deformation in flexion).
"""
# --- Symbols ---
E, I1, I2 = sym.symbols('E I1 I2', positive=True)
Lc = sym.symbols('Lc', positive=True)                 # story height
kx1, kx2 = sym.symbols('kx_1 kx_2', nonnegative=True) # rotational springs about x (affect bending in y, i.e., Uy)
ky1, ky2 = sym.symbols('ky_1 ky_2', nonnegative=True) # rotational springs about y (affect bending in x, i.e., Ux)
B, H = sym.symbols('B H', positive=True)              # diaphragm plan dimensions

# --- Column lateral stiffness in each global translation ---
k_col_x = outils.column_lateral_stiffness_rot_springs(E, I2, Lc, ky1, ky2)  # governs Ux
k_col_y = outils.column_lateral_stiffness_rot_springs(E, I1, Lc, kx1, kx2)  # governs Uy

# --- Story (level) stiffness matrix in DOFs [Ux, Uy, Thetaz] ---
K_xx = 4 * k_col_x
K_yy = 4 * k_col_y
K_tt = k_col_x * H**2 + k_col_y * B**2

K_level = sym.diag(sym.simplify(K_xx), sym.simplify(K_yy), sym.simplify(K_tt))
