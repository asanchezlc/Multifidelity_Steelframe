
# -*- coding: utf-8 -*-
# Symbolic derivation of the horizontal force at node 2 needed to enforce u2 = 1
# for a vertical Euler-Bernoulli beam with rotational springs at both ends.

import helpers.outils as outils
import sympy as sym
from sympy.printing import latex

"""
Python script to derive horizontal stiffness matrix of a frame composed by:
- Two columns modeled as Euler-Bernoulli beams with rotational springs at both ends
and clamped supports
- One beam modeled as Euler-Bernoulli beam (no springs)

Theoretical values are obtained + comparison with numerical model in SAP2000

This file is used to define the function k_beam_rotational_springs in helpers/outils.py

IMPORTANT:
- This file accounts for the beam stiffness (does not assume infinite rigid); This is straightforward
for flexural behaviour around both axis, but not for torsional behaviour.
- It is used to obtain the following functions:
    - k_beam_rotational_springs_with_delta(E, I, h, k1, k2)
    - column_lateral_stiffness_with_beam_in_flexion(E, Ic, Ib, h, L, k1, k2)

SAP2000 MODEL TO CHECK AGAINST:
    - model_1_frame_plane_beam_not_rigid_rot_springs.sdb:
        For derivation of column_lateral_stiffness_with_beam_in_flexion(E, Ic, Ib, h, L, k1, k2)
    - model_1_frame_3D_beam_not_rigid_rot_springs.sdb:
        For derivation of stiffness_matrix_level_beam_in_flexion(E, Ic_y, Ic_x, Lc, Ib_1_y, Ib_2_y, kx1, kx2, ky1, ky2, B, H)
"""

# --- Symbols ---
E, I, I_c, I_b, h, L = sym.symbols('E I I_c I_b h L', positive=True, finite=True)
k1, k2 = sym.symbols('k1 k2', nonnegative=True,
                    finite=True)  # rotational springs

# Stiffness matrix of a beam with rotational springs at both ends
K6 = sym.Matrix([
    [12*E*I/L**3,   6*E*I/L**2,      0,          -12*E*I/L**3,   6*E*I/L**2,    0],
    [6*E*I/L**2,   4*E*I/L + k1,   -k1,          -6*E*I/L**2,    2*E*I/L,      0],
    [0,           -k1,        k1,                 0,             0,       0],
    [-12*E*I/L**3,  -6*E*I/L**2,      0,
        12*E*I/L**3,  -6*E*I/L**2,    0],
    [6*E*I/L**2,    2*E*I/L,        0,          -6*E*I/L**2,   4*E*I/L + k2, -k2],
    [0,             0,        0,                 0,           -k2,      k2]
])

# Condensation to 4 DOFS
K4, masters = outils.static_condensation(K6, slave_dofs=[1, 4])
K4 = sym.simplify(K4)
latex_str = r"\mathbf{K}_4 = " + latex(K4)
print(latex_str)
"""
We define Delta = 12*E**2*I**2 + 4*E*I*L*(k1 + k2) + L**2*k1*k2

As we see it is constant in the denominator of all terms of K4
"""
Delta = sym.symbols('Delta')

K4 = K4 * (12*E**2*I**2 + 4*E*I*L*(k1 + k2) + L**2*k1*k2)
K4 = sym.simplify(K4)
K4 = K4 / Delta
K4 = sym.simplify(K4)

latex_str = r"\mathbf{K}_4 = " + latex(K4)
print(latex_str)

"""
ASSEMBLING FOR FRAME
"""
# Local matrices
# k_col = outils.k_beam_rotational_springs_with_delta(E, I_c, h, k1, k2) -> USAR ESTA PARA OBTENER EXPRESIONES CON DELTA
k_col = outils.k_beam_rotational_springs(E, I_c, h, k1, k2)
Delta = 12*E**2*I_c**2 + 4*E*I_c*h*(k1 + k2) + h**2*k1*k2

k_vig = outils.beam_stiffness_sym(E, I_b, L)

# Submatrices
sel_top = [2, 3]  # not constrained dofs
k_col_top = k_col.extract(sel_top, sel_top)  # 2x2: acopla [u_top, th_top]
sel_rot = [1, 3]  # not constrained dofs (rotational; lateral are constrained for axial rigidity of columns)
k_vig_rot = k_vig.extract(sel_rot, sel_rot)  # 2x2: acopla [th1, th2]

# Assembling global stiffness matrix
K_frame_full = sym.zeros(3)

# Left column
map_col_L = [0, 1]
for i, I in enumerate(map_col_L):
    for j, J in enumerate(map_col_L):
        K_frame_full[I, J] += k_col_top[i, j]

# Right column
map_col_R = [0, 2]
for i, I in enumerate(map_col_R):
    for j, J in enumerate(map_col_R):
        K_frame_full[I, J] += k_col_top[i, j]

# Beam
map_vig = [1, 2]
for i, I in enumerate(map_vig):
    for j, J in enumerate(map_vig):
        K_frame_full[I, J] += k_vig_rot[i, j]

sym.simplify(K_frame_full)

sym.Matrix(K_frame_full)

latex_str = r"\mathbf{K_frame_full} = " + latex(K_frame_full)
print(latex_str)
latex_str = r"\Delta = " + latex(Delta)
print(latex_str)

# Condensation to horizontal DOF
K_frame_h, _ = outils.static_condensation(K_frame_full, slave_dofs=[1, 2])
K_frame_h = sym.simplify(K_frame_h)
K_frame_h = sym.simplify(K_frame_h)
latex_str = r"\mathbf{K_fh} = " + latex(K_frame_h[0,0])
print(latex_str)


"""
Numerical example (model_1_frame_plane_beam_not_rigid_rot_springs.sdb)
"""
E_subs = 2.100E+11  # Pa
I_c_subs = 2e-11    # m4
I_b_subs = 1/12*0.01*0.02**3  # m4
h_subs = 0.25      # m
L_subs = 0.5       # m
k_1_subs = 100    # N.m/rad
k_2_subs = 200    # N.m/rad
Delta_subs = Delta.subs({E: E_subs, I_c: I_c_subs, h: h_subs, k1: k_1_subs, k2: k_2_subs})
K_frame_h.subs({E: E_subs, I_c: I_c_subs, I_b: I_b_subs, h: h_subs, L: L_subs, k1: k_1_subs, k2: k_2_subs,
                Delta:Delta_subs})

# Behaviour of column lateral stiffness including beam in flexion (modelled as the half of the frame)
k_x = outils.column_lateral_stiffness_with_beam_in_flexion(E, I_c, I_b, h, L, k1, k2)
k_x.subs({E: E_subs, I_c: I_c_subs, I_b: I_b_subs, h: h_subs, L: L_subs, k1: k_1_subs, k2: k_2_subs})


Ic_x, I_b_2, L_2 = sym.symbols('Ic_x I_b_2 L_2', positive=True, finite=True)
k_y = outils.column_lateral_stiffness_with_beam_in_flexion(E, Ic_x, I_b_2, h, L_2, k1, k2)

K = outils.stiffness_matrix_level_beam_in_flexion(E=E, Ic_y=I_c, Ic_x=Ic_x, Lc=h, Ib_1_y=I_b, Ib_2_y=I_b_2, kx1=k1, kx2=k2, ky1=k1, ky2=k2, B=L, H=L_2)

I_c_x_subs = 1/12 * 0.002 * 0.03**3
I_b_2_subs = 1/12 * 0.01 * 0.001**3
L_2_subs = 0.25
K.subs({E: E_subs, I_c: I_c_subs, Ic_x: I_c_x_subs, h: h_subs,
        I_b: I_b_subs, I_b_2: I_b_2_subs,
        L: L_subs, L_2: L_2_subs,
        k1: k_1_subs, k2: k_2_subs})
