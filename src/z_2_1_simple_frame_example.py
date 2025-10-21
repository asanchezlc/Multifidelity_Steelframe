
import numpy as np
import os

import helpers.outils as outils
import helpers.sap2000 as sap2000

"""
File properties:

Computes the mass and stiffness matrices for a 3D frame (4 columns and 2 rigid beams)
and compares them with the matrices extracted from a SAP2000 model.

Next steps:
2) Assemble for multiple heights levels
3) State space equations

Cambiar t, level y m_motor
"""
# Material properties
E = 210e9                 # N/m2
rho = 7849.0476           # kg/m3

# Geometric properties
L_c = 0.25                # m
H = 0.25                  # m
B = 0.50                  # m

# Partial stiffnesses
ky1 = 1000                # N·m/rad (affects Ux -> uses I2)
ky2 = 5000                # N·m/rad (affects Ux -> uses I2)
kx1 = 2000                # N·m/rad (affects Uy -> uses I1)
kx2 = 8000                # N·m/rad (affects Uy -> uses I1)

# Section properties
b_mm = 2  # mm
h_mm = 30  # mm
A_b = 200 * 1e-6  # m2

# Other properties
t = 2e-3  # m
level = 5  # level number (from 1 to 5)
m_motor = 100/9.80665             # kg at center

# Derived properties
A = b_mm * h_mm * 1e-6  # m^2
I1 = b_mm * h_mm**3 / 12.0 * 1e-12  # m^4
I2 = h_mm * b_mm**3 / 12.0 * 1e-12  # m^4
L_b = B

# Mass computation
m_corner_b = outils.lumped_mass_from_beam(rho, A_b, L_b)
if level != 5:
    m_corner_c = 2 * outils.lumped_mass_from_beam(rho, A, L_c)
    m_center = 0
else:
    m_corner_c = outils.lumped_mass_from_beam(rho, A, L_c)
    m_center = m_motor
m_corner = m_corner_b + m_corner_c

# --- Build K_level using your function ---
K_level = outils.stiffness_matrix_level(
    E, I1, I2, L_c, kx1, kx2, ky1, ky2, B, H)
M_level = outils.mass_matrix_level(
    rho, t, B, H, m_corner=m_corner, m_center=m_center)

np.set_printoptions(precision=6, suppress=True)
print("\nK_level [Ux, Uy, Thetaz] in N/m, N/m, N·m/rad units respectively:")
print(K_level)
print("\nM_level [Ux, Uy, Thetaz] in kg, kg, kg·m² units respectively:")
print(M_level)












# SAP2000 CHECK (EVERYTHING WORKS FINE!)
# --- Read SAP2000 model and extract M, K, modal properties ---
# sapfile_name = "toy_beam_N_m.sdb"
sapfile_name = "model_1_frame_2_levels.sdb"
path = r"C:\Users\User\Documents\DOCTORADO\Multifidelity_Steelframe\sap2000"

path_log = outils.prepare_model(path, sapfile_name)
SapObject, SapModel = outils.open_SAP2000(
    path_log, sapfile_name)
sap2000.set_ISunits(SapModel)

frequencies, Phi, Phi_id = outils.get_modal_properties(
    SapModel)

full_modal_data = {
    'frequencies': frequencies,
    'Phi': Phi,
    'Phi_id': Phi_id
}
mass_matrix, stiffness_matrix, joints_matrix_index, constraints = outils.read_M_K_constraints_from_TX_files(
    sapfile_name, path_log)


Phi_upd, Phi_id_upd = outils.update_Phi_with_constraints(  # function checked - OK
    full_modal_data['Phi'], full_modal_data['Phi_id'], constraints)

outils.check_Phi_id_coverage(Phi_id_upd, joints_matrix_index)

Phi_K_M, Phi_id_K_M = outils.clean_phi_matrix(
    Phi_upd, Phi_id_upd, joints_matrix_index)
K = outils.reorder_K_M_matrix_to_phi_id(
    stiffness_matrix, Phi_id_K_M, joints_matrix_index)
M = outils.reorder_K_M_matrix_to_phi_id(
    mass_matrix, Phi_id_K_M, joints_matrix_index)

id_DIAPH = [16, 17, 18]

f, phi = outils.generalized_eig_singular(M, K)
f2, phi2 = outils.generalized_eig_singular(M_level, K_level)



# id_U2 = [1, 5]
# K_U2 = K[np.ix_(id_U2, id_U2)]
# K_i = outils.beam_stiffness(E, I2, L_c)
# K = outils.reorder_K_M_matrix_to_phi_id(
#     stiffness_matrix, ['2_U2', '2_R3'], joints_matrix_index)

K_beam_weak = outils.beam_stiffness(E, I2, L_c)
K_beam_strong = outils.beam_stiffness(E, I1, L_c)

K_beam_weak = K_beam_weak[np.ix_([2, 3], [2, 3])]
K_beam_strong = K_beam_strong[np.ix_([2, 3], [2, 3])]

id_BODY1 = [Phi_id_K_M.index('CBODY1_U1') ,Phi_id_K_M.index('CBODY2_U1')]
id_BODY2 = [Phi_id_K_M.index('CBODY1_U2') ,Phi_id_K_M.index('CBODY2_U2')]
id_BODY3 = [Phi_id_K_M.index('CBODY1_R3') ,Phi_id_K_M.index('CBODY2_R3')]

# id_BODY = [0,1,17]
# id_BODY = [Phi_id_K_M.index('9_U1') ,Phi_id_K_M.index('9_U2'),Phi_id_K_M.index('9_R3')]
# id_BODY1 = [Phi_id_K_M.index('CBODY1_U1') ,Phi_id_K_M.index('CBODY1_U2'),Phi_id_K_M.index('CBODY1_R3')]
# id_BODY2 = [Phi_id_K_M.index('CBODY2_U1') ,Phi_id_K_M.index('CBODY2_U2'),Phi_id_K_M.index('CBODY2_R3')]
print(K[np.ix_(id_DIAPH, id_DIAPH)])
print(K_level)
print(M[np.ix_(id_DIAPH, id_DIAPH)])
print(M_level)
print(K_beam_weak)
print(K_beam_strong)

output = SapModel.ConstraintDef.GetDiaphragm("DIAPH1")

K[:, 0] + K[:, 1] + K[:, 2] + K[:, 3]
K[:, 4] + K[:, 5] + K[:, 6] + K[:, 7]

0.141038
