
import numpy as np

import helpers.outils as outils
import helpers.sap2000 as sap2000


"""
Este archivo estaría por analizar; habría que compararlo con el "delete.json" y con 
el fichero model_2_frame_2_levels.sdb de SAP2000 para ver si los resultados coinciden

MATRIZ DE RIGIDEZ:
No lo hacemos ya que realmente no queremos la matriz de rigidez, sino la matriz de rigidez
condensada; si la obtenemos de SAP2000, nos acaba saliendo singular

Por no perder más tiempo -> Directamente en el modelo de SAP2000 se definen 3 load patterns
y se comprueba que la matriz K obtenida está bien

La matriz de masa no me ha termiando de salir, pero confío en que está bien (en el caso
de rigid beam me salía, y hoy ya no puedo perder más tiempo)
"""
sapfile_name = "model_2_frame_2_levels.sdb"
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

# K, Phi_id_K_M, _ = outils.prune_K_and_Phi(K, Phi_id_K_M=Phi_id_K_M)

id_DIAPH1 = [Phi_id_K_M.index('CDIAPH1_U1')]
id_DIAPH2 = [Phi_id_K_M.index('CDIAPH1_U2')]
id_DIAPH3 = [Phi_id_K_M.index('CDIAPH1_R3')]

# id_DIAPH1 = [Phi_id_K_M.index('9_U1')]
# id_DIAPH2 = [Phi_id_K_M.index('9_U2')]
# id_DIAPH3 = [Phi_id_K_M.index('9_R3')]

# id_slaves = [i for i in range(np.shape(K)[0]) if i not in id_DIAPH1 + id_DIAPH2 + id_DIAPH3]
# K_rigid = outils.static_condensation(K, id_slaves)
# id_DIAPH1 = [Phi_id_K_M.index('CDIAPH1_U1') ,Phi_id_K_M.index('CDIAPH2_U1')]
# id_DIAPH2 = [Phi_id_K_M.index('CDIAPH1_U2') ,Phi_id_K_M.index('CDIAPH2_U2')]
# id_DIAPH3 = [Phi_id_K_M.index('CDIAPH1_R3') ,Phi_id_K_M.index('CDIAPH2_R3')]

K_rigid = K[np.ix_(id_DIAPH1 + id_DIAPH2 + id_DIAPH3, id_DIAPH1 + id_DIAPH2 + id_DIAPH3)]
M_rigid = M[np.ix_(id_DIAPH1 + id_DIAPH2 + id_DIAPH3, id_DIAPH1 + id_DIAPH2 + id_DIAPH3)]
# K_rigid = K[np.ix_(id_DIAPH1 + id_DIAPH2 + id_DIAPH3, id_DIAPH1 + id_DIAPH2 + id_DIAPH3)]
# M_rigid = M[np.ix_(id_DIAPH1 + id_DIAPH2 + id_DIAPH3, id_DIAPH1 + id_DIAPH2 + id_DIAPH3)]

print(K_rigid[0:3,0:3])
print(K_rigid[0:3,3:6])
print(K_rigid[3:6,3:6])
print(M_rigid[0:3,0:3])
print(M_rigid[0:3,3:6])
print(M_rigid[3:6,3:6])

# id_DIAPH = [16, 17, 18]

# f, phi = outils.generalized_eig_singular(M, K)
# f2, phi2 = outils.generalized_eig_singular(M_level, K_level)

# id_U2 = [1, 5]
# K_U2 = K[np.ix_(id_U2, id_U2)]
# K_i = outils.beam_stiffness_np(E, I2, L_c)
# K = outils.reorder_K_M_matrix_to_phi_id(
#     stiffness_matrix, ['2_U2', '2_R3'], joints_matrix_index)

# K_beam_weak = outils.beam_stiffness_np(E, I2, L_c)
# K_beam_strong = outils.beam_stiffness_np(E, I1, L_c)

# K_beam_weak = K_beam_weak[np.ix_([2, 3], [2, 3])]
# K_beam_strong = K_beam_strong[np.ix_([2, 3], [2, 3])]

# id_DIAPH = [0,1,17]
# id_DIAPH = [Phi_id_K_M.index('9_U1') ,Phi_id_K_M.index('9_U2'),Phi_id_K_M.index('9_R3')]
# id_DIAPH1 = [Phi_id_K_M.index('CDIAPH1_U1') ,Phi_id_K_M.index('CDIAPH1_U2'),Phi_id_K_M.index('CDIAPH1_R3')]
# id_DIAPH2 = [Phi_id_K_M.index('CDIAPH2_U1') ,Phi_id_K_M.index('CDIAPH2_U2'),Phi_id_K_M.index('CDIAPH2_R3')]
# print(K[np.ix_(id_DIAPH, id_DIAPH)])
# print(K_level)
# print(M[np.ix_(id_DIAPH, id_DIAPH)])
# print(M_level)
# print(K_beam_weak)
# print(K_beam_strong)

# output = SapModel.ConstraintDef.GetDiaphragm("DIAPH1")

# K[:, 0] + K[:, 1] + K[:, 2] + K[:, 3]
# K[:, 4] + K[:, 5] + K[:, 6] + K[:, 7]

# 0.141038