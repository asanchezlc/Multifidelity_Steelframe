
import os
import numpy as np

import helpers.outils as outils
import helpers.sap2000 as sap2000


"""
ESTE ARCHIVO ESTÁ HECHO PARA OBTENER DIRECTAMENTE LA MATRIZ DE RIGIDEZ DESDE SAP2000
ES DECIR; TOMAMOS EL MODELO DEL PÓRTICO, LEEMOS SU MATRIZ DE RIGIDEZ, CONDENSAMOS A
15 DOFS, Y LISTO;

LA MATRIZ DE MASA LA OBTUVIMOS DEL CÓDIGO ANTERIOR (2_2_2_full_frame_assembling.py)

NOTA:
Faltaría, por hacerlo fino, haber obtenido la matriz de rigidez con 2_2_2_full_frame_assembling,
pero para ello necesitamos obtener la matriz de rigidez de un elemento con elementos más
gruesos en los extremos (que es lo que realmente tiene el modelo SAP2000).
"""
sapfile_name = "model_1.sdb"
path = r"C:\Users\User\Documents\DOCTORADO\Multifidelity_Steelframe\sap2000\steelframe"

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

Phi_id_masters = ['37_U1', '40_U1', '43_U1', '46_U1', '34_U1',
                  '37_U2', '40_U2', '43_U2', '46_U2', '34_U2',
                  '37_R3', '40_R3', '43_R3', '46_R3', '34_R3']

id_DIAPH1 = [Phi_id_K_M.index(m) for m in Phi_id_masters]

id_slaves = [i for i, _ in enumerate(Phi_id_K_M) if i not in id_DIAPH1]
id_masters = [i for i in range(np.shape(K)[0]) if i not in id_slaves]
K_reduced, id_masters_reduced = outils.static_condensation(K, id_slaves)
Phi_id_K_M_reduced = [Phi_id_K_M[i] for i in id_masters_reduced]

id_order = [Phi_id_K_M_reduced.index(node) for node in Phi_id_masters]
K_reordered = K_reduced[np.ix_(id_order, id_order)]
Phi_id_K_M_reordered = [Phi_id_K_M_reduced[i] for i in id_order]

K_reordered = 1000 * K_reordered  # to get N/m from kN/m

output_path = os.path.join('src', 'output')
base_name = sapfile_name.replace('.sdb', '')
np.savetxt(os.path.join(output_path, f'{base_name}_K.txt'), K_reordered, fmt="%.6e", delimiter="\t")

M_filename = 'steelframe_calibrated_v7_M.txt'
M = np.loadtxt(os.path.join(output_path, M_filename), delimiter="\t")
a, b = outils.generalized_eig_singular(M, K_reordered)

np.savetxt(os.path.join(output_path, f'{base_name}_M_from_manual_calculation.txt'), M, fmt="%.6e", delimiter="\t")

K_test = np.loadtxt(os.path.join(output_path, f'{base_name}_K.txt'), delimiter="\t")
M_test = np.loadtxt(os.path.join(output_path, f'{base_name}_M_from_manual_calculation.txt'), delimiter="\t")

a_test, b_test = outils.generalized_eig_singular(M_test, K_test)