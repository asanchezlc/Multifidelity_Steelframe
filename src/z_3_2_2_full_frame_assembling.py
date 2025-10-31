
import json
import numpy as np
import os

from helpers.classes import StoryLevelRigidUnions
import helpers.outils as outils

"""
File Duties:

SIMPLIFIED VERSION: RIGID UNION COLUMN-BEAM WITH RIGID BEAM
"""
filename_json = 'steelframe_rigid_X_Y.json'
input_path = os.path.join('src', 'data')
output_path = os.path.join('src', 'output')

# LOAD DATA
with open(os.path.join('src', 'data', filename_json), "r") as f:
    CONFIG = json.load(f)

n_floors = int(CONFIG["n_floors"])
base = CONFIG["base"]
overrides = CONFIG.get("levels", {})

# OBTAIN K, M FROM ASSEMBLED STORY LEVELS
levels = list()
for i in range(1, n_floors + 1):

    p = dict(base)
    p.update(overrides.get(str(i), {}))

    levels.append(
        StoryLevelRigidUnions(
            E=p["E"], rho=p["rho"],
            B=p["B"], H=p["H"], t=p["t"],
            Lc=p["Lc"],
            Ac=p["Ac"], Ic_y=p["Ic_y"], Ic_x=p["Ic_x"],
            A_b_B=p["A_b_B"], A_b_H=p["A_b_H"],
            is_highest_level=(i == n_floors),
            m_center_extra=p.get("m_center_extra", 0.0),
            m_corner_extra=p.get("m_corner_extra", 0.0),
        )
    )

K, M, dofs = outils.assemble_interstory_global_from_levels(levels)

base_name = filename_json.replace('.json', '')
np.savetxt(os.path.join(output_path, f'{base_name}_K.txt'), K, fmt="%.6e", delimiter="\t")
np.savetxt(os.path.join(output_path, f'{base_name}_M.txt'), M, fmt="%.6e", delimiter="\t")
np.savetxt(os.path.join(output_path, f'{base_name}_dofs.txt'), dofs, fmt="%s")

a, b = outils.generalized_eig_singular(M, K)