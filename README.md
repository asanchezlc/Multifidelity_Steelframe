# Synthetic OMA Signal Generator for a 3D Steel Frame

This project generates **synthetic acceleration time histories** at the locations where, in a real OMA campaign, accelerometers would be installed. It does this in two main steps:

1. **Build the structural model** (global stiffness `K`, mass `M`, and DOF labels) from a floor-by-floor description.
2. **Simulate the time response** of that model using a state–space formulation excited by white noise, and **re-map** the DOF accelerations to the real sensor locations defined in a MOVA-style geometry file.

The core scripts are:

1. `1_Obtain_K_M_matrices.py`
2. `2_Obtain_state_space.py`

There are also several **experimental / legacy** attempts (`z_...`) that can be ignored for the normal workflow but are kept in the repo for reference.

---

## 1. Folder Structure

```text
src/
 ├── data/
 │    ├── steelframe_rigid_X_cantilever_Y.json      ← main frame definition
 │    ├── geometry_OMA_Setup_1.txt                   ← sensor layout (MOVA style)
 │    └── z_* , model_* .sdb, ... (legacy attempts)
 ├── output/
 │    ├── steelframe_rigid_X_cantilever_Y_K.txt      ← global K
 │    ├── steelframe_rigid_X_cantilever_Y_M.txt      ← global M
 │    ├── steelframe_rigid_X_cantilever_Y_dofs.txt   ← DOF labels (e.g. `1_U1`, `3_R3`, …)
 │    └── steelframe_rigid_X_cantilever_Y_state_space.txt  ← synthetic accelerations per OMA channel
 ├── helpers/
 │    ├── outils.py
 │    └── classes.py
 ├── 1_Obtain_K_M_matrices.py
 └── 2_Obtain_state_space.py
