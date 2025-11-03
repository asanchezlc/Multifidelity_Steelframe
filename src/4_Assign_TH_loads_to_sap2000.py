

import helpers.sap2000 as sap2000
import helpers.outils as outils

import os

"""
File Duties

Define load case and assign time history loads to SAP2000 model

Remark:
    - First, the forces' functions must be generated (with 3_Forces_for_time_history.py)
    and stored in "src/sap2000/functions"
"""

sapfile_name = 'DA_uncalibrated_v7_updated.sdb'

# instead of all points; we do this after seen sap2000 gets stuck with many points
points_to_assign_dict = {
    "1": {"x": 0.0, "y": 0.0,   "z": 0.25},
    "2": {"x": 0.0, "y": 0.25,  "z": 0.5},
    "3": {"x": 0.0, "y": 0.0,   "z": 0.75},
    "4": {"x": 0.5, "y": 0.0,   "z": 1.0},
    "5": {"x": 0.5, "y": 0.25,  "z": 1.25},
}

# TIME HISTORY CONFIGURATION
xi = 0.02  # same as in lumped model
fs = 207  # [Hz] Sampling frequency
factor_dt = 10  # 1/factor_dt is the factor to increase the time step
T = 15*60  # [sec] Period of the time series (15 minutes)
load_case_name = 'TH_Loads'  # Name of the time history load case to be created

path_functions = r"C:\Users\User\Documents\DOCTORADO_CODES\Multifidelity_Steelframe\src\sap2000\functions"
files = os.listdir(path_functions)
sap2000_model_path = os.path.join('src', 'sap2000')
sapfile_name_out = sapfile_name.split('.sdb')[0] + '_with_TH.sdb'

# 1) Get the SAP2000 elements
FilePath = os.path.join(sap2000_model_path, sapfile_name)
mySapObject = sap2000.app_start()
SapModel = sap2000.open_file(mySapObject, FilePath)
sap2000.unlock_model(SapModel)
round_coordinates = True

Name_points_group, Name_elements_group = "ALL", "ALL"
all_points, all_elements, all_elements_stat = sap2000.getnames_point_elements(Name_points_group, Name_elements_group,
                                                                              SapModel)
all_points_coord = sap2000.get_pointcoordinates(
    all_points, SapModel, round_coordinates=round_coordinates)
all_elements_coord_connect = sap2000.get_frameconnectivity(all_points, all_elements,
                                                           SapModel, all_points_coord=all_points_coord)

# Create load pattern and functions
loads = list()
for file in files:
    # Load pattern and function name
    load_pattern = file.split('.')[0]
    function_name = load_pattern

    # Load values
    forces = [0]*6  # FX, FY, FZ, MX, MY, MZ
    force_label = file.split('_')[-1].split('.')[0]  # FZ.txt -> FZ
    forces[['FX', 'FY', 'FZ', 'MX', 'MY', 'MZ'].index(force_label)] = 1.0

    # Point object
    x, y, z = outils.parse_xyz(file)
    # point_name = outils.find_point_by_coord(all_points_coord, x, y, z)
    point_name = outils.find_point_by_coord(points_to_assign_dict, x, y, z)
    if point_name is None or force_label not in ['FX', 'FY']:
        continue

    # Def load pattern and assign unity force
    sap2000.add_load_pattern(SapModel, name=load_pattern)
    sap2000.set_point_force_load(SapModel, name=point_name, load_pattern=load_pattern,
                                 values=forces, replace=True)

    # Define function for the load
    sap2000.set_time_history_function_from_file(SapModel,
                                                name=function_name,
                                                file_name=os.path.join(path_functions, file),
                                                head_lines=0,
                                                pre_chars=0,
                                                points_per_line=1,
                                                value_type=2,
                                                free_format=True)
    load_i = {
        'type': 'Load',
        'name': load_pattern,
        'func': function_name,
        'sf': 1.0,
    }
    loads.append(load_i)

# Define time history load case
dt = 1/(factor_dt * fs)  # [sec] time resolution
N = int(T/dt)  # number of data points
sap2000.create_linear_modal_history_case(SapModel, name='TimeHistory', damping=xi, n_steps=N,
                                         dt=dt, modal_case="MODAL", loads=loads)
sap2000.save_model(SapModel, os.path.join(sap2000_model_path, sapfile_name_out))
sap2000.application_exit(mySapObject)