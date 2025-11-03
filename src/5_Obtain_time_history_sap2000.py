

import pandas as pd
import scipy as sp
import json
import numpy as np
import os
import warnings

import helpers.outils as outils
import helpers.sap2000 as sap2000

"""
File Duties:

Retrieve time history accelerations from SAP2000 model for a given load case
and save them in a text file.
"""
# 0. SAP2000 variables
sapfile_name = 'delete.sdb'
# sapfile_name = 'DA_uncalibrated_v7_updated_with_TH.sdb'
loadcase_name = 'test_TH'  # name of the load case in SAP2000
# loadcase_name = 'TimeHistory'  # name of the load case in SAP2000
load_pattern_channels = "DOFs"  # load pattern used to define the accelerometers in SAP2000
round_timesteps, round_coordinates = True, True

# Paths
output_path = os.path.join('src', 'output')
sap2000_model_path = os.path.join('src', 'sap2000')
sapfile_path = os.path.join(sap2000_model_path, sapfile_name)
output_filename = sapfile_name.replace('.sdb', f'_{loadcase_name}_accelerations_1.txt')

# Update output filename if it already exists
if output_filename in os.listdir(output_path):
    old_filename = str(output_filename)
    while output_filename in os.listdir(output_path):
        output_filename = outils.increment_filename(output_filename)
    warnings.warn(
        f'{old_filename} already exists. File is renamed to: {output_filename}')


# 1) Open SAP2000 model and get the channels corresponding to the accelerometers
mySapObject = sap2000.app_start(use_GUI=True)
SapModel = sap2000.open_file(mySapObject, sapfile_path)
sap2000.unlock_model(SapModel)

# Retrieve the channels
forces_setup = sap2000.get_point_forces(
    'ALL', SapModel, load_pattern=load_pattern_channels,
    return_kN=True)
sap2000.check_local_axes_channels(SapModel, list(set(forces_setup['PointObj'])))
acc_channels = outils.get_accelerometer_channels_from_forces(
    forces_setup)

# 2) Get the time history accelerations
time_history, t = outils.get_channels_time_history_accelerations(SapModel, load_case=loadcase_name, channels=acc_channels,
                                                                 round_timesteps=round_timesteps, rel_acceleration=True)

fs = 1/np.mean(np.diff(t))  # Hz
accelerations = np.zeros((len(t), len(acc_channels)))
for i, dof in enumerate(time_history):
    accelerations[:, i] = np.array(time_history[dof])

np.savetxt(os.path.join(output_path, output_filename), accelerations, delimiter="\t")

output_details = {
    'sapfile_name': sapfile_name,
    'loadcase_name': loadcase_name,
    'fs': fs,
    'channels': acc_channels
}

with open(os.path.join(output_path, output_filename.replace('.txt', '_details.json')), 'w') as f:
    json.dump(output_details, f, indent=4)

sap2000.application_exit(mySapObject)