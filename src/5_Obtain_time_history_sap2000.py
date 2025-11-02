
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import json
import numpy as np
import os
import warnings

import helpers.outils as outils
import helpers.sap2000 as sap2000

"""
FILE DUTIES:

Gets the strain time history data in some specific points of a SAP2000 model
    and stores it in a txt file.

Remark:
    Enhanced version of OMA_1_generate_timehistory.py, which generated tests 1, 2 and 3
    for the TFM

Required files:
    sensor_numbering_AVT_sg.json: containing the channels in which we want to record measurement
        (obtained with the file "OMA_4_1_read_geometry_sg.py")
    SAP2000.sdb file: it must contain a load case called loadcase_name (e.g., "simulated_noise")
        which contains the time history of the ground accelerations
    Remark: the ground accelerations have been generated as white noise with a code stored
        in the Draft Codes project
"""
# 0. Variables and Paths
average_values, round_coordinates = True, True


output_filename = 'test4.txt'  # output filename
sapfile_name = 'DA_uncalibrated_v7_updated_with_TH.sdb'

output_path = os.path.join('src', 'output')
sap2000_model_path = os.path.join('src', 'sap2000')


log_filepath = outils.prepare_log_folder(
    sap2000_model_path, sapfile_name)

loadcase_name = 'simulated_noise'
filename_sensors = 'sensor_numbering_AVT_sg.json'  # output file name
username = outils.get_username()
paths = outils.get_paths(os.path.join('src', 'paths', username + '.csv'))

datapath = paths['rawdata_footbridge_simulated_oma']
if output_filename in os.listdir(datapath):
    old_filename = str(output_filename)
    while output_filename in os.listdir(datapath):
        output_filename = outils.increment_filename(output_filename)
    warnings.warn(
        f'{old_filename} already exists. File is renamed to: {output_filename}')


# 1) Get the SAP2000 elements
FilePath = os.path.join(paths['sap2000_footbridge_model'], sapfile_name)
mySapObject = sap2000.app_start()
SapModel = sap2000.open_file(mySapObject, FilePath)
sap2000.unlock_model(SapModel)

Name_points_group, Name_elements_group = "ALL", "ALL"
all_points, all_elements, all_elements_stat = sap2000.getnames_point_elements(Name_points_group, Name_elements_group,
                                                                              SapModel)
all_points_coord = sap2000.get_pointcoordinates(
    all_points, SapModel, round_coordinates=round_coordinates)
all_elements_coord_connect = sap2000.get_frameconnectivity(all_points, all_elements,
                                                           SapModel, all_points_coord=all_points_coord)

# 2. Get the corresondance between physical strain gauges and FEM elements
filename = 'sensor_numbering_AVT_sg.json'
FilePath = os.path.join(paths['project'], 'src',
                        'oma_results_footbridge', filename)
with open(FilePath) as json_file:
    sensors_coordinates = json.load(json_file)

sensors_nodes_correspondance = outils.get_sg_elements_correspondance(
    sensors_coordinates, all_elements_coord_connect)

# 3. Run the model and get the strain time history data
sap2000.run_analysis(SapModel)
Name_elements_group = 'modeshape_frames'
selected_elements = [sensors_nodes_correspondance[i]['Element']
                     for i in sensors_nodes_correspondance]
modal_forces, step_time = sap2000.get_modalforces_timehistory(Name_elements_group, SapModel, loadcase_name,
                                                              average_values=average_values,
                                                              round_coordinates=round_coordinates,
                                                              selected_elements=selected_elements)
element_section = sap2000.get_elementsections(
    all_elements, all_elements_stat, SapModel)

all_sections = list(set([element_section[i] for i in list(element_section)]))
section_properties_material = sap2000.get_section_information(all_sections,
                                                              SapModel)
strain_timehistory = outils.get_strain_timehistory(
    modal_forces, element_section, section_properties_material)

# 4. Get the strains in the specific points of the elements in which gauges are located
df = pd.DataFrame(columns=list(sensors_nodes_correspondance))
for channel, data in sensors_nodes_correspondance.items():
    x_sg = data['x_sg']
    element_label = f"Element_{data['Element']}"
    location = data['location']
    if location == 'right':
        data_strain = np.array(
            strain_timehistory[element_label]['epsilon_1_2_right'])
    elif location == 'left':
        data_strain = np.array(
            strain_timehistory[element_label]['epsilon_1_2_left'])
    elif location == 'up':
        data_strain = np.array(
            strain_timehistory[element_label]['epsilon_1_3_up'])
    elif location == 'down':
        data_strain = np.array(
            strain_timehistory[element_label]['epsilon_1_3_down'])
    x_stations = np.array(strain_timehistory[element_label]['x'])
    eps_values = np.array([
        sp.interpolate.interp1d(
            x_stations, data_strain[:, i], kind='linear', fill_value='extrapolate')(x_sg)
        for i in range(data_strain.shape[1])
    ])
    df[channel] = eps_values

# 5. Save the data
df.to_csv(os.path.join(datapath, output_filename), sep=' ', index=False, header=False)

metadata_filename = output_filename.replace('.txt', '_metadata.json')
metadata = dict()
metadata['file'] = 'simulated_strain_timehistory'
metadata['loadcase'] = loadcase_name
metadata['sapfile'] = sapfile_name
metadata['fs'] = 1/np.mean(np.diff(step_time))
metadata['sg'] = sensors_nodes_correspondance

with open(os.path.join(datapath, metadata_filename), 'w') as f:
    json.dump(metadata, f)
