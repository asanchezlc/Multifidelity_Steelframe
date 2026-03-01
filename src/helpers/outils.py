
import ast
import copy
import json
import re
import os
import shutil
import psutil
import warnings

import scipy as sp
import numpy as np
import pandas as pd
import sympy as sym
import h5py

import helpers.sap2000 as sap2000

from dotenv import load_dotenv
from scipy.linalg import eigh
from typing import Any, List, Sequence

from collections import defaultdict
from datetime import datetime

from SALib.sample.sobol import sample


# Copied-pasted functions from TFM_codes

def build_Phi_v2(sensors_nodes_correspondance, disp_modeshapes,
                 return_positive_coordinates=True):
    """
    Builds Phi and Phi_id matrices for the given sensors and mode shapes.

    Parameters:
    ----------
    sensors_nodes_correspondance : dict
        Dictionary mapping sensor channel names to structural metadata.
        Example:
            {
                'Channel_1': {
                    'point': '5',
                    'dir': [-1, 0, 0],        # direction vector
                    'direction': '-U1' ...
                },
                ...
            }

    disp_modeshapes : dict
        Dictionary containing modal displacement data from SAP2000.
        Format:
            {
                'Mode_1': {
                    'U1': [...], 'U2': [...], 'U3': [...],
                    'R1': [...], 'R2': [...], 'R3': [...],
                    'Joint_id': [<joint_id_1>, <joint_id_2>, ...]
                },
                ...
            }

    return_positive_coordinates : bool, default=True
        If True, absolute values of direction vectors and positive direction labels are used
        (i.e., '-U1' becomes 'U1' in Phi_id and direction vector becomes [1,0,0]).

    Returns:
    -------
    Phi : np.ndarray
        Array of shape (n_channels, n_modes), containing mode shape projections for each sensor.

    Phi_id : list of str
        Identifiers for each sensor, formatted as "<JointID>_<direction>".
        These follow the order of channels and correspond to the rows of Phi.
    """
    n_channels = len(sensors_nodes_correspondance)
    n_modes = len(disp_modeshapes)
    Phi = np.zeros((n_channels, n_modes))
    Phi_id = list()
    for i_mode, mode in enumerate(disp_modeshapes):
        mode_data = disp_modeshapes[mode]
        joint_ids = mode_data['Joint_id']

        for i_channel, channel in enumerate(sensors_nodes_correspondance):
            channel_data = sensors_nodes_correspondance[channel]
            point = channel_data['point']
            dir_vec = channel_data['dir']
            direction = channel_data['direction']
            if return_positive_coordinates == True:
                # Get absolute value of the direction
                dir_vec = np.abs(dir_vec)
                direction = direction.replace('-', '')  # Remove negative sign
            Phi_id.append(
                f"{point}_{direction}") if i_mode == 0 else None  # Add once

            try:
                j = joint_ids.index(point)  # index of joint in this mode
            except ValueError:
                raise ValueError(
                    f"Point {point} not found in Joint_id for {mode}")

            # Get displacement components
            u = np.array([mode_data['U1'][j], mode_data['U2']
                         [j], mode_data['U3'][j]])
            projection = np.dot(dir_vec, u)
            Phi[i_channel, i_mode] = projection

    return Phi, Phi_id


def get_modal_response_SA(input_data, sg_channels, acc_channels,
                          element_section, section_properties_material, SapModel,
                          groups_dict=None, n_modes=None):
    """
    IMPORTANT!!!!
    In this case we are not updating properties of the sections neither
    Young modulus; consequently, 'section_properties_material' can be
    introduced as an input.
    If we updated it, then it should be recalculated within this function

    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        strain mode shapes

    Input:
        input_data: dictionary containing the parameters to be updated to each
            group of elements

        sg_channels and acc_channels: dictionaries containing the locations
            of sgs and accs

        element_section: dict
            Contains the section assigned to each element

        section_properties_material: dict
            Contains the material properties of each section
            (e.g., Young's modulus, Area, etc.; properties important to retrieve strain)

        n_modes : int, optional
            Number of modes to compute (used with modal cases).

    Remark:
        Specifically for the SA (Sensitivity Analysis)
    """
    # A) Unlock model and set IS units
    sap2000.unlock_model(SapModel)
    sap2000.set_ISunits(SapModel)

    # B) Set new material properties
    if 'MP' in input_data:
        material_dict = input_data['MP']
        sap2000.set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'JS' in input_data:
        group = groups_dict.get('JS') if groups_dict else True
        joint_dict = input_data['JS']
        sap2000.set_jointsprings(joint_dict, SapModel, group=group)

    # D) Modify partial fixity
    if 'FR' in input_data:
        group = groups_dict.get('FR') if groups_dict else True
        frame_releases_dict = input_data['FR']
        sap2000.set_framereleases(frame_releases_dict, SapModel, group=group)

    # E) Modify frame section
    if 'FS' in input_data:
        frame_dict = input_data['FS']
        sap2000.set_frame_property(frame_dict, SapModel)

    if 'FOM' in input_data:
        group = groups_dict.get('FOM') if groups_dict else True
        frame_obj_modifiers = input_data['FOM']
        sap2000.set_frame_obj_modifiers(frame_obj_modifiers, SapModel, group=group)

    # F) Modify area section
    if 'AS' in input_data:
        area_dict = input_data['AS']
        sap2000.set_areaproperty(area_dict, SapModel)

    # C) Run Analysis
    sap2000.run_analysis(SapModel, max_modes=n_modes)

    # D) Get Frequencies
    frequencies = sap2000.get_modalfrequencies(SapModel)
    frequencies = np.array([value['Frequency']
                           for key, value in frequencies.items()])

    # E) Get mode shapes in the locations given by acc_channels
    Name_points_group = "modeshape_points"
    disp_modeshapes = sap2000.get_displmodeshapes(Name_points_group, SapModel)
    Phi, Phi_id = build_Phi_v2(acc_channels, disp_modeshapes,
                               return_positive_coordinates=False)

    # F) Get section properties: Uncomment if any of those properties were modified!
    # Name_points_group, Name_elements_group = "allpoints", "allframes"
    # _, all_elements, all_elements_stat = sap2000.getnames_point_elements(Name_points_group,
    #                                                                             Name_elements_group,
    #                                                                             SapModel)

    # element_section = sap2000.get_elementsections(all_elements, all_elements_stat, SapModel)
    # all_sections = list(set([element_section[i] for i in list(element_section)]))
    # section_properties_material = sap2000.get_section_information(all_sections, SapModel)

    # Modal Forces and strain modeshapes
    Name_elements_group = 'modeshape_frames'
    modal_forces = sap2000.get_modalforces(Name_elements_group, SapModel)
    strain_modeshapes = get_strainmodeshapes(
        modal_forces, element_section, section_properties_material)
    Psi, Psi_id = build_Psi_v2(
        sg_channels, strain_modeshapes, interpolate_modeshapes=True)

    modal_results = {'frequencies': frequencies,
                     'Phi': Phi,
                     'Psi': Psi,
                     'Phi_id': Phi_id,
                     'Psi_id': Psi_id}

    return modal_results



def get_frame_property(SapModel, Name):
    """
    Attempts to retrieve frame section property data for the given Name.
    Tries different section types (I, Rectangular, Pipe, Circle, Tube, etc.)
    until one succeeds.

    Returns:
        Dictionary with property data, including flags:
        {
            'is_I': bool,
            'is_channel': bool,
            'is_tee': bool,
            'is_angle': bool,
            'is_double_angle': bool,
            'is_double_channel': bool,
            'is_pipe': bool,
            'is_tube': bool,
            ... (property fields depending on type)
        }
    """
    flags = {
        'is_I': False,
        'is_channel': False,
        'is_tee': False,
        'is_angle': False,
        'is_double_angle': False,
        'is_double_channel': False,
        'is_pipe': False,
        'is_tube': False,
        'is_rectangular': False,
        'is_SD': False,
        'is_circle': False,
        'is_steel_joist': False,
        'is_hybrid_I': False,
        'is_hybrid_U': False,
        'is_trapezoidal': False,
        'is_precastI': False,
        'is_precastU': False,
        'is_precastSuperT': False,
        'is_cold_C': False,
        'is_cold_Z': False,
        'is_cold_Box': False,
        'is_cold_I': False,
        'is_cold_L': False,
        'is_cold_T': False,
        'is_cold_Hat': False,
        'is_cold_Pipe': False,
        'is_General': False,
        'is_Nonprismatic': False,
        'is_cover_plated_I': False,
    }

    # Determine the type once
    section_type = sap2000.get_frame_section_type(SapModel, Name)
    type_name = section_type['TypeName']

    if f"is_{type_name}" not in flags:
        sap2000.raise_warning(
            f"Frame property '{Name}' has unknown type code {section_type['TypeCode']} ({type_name}).",
            1,
        )
        return {"flags": flags, "Type": section_type}

    # Mark the flag
    flags[f"is_{type_name}"] = True

    # Dispatch based on type
    output = None

    if flags['is_I']:
        output = sap2000.get_I_section(SapModel, Name)

    elif flags['is_channel']:
        output = sap2000.get_channel_section(SapModel, Name)

    elif flags['is_angle']:
        output = sap2000.get_angle_section(SapModel, Name)

    elif flags['is_double_angle']:
        output = sap2000.get_double_angle_section(SapModel, Name)

    elif flags['is_double_channel']:
        output = sap2000.get_double_channel_section(SapModel, Name)

    elif flags['is_pipe']:
        output = sap2000.get_pipe_section(SapModel, Name)

    elif flags['is_tube']:
        output = sap2000.get_tube_section(SapModel, Name)

    elif flags['is_rectangular']:
        output = sap2000.get_rectangular_section(SapModel, Name)

    elif flags['is_steel_joist']:
        # TODO: Steel joist function not available (or not found) in the OAPI
        # output = sap2000.get_steel_joist_section(SapModel, Name) !!
        pass

    elif flags['is_SD']:
        output = sap2000.get_SD_section(SapModel, Name)

    elif flags['is_circle']:
        output = sap2000.get_circle_section(SapModel, Name)

    elif flags['is_hybrid_I']:
        output = sap2000.get_hybrid_I_section(SapModel, Name)

    elif flags['is_hybrid_U']:
        # TODO
        # Hybrid section manages data differently (with an array)
        # IMPORTANT: IF ADDED, UPDATE THE allowed_properties in extract_FS_structure
        pass

    elif flags['is_trapezoidal']:
        output = sap2000.get_trapezoidal_section(SapModel, Name)

    elif flags['is_precastI']:
        # TODO
        # Precast section manages data differently
        # IMPORTANT: IF ADDED, UPDATE THE allowed_properties in extract_FS_structure
        pass

    elif flags['is_precastU']:
        # TODO
        # Precast section manages data differently
        # IMPORTANT: IF ADDED, UPDATE THE allowed_properties in extract_FS_structure
        pass

    elif flags['is_precastSuperT']:
        # TODO: Precast Super-T handling if required
        pass

    elif flags['is_cold_C']:
        output = sap2000.get_cold_C_section(SapModel, Name)

    elif flags['is_cold_Z']:
        output = sap2000.get_cold_Z_section(SapModel, Name)

    elif flags['is_cold_Box']:
        output = sap2000.get_cold_Box_section(SapModel, Name)

    elif flags['is_cold_I']:
        output = sap2000.get_cold_I_section(SapModel, Name)

    elif flags['is_cold_L']:
        output = sap2000.get_cold_L_section(SapModel, Name)

    elif flags['is_cold_T']:
        output = sap2000.get_cold_T_section(SapModel, Name)

    elif flags['is_cold_Hat']:
        output = sap2000.get_cold_Hat_section(SapModel, Name)

    elif flags['is_cold_Pipe']:
        output = sap2000.get_cold_pipe_section(SapModel, Name)

    elif flags['is_General']:
        output = sap2000.get_general_section(SapModel, Name)

    elif flags['is_Nonprismatic']:
        output = sap2000.get_nonprismatic_section(SapModel, Name)

    elif flags['is_cover_plated_I']:
        output = sap2000.get_cover_plated_I_section(SapModel, Name)

    if output is None:
        sap2000.raise_warning(
            f"Frame property '{Name}' of type '{type_name}' could not be retrieved.",
            1,
        )
        return {"flags": flags, "Type": section_type}

    # Attach flags and type info
    output.update(flags)
    output["Type"] = section_type
    return output


def build_Psi_v2(sensors_nodes_correspondance, strain_modeshapes, interpolate_modeshapes=True):
    """
    Function Duties:
        Obtains the matrix of mode shapes for the selected strain gauges
    Input:
        sensors_nodes_correspondance: dictionary with the correspondence between
            the sensors and the elements. It is like:
            {'Channel_1': {'x_sg': 0.1, 'location': 'right', 'Element': 1},
            ...
            }}
            x_sg: local x_coordinate of the strain gauge in the element
            location: location of the strain gauge in the element
            Element: SAP2000 element in which the sg is located
        strain_modeshapes: dictionary with the mode shapes coming from
            sap2000.get_strainmodeshapes
        interpolate_modeshapes: if True, the value of the strain coordinate is
            interpolated according to x_sg
    Output:
        Psi: matrix of mode shapes for the selected strain gauges (dimensions
            n_channels x n_modes)
        Psi_id: list of strings with the element mesh names and location
            (if interpolate_modeshapes is True, the value is not exactly
            the same as indicated in strain_modeshapes)
    """
    interpolate_modeshapes = True
    n_channels = len(sensors_nodes_correspondance)
    n_modes = len(strain_modeshapes)
    Psi = np.zeros((n_channels, n_modes))
    Psi_id = list()
    for channel, data in sensors_nodes_correspondance.items():
        i_channel = list(sensors_nodes_correspondance).index(channel)
        element, location, x_sg = data['Element'], data['location'], data['x_sg']
        element = f'Element_{element}'
        for i_mode, mode in enumerate(list(strain_modeshapes)):
            x_all = strain_modeshapes[mode][element]['x']
            mesh_id = strain_modeshapes[mode][element]['Mesh_id']
            if location == 'up':
                eps = strain_modeshapes[mode][element]['epsilon_1_3_up']
            elif location == 'down':
                eps = strain_modeshapes[mode][element]['epsilon_1_3_down']
            elif location == 'left':
                eps = strain_modeshapes[mode][element]['epsilon_1_2_left']
            elif location == 'right':
                eps = strain_modeshapes[mode][element]['epsilon_1_2_right']
            i = np.argmin(np.abs(np.array(x_all) - x_sg))
            if interpolate_modeshapes:
                f_x = sp.interpolate.interp1d(x_all, eps, kind='linear')
                eps_i = float(f_x(x_sg))
            else:
                eps_i = eps[i]
            Psi[i_channel, i_mode] = eps_i
            if i_mode == 0:
                Psi_id.append(f"{mesh_id[i]}_{location}")
    return Psi, Psi_id


def get_strainmodeshapes(modal_forces, element_section, section_properties_material):
    """
    Function duties:
    Computes the strain mode shapes as follow.
        At each point we have:
            Forces: P, M2, M3
            Section properties: Area, S22, S33
            Material properties: E
        Navier Formula is applied; for that, it is important to know
            what are the coordinates within the function.
    Remark:
        FOR THIS EXAMPLE, FUNCTION IS SUPPOSED TO BE RECTANGULAR
        STRESSES ARE COMPUTED AT UPPER AND DOWN CENTERED POINTS,
        AS WELL AS LEFT AND RIGHT CENTERED POINTS
    Remark II: see "SAP2000_sign_convention" in "docs" folder for understanding eps
        epsilon_up: eps in positive direction of X2; X3=0 (i.e. eps_2_pos)
        epsilon_down: eps in negative direction of X2; X3=0 (i.e. eps_2_neg)
        epsilon_right: eps in positive direction of X3; X2=0 (i.e. eps_3_pos)
        epsilon_left: eps in negative direction of X3; X2=0 (i.e. eps_3_neg)
    """
    strain_modes = dict()
    for mode_label in list(modal_forces):
        strain_modes[mode_label] = dict()
        for element_label in list(modal_forces[mode_label]):
            # Dictionary initialization
            strain_modes[mode_label][element_label] = dict()

            # Get section name
            element = element_label.replace("Element_", "")
            SectionName = element_section[element]

            # Section Properties
            Area = section_properties_material[SectionName]["Geometry"]["Area"]
            S22 = section_properties_material[SectionName]["Geometry"]["S22"]
            S33 = section_properties_material[SectionName]["Geometry"]["S33"]

            # Material Properties
            E = section_properties_material[SectionName]["Material"]["E"]

            # Forces and coordinates
            P = modal_forces[mode_label][element_label]['P']
            M3 = modal_forces[mode_label][element_label]['M3']
            M2 = modal_forces[mode_label][element_label]['M2']
            x = modal_forces[mode_label][element_label]['x']
            mesh_id = modal_forces[mode_label][element_label]['Mesh_id']

            # Stresses [!!!!! MADE FOR RECTANGULAR SECTION !!!!!!]
            sigma_1_2_right = [P[i]/Area - M2[i]/S22 for i, _ in enumerate(P)]
            sigma_1_2_left = [P[i]/Area + M2[i]/S22 for i, _ in enumerate(P)]
            sigma_1_3_up = [P[i]/Area - M3[i]/S33 for i, _ in enumerate(P)]
            sigma_1_3_down = [P[i]/Area + M3[i]/S33 for i, _ in enumerate(P)]

            # Strains
            epsilon_1_2_right = [i/E for i in sigma_1_2_right]
            epsilon_1_2_left = [i/E for i in sigma_1_2_left]
            epsilon_1_3_up = [i/E for i in sigma_1_3_up]
            epsilon_1_3_down = [i/E for i in sigma_1_3_down]

            # Save results
            strain_modes[mode_label][element_label]["x"] = x
            strain_modes[mode_label][element_label]["Mesh_id"] = mesh_id
            strain_modes[mode_label][element_label]["epsilon_1_2_right"] = epsilon_1_2_right
            strain_modes[mode_label][element_label]["epsilon_1_2_left"] = epsilon_1_2_left
            strain_modes[mode_label][element_label]["epsilon_1_3_up"] = epsilon_1_3_up
            strain_modes[mode_label][element_label]["epsilon_1_3_down"] = epsilon_1_3_down

    return strain_modes


def from_algorithm_parameters_to_sap2000_input(flat_input_data):
    """
    Parses a flat input dictionary into a structured dictionary with 'MP', 'JS', and 'FR' categories.
        'MP': Stands for Material Properties
        'JS': Stands for Joint Springs
        'FR': Stands for Frame Releases
        'FL': Stands for Frame Length
        'FS': Stands for Frame Section
        'AS': Stands for Area Section
        'FOM': Stands for Frame Object Modifiers

    Parameters
    ----------
    flat_input_data : dict
        Flat dictionary with keys like 'MP/Steel/E', 'JS/supports/U1', 'FR/group/M2/jj',
            'AS/section/Thickness', or 'FOM/1/A22', etc., and corresponding values.

    Returns
    -------
    input_data : dict
        Nested dictionary with structured input organized by 'MP', 'JS', and 'FR' sections.
    """
    input_data = {}

    # Extract material properties, joint springs, and frame releases
    input_data['MP'] = extract_MP_structure(flat_input_data)
    input_data['JS'] = extract_JS_structure(flat_input_data)
    input_data['FR'] = extract_FR_structure(flat_input_data)
    input_data['FOM'] = extract_FOM_structure(flat_input_data)
    input_data['FS'] = extract_FS_structure(flat_input_data)
    input_data['FL'] = extract_FL_structure(flat_input_data)
    input_data['AS'] = extract_AS_structure(flat_input_data)

    # Adapt JS and FR for proper input in SAP2000 functions
    input_data['JS'] = build_joint_spring_dict_from_JS(input_data['JS'])
    input_data['FR'] = build_frame_release_dict_from_FR(input_data['FR'])

    return input_data


def extract_FS_structure(input_dict, name_FS='FS'):
    """
    Extracts frame section properties from flat input keys of the form:
        'FS/<frame>/<property>' or 'FS/<frame>/<property>/<subproperty>'.

    Supports:
        - Modifiers with allowed subproperties (now combinable)
        - Direct geometrical/mechanical properties (now combinable)
        - Combined properties (concatenation of multiple allowed props)

    Parameters
    ----------
    input_dict : dict
        Flat dictionary with keys starting with 'name_FS'.

    Returns
    -------
    dict
        Nested dictionary in the format:
        {frame: {property: value or {subproperty: value}}}.
    """
    allowed_properties = [
        "t3", "t2", "tf", "tw", "t2b", "tfb",
        "Thickness", "Radius", "LipDepth", "LipAngle",
        "Area", "As2", "As3", "Torsion",
        "I22", "I33", "S22", "S33", "Z22", "Z33",
        "R22", "R33",
        "StartSec", "EndSec", "MyLength", "MyType", "EI22", "EI33",
        "FyTopFlange", "FyWeb", "FyBotFlange",   # yield strengths
        "tc", "bc", "tcb", "bcb",
    ]
    allowed_modifiers = {'A', 'A2', 'A3', 'J', 'I2', 'I3', 'M', 'W'}

    # all allowed properties are combinable
    prop_pattern = '|'.join(sorted(allowed_properties, key=len, reverse=True))
    mod_pattern = '|'.join(sorted(allowed_modifiers, key=len, reverse=True))

    fs_data = defaultdict(lambda: defaultdict(dict))

    for key, value in input_dict.items():
        parts = key.split('/')

        if parts[0] == name_FS and len(parts) in (3, 4):
            _, frame, prop, *rest = parts

            # Modifiers: FS/<section>/Modifiers/<subprop>
            if prop == 'Modifiers':
                if not rest:
                    raise ValueError(f"[FS] Missing subproperty for Modifiers in key '{key}'.")
                subprop = rest[0]

                matches = re.findall(f'({mod_pattern})', subprop)
                if matches and ''.join(matches) == subprop:
                    for m in matches:
                        fs_data[frame]['Modifiers'][m] = value
                elif subprop in allowed_modifiers:
                    fs_data[frame]['Modifiers'][subprop] = value
                else:
                    raise ValueError(
                        f"[FS] Invalid subproperty '{subprop}' for Modifiers in '{key}'. "
                        f"Allowed: {allowed_modifiers}"
                    )

            # Standard properties: FS/<section>/<prop>
            else:
                matches = re.findall(f'({prop_pattern})', prop)
                if matches and ''.join(matches) == prop:
                    for p in matches:
                        fs_data[frame][p] = value
                elif prop in allowed_properties:
                    fs_data[frame][prop] = value
                else:
                    raise ValueError(
                        f"[FS] Invalid property '{prop}' for frame '{frame}'. "
                        f"Allowed properties: {sorted(allowed_properties)}"
                    )

    return convert_to_dict(fs_data)


def extract_FOM_structure(input_dict, name_FOM='FOM'):
    """
    Extracts ONLY modifiers from flat input keys of the form:
        'FOM/<frame>/<modifier>'  (where <modifier> can be combined)

    Supports:
        - Modifiers with allowed subproperties (combinable)

    Parameters
    ----------
    input_dict : dict
        Flat dictionary with keys starting with 'name_FOM'.

    Returns
    -------
    dict
        Nested dictionary in the format:
        {frame: {'Modifiers': {modifier: value}}}.
    """
    allowed_modifiers = {'A', 'A2', 'A3', 'J', 'I2', 'I3', 'M', 'W'}
    mod_pattern = '|'.join(sorted(allowed_modifiers, key=len, reverse=True))

    fom_data = defaultdict(lambda: defaultdict(dict))

    for key, value in input_dict.items():
        parts = key.split('/')

        # Only accept: FOM/<frame>/<modifier>
        if parts[0] == name_FOM and len(parts) == 3:
            _, frame, subprop = parts

            matches = re.findall(f'({mod_pattern})', subprop)
            if matches and ''.join(matches) == subprop:
                for m in matches:
                    fom_data[frame]['Modifiers'][m] = value
            elif subprop in allowed_modifiers:
                fom_data[frame]['Modifiers'][subprop] = value
            else:
                raise ValueError(
                    f"[FOM] Invalid modifier '{subprop}' in '{key}'. "
                    f"Allowed: {allowed_modifiers}"
                )

        elif parts[0] == name_FOM:
            raise ValueError(
                f"[FOM] Invalid key format '{key}'. Expected: "
                f"'{name_FOM}/<frame>/<modifier>'."
            )

    return convert_to_dict(fom_data)


def extract_MP_structure(input_dict, name_MP='MP'):
    """
    Extracts material properties from flat input keys of the form 'MP/<material>/<property>'.

    Only allowed properties are:
        'E': modulus of elasticity
        'u': Poisson’s ratio
        'a': thermal coefficient
        'rho': density

    Parameters
    ----------
    input_dict : dict
        Flat dictionary with keys starting with 'name_MP'.

    Returns
    -------
    dict
        Nested dictionary in the format {material: {property: value}}.
    """
    allowed_properties = {'E', 'u', 'a', 'rho'}
    mp_data = defaultdict(dict)

    for key, value in input_dict.items():
        parts = key.split('/')
        if parts[0] == name_MP and len(parts) == 3:
            _, material, prop = parts
            if prop in allowed_properties:
                mp_data[material][prop] = value
            else:
                raise ValueError(
                    f"Invalid property '{prop}' for material '{material}'. "
                    f"Allowed properties are: {sorted(allowed_properties)}"
                )

    return dict(mp_data)


def extract_FL_structure(input_dict, name_FL='FL'):
    """
    Extracts frame length factor from flat input keys of the form 'FL/<group>/<ii or jj>'.

    Only allowed properties are:
        'ii': initial point moves (end point remains fixed)
        'jj': end point moves (initial point remains fixed)

    Parameters
    ----------
    input_dict : dict
        Flat dictionary with keys starting with 'name_FL'.

    Returns
    -------
    dict
        Nested dictionary in the format {group: {length_factor: value}}.
    """
    allowed_properties = {'ii', 'jj'}
    fl_data = defaultdict(dict)

    for key, value in input_dict.items():
        parts = key.split('/')
        if parts[0] == name_FL and len(parts) == 3:
            _, length_factor, prop = parts
            if prop in allowed_properties:
                fl_data[length_factor][prop] = value
            else:
                raise ValueError(
                    f"Invalid property '{prop}' for frame length '{length_factor}'. "
                    f"Allowed properties are: {sorted(allowed_properties)}"
                )

    return dict(fl_data)


def extract_AS_structure(input_dict, name_AS='AS'):
    """
    Extracts area properties from flat input keys of the form 'AS/<area>/<property>'.

    Supports assigning the same value to both 'Thickness' and 'Bending'
    via a combined key (e.g., 'ThicknessBending').

    Only allowed properties are:
        'MatAngle': material angle (deg)
        'Thickness': membrane thickness (if area shell) or plane thickness (if plane)
        'Bending': bending thickness (if shell)
        'Arc': The arc angle through which the area object is passed to define the asolid element [deg];
            A value of zero means 1 radian (approximately 57.3 degrees).

    Parameters
    ----------
    input_dict : dict
        Flat dictionary with keys starting with 'name_AS'.

    Returns
    -------
    dict
        Nested dictionary in the format {area: {property: value}}.
    """
    allowed_properties = {'MatAngle', 'Thickness', 'Bending', 'Arc'}
    combinable = {'Thickness', 'Bending'}
    prop_pattern = '|'.join(sorted(combinable, key=len, reverse=True))

    as_data = defaultdict(dict)

    for key, value in input_dict.items():
        parts = key.split('/')
        if parts[0] == name_AS and len(parts) == 3:
            _, area, prop = parts

            # Handle combined properties (e.g., 'ThicknessBending')
            matches = re.findall(f'({prop_pattern})', prop)
            if matches and ''.join(matches) == prop:
                for p in matches:
                    if p not in allowed_properties:
                        raise ValueError(f"[AS] Invalid property '{p}' in '{key}'.")
                    as_data[area][p] = value
            elif prop in allowed_properties:
                as_data[area][prop] = value
            else:
                raise ValueError(
                    f"[AS] Invalid property '{prop}' for area '{area}'. "
                    f"Allowed properties: {sorted(allowed_properties)}"
                )

    return convert_to_dict(as_data)


def extract_FR_structure(input_dict, name_FR='FR'):
    """
    Extracts and structures force and moment data from a dictionary
    where keys follow the pattern '`name_FR`/<group>/<variables>/<ends>'.

    Only two valid variable groups are allowed:
    - Force: N, V2, V3
    - Moment: T, M2, M3

    Variables can be concatenated (e.g., 'M2M3' or 'NV2V3'), and the same value
    is assigned to each. Ends can be 'ii' (for frame release in the ini_point of the
    frame), 'jj' (for release in ending point), or both (e.g., 'iijj').

    Parameters:
        input_dict (dict): A dictionary with keys in the form 'FR/<group>/<variables>/<ends>' 
                           and corresponding numeric values.

    Returns:
        dict: A nested dictionary mapping each group to its end identifiers, 
              and those to their corresponding variables and values.

    Raises:
        ValueError: If any key contains invalid or mixed variable groups or invalid ends.
    """
    allowed_vars_F = {'N', 'V2', 'V3'}
    allowed_vars_M = {'T', 'M2', 'M3'}
    allowed_end = {'ii', 'jj'}

    var_F_pattern = '|'.join(sorted(allowed_vars_F, key=len, reverse=True))
    var_M_pattern = '|'.join(sorted(allowed_vars_M, key=len, reverse=True))
    end_pattern = '|'.join(sorted(allowed_end, key=len, reverse=True))

    fr_data = defaultdict(lambda: defaultdict(dict))

    for key, value in input_dict.items():
        parts = key.split('/')
        if parts[0] == name_FR and len(parts) == 4:
            _, group, var_concat, end_concat = parts

            # Match variables: check against both groups
            var_matches_F = re.findall(f'({var_F_pattern})', var_concat)
            var_matches_M = re.findall(f'({var_M_pattern})', var_concat)

            if ''.join(var_matches_F) == var_concat:
                var_matches = var_matches_F
            elif ''.join(var_matches_M) == var_concat:
                var_matches = var_matches_M
            else:
                raise ValueError(
                    f"[FR] Invalid or mixed variable group in key '{key}'. "
                    f"Allowed groups: {allowed_vars_F} or {allowed_vars_M}"
                )

            # Match and validate ends
            end_matches = re.findall(f'({end_pattern})', end_concat)
            if not end_matches or ''.join(end_matches) != end_concat:
                raise ValueError(
                    f"[FR] Invalid end(s) in key '{key}'. Allowed: {allowed_end}")

            # Assign value to each (end, variable) combination
            for e in end_matches:
                for v in var_matches:
                    fr_data[group][e][v] = value

    return convert_to_dict(fr_data)


def convert_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    return d


def extract_JS_structure(input_dict, name_JS='JS'):
    """
    Extracts and structures joint and rotational stiffness data from a dictionary 
    where keys follow the pattern '`name_JS`/<group>/<variables>'.

    Valid variable codes are U1, U2, U3, R1, R2, and R3. Multiple variables can be 
    concatenated (e.g., 'U1U2'), in which case the same value is assigned to each.

    Parameters:
        input_dict (dict): A dictionary with keys in the form 'JS/<group>/<variables>' 
                           and corresponding numeric values.

    Returns:
        dict: A nested dictionary where each group maps to its corresponding variables 
              and values.

    Raises:
        ValueError: If any key contains invalid variable codes.
    """
    allowed_vars = {'U1', 'U2', 'U3', 'R1', 'R2', 'R3'}

    var_pattern = '|'.join(sorted(allowed_vars, key=len, reverse=True))
    js_data = defaultdict(dict)
    for key, value in input_dict.items():
        parts = key.split('/')
        if parts[0] == name_JS and len(parts) == 3:
            _, group, var = parts
            matches = re.findall(f'({var_pattern})', var)
            if not matches or ''.join(matches) != var:
                raise ValueError(
                    f"[JS] Invalid variable(s) in key '{key}'. Allowed: {allowed_vars}")
            for v in matches:
                js_data[group][v] = value

    return convert_to_dict(js_data)


def build_joint_spring_dict_from_JS(JS_dict):
    """
    Converts a structured JS_dict into the format required by `set_jointsprings`.

    Parameters
    ----------
    JS_dict : dict
        Dictionary of joint spring parameters as produced by extract_JS_structure.

    Returns
    -------
    joint_spring_dict : dict
        Dictionary where each key is a group name and value is a 6-element list:
        [U1, U2, U3, R1, R2, R3]
    """
    joint_spring_dict = {}

    for group, dof_dict in JS_dict.items():
        # Start with zero stiffness for all 6 DOFs
        k = [None] * 6

        # Map input DOFs to their positions
        dof_map = {'U1': 0, 'U2': 1, 'U3': 2, 'R1': 3, 'R2': 4, 'R3': 5}

        for dof, value in dof_dict.items():
            if dof in dof_map:
                k[dof_map[dof]] = value
            else:
                raise ValueError(f"Invalid DOF '{dof}' in group '{group}'.")

        joint_spring_dict[group] = k

    return joint_spring_dict


def build_frame_release_dict_from_FR(FR_dict):
    """
    Converts a structured FR_dict into the format required by `set_framereleases`.

    Parameters
    ----------
    FR_dict : dict
        Dictionary as returned by `extract_FR_structure`.

    Returns
    -------
    frame_releases_dict : dict
        Dictionary formatted for `set_framereleases`, with:
        - ii, jj: lists of 6 booleans (release flags)
        - StartValue, EndValue: lists of corresponding stiffness values
    """
    dof_order = ['N', 'V2', 'V3', 'T', 'M2', 'M3']

    frame_releases_dict = {}

    for group, ends in FR_dict.items():
        group_data = {
            'ii': [False] * 6,
            'jj': [False] * 6,
            'StartValue': [0.0] * 6,
            'EndValue': [0.0] * 6
        }

        for end in ['ii', 'jj']:
            if end in ends:
                for var, value in ends[end].items():
                    if var not in dof_order:
                        raise ValueError(
                            f"Invalid release variable '{var}' in group '{group}/{end}'.")
                    idx = dof_order.index(var)
                    group_data[end][idx] = True
                    if end == 'ii':
                        group_data['StartValue'][idx] = value
                    else:
                        group_data['EndValue'][idx] = value

        frame_releases_dict[group] = group_data

    return frame_releases_dict


def check_updatable_parameters_groups(updatable_parameters, input_data_as_group, all_points, all_elements, SapModel):
    """
    Function Duties:
        Checks if the keys in updatable_parameters exists as group or elements. If so, returns:
            groups_dict:{key: bool} being True if the key is a group name, False otherwise (object
            name).
    Input:
        updatable_parameters: list of updatable parameters (e.g. 'FR/<key>/...) where key is
            expected to be a group name or an object name
        input_data_as_group: dictionary specifying the object type for each of the items in which
            apply
            e.g., {'JS': 'point', 'FR': 'frame'} (add more in the future)
        SapModel: SAP2000 model object
    Output:
        groups_dict: dictionary like:
            {'JS': True, 'FR': False} ('JS' items are a group; 'FR' items are not a group)
        Raise error if:
        - A group name does not contain the object type (e.g. 'JS' group does not contain points)
        - The key is not a group neither an object in the model
    Future improvements:
        Substitute all_points and all_elements by a class containing sap info
    """
    parameters_dict = {key: None for key in updatable_parameters}
    input_data = from_algorithm_parameters_to_sap2000_input(
        parameters_dict)

    groups_dict = dict()

    # Part 1: Check if groups exist and contain the object type
    all_groups = sap2000.get_group_names(SapModel)
    for item, obj_type in input_data_as_group.items():
        groups_dict[item] = True  # by default we have group names
        for key in input_data[item]:
            if key in all_groups:
                if not sap2000.group_contains_object_type(SapModel, key, obj_type):
                    raise ValueError(f"Group {key} does not contain objects of type {obj_type} necessary to update {item}")
            else:
                groups_dict[item] = False  # assume keys are object names
                break

    # Part 2: Check if the objects exist in the model
    for item, obj_type in input_data_as_group.items():
        if not groups_dict[item]:
            for key in input_data[item]:
                list_obj = all_points if input_data_as_group[item] == 'point' else all_elements
                if key not in list_obj:
                    raise ValueError(
                        f"Some names in input_data['{item}'] do not exist as group neither {obj_type} objects in the model.")
                else:
                    groups_dict[item] = False

    return groups_dict



def process_modal_results_fast(StepNum, Element, Station, P, M2, M3, Element_unique,
                               StepNum_unique, average_values=True):
    """
    Function to be used inside sap2000.get_modal_forces()
    Optimized version of the original code assisted by GPT to run faster
    """
    results = dict()
    StepNum = np.array(StepNum)
    Element = np.array(Element)
    Station = np.array(Station)
    P = np.array(P)
    M2 = np.array(M2)
    M3 = np.array(M3)

    for mode in StepNum_unique:
        mode_label = f'Mode_{int(mode)}'
        results[mode_label] = dict()

        # Precompute mode mask
        mode_mask = StepNum == mode

        for element in Element_unique:
            element_label = f'Element_{int(element)}'

            # Combined boolean mask
            mask = mode_mask & (Element == element)
            if not np.any(mask):
                continue  # Skip if no data

            station_vals = Station[mask]
            p_vals = P[mask]
            m2_vals = M2[mask]
            m3_vals = M3[mask]

            if average_values:
                # Group by unique station values
                unique_stations, inv_idx = np.unique(
                    station_vals, return_inverse=True)
                p_def = np.zeros_like(unique_stations, dtype=float)
                m2_def = np.zeros_like(unique_stations, dtype=float)
                m3_def = np.zeros_like(unique_stations, dtype=float)

                for i in range(len(unique_stations)):
                    idx = (inv_idx == i)
                    p_def[i] = np.mean(p_vals[idx])
                    m2_def[i] = np.mean(m2_vals[idx])
                    m3_def[i] = np.mean(m3_vals[idx])

                x = unique_stations.tolist()
            else:
                # Left-most appearance of each station value
                _, index = np.unique(station_vals, return_index=True)
                x = station_vals[index].tolist()
                p_def = p_vals[index].tolist()
                m2_def = m2_vals[index].tolist()
                m3_def = m3_vals[index].tolist()

            mesh = [f'{element}.{i}' for i in range(len(x))]

            # Save results
            results[mode_label][element_label] = {
                'x': x,
                'P': p_def,
                'M2': m2_def,
                'M3': m3_def,
                'Mesh_id': mesh
            }

    return results


def generate_sobol_samples_from_bounds(bounds_dict, n_samples, calc_second_order=False):
    """
    Generate Sobol samples for global sensitivity analysis from a dictionary of parameter bounds.

    Parameters:
    -----------
    bounds_dict : dict
        Dictionary of parameter bounds in the form:
        {
            'param_name1': [lower, upper],
            'param_name2': [lower, upper],
            ...
        }

    n_samples : int
        Desired approximate total number of model evaluations.
        The function will adjust this to match the structure required by SALib.

    calc_second_order : bool, default=False
        If True, includes second-order interaction effects in the sampling design.

    Returns:
    --------
    param_samples_dict : dict
        Dictionary mapping each parameter name to an array of sampled values.
    param_values : np.ndarray
        The full 2D array of shape (n_evals, num_params) with all sample combinations.
    """
    # SALib parameters
    param_names = list(bounds_dict.keys())
    bounds_list = [bounds_dict[name] for name in param_names]
    p = len(param_names)

    # Adjust base N to match target sample size
    denominator = (2 * p + 2) if calc_second_order else (p + 2)
    N_base = int(2 ** np.ceil(np.log2(n_samples / denominator)))
    actual_samples = N_base * denominator

    print(
        f'[INFO] Adjusted total number of samples: {actual_samples} (N base = {N_base})')

    # Define problem for SALib
    problem = {
        'num_vars': p,
        'names': param_names,
        'bounds': bounds_list
    }

    # Generate samples
    param_values = sample(problem, N_base, calc_second_order=calc_second_order)

    # Build dictionary of sampled values
    param_samples_dict = {name: param_values[:, i]
                          for i, name in enumerate(param_names)}

    return param_samples_dict, param_values


def generate_parameter_samples_from_algorithm_parameters(algorithm_parameters):
    """
    Generates a dictionary of parameter samples using the specified algorithm in `algorithm_parameters`.
    Currently supports only 'sobol' sampling.

    Parameters
    ----------
    algorithm_parameters : dict
        Dictionary specifying the sampling configuration. It must include:
            - 'algorithm': Name of the sampling algorithm (e.g., 'sobol').
            - 'nsamples': Desired number of samples to generate.
            - Parameter bounds: Keys ending having 'interv' on them that define lower and upper limits 
            for each parameter as a list [min, max].

        To apply sampling in logarithmic (base-10) space, append '_log10' to the parameter name.
        This triggers log-scaling during sample generation and automatic inverse scaling afterwards.

    Returns
    -------
    samples_dict : dict
        Dictionary mapping FEM input parameter names to arrays of sampled values.
    n_samples : int
        The number of samples generated.

    Remark:
    The main function to be used is generate_sobol_samples_from_bounds
    """
    # Extract parameters
    algorithm = algorithm_parameters['algorithm']
    n_samples = algorithm_parameters['nsamples']

    bounds_dict = {
        key.replace('/interv', ''): value for key, value in algorithm_parameters.items()
        if '/interv' in key and isinstance(value, list)
    }

    log10_flags = {key.replace(
        '/log10', ''): True for key in bounds_dict if '/log10' in key}

    # Remove '_log10' suffix from keys and apply log10 to interv
    bounds_dict = {key.replace('/log10', ''): value for key,
                   value in bounds_dict.items()}

    for key in bounds_dict:
        if log10_flags.get(key, False):
            bounds_dict[key] = np.log10(np.maximum(bounds_dict[key], 1e-12))

    # Generate samples using the specified algorithm
    if algorithm == 'sobol':
        # Extract parameters
        calc_second_order = algorithm_parameters.get(
            'calc_second_order', False)

        # Generate samples
        samples_dict, _ = generate_sobol_samples_from_bounds(
            bounds_dict,
            n_samples,
            calc_second_order=calc_second_order
        )
        n_samples_generated = len(samples_dict[list(samples_dict.keys())[0]])

        # Scale samples
        for key in bounds_dict:
            if log10_flags.get(key, False):
                samples_dict[key] = 10**samples_dict[key]

        return samples_dict, n_samples_generated

    else:
        raise ValueError(
            f"Algorithm '{algorithm}' is not supported. Supported algorithms are: 'sobol'.")


def generate_parameter_samples_from_algorithm_parameters_discrete(algorithm_parameters):
    """
    Similar logic as generate_parameter_samples_from_algorithm_parameters
    Instead of generating with sobol, we hardcode to properly define damages
    """
    # Extract parameters
    algorithm = algorithm_parameters['algorithm']
    n_samples = algorithm_parameters['nsamples']

    bounds_dict_aux = {
        key.replace('/interv', ''): value for key, value in algorithm_parameters.items()
        if '/interv' in key and isinstance(value, list)
    }

    if not all([bounds_dict_aux[list(bounds_dict_aux)[0]] == bounds_dict_aux[list(bounds_dict_aux)[i]] for i in range(len(bounds_dict_aux))]):
        raise ValueError("All parameters must have the same number of discrete values for sampling.")

    bounds_dict = {'values': bounds_dict_aux[list(bounds_dict_aux)[0]],
                    'numbers': [0, len(bounds_dict_aux)]}

    log10_flags = {key.replace(
        '/log10', ''): True for key in bounds_dict if '/log10' in key}

    # Remove '_log10' suffix from keys and apply log10 to interv
    bounds_dict = {key.replace('/log10', ''): value for key,
                   value in bounds_dict.items()}

    for key in bounds_dict:
        if log10_flags.get(key, False):
            bounds_dict[key] = np.log10(np.maximum(bounds_dict[key], 1e-12))

    # Generate samples using the specified algorithm
    if algorithm == 'sobol':
        # Extract parameters
        n_elements = len(bounds_dict_aux)
        n_values_element = int((n_samples / n_elements))
        values = np.linspace(bounds_dict['values'][0], bounds_dict['values'][-1], n_values_element, endpoint=False)
        values_expanded = np.tile(values, n_elements)
        numbers_expanded = np.repeat(np.arange(n_elements), n_values_element)
        n_samples_generated = len(values_expanded)

        samples_dict = {'values': values_expanded, 'numbers': numbers_expanded}
        # Scale samples
        for key in bounds_dict:
            if log10_flags.get(key, False):
                samples_dict[key] = 10**samples_dict[key]

        samples_dict_discrete = adapt_samples_for_discretizing(samples_dict, bounds_dict_aux)

        return samples_dict_discrete, n_samples_generated

    else:
        raise ValueError(
            f"Algorithm '{algorithm}' is not supported. Supported algorithms are: 'sobol'.")


def adapt_samples_for_discretizing(samples_dict, bounds_dict_aux):
    """
    Build a per-bin representation of sample values for discretization.

    Given `samples_dict` with NumPy arrays `numbers` (typically in [0, 40]) and
    `values` (typically in [0, 1]), and a `bounds_dict_aux` whose keys define the
    bin order (e.g., 40 bins of width 1), return a dict mapping each key to an
    array `out` of the same length as `values` where:
    - out[j] = values[j] if numbers[j] falls in that bin ([i-1, i) except last bin
    which is [n_bins-1, n_bins] inclusive),
    - out[j] = 1 otherwise.
    """
    numbers = samples_dict["numbers"]
    values = samples_dict["values"]

    keys = list(bounds_dict_aux.keys())
    n_bins = len(keys)

    samples_dict_adapted = {}

    for i, k in enumerate(keys, start=1):
        low = i - 1
        high = i

        if i < n_bins:
            mask = (numbers >= low) & (numbers < high)
        else:
            mask = (numbers >= low) & (numbers <= high)

        out = np.ones_like(values, dtype=values.dtype)
        out[mask] = values[mask]

        samples_dict_adapted[k] = out

    return samples_dict_adapted


def get_SGs_positions_dictionary(point_forces, load_pattern, all_elements_coord_connect):
    """
    Function Duties:
        Constructs the dictionary of SGs positions.
    Input:
        point_forces : dict
            Dictionary of point loads retrieved from SAP2000 (via get_point_loads_on_frame),
            structured as: {element_name: {load_pattern: load_data}}.
        load_pattern : str
            The specific load pattern to extract information from (e.g. "DOFs_sg").
        all_elements_coord_connect : dict
            Dictionary containing coordinates of start (`Point_0`) and end (`Point_f`)
            nodes of each frame element.
    Output:
        sg_channels : dict
            Dictionary of SG channel properties. Each entry includes:
            - x_sg: Location along the element where the SG is placed (absolute distance)
            - location: Direction label ('up', 'down', 'left', 'right') based on force direction
            - Point_0: Coordinates of the element's I-End
            - Point_f: Coordinates of the element's J-End
            - Element: Frame element name
    Notes:
        - Only local directions 2 and 3 are interpreted.
        - If Dir > 3, a warning is printed (indicating use of global/projected coordinates).
        - SG channels are sorted numerically by their channel number.
    """
    sg_channels = dict()

    for element in point_forces:
        info = point_forces[element][load_pattern]
        Val = info['Val']
        Dir = info['Dir']
        x_sg = info['Dist']
        if Dir == 2:
            location = 'up' if Val < 0 else 'down'
        elif Dir == 3:
            location = 'right' if Val < 0 else 'left'
        elif Dir > 3:
            print(f"Warning: FORCES MUST BE DEFINED IN LOCAL COORDINATES")
        Point_0_aux = all_elements_coord_connect[element]['Point_0']
        Point_f_aux = all_elements_coord_connect[element]['Point_f']
        Point_0 = {'x': Point_0_aux['x'],
                   'y': Point_0_aux['y'], 'z': Point_0_aux['z']}
        Point_f = {'x': Point_f_aux['x'],
                   'y': Point_f_aux['y'], 'z': Point_f_aux['z']}
        sg_channels[f'Channel_{int(abs(Val))}'] = {
            'x_sg': x_sg, 'location': location, 'Point_0': Point_0,
            'Point_f': Point_f, 'Element': element
        }
    # Sort dictionary
    sorted_items = sort_string_separated_by(list(sg_channels), separator='_')
    sg_channels = {key: sg_channels[key] for key in sorted_items}

    return sg_channels


def read_parameters(file_path):
    """
    Function Duties:
        Read bayesian_inference_parameters.txt file
    """
    parameters = dict()
    constraint_counter = 0
    with open(file_path, 'r') as file:
        for line in file:
            # Remove comments and strip whitespace
            line = line.split('#')[0].strip()
            if line:  # Skip empty lines
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                if value == 'None':
                    parameters[key] = None
                elif 'filename' in key:
                    parameters[key] = value
                elif 'constraint' in key and any(op in value for op in ['<', '>']):
                    constraint_counter += 1
                    key = f'constraint_{constraint_counter}'
                    parameters[key] = value
                elif 'constraint' in key and '==' in value:
                    raise ValueError(
                        f"Invalid constraint format in {file_path}: '{key}' uses the unsupported '==' operator.\n"
                        "Note: If two variables should take the same value, place them on the same group and/or assign them jointly in the input file.\n"
                        "For example:\n"
                        "    FR/releasebeams_top/M2M3/ii/ini_guess = 1e5"
                    )
                else:
                    parameters[key] = safe_eval(value)
    return parameters


def safe_eval(value):
    """
    Function Duties:
        Safely evaluate a string as a literal or expression
    """
    try:
        # Try to evaluate the value as a literal
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If that fails, use eval to handle expressions like 10**6
        return eval(value)


def read_matrices_h5py(h5py_filename, groups):
    """
    Read matrices from a h5py file
    """
    with h5py.File(h5py_filename, 'r') as h5py_file:
        matrices = dict()
        for group in groups:
            matrices[group] = h5py_file[group][:]
    return matrices



def save_json_serialized(obj, filepath, omit_keys=None) -> None:
    serial = serialize_dictionary_v2(obj, omit_keys=omit_keys)
    with open(filepath, 'w') as f:
        json.dump(serial, f, indent=2)


def load_json_serialized(filepath):
    with open(filepath, 'r') as f:
        return from_serializable(json.load(f))


def from_serializable(obj):
    """
    Deserializes objects encoded by `to_serializable`, restoring NumPy arrays,
    complex numbers, and nested structures to native Python types.

    Remark: enhanced version of deserilize_dict
    """
    if isinstance(obj, dict):
        if "__complex__" in obj:
            return complex(obj["real"], obj["imag"])
        elif "__complex_array__" in obj:
            return np.array(obj["real"]) + 1j * np.array(obj["imag"])
        else:
            return {k: from_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Try converting to ndarray if list of numbers or complex values
        converted = [from_serializable(v) for v in obj]
        if all(isinstance(x, (float, int, complex, np.number)) for x in converted):
            return np.array(converted)
        elif all(isinstance(x, np.ndarray) for x in converted):
            try:
                return np.stack(converted)
            except Exception:
                return converted  # fallback: list of arrays
        return converted
    return obj


def serialize_dictionary_v2(test_dict, omit_keys=None):
    """
    Converts a dictionary into a JSON-serializable format, handling complex numbers,
    NumPy arrays, and other non-native JSON types.

    Parameters
    ----------
    test_dict : dict
        The dictionary containing results for one test (e.g., signal processing, FDD results).
    omit_keys : str or list of str, optional
        Key(s) to exclude from serialization (e.g., 'FDD' to avoid saving bulky internal data).

    Returns
    -------
    test_dict_serializable : dict
        A cleaned and fully JSON-compatible dictionary, suitable for writing to file.
    """
    if omit_keys is None:
        omit_keys = []
    elif isinstance(omit_keys, str):
        omit_keys = [omit_keys]

    test_dict_serializable = {}
    for key, value in test_dict.items():
        if key in omit_keys:
            continue
        test_dict_serializable[key] = to_serializable(value)

    return test_dict_serializable


def to_serializable(obj):
    """
    Converts NumPy arrays, complex numbers, and nested structures into
    JSON-compatible formats for safe serialization.
    """
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "__complex_array__": True,
                "real": obj.real.tolist(),
                "imag": obj.imag.tolist()
            }
        else:
            return obj.tolist()
    elif isinstance(obj, complex):
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def backup_existing_output_files(output_path, files_to_backup) -> None:
    """
    Creates a timestamped backup directory inside `output_path` and moves existing files into it.

    Parameters
    ----------
    output_path : str
        Path where the output files are stored and where the backup folder will be created.

    files_to_backup : list of str
        List of filenames (not full paths) to check and move into the backup folder if they exist.
    """
    # Create timestamped backup folder
    date_time_str = datetime.now().strftime('%Y%m%d_%H%M')
    backup_path = os.path.join(output_path, f'backup_{date_time_str}')
    os.makedirs(backup_path, exist_ok=True)

    for file in files_to_backup:
        file_path = os.path.join(output_path, file)
        if file in os.listdir(output_path):
            backup_file_path = os.path.join(backup_path, file)
            if file in os.listdir(backup_path):
                os.remove(backup_file_path)
            shutil.move(file_path, backup_path)


def save_state(state, filepath):
    """Save the current state to a file."""
    with open(filepath, 'w') as f:
        json.dump(state, f)
    print(f"State saved to {filepath}")


def load_state(filepath):
    """Load the state from a file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        print(f"State loaded from {filepath}")
        return state
    return None


def remove_file(file_path):
    """Remove a file if it exists."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Removed file: {file_path}")

def get_username():
    """
    Get the username from the .env file
    """
    load_dotenv(override=True)  # loads variables from .env file
    username = os.getenv('USERNAME')
    return username



def get_sectionproperties_material(section_properties, section_material,
                                   material_properties):
    """
    Input:
        section_properties: dictionary containing geometric properties
            associated to each section (from get_sectionproperties function)
        section_material: dictionary containing which material is associated
            to each section (from get_material_I_section)
        material_properties: dictionary containing physical properties
            associated to each material (from get_material_properties function)
    Return:
        Dictionary with all aggregated information
    """

    section_properties_material = dict()
    for section in list(section_properties):
        material = section_material[section]
        mat_props = material_properties[material]
        sect_props = section_properties[section]
        section_properties_material[section] = dict()
        section_properties_material[section]['Geometry'] = sect_props
        section_properties_material[section]['Material'] = mat_props

    return section_properties_material


def get_areaproperties_material(area_section_properties,
                                material_properties):
    """
    Input:
        area_section_properties: dictionary containing all geometric properties
        material_properties: dictionary containing physical properties
            associated to each material (from get_material_properties function)
    Return:
        Dictionary with all aggregated information (respecting the formatting used for
        frame elements)
    """
    area_properties_material = dict()
    for area in area_section_properties:
        area_properties_material[area] = {
            'Geometry': copy.deepcopy(area_section_properties[area]),
            'Material': copy.deepcopy(material_properties[area_section_properties[area]['MatProp']])
        }

    return area_properties_material


def get_paths(csv):
    """
    Function duties:
        Get dictionary with paths
    """

    paths_df = pd.read_csv(csv, sep=';', comment='#')
    paths = dict()

    for var in list(paths_df.VarName):
        i = list(paths_df.VarName).index(var)
        path_value = paths_df.Path[i]
        if ":" not in path_value:
            path_value = os.path.join(
                paths_df[paths_df['VarName'] == 'project']['Path'].iloc[0], path_value)
        paths[var] = path_value

    return paths

def kill_process_advanced(process_name):
    """Kills the specified process and its subprocesses."""
    found_process = False
    for proc in psutil.process_iter():
        try:
            if process_name.lower() in proc.name().lower():
                found_process = True
                print(f"Killing process '{proc.name()}' with PID {proc.pid}")
                proc.kill()
                proc.wait()  # Wait for the process to fully terminate

                # Check for any child processes and kill them
                for child in proc.children(recursive=True):
                    print(
                        f"Killing subprocess '{child.name()}' with PID {child.pid}")
                    child.kill()
                    child.wait()

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
            print(f"Error: {e}")

    if not found_process:
        print(f"The process '{process_name}' is not running.")


def has_match(row, df):
    """
    Determine whether a constraint row shares all non-NaN values with any other row in the DataFrame.

    Parameters
    ----------
    row : pandas.Series
        A single row (constraint) from the DataFrame whose non-NaN entries will be compared.
    df : pandas.DataFrame
        The full DataFrame in which to search for matching rows.

    Returns
    -------
    bool
        True if there exists at least one other row in `df` with identical values in every non-NaN column of `row`; False otherwise.
    """
    cols = row.index[1:][row.iloc[1:].notna()]
    matches = (df[cols] == row[cols]).all(axis=1) & (df.index != row.name)

    return matches.any()


def clean_joint_index_unused_constraints(joints_matrix_index):
    """
    Remove redundant constraint rows (rows starting with 'C') from a joint index matrix.
    Redundant means that have the exactly the same equations as other nodes

    Parameters
    ----------
    joints_matrix_index : pandas.DataFrame
        DataFrame containing joint labels and associated numeric values. Must include:
        - A 'Joint_Label' column (string identifiers for joints or constraints)
        - Numeric columns: 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame with redundant constraint rows removed.

    Notes
    -----
    - Matching is performed only on overlapping non-NaN values between the constraint
      and non-constraint rows.
    """
    # 2) Only rows starting by 'C'
    df = joints_matrix_index.copy()  # pandas.DataFrame.copy
    mask_C = df['Joint_Label'].str.startswith(
        'C', na=False)  # pandas.Series.str.startswith

    rows_to_delete = df[mask_C].index[
        df[mask_C].apply(lambda row: has_match(row, df), axis=1)
    ]
    df.drop(index=rows_to_delete, inplace=True)

    return df


def get_point_obj_constraints_dict(FilePath, joints_matrix_index):
    """
    Build a dictionary mapping negative DOFs (point-object constraints) to related constraint (e.g. CDIAPH)
        DOFs and their coefficients.

    Parameters
    ----------
    FilePath : str
        Path to the constraints file.
    joints_matrix_index : pandas.DataFrame
        DataFrame representing the joint matrix index with columns:
        ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'].
        - Negative values indicate constraint IDs.
        - Positive values indicate active DOFs (equation numbers are value + 1).
        - Rows starting with 'C' (e.g., 'CDIAPH1') are used to identify CDIAPH DOFs.

    Returns
    -------
    dict
        Dictionary of the form:
        {
            "3_U1": {"CDIAPH1_U1": 1.0, "CDIAPH1_R3": 3.0},
            "3_U2": {"CDIAPH1_U2": 1.0, "CDIAPH1_R3": -3.0},
            ...
        }
        Keys are joint DOFs (negative ones), and values are mappings of CDIAPH DOFs with their coefficients.
    """
    constraints = pd.read_csv(
        FilePath,
        sep='\t',
        skiprows=1,
        header=None,
        names=["constraint_id", "equation_number", "coefficient"]
    )

    # Iterate rows to map each positive equation index (value+1) to joint_DOFLABEL
    eq_to_dof = dict()
    for _, row in joints_matrix_index.iterrows():
        joint_label = str(row["Joint_Label"])
        if joint_label.startswith('C'):  # constraints always start with 'C'
            for dof in ["U1", "U2", "U3", "R1", "R2", "R3"]:
                val = row[dof]
                if pd.notna(val) and val >= 0:  # Positive index
                    equation = int(val) + 1     # Convert to equation number
                    eq_to_dof[equation] = f"{joint_label}_{dof}"

    # Build relationships for negative DOFs (constraints) ---
    final_dict = dict()

    # Loop again to locate negative DOFs
    for _, row in joints_matrix_index.iterrows():
        joint_label = str(row["Joint_Label"])
        for dof in ["U1", "U2", "U3", "R1", "R2", "R3"]:
            val = row[dof]
            if pd.notna(val) and val < 0:  # Negative = constraint ID
                constraint_id = int(val)

                # Find equations related to this constraint
                related_eqs = constraints[constraints["constraint_id"]
                                          == constraint_id]

                # Map each equation to its DOF name and coefficient
                relation = dict()
                for _, eq_row in related_eqs.iterrows():
                    eq_num = eq_row["equation_number"]
                    coeff = eq_row["coefficient"]
                    if eq_num in eq_to_dof:  # Only map if equation exists in DOF map
                        relation[eq_to_dof[eq_num]] = coeff
                    else:
                        print(
                            f"[WARNING] Equation {eq_num} not found for joint {joint_label}_{dof}")

                # Store result
                final_dict[f"{joint_label}_{dof}"] = relation

    return final_dict


def sort_dofs_custom(dofs):
    """
    Sort DOFs first by base name (all parts before the last underscore) and then
    by a fixed order of DOF suffixes: U1, U2, U3, R1, R2, R3.

    Parameters
    ----------
    dofs : iterable of str
        DOF labels in the form 'BaseName_Ux' or 'BaseName_Rx'.

    Returns
    -------
    list of str
        Sorted DOF labels.
    """
    dof_order = {"U1": 0, "U2": 1, "U3": 2, "R1": 3, "R2": 4, "R3": 5}

    def sort_key(dof):
        parts = dof.split("_")
        base = "_".join(parts[:-1])       # everything before last underscore
        suffix = parts[-1]                # last part (e.g., U1, R3)
        return (base, dof_order.get(suffix, 999))  # unknown suffix last

    return sorted(dofs, key=sort_key)


def get_constraints_matrix(constraints_dict):
    """
    Build the constraint coefficient matrix (C) and its pseudoinverse (C_inv) from a constraints dictionary.

    This function transforms a dictionary of relationships between slave DOFs and master (constraint) DOFs
    into a matrix form suitable for linear algebra operations:
        a = C * b   ->   b = C_inv * a
    where:
        - a: vector of slave DOFs (negative DOFs).
        - b: vector of master DOFs (e.g., CDIAPH DOFs).
        - C: coefficient matrix built from the input dictionary.
        - C_inv: Moore–Penrose pseudoinverse of C (handles non-square or singular matrices).

    Parameters
    ----------
    constraints_dict : dict
        Dictionary of the form:
        {
            "3_U1": {"CDIAPH1_U1": 1.0, "CDIAPH1_R3": 3.0},
            "3_U2": {"CDIAPH1_U2": 1.0, "CDIAPH1_R3": -3.0},
            ...
        }
        Keys represent slave DOFs, values map to master DOFs with coefficients.

    Returns
    -------
    slaves_dofs : list of str
        Ordered list of slave DOF labels (rows of C).
    constraints_dofs : list of str
        Ordered list of master DOF labels (columns of C).
    C : numpy.ndarray
        Constraint coefficient matrix of shape (len(slaves_dofs), len(constraints_dofs)).
    C_inv : numpy.ndarray
        Pseudoinverse of C, suitable for computing master DOFs (b) from slave DOFs (a).
    """
    slaves_dofs = list(constraints_dict.keys())  # filas
    constraints_dofs = sort_dofs_custom(
        list({dof for subdict in constraints_dict.values() for dof in subdict.keys()}))
    C = np.zeros((len(slaves_dofs), len(constraints_dofs)))

    for i, a_key in enumerate(slaves_dofs):
        for j, b_key in enumerate(constraints_dofs):
            C[i, j] = constraints_dict[a_key].get(b_key, 0.0)

    C_inv = np.linalg.pinv(C)

    return slaves_dofs, constraints_dofs, C, C_inv


def get_constraints_update_matrix_index(FilePath, joints_matrix_index):
    """
    Build the constraints matrices (C and C_inv)
    Return an updated joint matrix with negatives replaced by NaN (negative
    DOFs are constraints, so they do not appear in M, K matrices) and without
    redundant constraints.

    Parameters
    ----------
    FilePath : str
        Path to the constraints file (tab-separated) containing columns:
        [constraint_id, equation_number, coefficient].
    joints_matrix_index : pandas.DataFrame
        DataFrame representing the joint matrix index with columns:
        ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'].
        - Negative values indicate constraint DOFs.
        - Positive values indicate active DOFs.

    Returns
    -------
    constraints : dict
        Dictionary containing:
        {
            'slave_dofs': list of str  # Labels for slave DOFs (rows of C),
            'constraints_dofs': list of str  # Labels for master DOFs (columns of C),
            'C': numpy.ndarray  # Coefficient matrix (slave_dofs x constraints_dofs),
            'C_inv': numpy.ndarray  # Pseudoinverse of C
        }
    joints_matrix_updated : pandas.DataFrame
        Copy of `joints_matrix_index` where negative DOF values are replaced by NaN.
        Positive and NaN values remain unchanged.
    """
    if os.path.exists(FilePath):
        constraints_dict = get_point_obj_constraints_dict(
            FilePath, joints_matrix_index)
        slaves_dofs, constraints_dofs, C, C_inv = get_constraints_matrix(
            constraints_dict)
    else:
        slaves_dofs, constraints_dofs, C, C_inv = list(
        ), list(), np.empty((0, 0)), np.empty((0, 0))
    constraints = {'slave_dofs': slaves_dofs,
                   'constraints_dofs': constraints_dofs, 'C': C, 'C_inv': C_inv}

    # Remove constraints that are not used (i.e. have the same equations as other nodes)
    joints_matrix_index = clean_joint_index_unused_constraints(
        joints_matrix_index)

    # Set NaN values to negative index (negative refers to constraints, not to K, M matrices)
    joints_matrix_updated = joints_matrix_index.copy()
    dof_columns = ["U1", "U2", "U3", "R1", "R2", "R3"]
    joints_matrix_updated[dof_columns] = joints_matrix_updated[dof_columns].mask(
        joints_matrix_updated[dof_columns] < 0)

    return constraints, joints_matrix_updated


def get_joint_matrix_index(FilePath):
    """
    Function Duties:
        Read the joint matrix index from a file
    Input:
        FilePath: Path to the .TXE file
    Output:
        joints_matrix_index: DataFrame containing the joint matrix index
            The columns are: ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'];
            joints_matrix_index.iloc[i] = [joint_label, U1, U2, U3, R1, R2, R3] being:
                joint_label: the name of the joint
                Uk: the number of the equation for the k-th traslational dof in the M, K matrices
                Rk: the number of the equation for the k-th rotational dof in the M, K matrices
                If the value is NaN it means that the DOF is restrained
    REMARK I:
        If the number is 0 in the TXE file -> it is converted to NaN in joints_index_matrix
        At the end, the number of joints_index_matrix are adjusted to Python indexing (start at 0),
            so in the original .TXE file the equation number 1 is the 0-th equation in the matrix
            and is also 0 in the joints_index_matrix
    REMARK II:
        Negative values correspond to constraint equations (see the .TXA file)
        IF PRESENT, THIS FUNCTION MUST BE UPDATED / COMPLETED / FINISHED
    """
    # Read the file
    with open(FilePath, 'r') as file:
        lines = file.readlines()

    # Process the lines (skipping the header line)
    processed_lines = []
    temp_line = ""
    header_skipped = False
    for line in lines:
        stripped_line = line.strip()
        if not header_skipped:
            header_skipped = True
            continue  # Skip the header line
        if '\t' not in stripped_line:  # Check if the line is a continuation
            temp_line += " \t" + stripped_line  # Continue the previous record
        else:
            if temp_line:
                processed_lines.append(temp_line)  # Add the completed record
            temp_line = stripped_line  # Start a new record

    if temp_line:  # Append the last line if it was in progress
        processed_lines.append(temp_line)

    # Split the processed lines into columns and remove whitespaces
    data_aux = [line.split('\t') for line in processed_lines]
    data = [[value.strip() for value in line] for line in data_aux]

    # Convert to DataFrame
    columns = ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3']
    joints_matrix_index = pd.DataFrame(data, columns=columns)

    # Transform the values to integers and adjust indexing
    def transform_value(value):
        if value == '0':  # restrained nodes are labeled as 0
            return np.nan
        elif int(value) > 0:
            return int(int(value) - 1)  # python index start at 0
        elif int(value) < 0:
            return int(int(value))

    for col in columns[1:]:  # Skip the first column 'Joint_Label'
        joints_matrix_index[col] = joints_matrix_index[col].apply(
            transform_value)

    return joints_matrix_index


def get_mass_stiffness_matrix(FilePath):
    """
    Function Duties:
        Read the mass or stiffness matrix from the .TXM or .TXK file
    Input:
        FilePath: Path to the .TXM (for mass) or .TXK (for stiffness) file
    Output:
        matrix: Mass or stiffness matrix in NumPy array format
    REMARK:
        The correspondance between the K, M matrices with the DOFS is given
        by the joints_matrix_index matrix obtained with get_joint_matrix_index function
    """
    # Read the stiffness matrix
    with open(FilePath, 'r') as file:
        lines = file.readlines()

    data_aux = [line.split('\t') for line in lines[1:]]
    data = [[value.strip() for value in line] for line in data_aux]

    # convert to numpy array
    data = [[int(row[0]), int(row[1]), float(row[2])] for row in data]

    # Determine the shape of the matrix (max row and column indices)
    max_row = max(row[0] for row in data)
    max_col = max(row[1] for row in data)

    # Create an empty NumPy array with the determined shape
    matrix = np.zeros((max_row, max_col))

    # Populate the matrix with the values from the data list
    for row, col, value in data:
        matrix[row-1, col-1] = value  # Adjust for zero-based indexing
        matrix[col-1, row-1] = value  # Symmetric matrix

    return matrix


def read_M_K_constraints_from_TX_files(sapfile_name, paths):
    """
    Read M, K and joints_matrix_index from SAP2000 TX files.
    """
    # C) Read matrices
    # C.1 Matrix index
    equations_file = sapfile_name.replace('.sdb', '.TXE')
    FilePath = os.path.join(paths, equations_file)
    joints_matrix_index = get_joint_matrix_index(FilePath)

    # C.2 Constraints equations
    constraints_file = sapfile_name.replace('.sdb', '.TXC')
    FilePath = os.path.join(paths, constraints_file)
    constraints, joints_matrix_index = get_constraints_update_matrix_index(
        FilePath, joints_matrix_index)

    # C.2 Get the mass matrix
    mass_matrix_file = sapfile_name.replace('.sdb', '.TXM')
    FilePath = os.path.join(paths, mass_matrix_file)
    mass_matrix = get_mass_stiffness_matrix(FilePath)

    # C.3 Get the stiffness matrix
    stiffness_matrix_file = sapfile_name.replace('.sdb', '.TXK')
    FilePath = os.path.join(paths, stiffness_matrix_file)
    stiffness_matrix = get_mass_stiffness_matrix(FilePath)

    return mass_matrix, stiffness_matrix, joints_matrix_index, constraints


def get_modal_properties_from_K_M(K, M, tol=1e-10):
    """
    Function Duties:
        Computes the modal frequencies and mode shapes of a system
        using the eigenvalue problem.
    Remark: phi is normalized to unity max displacement.
    """
    if np.linalg.det(M) < tol or np.linalg.det(K) < tol:
        raise ValueError("The mass or stiffness matrix is singular.")

    # Solve eigenvalue problem
    w2, phi = np.linalg.eig(np.linalg.inv(M) @ K)
    f = np.sqrt(w2)/(2*np.pi)

    # Sort eigenvalues and eigenvectors
    idx_sorted = np.argsort(f)
    f = f[idx_sorted]
    phi = phi[:, idx_sorted]

    # normalize phi to unity max displacement
    idx_max = np.argmax(np.abs(phi), axis=0)
    phi = phi / np.array([phi[idx_max[j], j] for j in range(np.shape(phi)[0])])

    return f, phi


def static_condensation(K, slave_dofs, F=None):
    """
    Static condensation of a matrix K eliminating the DOFs in slave_dofs.
    """
    # K = sym.Matrix(K)
    n = K.shape[0] if hasattr(K, 'shape') else len(K)
    Islaves = sorted(set(slave_dofs))
    Imasters = sorted(set(range(n)) - set(Islaves))
    if isinstance(K, sym.MatrixBase):
        K = sym.Matrix(K)
        # Blocs
        Kmm = K.extract(Imasters, Imasters)
        Kms = K.extract(Imasters, Islaves)
        Ksm = K.extract(Islaves, Imasters)
        Kss = K.extract(Islaves, Islaves)

        # Condensation
        X = Kss.LUsolve(Ksm) if Kss.shape[0] > 0 else sym.zeros(0, len(Imasters))
        Kc = Kmm - Kms * X

    elif isinstance(K, np.ndarray):
        K = np.array(K, dtype=float, copy=False)
        Isl = np.array(Islaves, dtype=int)
        Ima = np.array(Imasters, dtype=int)

        Kee = K[np.ix_(Ima, Ima)]
        Kei = K[np.ix_(Ima, Isl)]
        Kie = K[np.ix_(Isl, Ima)]
        Kii = K[np.ix_(Isl, Isl)]

        X = np.linalg.solve(Kii, Kie)
        Kc = Kee - Kei @ X

    return Kc, Imasters


def beam_stiffness_np(E, I, L):
    """B-E beam element stiffness matrix for bending in a single plane."""
    k = (E * I) / (L**3)
    K = k * np.array([
        [12,    6*L,   -12,    6*L],
        [6*L,  4*L**2, -6*L,  2*L**2],
        [-12,  -6*L,    12,   -6*L],
        [6*L,  2*L**2, -6*L,  4*L**2]
    ], dtype=float)
    return K


def assemble_global_beam_sym(E, I, Le, n_el, keep_factor=True, simplify_result=True):
    """
    Ensambla la matriz de rigidez global (simbólica) para n_el elementos EB en serie.
    - Nodos = n_el + 1, 2 GDL por nodo: [w_i, theta_i]
    Devuelve:
      K_full  : matriz global completa de tamaño (2*(n_el+1)) x (2*(n_el+1))
      map_dof : lista con el significado de cada DOF global (tupla (node, dof_name))
    """
    k = beam_stiffness_sym(E, I, Le)
    n_nodes = n_el + 1
    ndof = 2 * n_nodes
    K = sym.zeros(ndof, ndof)

    # Ensamblaje por superposición
    for e in range(n_el):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]  # [w_e, θ_e, w_{e+1}, θ_{e+1}]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k[i, j]

    if simplify_result:
        K = sym.simplify(K)

    if keep_factor:
        factor = (E*I)/(Le**3)
        K = sym.simplify(K / factor) * factor  # mantiene explícito EI/Le^3

    map_dof = [(i//2, 'w' if i % 2 == 0 else 'theta') for i in range(ndof)]
    return K, map_dof


def assemble_global_beam_np(E, I, Le, n_el):
    """
    Ensambla la matriz de rigidez global (numérica) para n_el elementos EB en serie.
    - Nodos = n_el + 1, 2 GDL por nodo: [w_i, theta_i]
    Devuelve:
      K_full  : matriz global completa de tamaño (2*(n_el+1)) x (2*(n_el+1))
      map_dof : lista con el significado de cada DOF global (tupla (node, dof_name))
    """
    k = beam_stiffness_np(E, I, Le)        # 4x4 numérica
    n_nodes = n_el + 1
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof), dtype=float)

    # Ensamblaje por superposición
    for e in range(n_el):
        dofs = [2*e, 2*e+1, 2*e+2, 2*e+3]  # [w_e, θ_e, w_{e+1}, θ_{e+1}]
        for i in range(4):
            for j in range(4):
                K[dofs[i], dofs[j]] += k[i, j]

    map_dof = [(i//2, 'w' if i % 2 == 0 else 'theta') for i in range(ndof)]
    return K, map_dof


def reduce_cantilever_sym(K_full):
    """
    Aplica el empotramiento en el nodo 0 (w0 = theta0 = 0).
    Devuelve:
      K_red   : submatriz de rigidez en los GDL libres
      free_ix : índices globales de los GDL libres (corresponden a K_red)
    """
    ndof = K_full.shape[0]
    fixed = [0, 1]                 # w0, theta0
    free_ix = [i for i in range(ndof) if i not in fixed]
    K_red = K_full.extract(free_ix, free_ix)
    return K_red, free_ix


def reduce_cantilever_np(K_full):
    """
    Aplica el empotramiento en el nodo 0 (w0 = theta0 = 0).
    Devuelve:
      K_red   : submatriz de rigidez en los GDL libres
      free_ix : índices globales de los GDL libres (corresponden a K_red)
    """
    ndof = K_full.shape[0]
    fixed = [0, 1]  # w0, theta0
    free_ix = [i for i in range(ndof) if i not in fixed]
    K_red = K_full[np.ix_(free_ix, free_ix)]
    return K_red, free_ix


def _is_symbolic(*vals) -> bool:
    return any(getattr(v, "free_symbols", set()) for v in vals)


def beam_stiffness_sym(E, I, Le):
    """
    Local stiffness matrix for a Euler-Bernoulli beam.
    """
    EI = E*I
    return (EI/Le**3)*sym.Matrix([
        [12,    6*Le,  -12,    6*Le],
        [6*Le,  4*Le**2, -6*Le, 2*Le**2],
        [-12,   -6*Le,  12,   -6*Le],
        [6*Le,  2*Le**2, -6*Le, 4*Le**2]
    ])


def k_beam_rotational_springs(E, I, L, k1, k2):
    """
    Local stiffness matrix for a Euler-Bernoulli beam with rotational springs

    Delta = 12*E**2*I**2 + 4*E*I*L*(k1 + k2) + L**2*k1*k2
    """
    Delta = 12*E**2*I**2 + 4*E*I*L*(k1 + k2) + L**2*k1*k2
    EI = E*I

    K11 = 12*EI*(EI*(k1 + k2) + L*k1*k2) / (Delta*L**2)
    K12 = 6*EI*k1*(2*EI + L*k2) / (Delta*L)
    K13 = -K11
    K14 = 6*EI*k2*(2*EI + L*k1) / (Delta*L)

    K22 = 4*EI*k1*(3*EI + L*k2) / Delta
    K23 = -K12
    K24 = 2*EI*L*k1*k2 / Delta

    K33 = K11
    K34 = -K14

    K44 = 4*EI*k2*(3*EI + L*k1) / Delta

    K4 = sym.Matrix([
        [K11, K12, K13, K14],
        [K12, K22, K23, K24],
        [K13, K23, K33, K34],
        [K14, K24, K34, K44],
    ])
    return sym.simplify(K4)


def k_beam_rotational_springs_with_delta(E, I, L, k1, k2):
    """
    Local stiffness matrix for a Euler-Bernoulli beam with rotational springs

    Delta = 12*E**2*I**2 + 4*E*I*L*(k1 + k2) + L**2*k1*k2

    Obtained in the "equations_beam_rotational_k.py" file.
    """
    Delta = sym.symbols('Delta')
    EI = E*I

    K11 = 12*EI*(EI*(k1 + k2) + L*k1*k2) / (Delta*L**2)
    K12 = 6*EI*k1*(2*EI + L*k2) / (Delta*L)
    K13 = -K11
    K14 = 6*EI*k2*(2*EI + L*k1) / (Delta*L)

    K22 = 4*EI*k1*(3*EI + L*k2) / Delta
    K23 = -K12
    K24 = 2*EI*L*k1*k2 / Delta

    K33 = K11
    K34 = -K14

    K44 = 4*EI*k2*(3*EI + L*k1) / Delta

    K4 = sym.Matrix([
        [K11, K12, K13, K14],
        [K12, K22, K23, K24],
        [K13, K23, K33, K34],
        [K14, K24, K34, K44],
    ])
    return sym.simplify(K4)


def column_lateral_stiffness_rot_springs(E, I, L, k_lower, k_upper):
    """
    Lateral stiffness of a column modeled as an Euler-Bernoulli beam
    with rotational springs at both ends.

    Inputs:
        E, I : material/section properties
        L        : story height
        k_lower, k_upper  : rotational springs at bottom/top

    Returns:
        - SymPy Matrix with simplified entries if symbols are present.
        - numpy.ndarray (float) if all inputs are numeric.

    Obtained in file 1_equations_1_Fh.py.
    """
    a1 = k_lower * L / (E * I)
    a2 = k_upper * L / (E * I)
    base = E * I / L**3
    # F/u with u=1 at the top, u=0 at the base; symmetric spring supports
    k = base * (12 - 36*(4 + a1 + a2) / (12 + 4*(a1 + a2) + a1*a2))

    return sym.simplify(sym.factor(sym.together(k))) if _is_symbolic(E, I, L, k_lower, k_upper) else k


def column_stiffness(E, I, L):
    """
    Lateral stiffness of a column modeled as an Euler-Bernoulli beam

    Inputs:
        E, I : material/section properties
        L        : story height

    Returns:
        - SymPy Matrix with simplified entries if symbols are present.
        - numpy.ndarray (float) if all inputs are numeric.
    """
    k = 12 * E * I / L**3

    return sym.simplify(sym.factor(sym.together(k))) if _is_symbolic(E, I, L) else k


def column_stiffness_in_frame(E, Ic, Lc, Ib, Lb):
    """
    Lateral stiffness of a column modeled as an Euler-Bernoulli beam
    in a frame with a beam not infinitely rigid.

    See: "Optimal beam-to-column stiffness ratio of portal frames
    under lateral loads" (Pedro Silva et al) [or derive it from
    static condensation of a frame]
    """
    alpha = Ib / Ic
    kappa = Lb / Lc
    k = (6 * alpha + kappa) / (6 * alpha + 4 * kappa) * (24 * E * Ic / Lc**3)  # frame stiffness
    k = k / 2  # column stiffness

    return sym.simplify(sym.factor(sym.together(k))) if _is_symbolic(E, Ic, Lc, Ib, Lb) else k


def column_lateral_stiffness_with_beam_in_flexion(E, I_c, I_b, h, L, k1, k2):
    """
    Returns lateral stiffness of a column belonging to a frame with:
    - Two columns modeled as Euler-Bernoulli beams with rotational springs at both ends
    and clamped supports
    - One beam modeled as Euler-Bernoulli beam (no springs)
    - Infinite axial rigidity

    Remark: obtained in equations_frame_beam_rotational_k.py
    """
    k_x_frame = 12*E*I_c*(6*E*I_b*I_c*k1 + 6*E*I_b*I_c*k2 + 6*I_b*h*k1*k2 + I_c*L*k1*k2)/(h**2*(36*E**2*I_b*I_c**2 + 12*E*I_b*I_c*h*k1 + 12*E*I_b*I_c*h*k2 + 6*E*I_c**2*L*k2 + 3*I_b*h**2*k1*k2 + 2*I_c*L*h*k1*k2))
    k_x = k_x_frame / 2
    return k_x


def stiffness_matrix_level_rigid_frame(E, I1, I2, Lc, B, H):
    """
    Story stiffness matrix K_level for a rigid in-plane diaphragm with 4 identical corner columns.
    Global DOFs: [Ux, Uy, Thetaz].

    Inputs:
        E, I1, I2 : material/section properties (I1 about x-axis, I2 about y-axis)
        Lc        : story height
        B, H      : diaphragm plan dimensions (B along x, H along y)

    Returns:
        - SymPy Matrix with simplified entries if symbols are present.
        - numpy.ndarray (float) if all inputs are numeric.
    """
    # Column lateral stiffness in each global translation
    k_col_x = column_stiffness(E, I2, Lc)  # governs Ux
    k_col_y = column_stiffness(E, I1, Lc)  # governs Uy

    # Diagonal story stiffness (symmetry -> no couplings)
    K_xx = 4 * k_col_x
    K_yy = 4 * k_col_y
    K_tt = k_col_x * H**2 + k_col_y * B**2

    K_sym = sym.Matrix([[K_xx, 0, 0],
                       [0,   K_yy, 0],
                       [0,     0,  K_tt]])

    if _is_symbolic(E, I1, I2, Lc, B, H, k_col_x, k_col_y):
        # Simplify entry-wise to keep expressions compact
        K_simpl = K_sym.applyfunc(
            lambda e: sym.simplify(sym.factor(sym.together(e))))
        return K_simpl
    else:
        return np.array(K_sym.tolist(), dtype=float)


def stiffness_matrix_level_rigid_X_frame(E, Ic_x, Ic_y, Lc, Ib, B, H):
    """
    Story stiffness matrix K_level for a rigid in-plane diaphragm with 4 identical corner columns.
    Global DOFs: [Ux, Uy, Thetaz].

    IMPORTANT: for this case, Y and Torsional directions are not computed
    (will be done later)
    """
    # Column lateral stiffness in each global translation
    k_col_x = column_stiffness(E, Ic_y, Lc)  # governs Ux
    k_col_y = 0  # will be calculated later

    # Diagonal story stiffness (symmetry -> no couplings)
    K_xx = 4 * k_col_x
    K_yy = 0  # will be calculated later
    K_tt = 0   # will be calculated later

    K_sym = sym.Matrix([[K_xx, 0, 0],
                       [0,   K_yy, 0],
                       [0,     0,  K_tt]])

    if _is_symbolic(E, Ic_x, Ic_y, Lc, Ib, B, H):
        # Simplify entry-wise to keep expressions compact
        K_simpl = K_sym.applyfunc(
            lambda e: sym.simplify(sym.factor(sym.together(e))))
        return K_simpl
    else:
        return np.array(K_sym.tolist(), dtype=float)


def stiffness_matrix_level(E, I1, I2, Lc, kx1, kx2, ky1, ky2, B, H):
    """
    Story stiffness matrix K_level for a rigid in-plane diaphragm with 4 identical corner columns.
    Global DOFs: [Ux, Uy, Thetaz].

    Inputs:
        E, I1, I2 : material/section properties (I1 about x-axis, I2 about y-axis)
        Lc        : story height
        kx1, kx2  : rotational springs about x at bottom/top (affect bending in y -> Uy)
        ky1, ky2  : rotational springs about y at bottom/top (affect bending in x -> Ux)
        B, H      : diaphragm plan dimensions (B along x, H along y)

    Returns:
        - SymPy Matrix with simplified entries if symbols are present.
        - numpy.ndarray (float) if all inputs are numeric.

    Remark:
        Obtained in file 1_equations_2_lumped_model.py.
    """
    # Column lateral stiffness in each global translation
    k_col_x = column_lateral_stiffness_rot_springs(E, I2, Lc, ky1, ky2)  # governs Ux
    k_col_y = column_lateral_stiffness_rot_springs(E, I1, Lc, kx1, kx2)  # governs Uy

    # Diagonal story stiffness (symmetry -> no couplings)
    K_xx = 4 * k_col_x
    K_yy = 4 * k_col_y
    K_tt = k_col_x * H**2 + k_col_y * B**2

    K_sym = sym.Matrix([[K_xx, 0, 0],
                       [0,   K_yy, 0],
                       [0,     0,  K_tt]])

    if _is_symbolic(E, I1, I2, Lc, kx1, kx2, ky1, ky2, B, H, k_col_x, k_col_y):
        # Simplify entry-wise to keep expressions compact
        K_simpl = K_sym.applyfunc(
            lambda e: sym.simplify(sym.factor(sym.together(e))))
        return K_simpl
    else:
        return np.array(K_sym.tolist(), dtype=float)


def stiffness_matrix_level_beam_in_flexion(E, Ic_y, Ic_x, Lc, Ib_1_y, Ib_2_y, kx1, kx2, ky1, ky2, B, H):
    """
    Story stiffness matrix K_level for a 3D frame accounting for:
        - Beam stiffness
        - Rotational spring for column

    Inputs:
        E, I1, I2 : material/section properties (I1 about x-axis, I2 about y-axis)
        Lc        : story height
        kx1, kx2  : rotational springs about x at bottom/top (affect bending in y -> Uy)
        ky1, ky2  : rotational springs about y at bottom/top (affect bending in x -> Ux)
        B, H      : diaphragm plan dimensions (B along x, H along y)

    Returns:
        - SymPy Matrix with simplified entries if symbols are present.
        - numpy.ndarray (float) if all inputs are numeric.

    Remark:
        Obtained in file 1_equations_2_lumped_model.py.
    """
    # Column lateral stiffness in each global translation
    k_col_x = column_lateral_stiffness_with_beam_in_flexion(E, Ic_y, Ib_1_y, Lc, B, ky1, ky2)  # governs Ux
    k_col_y = column_lateral_stiffness_with_beam_in_flexion(E, Ic_x, Ib_2_y, Lc, H, kx1, kx2)   # governs Uy

    # Diagonal story stiffness (symmetry -> no couplings)
    K_xx = 4 * k_col_x
    K_yy = 4 * k_col_y
    K_tt = k_col_x * H**2 + k_col_y * B**2

    K_sym = sym.Matrix([[K_xx, 0, 0],
                       [0,   K_yy, 0],
                       [0,     0,  K_tt]])

    if _is_symbolic(E, Ic_y, Ic_x, Lc, Ib_1_y, Ib_2_y, kx1, kx2, ky1, ky2, B, H):
        # Simplify entry-wise to keep expressions compact
        K_simpl = K_sym.applyfunc(
            lambda e: sym.simplify(sym.factor(sym.together(e))))
        return K_simpl
    else:
        return np.array(K_sym.tolist(), dtype=float)


def open_SAP2000(path, sapfile_name):

    # A) Delete previous mass and stiffness matrices files (important for proper overwrite)
    delete_mass_stiffness_matrices_files(
        path, sapfile_name)

    # B) Open model -> Run analysis (generate files) -> Get frequencies and modeshapes
    FilePath = os.path.join(path, sapfile_name)
    mySapObject = sap2000.app_start()
    SapModel = sap2000.open_file(mySapObject, FilePath)
    sap2000.unlock_model(SapModel)

    return mySapObject, SapModel


def delete_mass_stiffness_matrices_files(filepath, filename) -> None:

    mass_matrix_file = filename.replace('.sdb', '.TXM')
    stiffness_matrix_file = filename.replace('.sdb', '.TXK')
    mass_matrix_path = os.path.join(filepath, mass_matrix_file)
    stiffness_matrix_path = os.path.join(filepath, stiffness_matrix_file)
    if os.path.exists(mass_matrix_path):
        os.remove(mass_matrix_path)
    if os.path.exists(stiffness_matrix_path):
        os.remove(stiffness_matrix_path)


def prepare_model(path, sapfile_name):
    """
    Verifies the existence of the SAP2000 model and creates a copy in the 'log' folder.

    Raises
    ------
    FileNotFoundError
        If the original .sdb file is not found in the specified path.
    """
    original_path = os.path.join(path, sapfile_name)
    if not os.path.isfile(original_path):
        raise FileNotFoundError(
            f"[ERROR] No se encontró el archivo SAP2000: {original_path}")

    path_log = prepare_log_folder_v2(
        path, sapfile_name)

    # copy_and_rename_file(
    #     path, sapfile_name, sapfile_reduced_name)
    # _ = prepare_log_folder_v2(path, sapfile_reduced_name)

    return path_log


def prepare_log_folder_v2(sap2000_model_path: str, sdb_filename: str) -> str:
    """
    Prepares the log folder by copying the original .sdb model into it.

    This function:
        - Creates the 'log/' folder inside sap2000_model_path if it doesn't exist.
        - Deletes any existing .sdb file in the 'log/' folder with the same name as sdb_filename.
        - Copies the original .sdb model from sap2000_model_path into 'log/'.

    Parameters
    ----------
    sap2000_model_path : str
        Path where the original .sdb file is located.

    sdb_filename : str
        Name of the SAP2000 model file (e.g., 'model.sdb').

    Returns
    -------
    log_filepath : str
        Full path to the copied .sdb file inside the 'log/' folder.

    Raises
    ------
    RuntimeError
        If the copy operation fails.
    """
    original_filepath = os.path.join(sap2000_model_path, sdb_filename)
    log_folder = os.path.join(sap2000_model_path, 'log')
    log_filepath = os.path.join(log_folder, sdb_filename)

    # Ensure log folder exists
    os.makedirs(log_folder, exist_ok=True)

    # Remove existing .sdb in log folder if it exists
    try:
        if os.path.isfile(log_filepath):
            os.remove(log_filepath)
    except OSError as e:
        print(f"[WARN] Could not delete previous .sdb in log/: {e}")

    # Copy the original .sdb to log folder
    try:
        shutil.copy(original_filepath, log_filepath)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to copy .sdb file to log/: {e}")

    return log_folder


def copy_and_rename_file(path: str, original_name: str, new_name: str) -> str:
    """
    Copy a file from the specified path and rename it.
    """
    source = os.path.join(path, original_name)
    destination = os.path.join(path, new_name)
    shutil.copy2(source, destination)
    return destination


def get_modal_properties(SapModel):

    if sap2000.get_case_status(SapModel, 'MODAL')['status_code'] != 4:
        # Ensure the model is run before retrieving names
        sap2000.run_analysis(SapModel)

    frequencies = sap2000.get_modalfrequencies(SapModel)
    Name_points_group = "ALL"
    disp_modeshapes_all = sap2000.get_displmodeshapes(
        Name_points_group, SapModel)
    Phi, Phi_id = build_Phi(disp_modeshapes_all, active_dofs=[
        'U1', 'U2', 'U3', 'R1', 'R2', 'R3'])
    frequencies_list = [frequencies[key]['Frequency']
                        for key in list(frequencies)]

    return frequencies_list, Phi, Phi_id


def build_Phi(disp_modeshapes, active_dofs=['U1', 'U2', 'U3'], num_modes=None):
    """
    Function Duties:
        Get the matrix of mode shapes for the selected DOFs
    Input:
        disp_modeshapes: dictionary with the mode shapes coming from
            sap2000.get_displmodeshapes
        active_dofs: list of strings with the DOFS to be included in Phi
        num_modes: number of modes to be included in Phi
    Return:
        Phi: matrix of mode shapes for the selected DOFs and number of modes
        Phi_id: list of strings with the joint names and DOFs for each
            column of Phi
    """
    if num_modes is None:
        num_modes = len(disp_modeshapes)

    joint_id_sorted = sort_list_string(
        disp_modeshapes[list(disp_modeshapes)[0]]['Joint_id'])
    Phi = np.zeros((len(joint_id_sorted) * len(active_dofs), num_modes))
    Phi_id = list()
    for count, dof in enumerate(active_dofs):
        Phi_id += [i + f'_{dof}' for i in joint_id_sorted]
        for mode in range(num_modes):
            mode_label = list(disp_modeshapes)[mode]
            joint_order = [disp_modeshapes[mode_label]
                           ['Joint_id'].index(item) for item in joint_id_sorted]
            Phi[len(joint_id_sorted)*count:len(joint_id_sorted)*(count+1),
                mode] = [disp_modeshapes[mode_label][dof][i] for i in joint_order]

    return Phi, Phi_id


def sort_list_string(list_str):
    """
    Function duties:
        Convert a list of string-numbers, that can start by "~",
        into a sorted list.
    Example:
        list_str = ['6', '7', '~3', None, '~5', '~10']
        sorted_list = ['6', '7', '~3', '~5', '~10', None]
    """
    none_elements = len([i for i in list_str if i is None])
    list_str = [i for i in list_str if i is not None]
    if isinstance(list_str[0], float):
        sorted_list = sorted(list_str)
    else:
        regular_numbers = [s for s in list_str if not s.startswith('~')]
        tilde_numbers = [s for s in list_str if s.startswith('~')]
        regular_numbers = sorted(regular_numbers, key=lambda x: float(x))
        tilde_numbers = sorted(tilde_numbers, key=lambda x: float(x[1:]))
        sorted_list = regular_numbers + tilde_numbers

    sorted_list += [None]*none_elements

    return sorted_list


def update_Phi_with_constraints(Phi, Phi_id, constraints):
    """
    Update a mode shape matrix (Phi) by removing slave DOFs and appending equivalent constraint DOFs.

    This function transforms a mode shape matrix based on a set of point-object constraints:
    - Removes rows corresponding to slave DOFs.
    - Computes equivalent constraint DOFs using the pseudoinverse of the constraint matrix.
    - Appends the computed constraint DOFs to the reduced matrix.
    - Returns the updated matrix and corresponding DOF identifiers.

    Parameters
    ----------
    Phi : numpy.ndarray
        Mode shape matrix of shape (n_dofs, n_modes), where each row corresponds to a DOF in `Phi_id`.
    Phi_id : list of str
        Labels of the DOFs corresponding to the rows of `Phi` (e.g., ["3_U1", "3_U2", ...]).
    constraints : dict
        Dictionary containing constraint data with the following keys:
            - 'slave_dofs': list of str
                DOF labels to be removed (slaves).
            - 'constraints_dofs': list of str
                DOF labels representing the constraints (masters).
            - 'C_inv': numpy.ndarray
                Pseudoinverse of the constraint matrix, used to map slave DOFs to constraint DOFs.

    Returns
    -------
    Phi_upd : numpy.ndarray
        Updated mode shape matrix with slave rows removed and constraint rows appended.
    Phi_id_upd : list of str
        Updated list of DOF labels corresponding to `Phi_upd`.
    """
    # Retrieve variables
    slave_dofs = constraints['slave_dofs']
    constraints_dofs = constraints['constraints_dofs']
    C_inv = constraints['C_inv']

    # Put the variables sorted w.r.t. Phi_id
    order = [slave_dofs.index(dof) for dof in Phi_id if dof in slave_dofs]
    slave_dofs_ordered = [slave_dofs[i] for i in order]
    C_inv_ordered = C_inv[:, order]  # reorder columns of C_inv

    idx_slaves = [Phi_id.index(i) for i in slave_dofs_ordered]
    Phi_constraints = C_inv_ordered @ Phi[idx_slaves, :]
    Phi_upd = np.delete(Phi, idx_slaves, axis=0)
    Phi_upd = np.vstack((Phi_upd, Phi_constraints))
    Phi_id_upd = [
        i for i in Phi_id if i not in slave_dofs_ordered] + constraints_dofs

    return Phi_upd, Phi_id_upd


def check_Phi_id_coverage(Phi_id, joints_matrix_index) -> None:
    """
    Checks that all non-NaN DOFs in joints_matrix_index are present in Phi_id.

    Parameters:
        Phi_id (List[str]): List of DOF labels (e.g., '1_U1') defining the modal shape order.
        joints_matrix_index (pd.DataFrame): DataFrame with columns
            ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'].

    Raises ValueError:
    """
    expected_dofs = []

    dof_columns = ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']
    for _, row in joints_matrix_index.iterrows():
        joint_label = row['Joint_Label']
        for dof in dof_columns:
            if not np.isnan(row[dof]):
                dof_label = f"{joint_label}_{dof}"
                expected_dofs.append(dof_label)

    # Identify missing DOFs
    missing_dofs = [dof for dof in expected_dofs if dof not in Phi_id]

    # Remove constrained if they are defined but do not affect

    if len(missing_dofs) > 0:
        raise ValueError(
            f"Missing DOFs in Phi_id: {missing_dofs}. "
            "Ensure that all non-NaN DOFs in joints_matrix_index are included in Phi_id."
            "Hint: outils.build_Phi may require active_dofs=['U1', 'U2', 'U3', 'R1', 'R2', 'R3']."
        )


def generalized_eig_singular(M, K, tol=1e-12):
    # SVD de la matriz de masa
    U, s, _ = np.linalg.svd(M)

    # Determinar rango efectivo
    r = np.sum(s > tol)

    # Base ortonormal del subespacio de rango
    Ur = U[:, :r]

    # Proyección de las matrices
    M_r = Ur.T @ M @ Ur
    K_r = Ur.T @ K @ Ur

    # Resolver el problema generalizado en el subespacio reducido
    lam, phi_r = eigh(K_r, M_r)

    # Reconstruir modos en espacio original
    phi = Ur @ phi_r
    f = np.sqrt(np.abs(lam)) / (2*np.pi)  # frecuencias en Hz

    return f, phi


def clean_phi_matrix(Phi, Phi_id, joints_matrix_index):
    """
    Removes rows in Phi corresponding to inactive DOFs (NaNs in joints_matrix_index).

    Parameters:
        Phi (np.ndarray): Original mode shape matrix of shape (n_dofs, n_modes).
        Phi_id (List[str]): List of DOF labels corresponding to rows of Phi (e.g., '1_U1').
        joints_matrix_index (pd.DataFrame): DataFrame with columns
            ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'] indicating active DOFs.

    Returns:
        Phi_reduced (np.ndarray): Mode shape matrix with only active DOFs.
        Phi_id_reduced (List[str]): List of DOF labels corresponding to active rows.
    """
    valid_indices = []
    Phi_id_reduced = []

    for idx, dof_label in enumerate(Phi_id):
        joint, dof = dof_label.split('_')
        row = joints_matrix_index[joints_matrix_index['Joint_Label'] == joint]

        if not row.empty and not np.isnan(row[dof].values[0]):
            valid_indices.append(idx)
            Phi_id_reduced.append(dof_label)

    Phi_reduced = Phi[valid_indices, :]

    return Phi_reduced, Phi_id_reduced



def reorder_K_M_matrix_to_phi_id(input_matrix, Phi_id_reduced, joints_matrix_index):
    """
    Reorders a mass or stiffness matrix to match the order defined in Phi_id_reduced,
    considering only the active DOFs (non-NaN entries in joints_matrix_index).

    Parameters:
        input_matrix (np.ndarray): Original matrix (mass or stiffness) with only active DOFs.
        Phi_id_reduced (List[str]): List of active DOF labels (e.g., '1_U1'), typically from clean_phi_matrix.
        joints_matrix_index (pd.DataFrame): DataFrame with columns
            ['Joint_Label', 'U1', 'U2', 'U3', 'R1', 'R2', 'R3'].

    Returns:
        reordered_matrix (np.ndarray): Matrix reordered according to the order in Phi_id_reduced.
    """
    # Build mapping from Phi_id to global DOF indices
    dof_map = []
    for dof_label in Phi_id_reduced:
        joint, dof = dof_label.split('_')
        row = joints_matrix_index[joints_matrix_index['Joint_Label'] == joint]
        if not row.empty and not np.isnan(row[dof].values[0]):
            dof_index = int(row[dof].values[0])
            dof_map.append(dof_index)
        else:
            raise ValueError(f"DOF '{dof_label}' not found or is NaN in joints_matrix_index.")

    input_matrix = input_matrix[np.ix_(dof_map, dof_map)]

    return input_matrix


def mass_matrix_level(rho, t, B, H, m_corner=0, m_center=0):
    """
    Build the 3x3 mass matrix M for a rigid in-plane diaphragm with DOFs [Ux, Uy, Thetaz].

    Parameters
    ----------
    rho : volumetric density (kg/m^3)
    t   : slab thickness (m)
    B,H : slab plan dimensions (m)
    m_corner : point mass at each corner (kg) -> 4 corners total
    m_center : point mass at center (kg)

    Returns
    -------
    SymPy Matrix if any symbolic inputs are present; numpy.ndarray otherwise.
    """
    # Slab mass and its polar inertia about the geometric center
    m_slab = rho * t * B * H
    I_slab_center = m_slab * (B**2 + H**2) / 12

    # Point masses (Four equal corner masses at (±B/2, ±H/2)):
    I_corners_center = m_corner * (B**2 + H**2)

    # Totals referred to the geometric center
    m_tot = m_slab + 4*m_corner + m_center
    I_center = I_slab_center + I_corners_center  # (m_center contributes zero at center)

    # Components
    M11, M22 = m_tot, m_tot
    M12, M13, M23 = 0, 0, 0
    M33 = I_center

    M_sym = sym.Matrix([
        [M11,  M12, M13],
        [M12,  M22, M23],
        [M13,  M23, M33]
    ])

    # Symbolic vs numeric return
    if _is_symbolic(rho, t, B, H, m_corner, m_center):
        return M_sym.applyfunc(lambda e: sym.simplify(sym.together(sym.factor(e))))
    else:
        return np.array(M_sym.tolist(), dtype=float)


def lumped_mass_from_beam(rho, A, L):
    """
    Calculate lumped mass at one node.

    Remark: simplified method (for exact one, use mass matrix of beam element).

    Parameters
    ----------
    rho : volumetric density (kg/m^3)
    A   : cross-sectional area (m^2)
    L   : length of the beam (m)

    Returns
    -------
    m_lumped : mass at each node (kg)
    """
    total_mass = rho * A * L
    m_lumped = total_mass / 2
    return m_lumped


def _build_tridiag_from_interstories(k_list: Sequence[Any]):
    """Return NxN tridiagonal matrix for a single direction given [k1,...,kN] interstory constants."""
    N = len(k_list)
    if N == 0:
        raise ValueError("Empty interstory list.")
    symb = _is_symbolic(*k_list)
    # diagonal and off-diagonal
    diag = []
    for i in range(N):
        if i == 0:
            val = k_list[0] if N == 1 else (k_list[0] + k_list[1])
        elif i == N - 1:
            val = k_list[N - 1]
        else:
            val = k_list[i] + k_list[i + 1]
        diag.append(val)
    off = [-k_list[i + 1] for i in range(N - 1)]

    if symb:
        K = sym.zeros(N)
        for i in range(N):
            K[i, i] = sym.simplify(sym.together(sym.factor(diag[i])))
        for i in range(N - 1):
            K[i, i + 1] = -sym.simplify(sym.together(sym.factor(k_list[i + 1])))
            K[i + 1, i] = K[i, i + 1]
        return K
    else:
        K = np.zeros((N, N), dtype=float)
        for i in range(N):
            K[i, i] = float(diag[i])
        for i in range(N - 1):
            v = float(off[i])
            K[i, i + 1] = v
            K[i + 1, i] = v
        return K


def _build_diag_from_levels(val_list):
    """
    Return an NxN diagonal matrix from [v1,...,vN].
    Uses SymPy if any symbolic entry; otherwise returns numpy.ndarray.
    """
    N = len(val_list)
    if N == 0:
        raise ValueError("Empty list for diagonal assembly.")
    symb = _is_symbolic(*val_list)

    if symb:
        D = sym.zeros(N)
        for i, v in enumerate(val_list):
            D[i, i] = sym.simplify(sym.together(sym.factor(v)))
        return D
    else:
        D = np.zeros((N, N), dtype=float)
        for i, v in enumerate(val_list):
            D[i, i] = float(v)
        return D


def assemble_interstory_global_from_levels(levels):
    """
    Assemble the global interstory stiffness K (3N x 3N) with DOF order:
    [Ux_1..Ux_N | Uy_1..Uy_N | Thetaz_1..Thetaz_N].

    levels is a list of StoryLevelRigidBeam objects.
    """
    N = len(levels)

    if N == 0:
        raise ValueError("Levels list is empty.")

    # Stiffness / mass triplets per level
    kx_list, ky_list, kt_list = [], [], []
    mx_list, my_list, mt_list = [], [], []
    for lev in levels:
        kx, ky, kt, mx, my, mt = lev.level_diagonals()
        kx_list.append(kx)
        ky_list.append(ky)
        kt_list.append(kt)
        mx_list.append(mx)
        my_list.append(my)
        mt_list.append(mt)

    symb = _is_symbolic(*kx_list, *ky_list, *kt_list, *mx_list, *my_list, *mt_list)

    Kx = _build_tridiag_from_interstories(kx_list)
    Ky = _build_tridiag_from_interstories(ky_list)
    Kt = _build_tridiag_from_interstories(kt_list)
    Mx = _build_diag_from_levels(mx_list)
    My = _build_diag_from_levels(my_list)
    Mt = _build_diag_from_levels(mt_list)
    dofs = [f'{i+1}_U1' for i in range(N)] + \
           [f'{i+1}_U2' for i in range(N)] + \
           [f'{i+1}_R3' for i in range(N)]

    # block-diagonal with order [X-block | Y-block | Theta-block]
    if symb:
        K = sym.BlockDiagMatrix(Kx, Ky, Kt)
        M = sym.BlockDiagMatrix(Mx, My, Mt)       
    else:
        K = np.zeros((3*N, 3*N))
        K[0:N, 0:N] = Kx
        K[N:2 * N, N:2 * N] = Ky
        K[2 * N:3 * N, 2 * N:3 * N] = Kt
        M = np.zeros((3*N, 3*N), dtype=float)
        M[0:N,       0:N]       = Mx
        M[N:2*N,     N:2*N]     = My
        M[2*N:3*N,   2*N:3*N]   = Mt

    return K, M, dofs


def prune_K_and_Phi(K: np.ndarray, Phi_id_K_M: list[str]):
    """
    Elimina de K las filas/columnas que son todo ceros
    y elimina las entradas correspondientes de Phi_id_K_M.

    Parámetros
    ----------
    K : np.ndarray (matriz cuadrada)
    Phi_id_K_M : list[str] con misma longitud que K.shape[0]

    Returns
    -------
    K_reduced : np.ndarray
    Phi_reduced : list[str]
    mask_kept : np.ndarray[bool]  # por si luego quieres mapear índices originales
    """
    nonzero_rows = ~(np.all(K == 0, axis=1))
    nonzero_cols = ~(np.all(K == 0, axis=0))
    keep_mask = nonzero_rows & nonzero_cols
    K_reduced = K[np.ix_(keep_mask, keep_mask)]
    Phi_reduced = [phi for phi, keep in zip(Phi_id_K_M, keep_mask) if keep]

    return K_reduced, Phi_reduced, keep_mask


def generalized_eig_singular(M, K, tol=1e-12):
    # SVD de la matriz de masa
    U, s, _ = np.linalg.svd(M)

    # Determinar rango efectivo
    r = np.sum(s > tol)

    # Base ortonormal del subespacio de rango
    Ur = U[:, :r]

    # Proyección de las matrices
    M_r = Ur.T @ M @ Ur
    K_r = Ur.T @ K @ Ur

    # Resolver el problema generalizado en el subespacio reducido
    lam, phi_r = eigh(K_r, M_r)

    # Reconstruir modos en espacio original
    phi = Ur @ phi_r
    f = np.sqrt(np.abs(lam)) / (2*np.pi)  # frecuencias en Hz

    return f, phi



def read_sensors_from_MOVA_file(filename):
    """
    Function Duties:
        Extracts the sensors from the given file.
    Input:
        filename (str): Path to the text file.
    Output:
        sensors: dictionary containing the sensors
            (related to the file geometry).
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Locate the start of the SENSORS section
    start_marker = "SENSORS"
    start_index = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i+1  # Skip the section title and the separator line
            break

    if start_index is None:
        raise ValueError("SENSORS section not found in the file.")

    # Read the data lines until the next empty line or section
    sensors = dict()
    for line in lines[start_index:]:
        line = line.strip()
        if not line and len(sensors) > 0:  # Stop reading at an empty line
            break
        line = line.split()
        if len(line) == 1:  # skip the separator indicating the total number of sensors
            continue
        if line:
            ch = len(sensors) + 1
            node = line[0]
            direction = [int(i) for i in line[1:]]
            sensors[f'Channel_{ch}'] = {'node': node, 'dir': direction}
            # if line[0] in list(sensors):  # this node already has a sensor
            #     old_dofs = sensors[line[0]]
            #     new_dofs = [float(i) for i in line[1:]]
            #     sensors[line[0]] = list(np.array(old_dofs) + np.array(new_dofs))
            # sensors[line[0]] = [float(i) for i in line[1:]]

    return sensors


def extract_info_from_name(test_name: str) -> dict:
    """
    Function Duties:
        - Extract information from the test name
    """
    information = dict()
    # Check for standalone numbers
    match = re.search(r'\d+', test_name)
    if match:
        information['test_number'] = int(match.group())
    else:
        raise ValueError('No test number found in the test name')

    # Assign data type
    if 'acc' in test_name:
        information['data_type'] = 'Acceleration'
    else:
        information['data_type'] = 'Strain'

    # Check for 'Nsg'
    match_sg = re.search(r'(\d+)sg', test_name)
    if match_sg:
        information['num_channels'] = int(match_sg.group(1))
    else:
        information['num_channels'] = None

    # Check for 'suboptN'
    match_subopt = re.search(r'subopt(\d+)', test_name)
    if match_subopt:
        information['num_subopt'] = int(match_subopt.group(1))
    else:
        information['num_subopt'] = None

    return information


def MOVA_read_geometry(filename):
    """
    Function to read and parse geometry file from MOVA
    """
    nodes, lines, planes, color_planes = dict(), dict(), dict(), dict()
    line_id, plane_id, color_id = 1, 1, 1
    mode = None  # Track which section we are in

    with open(filename, 'r') as file:

        for line in file:
            line = line.strip()

            # Identify sections
            if "NODES" in line:
                mode = "nodes"
                continue
            elif "LINES" in line:
                mode = "lines"
                continue
            elif "SENSORS" in line:
                mode = "sensors"
                continue
            elif "PLANES" in line:
                mode = "planes"
                continue
            elif "COLOR" in line:
                mode = "color"
                continue
            elif not line or line.startswith("//"):
                continue  # Skip empty lines and comments

            # Parse Nodes
            if mode == "nodes":
                parts = line.split()
                node_id = int(parts[0])
                x, y, z = map(float, parts[1:])
                nodes[str(node_id)] = (x, y, z)

            # Parse Lines
            elif mode == "lines":
                parts = list(map(int, line.split()))
                if len(parts) == 2:
                    lines[str(line_id)] = (str(parts[0]), str(parts[1]))
                    line_id += 1

            # Parse Planes
            elif mode == "planes":
                parts = list(map(int, line.split()))
                if len(parts) >= 3:  # At least 3 points
                    planes[str(plane_id)] = [str(p) for p in parts]
                    plane_id += 1

            # Parse Planes
            elif mode == "color":
                parts = list(map(int, line.split()))
                if len(parts) >= 3:  # At least 3 points
                    color_planes[str(color_id)] = [str(p) for p in parts]
                    color_id += 1

    return nodes, lines, planes, color_planes


def compute_Y_torsional_stiffness_frame_rigid_X(K, levels):
    """
    Computes the global stiffness matrix of a steel frame with rigid X diaphragms
    and cantilever behaviour in Y direction.

    Takes as input the global stiffness matrix K with only Kxx terms computed.

    Remark:
        - It can be verified against steelframe_rigid_unions_X.sdb that the
        computation is exact
    """
    n_levels = len(levels)

    B_levels = [levels[i].B for i in range(n_levels)]
    H_levels = [levels[i].H for i in range(n_levels)]
    E_levels = [levels[i].E for i in range(n_levels)]
    Ic_x_levels = [levels[i].Ic_x for i in range(n_levels)]
    Lc_levels = [levels[i].Lc for i in range(n_levels)]

    if any ([len(set(list_i)) > 1 for list_i in [B_levels, H_levels, E_levels, Ic_x_levels, Lc_levels]]):
        print('TO BE DONE: different E, Ic_x, Lc per level')
        raise NotImplementedError
    else:
        B, H, E, Ic_x, Lc = B_levels[0], H_levels[0], E_levels[0], Ic_x_levels[0], Lc_levels[0]

    Kyy_column, _ = assemble_global_beam_np(E, Ic_x, Lc, n_levels)  # one column
    Kyy_column, _ = reduce_cantilever_np(Kyy_column)
    rot_dofs = [i for i in range(Kyy_column.shape[0]) if i % 2 == 1]
    Kyy_column, _ = static_condensation(Kyy_column, slave_dofs=rot_dofs)

    Kxx = K[0:n_levels, 0:n_levels]
    Kyy = 4 * Kyy_column

    Ktt = (H**2 / 4.0) * Kxx + (B**2 / 4.0) * Kyy

    Kxy, Kxt, Kty = np.zeros((n_levels, n_levels)), np.zeros((n_levels, n_levels)), np.zeros((n_levels, n_levels))
    K_upd = np.block([[Kxx, Kxy, Kxt],
                      [Kxy.T, Kyy, Kty],
                      [Kxt.T, Kty.T, Ktt]])
    return K_upd


def round_6_sign_digits(number):
    """
    Function duties:
        Rounds a float to 6 significant digits
    """
    formatted_number = "{:.6g}".format(number)
    rounded_number = float(formatted_number)

    return rounded_number


def prepare_log_folder(sap2000_model_path: str, sdb_filename: str) -> str:
    """
    Prepares the log folder by copying the original .sdb model into it.

    This function:
        - Creates the 'log/' folder inside sap2000_model_path if it doesn't exist.
        - Deletes any existing .sdb file in the 'log/' folder.
        - Copies the original .sdb model from sap2000_model_path into 'log/'.

    Parameters
    ----------
    sap2000_model_path : str
        Path where the original .sdb file is located.

    sdb_filename : str
        Name of the SAP2000 model file (e.g., 'model.sdb').

    Returns
    -------
    log_filepath : str
        Full path to the copied .sdb file inside the 'log/' folder.

    Raises
    ------
    RuntimeError
        If the copy operation fails.
    """
    original_filepath = os.path.join(sap2000_model_path, sdb_filename)
    log_folder = os.path.join(sap2000_model_path, 'log')
    log_filepath = os.path.join(log_folder, sdb_filename)

    # Ensure log folder exists
    os.makedirs(log_folder, exist_ok=True)

    # Remove existing .sdb in log folder if it exists
    try:
        if os.path.isfile(log_filepath):
            os.remove(log_filepath)
    except OSError as e:
        print(f"[WARN] Could not delete previous .sdb in log/: {e}")

    # Copy the original .sdb to log folder
    try:
        shutil.copy(original_filepath, log_filepath)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to copy .sdb file to log/: {e}")

    return log_filepath


def increment_filename(filename):
    """
    Function Duties:
        Increment a filename if it already exists in the folder.
    Example
    """
    base, ext = os.path.splitext(filename)
    match = re.search(r'(\d+)$', base)
    if match:  # If a number is found, increment it
        number = int(match.group(1))
        new_base = base[:match.start()] + str(number + 1)
    else:  # If no number is found, add '1' to the base name
        new_base = base + '1'

    # Generate the new filename
    new_filename = new_base + ext

    return new_filename


def get_accelerometer_channels_from_forces(forces_setup):
    """
    Function Duties:
        Processes nodal force data (e.g. from SAP2000) and builds a dictionary
        of accelerometer channels with their location and measurement direction.

    Input:
        forces_setup : dict
            Dictionary containing point force data:
            - 'PointObj' : list of node names
            - 'F1', 'F2', 'F3' : lists of force values in local axes 1, 2, 3

    Output:
        acc_channels : dict
            Dictionary with structure:
            {
                'Channel_1': {'point': 'NodeA', 'dir': [±1, 0, 0]},
                'Channel_2': {'point': 'NodeB', 'dir': [0, ±1, 0]},
                ...
            }
            - Direction vector is aligned with the nonzero force component.
            - Channel number is inferred from the absolute value of the force.
    Notes:
        - Each nonzero force component creates one channel.
        - The output is sorted by channel number for consistency.
    Remark:
        For fully coherence, this function might be enhanced to totally match
        the structure of get_SGs_positions_dictionary; for that, a new
        get_point_forces function to obtain forces_setup would be required
    """
    point_ids = forces_setup['PointObj']
    f1 = forces_setup['F1']
    f2 = forces_setup['F2']
    f3 = forces_setup['F3']
    n_items_f1 = len([i for i in range(len(f1)) if f1[i] != 0])
    n_items_f2 = len([i for i in range(len(f2)) if f2[i] != 0])
    n_items_f3 = len([i for i in range(len(f3)) if f3[i] != 0])
    n_items = n_items_f1 + n_items_f2 + n_items_f3

    acc_channels = dict()
    for i in range(len(point_ids)):

        if f1[i] != 0:
            direction = [0, 0, 0]
            direction[0] = int(np.sign(f1[i]))
            channel_id = int(abs(f1[i]))
            acc_channels[f'Channel_{channel_id}'] = {
                'point': point_ids[i],
                'dir': direction
            }
        if f2[i] != 0:
            direction = [0, 0, 0]
            direction[1] = int(np.sign(f2[i]))
            channel_id = int(abs(f2[i]))
            acc_channels[f'Channel_{channel_id}'] = {
                'point': point_ids[i],
                'dir': direction
            }
        if f3[i] != 0:
            direction = [0, 0, 0]
            direction[2] = int(np.sign(f3[i]))
            channel_id = int(abs(f3[i]))
            acc_channels[f'Channel_{channel_id}'] = {
                'point': point_ids[i],
                'dir': direction
            }

    sorted_items = sort_string_separated_by(list(acc_channels), separator='_')
    acc_channels = {key: acc_channels[key] for key in sorted_items}

    # Assign direction (i.e. U1, U2...)
    acc_channels = add_direction_to_acc_channels(acc_channels)

    return acc_channels


def sort_string_separated_by(str_list, separator='_'):
    """
    Function Duties:
        Sorts a string with numbers separated by a separator
        example: ['Element_1', 'Element_10', 'Element_2'] ->
            -> ['Element_1', 'Element_2', 'Element_10']
    Input:
        str_list: list of strings
        separator: character that separates numbers in the strings
    Output:
        str_sorted: sorted list of strings
    """
    str_sorted = sorted(str_list, key=lambda x: int(x.split(separator)[-1]))

    return str_sorted



def add_direction_to_acc_channels(channels):
    """
    Function Duties:
    - Add the 'direction' of the accelerometers to the channels dictionary.
    Input:
        channels: a dictionary containing 'dir' key, which is like
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], or [0, 0, -1].
    Output:
        channels_with_direction: a copy of the input dictionary with an additional
        'direction' key for each channel, containing the direction as a str.
        e.g. if [-1, 0, 0] -> '-U1' -> , if [0, 1, 0] -> 'U2', etc.
    """
    channels_with_direction = copy.deepcopy(channels)
    for ch in list(channels_with_direction):
        dir_vec = channels_with_direction[ch]['dir']
        if dir_vec[0] == 1:
            channels_with_direction[ch]['direction'] = 'U1'
        elif dir_vec[1] == 1:
            channels_with_direction[ch]['direction'] = 'U2'
        elif dir_vec[2] == 1:
            channels_with_direction[ch]['direction'] = 'U3'
        elif dir_vec[0] == -1:
            channels_with_direction[ch]['direction'] = '-U1'
        elif dir_vec[1] == -1:
            channels_with_direction[ch]['direction'] = '-U2'
        elif dir_vec[2] == -1:
            channels_with_direction[ch]['direction'] = '-U3'

    return channels_with_direction


def parse_xyz(filename: str):
    """
    Parse filenames like 'x=0-5_y=0-25_z=1-25_FZ.txt' and return floats.
    Hyphen '-' inside the number is treated as a decimal point.
    """
    PATTERN = re.compile(
        r"x=(?P<x>[+-]?\d+(?:-\d+)?)_y=(?P<y>[+-]?\d+(?:-\d+)?)_z=(?P<z>[+-]?\d+(?:-\d+)?)_"
        )
    m = PATTERN.search(filename)
    if not m:
        raise ValueError(f"Cannot find x/y/z pattern in: {filename}")

    def to_float(s: str) -> float:
        return float(s.replace("-", "."))

    x = to_float(m.group("x"))
    y = to_float(m.group("y"))
    z = to_float(m.group("z"))
    return x, y, z


def find_point_by_coord(all_points_coord: dict,
                        x: float,
                        y: float,
                        z: float,
                        tol: float = 1e-6):
    """
    Returns the point name whose coordinates match (x, y, z) within `tol`.
    If several points match, returns a list of all of them.
    If none match, returns None.
    """
    matches = []

    for pt_name, coord in all_points_coord.items():
        if (abs(coord["x"] - x) <= tol and
            abs(coord["y"] - y) <= tol and
            abs(coord["z"] - z) <= tol):
            matches.append(pt_name)

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return matches



def _parse_direction(direction: str):
    """
    Returns (component, sign). E.g. '-U1' -> ('U1', -1), 'R3' -> ('R3', 1).
    """
    direction = direction.strip()
    sign = -1 if direction.startswith("-") else 1
    comp = direction[1:] if direction.startswith(("-", "+")) else direction
    valid = {"U1", "U2", "U3", "R1", "R2", "R3"}
    if comp not in valid:
        raise ValueError(f"Invalid direction '{direction}'. Expected one of {sorted(valid)} with optional '+'/'-'.")
    return comp, sign


def _extract_time_steps_from_results(res: dict, load_case: str):
    """
    Build an ordered list of time steps for the given load case, considering only 'Time' steps.
    Keeps first-seen order to avoid floating rounding re-sorting artifacts.
    """
    steps = []
    seen = set()
    for lc, stype, t in zip(res["LoadCase"], res["StepType"], res["StepNum"]):
        if lc == load_case and stype == "Time" and t not in seen:
            seen.add(t)
            steps.append(t)
    if not steps:
        raise RuntimeError(f"No 'Time' steps found for load case '{load_case}'. "
                           "Ensure modal-history output is set to step-by-step.")
    return steps


def _series_for_point_component(res: dict, load_case: str, point: str, comp: str, time_steps: list[float], round_timesteps: bool = True):
    """
    Collects the series for a given point/component following the provided `time_steps`.
    """
    # Map component -> array
    comp_map = {
        "U1": res["U1"], "U2": res["U2"], "U3": res["U3"],
        "R1": res["R1"], "R2": res["R2"], "R3": res["R3"],
    }

    # Index the (StepNum -> value) for the specific point & load case & Time steps
    step_to_val = {}
    for obj, lc, stype, t, val in zip(res["Obj"], res["LoadCase"], res["StepType"], res["StepNum"], comp_map[comp]):
        if obj == point and lc == load_case and stype == "Time":
            if round_timesteps:
                t = round_6_sign_digits(t)
            step_to_val[t] = val

    if len(step_to_val) < len(time_steps):
        # We’ll insert None for missing samples.
        message = f"Point '{point}' component '{comp}' is missing some time steps for load case '{load_case}'. HINT: A problem with rounding time steps might be occurring."
        warnings.warn(message, UserWarning)

    return [step_to_val.get(t, None) for t in time_steps]


def get_channels_time_history_accelerations(
    SapModel,
    load_case: str,
    channels: dict,
    round_timesteps: bool = True,
    rel_acceleration: bool = True,
):
    """
    Returns per-channel acceleration time series for a given load case.

    Inputs
    ------
    SapModel : cSapModel
    load_case : str
        Name of the time-history load case (e.g., 'test_TH').
    channels : dict
        {'Channel_1': {'point': '27', 'direction': '-U1'}, ...}
    round_timesteps : bool, default=True
        Whether to round time steps to 6 significant digits to avoid floating point issues.
    rel_acceleration : bool, default=True
        If True, returns relative acceleration is within the reference frame local to the
            structure
        If False, absolute acceleration is given as the sum of relative acceleration and ground
            acceleration
        See https://web.wiki.csiamerica.com/wiki/spaces/kb/pages/2000234/Acceleration+FAQ

    Outputs
    -------
    results_dict : dict
        Keyed as '{point}_{direction}' (e.g., '27_-U1') with list of values.
        IMPORTANT: values are given in local axes of the point object.
    time_steps : list[float]
        The time vector corresponding to the values.
    """
    # Run the model if it is not locked
    if not sap2000.is_model_locked(SapModel):
        sap2000.run_analysis(SapModel)

    # Ensure output setup is correct for modal history
    sap2000.set_results_to_step_by_step(SapModel, load_case)

    # Query results point-by-point (robust and simple)
    # Collect results for the set of unique points referenced by channels
    unique_points = sorted({ch['point'] for ch in channels.values()})
    per_point_results = {}
    for pt in unique_points:
        res = sap2000.get_joint_accelerations(SapModel, name=str(pt), item_type=0,  # ObjectElm
                                              round_timesteps=round_timesteps,
                                              rel_acceleration=rel_acceleration)
        per_point_results[pt] = res

    # Build a canonical time vector using the first available point that has data
    time_steps = None
    for pt in unique_points:
        res = per_point_results[pt]
        try:
            time_steps = _extract_time_steps_from_results(res, load_case)
            break
        except:
            message = f"No time steps found for load case '{load_case}' at point '{pt}'. Trying next point."
            warnings.warn(message, UserWarning)
            continue
    if time_steps is None:
        raise RuntimeError(f"No time steps found for load case '{load_case}' in any requested point.")

    # Assemble per-channel series
    results_dict = {}
    for _, cfg in channels.items():
        point = str(cfg["point"])
        comp, sign = _parse_direction(cfg["direction"])
        res = per_point_results.get(point)
        if res is None:
            raise RuntimeError(f"Missing results for point '{point}'.")
        series = _series_for_point_component(res, load_case, point, comp, time_steps, round_timesteps=round_timesteps)
        # Apply sign
        series = [None if v is None else sign * v for v in series]
        key = f"{point}_{cfg['direction']}"
        results_dict[key] = series

    return results_dict, time_steps


def replace_dots_with_commas_in_file(filepath, filename) -> None:
    """
    Replace dots with commas in a text file.
    """
    full_path = os.path.join(filepath, filename)
    # Replace dots with commas
    with open(full_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = text.replace(".", ",")

    with open(full_path, "w", encoding="utf-8") as file:
        file.write(text)
