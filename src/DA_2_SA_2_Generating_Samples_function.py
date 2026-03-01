

import h5py

import numpy as np
import os
import shutil
import threading
import time

import helpers.outils as outils
import helpers.sap2000 as sap2000


"""
File content:
    Main functions used by the DA_2_LAUNCHER.py file
"""


def timeout_handler(timeout_duration, stop_event) -> None:
    """
    Function Duties:
        Process monitoring; Kills SAP2000 if it exceeds the timeout
    It is defined to be inside a thread
    """
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_duration:
            print("Timeout exceeded, killing SAP2000...")
            process_name = "SAP2000.exe"
            outils.kill_process_advanced(process_name)
            stop_event.set()  # stop_event is set to True
            break
        time.sleep(0.05)  # Verify every 0.05 seconds


def run_samples_generation_for_SA(sap2000_model_path, algorithm_parameters_filename,
                                  algorithm_parameters_path, algorithm_output_path,
                                  load_previous_data,
                                  discrete_samples=False,
                                  seed=None, use_GUI=False,
                                  print_process=False, use_threads=False):
    """
    Runs a sensitivity analysis on a FEM model by varying input parameters
    (e.g., frame releases) using pre-defined sampling (e.g., Sobol sequences)
    and evaluating the resulting modal outputs using SAP2000.

    Files updated during execution:
        - HDF5 (.h5) file storing:
            • frequencies: modal frequencies for each simulation
            • phi: displacement mode shapes
            • psi: strain mode shapes
        - fem_parameters.json: location of accelerometers and strain gauges
        - input_parameters.json: input parameters used in the analysis
        - process.txt: timestamp and duration log for each iteration
        - errors.txt: errors encountered during analysis (if any)
        - backup_<timestamp>/: folder where overwritten files are saved

    Parameters
    ----------
    sap2000_model_path : str
        Directory where the .sdb file (called sdb_filename)
        is located.

    algorithm_parameters_filename : str
        Filename of the JSON file containing the algorithm's configuration.

    algorithm_parameters_path : str
        Directory where the algorithm_parameters JSON file is located.

    algorithm_output_path : str
        Directory where outputs will be written and/or loaded from.

    load_previous_data : bool
        If True, resumes a previous run from existing data.

    use_GUI : bool, optional
        If True, SAP2000 is run with its graphical interface (default is False).

    print_process : bool, optional
        If True, prints progress and timing of each iteration (default is False).

    use_threads : bool, optional
        If True, enables a background watchdog to kill SAP2000 if it hangs (default is False).

    Returns
    -------
    Success : bool
        Whether the full sensitivity analysis completed successfully.

    i : int
        The final iteration reached (useful for debugging or tracking progress).
    """
    # Basic variable which should be updated when more input data is added
    input_data_as_group = {'JS': 'point', 'FR': 'frame', 'FOM': 'frame'}

    # 1. Read parameters and configure additional variables
    parameters_filepath = os.path.join(
        algorithm_parameters_path, algorithm_parameters_filename)
    algorithm_parameters = outils.read_parameters(parameters_filepath)
    SolverProcessType = 1 if use_GUI else 2
    if seed is not None:
        np.random.seed(seed)
        algorithm_parameters['seed'] = seed
    algorithm_parameters_outputname = algorithm_parameters_filename.replace(
        '.txt', '.json')

    # 2. Start SAP2000 (in a try-except block to handle SAP2000 errors)
    try:
        # 0. Kill SAP2000 if it is being runned (otherwise raises error)
        process_name = "SAP2000.exe"
        outils.kill_process_advanced(process_name)

        # 1. Create a log folder where the process will be run
        sdb_filename = algorithm_parameters['sap2000_filename']
        log_filepath = outils.prepare_log_folder(sap2000_model_path, sdb_filename)

        # 2. Open SAP2000
        mySapObject = sap2000.app_start(use_GUI)
        SapModel = sap2000.open_file(mySapObject, log_filepath)
        sap2000.unlock_model(SapModel)
        sap2000.set_solver(SapModel, SolverType=2, SolverProcessType=SolverProcessType,
                           NumberParallelRuns=0, ResponseFileSizeMaxMB=-1, NumberAnalysisThreads=-1,
                           StiffCase="MODAL")
        sap2000.set_ISunits(SapModel)

        # 3. Retrieve important data from the model
        # A) Retrieve geometry of the model (joints and frames)
        round_coordinates = True
        Name_points_group, Name_elements_group = "ALL", "ALL"
        all_points, all_elements, all_elements_stat = sap2000.getnames_point_elements(Name_points_group, Name_elements_group,
                                                                    SapModel)
        all_points_coord = sap2000.get_pointcoordinates(
            all_points, SapModel, round_coordinates=round_coordinates)
        all_elements_coord_connect = sap2000.get_frameconnectivity(all_points, all_elements,
                                                                SapModel, all_points_coord=all_points_coord)

        # B) Retrieve properties of each section (useful for retrieving strain mode shapes)
        element_section = sap2000.get_elementsections(all_elements, all_elements_stat, SapModel)
        all_sections = list(set([element_section[i] for i in list(element_section)]))
        section_properties_material = sap2000.get_section_information(all_sections, SapModel)

        # 3. Prepare data for the Algorithm
        # 3.1 Load previous data
        if load_previous_data:
            # 3.1 Sensitivity Analysis Data
            # A) FEM Parameters
            fem_channels = outils.load_json_serialized(
                os.path.join(algorithm_output_path, algorithm_parameters['fem_parameters_filename']))
            acc_channels = fem_channels['acc_channels']
            sg_channels = fem_channels['sg_channels']

            # B) Input parameters
            input_parameters = outils.load_json_serialized(
                os.path.join(algorithm_output_path, algorithm_parameters['input_parameters_filename']))

            # C) Output modal data
            h5py_file = os.path.join(
                algorithm_output_path, algorithm_parameters['sa_output_filename'])
            groups = ['frequencies']
            matrices = outils.read_matrices_h5py(h5py_file, groups)
            frequencies = matrices[groups[0]]

            # D) Retrieve the iteration in which we are
            starting_i = np.where(np.any(frequencies != 0, axis=(1, 2)))[0][-1] + 1
            nsamples_sobol = len(input_parameters[list(input_parameters)[0]])
            n_modes = algorithm_parameters['n_modes']
            n_sg, n_acc = len(sg_channels), len(acc_channels)

            # Save the number of samples
            algorithm_parameters['nsamples_sobol'] = nsamples_sobol
        else:
            # 0) Save all files that might be overwritten in a backup folder
            files_to_backup = [algorithm_parameters[key]
                               for key in algorithm_parameters if 'filename' in key]
            files_to_backup.append(algorithm_parameters_outputname)
            outils.backup_existing_output_files(
                algorithm_output_path, files_to_backup)

            # A) FEM Parameters
            # SGs' locations from SAP2000 load pattern
            load_pattern = "DOFs_sg"  # load pattern containing SGs
            point_forces = sap2000.get_point_loads_on_frame(
                SapModel, 'ALL', item_type=1, round_coordinates=round_coordinates,
                return_kN=True)
            sg_channels = outils.get_SGs_positions_dictionary(
                point_forces, load_pattern, all_elements_coord_connect)

            # Accelerometers' locations from SAP2000 load pattern
            load_pattern = "DOFs"
            forces_setup = sap2000.get_point_forces(
                'ALL', SapModel, load_pattern=load_pattern,
                return_kN=True)
            acc_channels = outils.get_accelerometer_channels_from_forces(
                forces_setup)

            # Save dict
            fem_channels = {'acc_channels': acc_channels,
                            'sg_channels': sg_channels}
            fem_channels_filepath = os.path.join(
                algorithm_output_path, algorithm_parameters['fem_parameters_filename'])
            outils.save_json_serialized(fem_channels, fem_channels_filepath)

            # B) Generate the input parameters
            if discrete_samples:
                input_parameters, nsamples_sobol = outils.generate_parameter_samples_from_algorithm_parameters_discrete(
                    algorithm_parameters)
            else:
                input_parameters, nsamples_sobol = outils.generate_parameter_samples_from_algorithm_parameters(
                    algorithm_parameters)
            input_parameters_filepath = os.path.join(
                algorithm_output_path, algorithm_parameters['input_parameters_filename'])
            outils.save_json_serialized(
                input_parameters, input_parameters_filepath)
            # Save the number of samples
            algorithm_parameters['nsamples_sobol'] = nsamples_sobol

            # C) Prepare variables for creating the dataset (h5py_file.create_dataset) when i==0
            n_modes = algorithm_parameters['n_modes']
            n_sg, n_acc = len(sg_channels), len(acc_channels)
            names = ['frequencies', 'phi', 'psi']
            dimensions = [(nsamples_sobol, n_modes, 1), (nsamples_sobol, n_acc,
                                                n_modes), (nsamples_sobol, n_sg, n_modes)]

            # D) Set first iteration
            starting_i = 0

    except Exception as e:
        errors_filepath = os.path.join(
            algorithm_output_path, algorithm_parameters['errors_filename'])
        with open(errors_filepath, 'a') as file:
            file.write(f'Error starting SAP2000: {e}\n')

        Success = False

        return Success, starting_i

    # groups data (reallocate in the best place)
    groups_dict = outils.check_updatable_parameters_groups(input_parameters, input_data_as_group,
                                                           all_points, all_elements, SapModel)

    # 3. Run algorithm
    sa_output_filepath = os.path.join(
        algorithm_output_path, algorithm_parameters['sa_output_filename'])
    opening_mode = 'w' if not load_previous_data else 'r+'
    with h5py.File(sa_output_filepath, opening_mode) as h5py_file:

        for i in range(starting_i, nsamples_sobol):
            # 1. Generate parameters
            start_time = time.time()  # Record the start time
            if i == 0:  # initial case: start by initial parameters
                for name, dim in zip(names, dimensions):
                    h5py_file.create_dataset(name, dim, dtype='float64')

                # Save algorithm parameters used
                algorithm_parameters_filepath = os.path.join(
                    algorithm_output_path, algorithm_parameters_outputname)
                outils.save_json_serialized(
                    algorithm_parameters, algorithm_parameters_filepath)

            # 1.1 Start timer to kill SAP2000 if it exceeds the timeout
            if use_threads:
                stop_event = threading.Event()
                timeout_duration = 60*2  # 2 minutes allowed to work
                timer_thread = threading.Thread(target=timeout_handler,
                                                args=(timeout_duration, stop_event))
                timer_thread.start()

            # Retrieve input parameters and adapt as required for SAP2000 function
            parameters_dict = {key: value[i]
                               for key, value in input_parameters.items()}
            input_data = outils.from_algorithm_parameters_to_sap2000_input(parameters_dict)

            # 2. Run analysis and get modal response
            try:
                # Run SAP2000 (which might get stuck)
                modal_results = outils.get_modal_response_SA(
                    input_data, sg_channels, acc_channels,
                    element_section, section_properties_material,
                    SapModel, groups_dict=groups_dict,
                    n_modes=n_modes)

                # Stop the timer_thread as everything worked properly
                if use_threads:
                    stop_event.set()  # so that timer_thread stops
                    timer_thread.join()  # wait until the thread finishes

            except Exception as e:
                # Stop the timer_thread before restarting
                if use_threads:
                    stop_event.set()  # so that timer_thread stops
                    timer_thread.join()  # wait until the thread finishes
                errors_filepath = os.path.join(
                    algorithm_output_path, algorithm_parameters['errors_filename'])
                with open(errors_filepath, 'a') as file:
                    parameters_string = ', '.join(
                        [f"{key}={val:.4g}" for key, val in parameters_dict.items()])
                    file.write(f'Iteration {i}: {e}; {parameters_string}\n')

                Success = False

                return Success, i

            # B) Save data
            # b.2 FEM output data
            h5py_file['frequencies'][i, :, 0] = modal_results['frequencies']
            h5py_file['phi'][i, :, :] = modal_results['Phi']
            h5py_file['psi'][i, :, :] = modal_results['Psi']

            # 8. Save process
            if print_process:
                if np.mod(i, 1) == 0:
                    print(f'Iteration {i+1}/{nsamples_sobol}')
            end_time = time.time()
            time_required = end_time - start_time
            current_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(end_time))
            process_filepath = os.path.join(
                algorithm_output_path, algorithm_parameters['process_filename'])
            with open(process_filepath, "a") as file:
                text = f"Iteration {i}: time required: {time_required:.2f} seconds, endtime: {current_time}\n"
                file.write(text)

    mySapObject.ApplicationExit(False)
    Success = True

    return Success, i
