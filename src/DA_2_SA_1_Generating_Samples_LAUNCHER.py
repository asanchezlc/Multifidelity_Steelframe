
import os
import time
import sys

from helpers.outils import get_paths, get_username, load_state, save_state, remove_file
from DA_2_SA_2_Generating_Samples_function import run_samples_generation_for_SA

"""
FILE DESCRIPTION:
    Generate samples for a subsequent sensitivity analsysis (SA) in the context of generating a
    surrogate FEM model using SAP2000.

    The analysis evaluates modal responses (frequencies and modeshapes) over a set of
    input parameter samples (e.g., via Sobol sampling) and stores results in HDF5 format.

PROCESS OVERVIEW:
    - Loads or generates sampling parameters
    - Runs SAP2000 in batch mode for each sample
    - Logs outputs (frequencies, mode shapes) per iteration
    - Automatically handles interruptions and resumes from saved state

CONFIGURABLE PARAMETERS:
    - load_previous_data: If True, resumes from existing output files
    - use_GUI: Enables or disables SAP2000 GUI (useful for debugging)
    - use_threads: Enables timeout handling for SAP2000 hangs (recommended for robustness)
    - print_process: Displays iteration progress
    - max_retries: Maximum attempts before restarting the entire script

FAIL-SAFE MECHANISMS:
    - State is saved to 'state.json' upon failure
    - Restarts script if failure persists for multiple consecutive iterations
    - Backs up overwritten outputs to timestamped folders

NOTES:
    The core logic is in `run_samples_generation_for_SA()`, which performs all model runs and data management.
"""
###################################################################
# UPDATABLE PARAMETERS
###################################################################
# 0. General data
# A) Output Folder Name and Paths
sensitivity_analysis_name = 'HF_Samples'  # Assigned Folder Name
discrete_samples = True  # IMP! Only one parameter is altered at each time
# sensitivity_analysis_name = 'sensitivity_analysis_1'  # Assigned Folder Name
username = get_username()  # File containing paths
algorithm_parameters_filename = 'samples_generation_for_surrogate.txt'
# algorithm_parameters_filename = 'sensitivity_analysis_parameters.txt'
paths = get_paths(os.path.join('src', 'paths', username + '.csv'))
algorithm_parameters_path = os.path.join(paths['project'],
                                         'src', 'DA_Damage_Analysis')
algorithm_output_path = os.path.join(paths['files_Multifidelity_output'], sensitivity_analysis_name)
# algorithm_output_path = os.path.join(paths['files_DA_output'],
#                                      'sensitivity_analysis', sensitivity_analysis_name)
# sap2000_model_path = paths['sap2000_DA']
sap2000_model_path = paths['sap2000_HF']

# B) Execution parameters
load_previous_data = True  # False for a NEW Data set generation
use_GUI = False
print_process = True
seed = 2  # so that we can reproduce all results
use_threads = False  # if True, iterations are slower but process is more robust
max_retries = 20  # maximum number of retries before restarting the full code
###################################################################

###################################################################
# MAIN PROGRAM
###################################################################
# 1. Initializing variables
process_finished = False
retries, iteration_prev = 0, 0

state_filepath = os.path.join(paths['project'], 'state.json')
state = load_state(state_filepath)  # if the process has been interrupted
if state:  # if the process has been interrupted, load previous data
    if not load_previous_data:
        input(
            f'There is an existing state.json file located in {paths["project"]} which indicates that the sampling generation will continue by loading previous data. If this is not desired, interrupt the execution and delete the file. Press Enter to continue...')
    load_previous_data = state.get('load_previous_data', True)

# 2. Sensitivity Analysis Process
print('------- Starting Program for Sensitivity Analysis -------')
while not process_finished:
    process_finished, iteration = run_samples_generation_for_SA(sap2000_model_path, algorithm_parameters_filename,
                                                                algorithm_parameters_path, algorithm_output_path,
                                                                load_previous_data, seed=seed,
                                                                discrete_samples=discrete_samples,
                                                                use_GUI=use_GUI, print_process=print_process,
                                                                use_threads=use_threads)
    if not process_finished:  # Some problem occurs
        retries = retries + 1 if iteration == iteration_prev else 0
        if retries < max_retries:  # If the error occurs for the first time, retry
            print(
                f"An error occurred. Retry attempt {retries} of {max_retries}. Re-launching the main function.")
            load_previous_data, iteration_prev = True, int(iteration)
            time.sleep(5)
        else:  # restart script
            print(
                f"An error occurred. Retry attempt {retries} of {max_retries}. Restarting the FULL process.")
            save_state({'load_previous_data': True},
                       state_filepath)
            time.sleep(5)
            os.execv(sys.executable, ['python'] + sys.argv)
    else:  # Process finished successfully
        remove_file(state_filepath)
        print(
            f'Generating Samples for Sensitivity Analysis Process Completed Successfully; {iteration} iterations completed.')

###################################################################
# END OF FILE
###################################################################
