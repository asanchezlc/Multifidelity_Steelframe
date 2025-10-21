

import comtypes.client
import numpy as np
import warnings


def app_start(use_GUI=True):
    """
    Function duties:
        Starts sap2000 application
    """
    # create API helper object
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")

    # start SAP2000 application
    mySapObject.ApplicationStart(3, use_GUI, "")

    return mySapObject


def raise_warning(process_name, ret) -> None:
    """
    Raise warning if ret=1
    """
    if ret == 1:
        message = process_name + ' was not properly retrieved in SAP2000'
        warnings.warn(message, UserWarning)


def open_file(mySapObject, FilePath):
    """
    Function duties:
        Once the application has started, open an existing SAP2000 file
    """
    # create SapModel object
    SapModel = mySapObject.SapModel

    # initialize model
    ret = SapModel.InitializeNewModel()
    raise_warning('Initialize SAP2000', ret)

    # open existing file
    ret = SapModel.File.OpenFile(FilePath)
    raise_warning('Open file', ret)

    return SapModel


def unlock_model(SapModel, lock=False) -> None:
    """
    Function duties:
        If lock=False: unlocks model
        Else: locks model
    """
    ret = SapModel.SetModelIsLocked(False)
    if lock:
        raise_warning("Unlock", ret)
    else:
        raise_warning("Lock", ret)


def get_case_status(SapModel, case_name='MODAL'):
    """
    Retrieves the status of a specific analysis case in SAP2000.

    Parameters:
        SapModel: The SAP2000 model object (SapModel from the OAPI).
        case_name (str): Name of the analysis case to check.

    Returns:
        dict: Dictionary with keys:
            - 'status_code' (int): Numeric status code.
            - 'status_description' (str): Description of the status.

        Returns {'status_code': -1, 'status_description': 'Case not found'} if the case is not present.

    Notes:
        Status codes meaning (from SAP2000 API):
            1 = Not run
            2 = Could not start
            3 = Not finished
            4 = Finished
    """
    status_map = {
        1: 'Not run',
        2: 'Could not start',
        3: 'Not finished',
        4: 'Finished'
    }

    num_items, case_names, statuses, ret = SapModel.Analyze.GetCaseStatus()

    raise_warning(f'Read {case_name} status', ret)

    if case_name in case_names:
        idx = list(case_names).index(case_name)
        code = statuses[idx]
        return {'status_code': code, 'status_description': status_map.get(code, 'Unknown status')}
    else:
        return {'status_code': -1, 'status_description': 'Case not found'}



def run_analysis(SapModel, case_name='MODAL', max_modes=None, min_modes=None) -> None:
    """
    Runs the structural analysis in SAP2000.

    Optionally sets the number of modes for a modal eigen load case before running.

    Parameters
    ----------
    SapModel : object
        The active SAP2000 model object.

    case_name : str, optional
        Name of the modal eigen load case (e.g., 'MODAL'). If provided along with max_modes,
        the number of modes is configured before running the analysis.

    max_modes : int, optional
        Maximum number of modes to compute (used with modal cases).

    min_modes : int, optional
        Minimum number of modes to compute. Defaults to max_modes if not specified.

    Raises
    ------
    Warning
        If setting the number of modes or the analysis run fails.
    """
    # Set number of modes if modal eigen case configuration is requested
    if case_name is not None and max_modes is not None:
        set_number_of_modes(SapModel, case_name=case_name,
                            max_modes=max_modes, min_modes=min_modes)

    # Run the analysis
    ret = SapModel.Analyze.RunAnalysis()
    raise_warning('Run analysis', ret)



def set_number_of_modes(SapModel, case_name="MODAL", max_modes=6, min_modes=None) -> None:
    """
    Sets the number of modes requested for a modal eigen load case.

    Parameters
    ----------
    SapModel : object
        The active SAP2000 model object.

    case_name : str
        Name of the modal eigen load case (usually 'MODAL').

    max_modes : int
        The maximum number of modes to extract.

    min_modes : int or None, optional
        The minimum number of modes to extract. If None, defaults to max_modes.

    Raises
    ------
    Warning if the setting operation fails.
    """
    if min_modes is None:
        min_modes = max_modes

    ret = SapModel.LoadCases.ModalEigen.SetNumberModes(
        case_name, max_modes, min_modes)
    raise_warning("Set number of modes in modal eigen case", ret)



def get_modalfrequencies(SapModel):
    """
    Function duties:
        Retrieve frequency associated to each mode
    Input:
        NumberResults: Number of modes want to be retrieved
    Output:
        frequency_results: dictionary with results
    """
    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    NumberResults = 0
    LoadCase, StepType, StepNum, Period = [], [], [], []
    Frequency, CircFreq, EigenValue = [], [], []

    output = SapModel.Results.ModalPeriod(NumberResults, LoadCase, StepType,
                                          StepNum, Period, Frequency, CircFreq,
                                          EigenValue)
    [NumberResults, _, _, StepNum, _, Frequency, _, _, ret] = output
    raise_warning('Get displacement mode shapes', ret)

    StepNum = [int(i) for i in StepNum]  # modeshape id in integer format

    frequency_results = dict()
    for i, mode_id in enumerate(StepNum):
        mode_label = 'Mode_' + str(mode_id)
        frequency_results[mode_label] = dict()
        frequency_results[mode_label]['Frequency'] = Frequency[i]

    return frequency_results


def set_ISunits(SapModel) -> None:
    """
    Function Duties:
        Sets units in International Sistem

    Remark:
        Equivalent to use set_units(SapModel, 10)
    """
    N_m_C = 10
    ret = SapModel.SetPresentUnits(N_m_C)
    raise_warning('setting units', ret)


def get_displmodeshapes(Name_points_group, SapModel):
    """
    Function duties:
        Retrieves displacement mode shapes for a given group of points
    Input:
        name_points_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        joint_modeshape_results: dictionary of results
    Remark: bug found:
        If we manually visualize results (i.e., show tables) in SAP2000 (from
        the interface) this function stop working well, and the table has to be
        manually opened before executing this function
    """
    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # Get results from SAP2000
    # group element (if instead of group we want to get elements, then 1)
    GroupElm = 2
    Obj = 1  # I don't know why is it necessary (but it works with any number)
    LoadCase = "MODAL"
    StepType, StepNum = [], []
    U1, U2, U3 = [], [], []
    R1, R2, R3 = [], [], []

    output = SapModel.Results.ModeShape(Name_points_group,
                                        GroupElm, Obj, LoadCase,
                                        StepType, StepNum, U1,
                                        U2, U3, R1, R2, R3)

    raise_warning('Get displacement mode shapes', ret)

    [NumberResults, Obj, Elm, LoadCase, StepType,
        StepNum, U1, U2, U3, R1, R2, R3, ret] = output
    # [NumberResults, GroupElm, _, _, _, StepNum, U1, U2, U3, R1, R2, R3, ret] = output

    # Save results into dictionary
    StepNum = [int(i) for i in StepNum]  # modeshape id in integer format
    joint_modeshape_results = dict()

    for mode in list(set(StepNum)):
        mode_label = "Mode_" + str(mode)
        joint_modeshape_results[mode_label] = dict()
        mode_id = [mode == step for step in StepNum]

        joint_modeshape_results[mode_label]['U1'] = [
            _u for _i, _u in enumerate(list(U1)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['U2'] = [
            _u for _i, _u in enumerate(list(U2)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['U3'] = [
            _u for _i, _u in enumerate(list(U3)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['R1'] = [
            _r for _i, _r in enumerate(list(R1)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['R2'] = [
            _r for _i, _r in enumerate(list(R2)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['R3'] = [
            _r for _i, _r in enumerate(list(R3)) if mode_id[_i]]
        joint_modeshape_results[mode_label]['Joint_id'] = [
            _elm for _i, _elm in enumerate(list(Elm)) if mode_id[_i]]

    return joint_modeshape_results