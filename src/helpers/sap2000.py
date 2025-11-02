
"""
File Description:
    Contains helpers that use SAP2000 OAPI

File copied from TFM_codes repository
"""
from collections import defaultdict
import comtypes.client
import numpy as np
import warnings

import helpers.outils as outils


def raise_warning(process_name, ret) -> None:
    """
    Raise warning if ret=1
    """
    if ret == 1:
        message = process_name + ' was not properly retrieved in SAP2000'
        warnings.warn(message, UserWarning)


def app_start(use_GUI=True, version=None):
    """
    Function duties:
        Starts sap2000 application
    """
    # create API helper object
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)

    if version:
        prog_id = f"CSI.SAP2000.API.SapObject.{version}"
        mySapObject = helper.CreateObjectProgID(prog_id)
    else:
        mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")

    # start SAP2000 application
    mySapObject.ApplicationStart(3, use_GUI, "")

    return mySapObject


def application_exit(SapObject, file_save=False) -> None:
    """
    Closes the SAP2000 application, optionally saving the current model.

    Parameters
    ----------
    SapObject : object
        The SAP2000 main API object (cOAPI), typically obtained from `CreateObject` or `app_start`.

    file_save : bool, optional
        If True, saves the current model before closing. Default is False.
    """

    ret = SapObject.ApplicationExit(file_save)
    raise_warning('Unable to close SAP2000 application', ret)
    SapObject = None  # Release SAP2000 COM references (good practice)


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


def set_units(SapModel, units: int) -> None:
    """
    Sets the working units in SAP2000 using the provided unit code.

    Parameters
    ----------
    SapModel : object
        The active SAP2000 model object (cSapModel).

    units : int
        Unit code as defined in the SAP2000 eUnits enumeration.
        Valid values are:
            1  → lb_in_F     (Pound, Inch, Fahrenheit)
            2  → lb_ft_F     (Pound, Foot, Fahrenheit)
            3  → kip_in_F    (Kip, Inch, Fahrenheit)
            4  → kip_ft_F    (Kip, Foot, Fahrenheit)
            5  → kN_mm_C     (kN, Millimeter, Celsius)
            6  → kN_m_C      (kN, Meter, Celsius)
            7  → kgf_mm_C    (kgf, Millimeter, Celsius)
            8  → kgf_m_C     (kgf, Meter, Celsius)
            9  → N_mm_C      (Newton, Millimeter, Celsius)
            10 → N_m_C       (Newton, Meter, Celsius)
            11 → Ton_mm_C    (Ton, Millimeter, Celsius)
            12 → Ton_m_C     (Ton, Meter, Celsius)
            13 → kN_cm_C     (kN, Centimeter, Celsius)
            14 → kgf_cm_C    (kgf, Centimeter, Celsius)
            15 → N_cm_C      (Newton, Centimeter, Celsius)
            16 → Ton_cm_C    (Ton, Centimeter, Celsius)

    Raises
    ------
    Warning if setting units fails (non-zero return code).
    """
    ret = SapModel.SetPresentUnits(units)
    raise_warning('setting units', ret)


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


def set_kN_m_C_units(SapModel) -> None:
    """
    Function Duties:
        Sets units in kN, m, C
    Remark:
        Equivalent to use set_units(SapModel, 6)
    """
    set_kN_m_C = 6
    ret = SapModel.SetPresentUnits(set_kN_m_C)
    raise_warning('setting units', ret)


def get_units(SapModel):
    """
    Retrieves the current working units set in the SAP2000 model.

    Returns
    -------
    units_code : int
        The numerical unit code as defined by SAP2000's `eUnits` enumeration.

    units_name : str
        Name of the current unit setting.

    The following unit codes are possible:

        1  → lb_in_F     (Pound, Inch, Fahrenheit)
        2  → lb_ft_F     (Pound, Foot, Fahrenheit)
        3  → kip_in_F    (Kip, Inch, Fahrenheit)
        4  → kip_ft_F    (Kip, Foot, Fahrenheit)
        5  → kN_mm_C     (kN, Millimeter, Celsius)
        6  → kN_m_C      (kN, Meter, Celsius)
        7  → kgf_mm_C    (kgf, Millimeter, Celsius)
        8  → kgf_m_C     (kgf, Meter, Celsius)
        9  → N_mm_C      (Newton, Millimeter, Celsius)
        10 → N_m_C       (Newton, Meter, Celsius)
        11 → Ton_mm_C    (Ton, Millimeter, Celsius)
        12 → Ton_m_C     (Ton, Meter, Celsius)
        13 → kN_cm_C     (kN, Centimeter, Celsius)
        14 → kgf_cm_C    (kgf, Centimeter, Celsius)
        15 → N_cm_C      (Newton, Centimeter, Celsius)
        16 → Ton_cm_C    (Ton, Centimeter, Celsius)
    """
    # Map of SAP2000 unit codes to human-readable names
    unit_names = {
        1: 'lb_in_F',
        2: 'lb_ft_F',
        3: 'kip_in_F',
        4: 'kip_ft_F',
        5: 'kN_mm_C',
        6: 'kN_m_C',
        7: 'kgf_mm_C',
        8: 'kgf_m_C',
        9: 'N_mm_C',
        10: 'N_m_C',
        11: 'Ton_mm_C',
        12: 'Ton_m_C',
        13: 'kN_cm_C',
        14: 'kgf_cm_C',
        15: 'N_cm_C',
        16: 'Ton_cm_C'
    }

    units_code = SapModel.GetPresentUnits()
    units_name = unit_names.get(units_code, f"Unknown ({units_code})")

    return units_code, units_name


def set_materials(material_dict, SapModel) -> None:
    """
    Function Duties:
        Changes materials properties
    Input:
        material_dict: dictionary as follow:
            material_dict{mat1_dict, mat2_dict...}
            mat1_dict = {'E': E1, 'u': u1, 'a': a1, 'rho': rho1}
            Properties must be defined in International System
    """
    for material in list(material_dict):
        # 1. we read material properties:
        # A) E, u, a, g
        E_0, u_0, a_0, g_0 = 0, 0, 0, 0
        output = SapModel.PropMaterial.GetMPIsotropic(
            material, E_0, u_0, a_0, g_0)
        [E_0, u_0, a_0, g_0, ret] = output
        raise_warning(f'{material} properties reading', ret)

        E, u, a, _ = float(E_0), float(u_0), float(a_0), float(g_0)

        # B) rho
        w_0, rho_0 = 0, 0
        output = SapModel.PropMaterial.GetWeightAndMass(material, w_0, rho_0)
        [type_data, rho_0, ret] = output
        raise_warning(f'{material} density reading', ret)

        rho = float(rho_0)

        # 2. We update available properties
        # A) Retrieve properties from dictionary
        mat_list = [E, u, a, rho]
        for i, mat_prop in enumerate(['E', 'u', 'a', 'rho']):
            if mat_prop in list(material_dict[material]):
                mat_list[i] = material_dict[material][mat_prop]

        [E, u, a, rho] = mat_list

        # B) Set rho
        type_data = 2  # 1: weight/volume; 2: mass/volume
        ret = SapModel.PropMaterial.SetWeightAndMass(material, type_data, rho)
        raise_warning(f'{material} rho assignment', ret)

        # C) Set E, u, g, a
        ret = SapModel.PropMaterial.SetMPIsotropic(material, E, u, a)
        raise_warning(f'{material} E, u, a assignment', ret)


def set_jointsprings(joint_spring_dict, SapModel, group=True,
                     replace=True) -> None:
    """
    Function Duties:
        Sets springs to joint groups included in joint_spring_dict dict.
        (If group=False, then individual joint names are used)
    Input:
        joint_spring_dict: dictionary whose keys are names of joint groups
        (joint names if group=False).
        For each key, k array (of 6 elements) is defined:
            k[0], k[1], k[2] = U1 [F/L], U2 [F/L], U3 [F/L]
            k[3], k[4], k[5] = R1 [FL/rad], R2 [FL/rad], R3 [FL/rad]
            If None, then that component is not modified
        Replace: If True, replaces existing spring assignments; if False,
            adds to existing springs.
    """
    local_csys = True  # although SetSpring can be made in global, GetSrping is local always

    # We loop over all groups
    for name in list(joint_spring_dict):
        k = joint_spring_dict[name]  # new spring values
        if group:
            springs_ini = outils.get_springs_group(name, SapModel)
            if not springs_ini:
                warnings.warn(f'Group {name} does not contain point objects', UserWarning)
                continue
            points = list(springs_ini)
            same_k_for_all_group = all(tuple(springs_ini[p]) == tuple(springs_ini[points[0]]) for p in points[1:])

            if same_k_for_all_group:  # fast way to assign same k to all points
                ItemType = 1  # 1 = group, 0 = object
                k0 = list(springs_ini[points[0]])
                kf = outils.merge_k(k, k0)
                output = SapModel.PointObj.SetSpring(
                    name, kf, ItemType, local_csys, replace)
                [_, ret] = output
                raise_warning('Spring assignment', ret)

            else:  # we need to assign point by point
                ItemType = 0  # 1 = group, 0 = object
                for point in springs_ini:
                    k0 = list(springs_ini[point])
                    kf = outils.merge_k(k, k0)
                    output = SapModel.PointObj.SetSpring(
                        point, kf, ItemType, local_csys, replace)
                    [_, ret] = output
                    raise_warning('Spring assignment', ret)

        else:  # individual joint names
            ItemType = 0  # 1 = group, 0 = object
            k0 = list(get_springs_pointobj(name, SapModel))
            kf = outils.merge_k(k, k0)
            output = SapModel.PointObj.SetSpring(
                name, kf, ItemType, local_csys, replace)
            [_, ret] = output
            raise_warning('Spring assignment', ret)


def get_springs_pointobj(point_name, SapModel):
    """
    Retrieves spring values in LOCAL coordinate system for a specific point object.
    """
    k = [0.0]*6
    output = SapModel.PointObj.GetSpring(point_name, k)
    k, ret = output
    raise_warning(f'Retrieve spring for {point_name} point object', ret)

    return k


def set_jointmasses(joint_mass_dict, SapModel, group=True,
                    local_csys=True, replace=True) -> None:
    """
    Assigns mass values to joints or joint groups in a SAP2000 model.

    Args:
        joint_mass_dict (dict): Dictionary with joint or group names as keys.
            For each key, a mass array `m` of 6 elements is provided:
                m[0], m[1], m[2] = U1, U2, U3 translational masses [M]
                m[3], m[4], m[5] = R1, R2, R3 rotational masses [ML²]
        SapModel: SAP2000 cSapModel object.
        group (bool): If True, names are interpreted as group names;
                      if False, as individual joint names.
        local_csys (bool): If True, assigns mass in local coordinate system.
        Replace (bool): If True, replaces existing mass assignments.
    """
    ItemType = 1 if group else 0  # 1 = group, 0 = object

    for name in joint_mass_dict:
        m = joint_mass_dict[name]
        output = SapModel.PointObj.SetMass(name, m, ItemType, local_csys, replace)
        [_, ret] = output
        raise_warning('Mass assignment', ret)


def set_framereleases(frame_releases_dict, SapModel, group=True) -> None:
    """
    Function Duties:
        Applies releases to frame groups included in frame_releases_dict
        (If group=False, then individual frame names are used)
    Input:
        frame_releases_dict:
            Dictionary whose keys are names of frame groups (obj. names if group=False).
            For each key:
                ii: list of 6 boolean elements for i extreme. Each element corresponds
                    to U1, U2, U3, R1, R2, R3: True if released, False elsewhere)
                jj: list of 6 boolean elements for j extreme. Each element corresponds
                    to U1, U2, U3, R1, R2, R3: True if released, False elsewhere)
                StartValue: list of 6 elements, corresponding to partial fixity in ext. i;
                    U1 [F/L], U2 [F/L], U3 [F/L], R1 [FL/rad], R2 [FL/rad], R3 [FL/rad]
                EndValue: analogous to StartValue; extreme j
    """
    if group:
        ItemType = 1  # group name
    else:
        ItemType = 0  # individual frame object name

    for name in list(frame_releases_dict):
        ii = frame_releases_dict[name]['ii']
        jj = frame_releases_dict[name]['jj']
        StartValue = frame_releases_dict[name]['StartValue']
        EndValue = frame_releases_dict[name]['EndValue']
        output = SapModel.FrameObj.SetReleases(
            name, ii, jj, StartValue, EndValue, ItemType)
        [ii, jj, StartValue, EndValue, ret] = output
        raise_warning('Partial fixity assignment', ret)


def get_frame_releases(SapModel, frame_obj_name):
    """
    Retrieves the partial fixity values for a specific frame object in SAP2000.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model object
    
    frame_obj_name : str
        The name of the individual frame object for which the release data will be retrieved
        (e.g. "3").

    Returns
    -------
    dict
        A dictionary containing the release and partial fixity data with the following keys:
        
        - 'ii': list of 6 booleans
            Release conditions at the I-end of the frame [U1, U2, U3, R1, R2, R3] (True if released
            else False).
        - 'jj': list of 6 booleans
            Release conditions at the J-end of the frame [U1, U2, U3, R1, R2, R3] (True if released
            else False).
        - 'StartValue': list of 6 floats
            Partial fixity values at the I-end, in units [F/L, F/L, F/L, FL/rad, FL/rad, FL/rad].
        - 'EndValue': list of 6 floats
            Partial fixity values at the J-end, in units [F/L, F/L, F/L, FL/rad, FL/rad, FL/rad].
    """
    ii = [False] * 6
    jj = [False] * 6
    StartValue = [0.0] * 6
    EndValue = [0.0] * 6

    output = SapModel.FrameObj.GetReleases(frame_obj_name, ii, jj, StartValue, EndValue)
    [ii, jj, StartValue, EndValue, ret] = output
    raise_warning(f'Retrieve partial fixity for {frame_obj_name} frame object', ret)

    frame_releases_dict = {
        'ii': list(ii),
        'jj': list(jj),
        'StartValue': list(StartValue),
        'EndValue': list(EndValue)
    }

    return frame_releases_dict


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


def getnames_point_elements(Name_points_group, Name_elements_group, SapModel, get_area_properties=False):
    """
    Function duties:
        Retrieves all point labels
    Input:
        Name_points_group: This name comes from SAP2000! (Check that the group exists)
        Name_elements_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        all_points: list with all points
        all_elements: list with all points
    Remark:
        This function has been enhanced to additionally return area objects & elements
        (if get_area_properties=True)
    Remark: bug found:
        Once the .sdb has been file we change it manually (from the SAP2000
        interface) this function does not work well, and the table has to be
        manually opened before executing this function
    """
    # Run the model if it is not locked
    if not (SapModel.GetModelIsLocked()):
        run_analysis(SapModel)

    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # A) Get all names of points
    # group element (if instead of group we want to get elements, then 1)
    GroupElm = 2
    Obj = 1  # I don't know why is it necessary (but it works with any number)
    LoadCase, StepType, StepNum = [], [], []
    U1, U2, U3 = [], [], []
    R1, R2, R3 = [], [], []

    output = SapModel.Results.ModeShape(Name_points_group,
                                        GroupElm, Obj, LoadCase,
                                        StepType, StepNum, U1,
                                        U2, U3, R1, R2, R3)
    [_, Point, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get all points names', ret)

    # B) Get all names of elements and element stations
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [_, Element, _, ElementStat, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get all element names', ret)

    all_points = list(set(Point))
    all_elements = list(set(Element))
    all_elements_stat = list(set(ElementStat))

    if get_area_properties:
        GroupElm = 2  # eItemTypeElm.GroupElm
        NumberResults = 0
        Obj, Elm, PointElm, LoadCase, StepType, StepNum = [], [], [], [], [], []
        F11, F22, F12, FMax, FMin, FAngle, FVM = [], [], [], [], [], [], []
        M11, M22, M12, MMax, MMin, MAngle, V13, V23, VMax, VAngle = [
        ], [], [], [], [], [], [], [], [], []
        output = SapModel.Results.AreaForceShell(
            Name_elements_group, GroupElm, NumberResults,
            Obj, Elm, PointElm,
            LoadCase, StepType, StepNum,
            F11, F22, F12, FMax, FMin, FAngle, FVM,
            M11, M22, M12, MMax, MMin, MAngle,
            V13, V23, VMax, VAngle
        )

        [_, AreaObj, AreaElm, _, _, _, _, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _, _, _, _, ret] = output

        raise_warning('Get all shell (area) element results', ret)

        all_area_objects = list(set(AreaObj))
        all_area_elements = list(set(AreaElm))

    if get_area_properties:
        return all_points, all_elements, all_elements_stat, all_area_objects, all_area_elements
    else:
        return all_points, all_elements, all_elements_stat


def get_frame_elem_names(SapModel, Name_elements_group='ALL'):
    """
    Function duties:
        Retrieves all frame elements (i.e. frames after meshing and analyzing)
    Input:
        Name_elements_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        all_elements (list): List of frame element IDs (from the analysis mesh)
    Remark: bug found:
        Once the .sdb has been file we change it manually (from the SAP2000
        interface) this function does not work well, and the table has to be
        manually opened before executing this function
    """
    # Run the model if it is not locked
    if not (SapModel.GetModelIsLocked()):
        run_analysis(SapModel)

    # clear all case and combo and set the modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # Retrieve frame element names
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [_, Element, _, ElementStat, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get frame element names', ret)

    all_elements_stat = list(set(ElementStat))

    return all_elements_stat


def get_area_elem_names(SapModel, Name_elements_group='ALL'):
    """
    Function duties:
        Retrieves all area elements (i.e. areas after meshing and analyzing)
    Input:
        Name_elements_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        all_area_elements (list): List of area element IDs (from the analysis mesh)
    Remark: bug found:
        Once the .sdb has been file we change it manually (from the SAP2000
        interface) this function does not work well, and the table has to be
        manually opened before executing this function
    """
    # Run the model if it is not locked
    if not (SapModel.GetModelIsLocked()):
        run_analysis(SapModel)

    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    GroupElm = 2  # eItemTypeElm.GroupElm
    NumberResults = 0
    Obj, Elm, PointElm, LoadCase, StepType, StepNum = [], [], [], [], [], []
    F11, F22, F12, FMax, FMin, FAngle, FVM = [], [], [], [], [], [], []
    M11, M22, M12, MMax, MMin, MAngle, V13, V23, VMax, VAngle = [
    ], [], [], [], [], [], [], [], [], []
    output = SapModel.Results.AreaForceShell(
        Name_elements_group, GroupElm, NumberResults,
        Obj, Elm, PointElm,
        LoadCase, StepType, StepNum,
        F11, F22, F12, FMax, FMin, FAngle, FVM,
        M11, M22, M12, MMax, MMin, MAngle,
        V13, V23, VMax, VAngle
    )

    [_, AreaObj, AreaElm, _, _, _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, _, _, _, ret] = output

    raise_warning('Get all shell (area) element results', ret)

    all_area_elements = list(set(AreaElm))

    return all_area_elements


def get_all_material_names(SapModel):
    """
    Retrieves all material property names defined in the SAP2000 model.

    Parameters:
        SapModel: COM object from SAP2000 API (e.g., SapObject.SapModel)

    Returns:
        all_materials (list): List of all defined material names

    Raises:
        RuntimeError: If the retrieval fails
    """
    NumberNames = 0
    MaterialNames = []
    output = SapModel.PropMaterial.GetNameList()
    NumberNames, MaterialNames, ret = output
    raise_warning('Get all material names', ret)

    return list(MaterialNames)


def get_point_obj_names(SapModel):
    """
    Retrieves the names of all defined point (joint) objects in the SAP2000 model.

    Parameters:
        SapModel: COM object from SAP2000 API (e.g., SapObject.SapModel)

    Returns:
        all_points (list): List of all point object names in the model

    Raises:
        RuntimeError: If the retrieval fails
    """
    NumberNames = 0
    MyName = []
    output = SapModel.PointObj.GetNameList()
    NumberNames, MyName, ret = output
    raise_warning('Get all point names', ret)

    # Return as list
    return list(MyName)


def get_frame_obj_names(SapModel):
    """
    Retrieves the names of all defined frame (line) objects in the SAP2000 model.

    Parameters:
        SapModel: COM object from SAP2000 API (e.g., SapObject.SapModel)

    Returns:
        all_frames (list): List of all frame object names in the model

    Raises:
        RuntimeError: If the retrieval fails
    """
    NumberNames = 0
    FrameNames = []
    output = SapModel.FrameObj.GetNameList()
    NumberNames, FrameNames, ret = output
    raise_warning('Get all frame names', ret)

    # Return as list
    return list(FrameNames)


def get_area_obj_names(SapModel):
    """
    Retrieves the names of all defined area objects in the SAP2000 model.

    Parameters:
        SapModel: COM object from SAP2000 API (e.g., SapObject.SapModel)

    Returns:
        all_areas (list): List of all area object names in the model

    Raises:
        RuntimeError: If the retrieval fails
    """
    NumberNames = 0
    AreaNames = []
    output = SapModel.AreaObj.GetNameList()
    NumberNames, AreaNames, ret = output
    raise_warning('Get all area names', ret)

    return list(AreaNames)


def get_point_names(Name_points_group, Name_elements_group, SapModel, get_area_properties=False):
    """
    Function duties:
        Retrieves all point labels
    Input:
        Name_points_group: This name comes from SAP2000! (Check that the group exists)
        Name_elements_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        all_points: list with all points
        all_elements: list with all points
    Remark:
        This function has been enhanced to additionally return area objects & elements
        (if get_area_properties=True)
    Remark: bug found:
        Once the .sdb has been file we change it manually (from the SAP2000
        interface) this function does not work well, and the table has to be
        manually opened before executing this function
    """
    # Run the model if it is not locked
    if not (SapModel.GetModelIsLocked()):
        run_analysis(SapModel)

    # clear all case and combo and set modal one
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # A) Get all names of points
    # group element (if instead of group we want to get elements, then 1)
    GroupElm = 2
    Obj = 1  # I don't know why is it necessary (but it works with any number)
    LoadCase, StepType, StepNum = [], [], []
    U1, U2, U3 = [], [], []
    R1, R2, R3 = [], [], []

    output = SapModel.Results.ModeShape(Name_points_group,
                                        GroupElm, Obj, LoadCase,
                                        StepType, StepNum, U1,
                                        U2, U3, R1, R2, R3)
    [_, Point, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get all points names', ret)
    all_points = list(set(Point))

    # B) Get all names of elements and element stations
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [_, Element, _, ElementStat, _, _, _, _, _, _, _, _, _, _, ret] = output
    raise_warning('Get all element names', ret)

    
    all_elements = list(set(Element))
    all_elements_stat = list(set(ElementStat))

    if get_area_properties:
        GroupElm = 2  # eItemTypeElm.GroupElm
        NumberResults = 0
        Obj, Elm, PointElm, LoadCase, StepType, StepNum = [], [], [], [], [], []
        F11, F22, F12, FMax, FMin, FAngle, FVM = [], [], [], [], [], [], []
        M11, M22, M12, MMax, MMin, MAngle, V13, V23, VMax, VAngle = [
        ], [], [], [], [], [], [], [], [], []
        output = SapModel.Results.AreaForceShell(
            Name_elements_group, GroupElm, NumberResults,
            Obj, Elm, PointElm,
            LoadCase, StepType, StepNum,
            F11, F22, F12, FMax, FMin, FAngle, FVM,
            M11, M22, M12, MMax, MMin, MAngle,
            V13, V23, VMax, VAngle
        )

        [_, AreaObj, AreaElm, _, _, _, _, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _, _, _, _, ret] = output

        raise_warning('Get all shell (area) element results', ret)

        all_area_objects = list(set(AreaObj))
        all_area_elements = list(set(AreaElm))

    if get_area_properties:
        return all_points, all_elements, all_elements_stat, all_area_objects, all_area_elements
    else:
        return all_points, all_elements, all_elements_stat


def get_displmodeshapes(Name_points_group, SapModel):
    """
    Function duties:
        Retrieves displacement mode shapes for a given group of points
    Input:
        name_points_group: This name comes from SAP2000! (Check that the group exists)
    Return:
        joint_modeshape_results: dictionary of results
    Remark:
        dofs (U1-3, R1-3) are LOCAL, not global directions!
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


def get_pointcoordinates(all_points, SapModel, round_coordinates=True):
    """
    Function duties:
        Returns a dictionary with x-y-z coordinates for each point
        in all_points list
    Input:
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
    """
    pointcoord_dict = dict()
    for point in outils.sort_list_string(all_points):
        pointcoord_dict[point] = dict()
        x, y, z = 0, 0, 0
        output = SapModel.PointObj.GetCoordCartesian(point, x, y, z)
        [x, y, z, ret] = output
        raise_warning('Point coordinates', ret)

        if round_coordinates:
            x = outils.round_6_sign_digits(x)
            y = outils.round_6_sign_digits(y)
            z = outils.round_6_sign_digits(z)

        pointcoord_dict[point]['x'] = x
        pointcoord_dict[point]['y'] = y
        pointcoord_dict[point]['z'] = z

    return pointcoord_dict


def get_frameconnectivity(all_points, all_elements,
                          SapModel, all_points_coord={}):
    """
    Function duties:
        For each frame, it gives initial and end point names;
        If all_points_coord dictionary is introduced, also coordinates of each point are provided
    Input:
        all_points: list with all points
        all_elements: list with all elements
        all_points_coord: if provided, dict containing x, y, z coordinates for each point
            (this dictionary comes from sap2000_getpointcoordinates function)
    """
    if len(all_points_coord) > 0:
        coord_defined = True
    else:
        coord_defined = False

    frames_dict = dict()
    key_ini, key_end = 'Point_0', 'Point_f'
    keys = [key_ini, key_end]
    for element in outils.sort_list_string(all_elements):
        frames_dict[element] = {key: None for key in keys}

    for PointName in all_points:
        NumberItems = 0
        ObjectType, ObjectName, PointNumber = [], [], []
        output = SapModel.PointObj.GetConnectivity(PointName, NumberItems,
                                                   ObjectType, ObjectName, PointNumber)
        [NumberItems, ObjectType, ObjectName, PointNumber, ret] = output
        raise_warning('Get connectivity', ret)

        # element_joint_connect = dict()
        ObjectType, ObjectName, PointNumber = list(
            ObjectType), list(ObjectName), list(PointNumber)
        FRAME_ID = 2  # stablished by SAP2000
        OBJECT_INI, OBJECT_END = 1, 2  # stablished by SAP2000

        # Retrieve frames (other elements could be connected)
        frames = [name for i, name in enumerate(
            ObjectName) if ObjectType[i] == FRAME_ID]

        for i, frame in enumerate(frames):
            coord = {}
            if PointNumber[i] == OBJECT_INI:
                if frames_dict[frame][key_ini] is None:
                    if coord_defined:
                        coord = all_points_coord[PointName]
                    frames_dict[frame][key_ini] = {
                        **{'PointName': PointName}, **coord}
                else:
                    if frames_dict[frame][key_ini] != PointName:
                        message = f'{frame} frame has 2 different {key_ini} assigments: ({PointName} and {frames_dict[frame][key_ini]}) '
                        warnings.warn(message, UserWarning)
            elif PointNumber[i] == OBJECT_END:
                if frames_dict[frame][key_end] is None:
                    if coord_defined:
                        coord = all_points_coord[PointName]
                    frames_dict[frame][key_end] = {
                        **{'PointName': PointName}, **coord}
                else:
                    if frames_dict[frame][key_end] != PointName:
                        message = f'{frame} frame has 2 different {key_end} assigments: ({PointName} and {frames_dict[frame][key_end]}) '
                        warnings.warn(message, UserWarning)

    return frames_dict


def get_areaconnectivity(all_points, all_areas, SapModel, all_points_coord={}):
    """
    For each area object, returns a dictionary with numbered point keys.
    If all_points_coord is given, includes coordinates for each point.

    Output format:
    {
        'A1': {
            'Point_1': {PointName: ..., x: ..., y: ..., z: ...},
            'Point_2': {...},
            ...
        },
        ...
    }
    """
    AREA_ID = 5  # SAP2000 object type for Area
    areas_dict = {area: {} for area in all_areas}
    coord_defined = len(all_points_coord) > 0

    for PointName in all_points:
        NumberItems = 0
        ObjectType, ObjectName, PointNumber = [], [], []
        output = SapModel.PointObj.GetConnectivity(PointName, NumberItems,
                                                   ObjectType, ObjectName, PointNumber)
        [NumberItems, ObjectType, ObjectName, PointNumber, ret] = output
        raise_warning('Get connectivity', ret)

        ObjectType = list(ObjectType)
        ObjectName = list(ObjectName)

        for i, obj_type in enumerate(ObjectType):
            if obj_type == AREA_ID:
                area_name = ObjectName[i]
                if area_name in areas_dict:
                    point_info = {'PointName': PointName}
                    if coord_defined:
                        point_info.update(all_points_coord.get(PointName, {}))

                    # Avoid duplicate points
                    existing_points = [entry['PointName']
                                       for entry in areas_dict[area_name].values()]
                    if PointName not in existing_points:
                        point_index = len(areas_dict[area_name]) + 1
                        key = f'Point_{point_index}'
                        areas_dict[area_name][key] = point_info

    return areas_dict


def get_point_forces(Name_points_group, SapModel, load_pattern=None,
                     return_kN=False):
    """
    Function Duties:
        Retrieves forces applied to a group of points in the model.
    Input:
        Name_points_group: Name of the group of points in the model
        SapModel: SAP2000 model object
        load_pattern: if specified, only retrieves results for it
            (default is None, which retrieves all load patterns)
        return_kN: if True, the function will convert the units to kN
    Output:
        output_dict: Dictionary with the forces applied to the points
            in the group
    """
    if return_kN:
        actual_units, _ = get_units(SapModel)
        set_kN_m_C_units(SapModel)

    PointName = ""
    NumberItems = 0
    LoadPat = ""
    LCStep = []
    CSys = []
    F1 = []
    F2 = []
    F3 = []
    M1 = []
    M2 = []
    M3 = []
    ItemType = 1  # 1 for Group, 0 for Object

    output = SapModel.PointObj.GetLoadForce(
        Name_points_group, NumberItems, PointName, LoadPat, LCStep, CSys, F1, F2, F3, M1, M2, M3, ItemType
    )

    [NumberItems, PointName, LoadPat, LCStep,
        CSys, F1, F2, F3, M1, M2, M3, ret] = output

    raise_warning('Get point forces', ret)

    if not (load_pattern is None):
        bool = [i == load_pattern for i in list(LoadPat)]
    else:
        bool = [True for i in list(LoadPat)]

    PointName = [PointName[i] for i, b in enumerate(bool) if b]
    LoadPat = [LoadPat[i] for i, b in enumerate(bool) if b]
    F1 = [F1[i] for i, b in enumerate(bool) if b]
    F2 = [F2[i] for i, b in enumerate(bool) if b]
    F3 = [F3[i] for i, b in enumerate(bool) if b]
    M1 = [M1[i] for i, b in enumerate(bool) if b]
    M2 = [M2[i] for i, b in enumerate(bool) if b]
    M3 = [M3[i] for i, b in enumerate(bool) if b]

    output_dict = {'PointObj': PointName, 'LoadPat': LoadPat,
                   'F1': F1, 'F2': F2, 'F3': F3,
                   'M1': M1, 'M2': M2, 'M3': M3}

    # Restore the original units
    if return_kN:
        set_units(SapModel, actual_units)

    return output_dict


def get_distributed_loads(SapModel, Name_elements_group, item_type=1):
    """
    Function Duties:
        Retrieves distributed load assignments for specified frame objects or groups in SAP2000.
    Input:
        SapModel : object
            The SAP2000 model object to interact with the API.
        Name_elements_group : str
            The name of an existing frame object, group, or can be ignored if `item_type` is 2 (SelectedObjects).
        item_type : int, optional
            Specifies the selection type:
            - 0: Object - Retrieves assignments for the specified frame object.
            - 1: Group (default) - Retrieves assignments for all frame objects in the specified group.
            - 2: SelectedObjects - Retrieves assignments for all selected frame objects, ignoring the `Name_elements_group` parameter.
    Output: frames_dict (dict)
        A dictionary containing forces for each frame and load pattern:
        - 'MyType': List indicating load type (1 = Force, 2 = Moment).
        - 'CSys': List of coordinate systems used (Local or defined system).
        - 'Dir': List indicating load direction (1–11):
            1, 2, 3: Local 1, 2, 3 axis (only applies when CSys is Local)
            4, 5, 6: X, Y, Z direction (does not applie when CSys is Local)
            7, 8, 9: Projected X, Y, Z direction (does not apply when CSys is Local)
            10: Gravity direction (only applies when CSys is Global)
            11: Projected Gravity direction (only applies when CSys is Global)
        - 'RD1', 'RD2': Lists of relative distances from the I-End to load start and end.
        - 'Dist1', 'Dist2': Lists of actual distances from the I-End to load start and end.
        - 'Val1', 'Val2': Lists of load values at the start and end of the distributed load.
    """
    NumberItems = 0
    FrameName, LoadPat, MyType, CSys, Dir = [], [], [], [], []
    RD1, RD2, Dist1, Dist2, Val1, Val2 = [], [], [], [], [], []

    output = SapModel.FrameObj.GetLoadDistributed(
        Name_elements_group, NumberItems, FrameName, LoadPat, MyType, CSys, Dir,
        RD1, RD2, Dist1, Dist2, Val1, Val2, item_type
    )

    NumberItems, FrameName, LoadPat, MyType, CSys, Dir, RD1, RD2, Dist1, Dist2, Val1, Val2, ret = output
    raise_warning('Get distributed frame forces', ret)

    frames_dict = dict()
    for i in range(NumberItems):
        if FrameName[i] not in frames_dict:
            frames_dict[FrameName[i]] = dict()
        frames_dict[FrameName[i]][LoadPat[i]] = {
            'MyType': MyType[i],
            'CSys': CSys[i],
            'Dir': Dir[i],
            'RD1': RD1[i],
            'RD2': RD2[i],
            'Dist1': Dist1[i],
            'Dist2': Dist2[i],
            'Val1': Val1[i],
            'Val2': Val2[i]
        }

    return frames_dict


def get_distributed_loads_v2(SapModel, Name_elements_group, item_type=1):
    """
    Analogous to get_distributed_loads, but solves the following issue:
        when two loads were associated to the same element, only retrieved one
    """
    NumberItems = 0
    FrameName, LoadPat, MyType, CSys, Dir = [], [], [], [], []
    RD1, RD2, Dist1, Dist2, Val1, Val2 = [], [], [], [], [], []

    output = SapModel.FrameObj.GetLoadDistributed(
        Name_elements_group, NumberItems, FrameName, LoadPat, MyType, CSys, Dir,
        RD1, RD2, Dist1, Dist2, Val1, Val2, item_type
    )

    NumberItems, FrameName, LoadPat, MyType, CSys, Dir, RD1, RD2, Dist1, Dist2, Val1, Val2, ret = output
    raise_warning('Get distributed frame forces', ret)

    frames_dict = dict()
    for i in range(NumberItems):
        if FrameName[i] not in frames_dict:
            frames_dict[FrameName[i]] = dict()

        # Ensure load pattern key exists
        if LoadPat[i] not in frames_dict[FrameName[i]]:
            frames_dict[FrameName[i]][LoadPat[i]] = []

        # Append this load entry (so we can have multiple directions, etc.)
        frames_dict[FrameName[i]][LoadPat[i]].append({
            'MyType': MyType[i],
            'CSys': CSys[i],
            'Dir': Dir[i],
            'RD1': RD1[i],
            'RD2': RD2[i],
            'Dist1': Dist1[i],
            'Dist2': Dist2[i],
            'Val1': Val1[i],
            'Val2': Val2[i]
        })

    return frames_dict


def get_point_loads_on_frame(SapModel, Name_elements_group, item_type=1, round_coordinates=True,
                             return_kN=False):
    """
    Function Duties:
        Retrieves distributed load assignments for specified frame objects or groups in SAP2000.
    Input:
        SapModel : object
            The SAP2000 model object to interact with the API.
        Name_elements_group : str
            The name of an existing frame object, group, or can be ignored if `item_type` is 2 (SelectedObjects).
        item_type : int, optional
            Specifies the selection type (i.e. how to interpret Name_elements_group):
            - 0: Object - Retrieves assignments for the specified frame object.
            - 1: Group (default) - Retrieves assignments for all frame objects in the specified group.
            - 2: SelectedObjects - Retrieves assignments for all selected frame objects, ignoring the `Name_elements_group` parameter.
        return_kN: if True, the function will convert the units to kN
    Output: frames_dict (dict)
        A dictionary containing forces for each frame and load pattern:
        - 'MyType': List indicating load type (1 = Force, 2 = Moment).
        - 'CSys': List of coordinate systems used (Local or defined system).
        - 'Dir': List indicating load direction (1–11):
            1, 2, 3: Local 1, 2, 3 axis (only applies when CSys is Local)
            4, 5, 6: X, Y, Z direction (does not applie when CSys is Local)
            7, 8, 9: Projected X, Y, Z direction (does not apply when CSys is Local)
            10: Gravity direction (only applies when CSys is Global)
            11: Projected Gravity direction (only applies when CSys is Global)
        - RelDist: Relative distance from I-End (0–1)
        - Dist: Absolute distance from I-End
        - Val: Magnitude of the point load
    """
    if return_kN:
        actual_units, _ = get_units(SapModel)
        set_kN_m_C_units(SapModel)
    NumberItems = 0
    FrameName, LoadPat, MyType, CSys, Dir = [], [], [], [], []
    RelDist, Dist, Val = [], [], []

    output = SapModel.FrameObj.GetLoadPoint(
        Name_elements_group, NumberItems,
        FrameName, LoadPat, MyType, CSys, Dir,
        RelDist, Dist, Val, item_type
    )

    # Unpack and check return
    NumberItems, FrameName, LoadPat, MyType, CSys, Dir, RelDist, Dist, Val, ret = output
    raise_warning('Get point loads on frame', ret)

    if round_coordinates:
        RelDist = list([outils.round_6_sign_digits(i) for i in RelDist])
        Dist = list([outils.round_6_sign_digits(i) for i in Dist])

    # Organize output into a dictionary
    frames_dict = dict()
    for i in range(NumberItems):
        if FrameName[i] not in frames_dict:
            frames_dict[FrameName[i]] = dict()
        frames_dict[FrameName[i]][LoadPat[i]] = {
            'MyType': MyType[i],
            'CSys': CSys[i],
            'Dir': Dir[i],
            'RelDist': RelDist[i],
            'Dist': Dist[i],
            'Val': Val[i]
        }

    # Restore the original units
    if return_kN:
        set_units(SapModel, actual_units)

    return frames_dict


def get_modalforces(Name_elements_group, SapModel,
                    average_values=True, round_coordinates=True,
                    considered_modes=None):
    """
    Input:
        Name_elements_group: name of the group containing elements whose
            forces will be retrieved
                Remark: This name is defined in SAP2000 .sdb model!!
        SapModel: SAP2000 model
        average_values: if True, values in elements with the same x value are averaged
            (Set to False when comparing with SAP2000 interface tables)
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
        considered_modes:
            if None, all modes are considered
            if list, only modes in the list are considered (starting from 1 to N (SAP2000 mode numbering, not python))
    Output:
        results: dictionary containing information for each mode as follow:
            Mode_i: element_1 to element_N:
                element_1: P, M2 and M3 per each x coordinate within the element
    """
    # Select modal case
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput("MODAL")
    raise_warning('Select modal case', ret)

    # Read information from SAP2000
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumResults, Element, Station, Mesh, MeshStation, LoadCase,
        StepType, StepNum, P, V2, V3, T, M2, M3, ret] = output
    raise_warning('Get element forces', ret)

    StepNum = list(StepNum)
    Element = list(Element)
    if round_coordinates:
        Station = list([outils.round_6_sign_digits(i) for i in Station])
        MeshStation = list([outils.round_6_sign_digits(i)
                           for i in MeshStation])
    P, M2, M3 = list(P), list(M2), list(M3)

    # Save information into a dictionary
    StepNum_unique = list(set(StepNum))
    StepNum_unique.sort()
    Element_unique = outils.sort_list_string(list(set(Element)))
    if considered_modes is not None:
        StepNum_unique = [
            mode for mode in StepNum_unique if mode in considered_modes]

    results = outils.process_modal_results_fast(StepNum, Element, Station, P, M2, M3, Element_unique,
                                                StepNum_unique, average_values=average_values)

    return results

    # LAST PART OF THE OLD FUNCTION
    # results = dict()
    # for mode in StepNum_unique:
    #     mode_label = 'Mode_' + str(int(mode))
    #     results[mode_label] = dict()
    #     for element in Element_unique:
    #         element_label = 'Element_' + str(int(element))
    #         results[mode_label][element_label] = dict()

    #         # Bool list for for specific mode and element
    #         id_mode_element = [el == element and StepNum[i]
    #                            == mode for i, el in enumerate(Element)]

    #         # Coordinates and forces retrievement
    #         station_mode_element = [st for i, st in enumerate(
    #             Station) if id_mode_element[i]]
    #         p = [force for i, force in enumerate(P) if id_mode_element[i]]
    #         m2 = [force for i, force in enumerate(M2) if id_mode_element[i]]
    #         m3 = [force for i, force in enumerate(M3) if id_mode_element[i]]

    #         if average_values:
    #             # values in elements with the same x value are averaged
    #             x = outils.sort_list_string(list(set(station_mode_element)))
    #             p_def, m2_def, m3_def = list(), list(), list()
    #             for i in x:
    #                 average_id = [i == x for x in station_mode_element]
    #                 p_aux = [p[j] for j, _ in enumerate(p) if average_id[j]]
    #                 m2_aux = [m2[j] for j, _ in enumerate(m2) if average_id[j]]
    #                 m3_aux = [m3[j] for j, _ in enumerate(m3) if average_id[j]]
    #                 p_def.append(np.mean(p_aux))
    #                 m2_def.append(np.mean(m2_aux))
    #                 m3_def.append(np.mean(m3_aux))
    #         else:
    #             # retrieve left-hand value in elements with the same x value (for validating purposes)
    #             x_coordinate_id = [st != station_mode_element[i-1] for i, st in enumerate(station_mode_element)]
    #             x = [x for i, x in enumerate(station_mode_element) if x_coordinate_id[i]]
    #             p_def = [force for i, force in enumerate(p) if x_coordinate_id[i]]
    #             m2_def = [force for i, force in enumerate(m2) if x_coordinate_id[i]]
    #             m3_def = [force for i, force in enumerate(m3) if x_coordinate_id[i]]

    #         mesh = [f'{element}.{mesh_id}' for mesh_id in range(len(x))]

    #         # Save into results dictionary
    #         results[mode_label][element_label]['x'] = x
    #         results[mode_label][element_label]['P'] = p_def
    #         results[mode_label][element_label]['M2'] = m2_def
    #         results[mode_label][element_label]['M3'] = m3_def
    #         results[mode_label][element_label]['Mesh_id'] = mesh


def get_modalforces_timehistory(Name_elements_group, SapModel,
                                loadcase_name, average_values=True,
                                round_coordinates=True,
                                selected_elements=None):
    """
    Input:
        Name_elements_group: name of the group containing elements whose
            forces will be retrieved
                Remark: This name is defined in SAP2000 .sdb model!!
        SapModel: SAP2000 model
        loadcase_name: name of the time history load case
        average_values: each element has 2 values per station and time step. If True,
            these values are averaged; if False, the left-hand value is taken
            (Set to False when comparing with SAP2000 interface tables); True is recommended
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
        considered_modes:
            if None, all modes are considered
            if list, only modes in the list are considered
        selected_elements: a list of elements to be considered (if None, all elements are considered)
            it is used to filter the results (much smaller dictionary)
    Output:
        results: dictionary containing information for the time history as follow:
            Mode_i: element_1
            element_1 to element_N:
                x: x coordinates
                P, M2 and M3: arrays whose i,j elements refer to:
                    i: x coordinate
                    j: time step (from StepNum)
        StepNum: list with time steps (same for all elements)
    """
    # Select modal case
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput(loadcase_name)
    raise_warning('Select modal case', ret)

    # Change display table options (only way I found to show step-by-step results)
    Envelopes, Step_by_Step = 1, 2
    BaseReactionGX, BaseReactionGY, BaseReactionGZ = 0, 0, 0
    IsAllModes, StartMode, EndMode = False, 1, 1
    IsAllBucklingModes, StartBuckingMode, EndBucklingMode = False, 1, 1
    ModalHistory, DirectHistory, NonLinearStatic = Step_by_Step, Step_by_Step, Step_by_Step
    MultistepStaticStatic, SteadyState = Step_by_Step, Envelopes
    SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing = 1, 1, 1, 1
    ret = SapModel.DatabaseTables.SetTableOutputOptionsForDisplay(BaseReactionGX, BaseReactionGY, BaseReactionGZ, IsAllModes, StartMode, EndMode, IsAllBucklingModes, StartBuckingMode,
                                                                  EndBucklingMode, ModalHistory, DirectHistory, NonLinearStatic, MultistepStaticStatic, SteadyState, SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing)
    raise_warning('Display tables - options', ret)

    # Read information from SAP2000
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumResults, Element, Station, Mesh, MeshStation, LoadCase,
        StepType, StepNum, P, V2, V3, T, M2, M3, ret] = output
    raise_warning('Get element forces', ret)

    # Save into a dictionary (enhanced version of get_modalforces function for faster performance)
    Element, Station, StepNum, LoadCase = np.array(Element), np.array(
        Station), np.array(StepNum), np.array(LoadCase)
    P, M2, M3 = np.array(P), np.array(M2), np.array(M3)

    # Round if necessary
    if round_coordinates:
        Station = np.vectorize(outils.round_6_sign_digits)(Station)
        MeshStation = np.vectorize(outils.round_6_sign_digits)(MeshStation)
        StepNum = np.vectorize(outils.round_6_sign_digits)(StepNum)

    # Filtering for the specific load case
    loadcase_mask = (LoadCase == loadcase_name)
    Element = Element[loadcase_mask]
    Station = Station[loadcase_mask]
    StepNum = StepNum[loadcase_mask]
    P = P[loadcase_mask]
    M2 = M2[loadcase_mask]
    M3 = M3[loadcase_mask]

    # Unique Elements and Step Numbers
    Element_unique = np.array(outils.sort_list_string(list(set(Element))))
    StepNum_unique = np.unique(StepNum)

    results = defaultdict(lambda: {'x': [], 'P': [], 'M2': [], 'M3': []})

    # Group data by element
    if selected_elements is None:
        selected_elements = Element_unique

    for element in selected_elements:
        element_mask = (Element == element)
        station_element = Station[element_mask]
        step_element = StepNum[element_mask]
        p_element = P[element_mask]
        m2_element = M2[element_mask]
        m3_element = M3[element_mask]

        x_unique, x_indices = np.unique(station_element, return_inverse=True)
        num_stations = len(x_unique)
        num_steps = len(StepNum_unique)

        p_def = np.full((num_stations, num_steps), np.nan)
        m2_def = np.full((num_stations, num_steps), np.nan)
        m3_def = np.full((num_stations, num_steps), np.nan)

        for i_step, step in enumerate(StepNum_unique):
            step_mask = (step_element == step)
            p_step = p_element[step_mask]
            m2_step = m2_element[step_mask]
            m3_step = m3_element[step_mask]
            station_step = x_indices[step_mask]

            for i_station in range(num_stations):
                station_mask = (station_step == i_station)
                if station_mask.any():
                    if average_values:  # 2 values are provided per station and step
                        p_def[i_station, i_step] = np.mean(
                            p_step[station_mask])
                        m2_def[i_station, i_step] = np.mean(
                            m2_step[station_mask])
                        m3_def[i_station, i_step] = np.mean(
                            m3_step[station_mask])
                    else:  # for comparing with SAP2000 tables
                        p_def[i_station, i_step] = p_step[station_mask][0]
                        m2_def[i_station, i_step] = m2_step[station_mask][0]
                        m3_def[i_station, i_step] = m3_step[station_mask][0]

        element_label = f'Element_{int(element)}'
        results[element_label]['x'] = x_unique.tolist()
        results[element_label]['P'] = p_def.tolist()
        results[element_label]['M2'] = m2_def.tolist()
        results[element_label]['M3'] = m3_def.tolist()
        # results[element_label]['Mesh_id'] = [f'{element}.{mesh_id}' for mesh_id in range(len(x_unique))]

    # Convert defaultdict to dict
    results = dict(results)

    # Come back to default display table options
    Envelopes, Step_by_Step = 1, 2
    BaseReactionGX, BaseReactionGY, BaseReactionGZ = 0, 0, 0
    IsAllModes, StartMode, EndMode = False, 1, 1
    IsAllBucklingModes, StartBuckingMode, EndBucklingMode = False, 1, 1
    ModalHistory, DirectHistory, NonLinearStatic = Envelopes, Envelopes, Envelopes
    MultistepStaticStatic, SteadyState = Envelopes, Envelopes
    SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing = 1, 1, 1, 1
    ret = SapModel.DatabaseTables.SetTableOutputOptionsForDisplay(BaseReactionGX, BaseReactionGY, BaseReactionGZ, IsAllModes, StartMode, EndMode, IsAllBucklingModes, StartBuckingMode,
                                                                  EndBucklingMode, ModalHistory, DirectHistory, NonLinearStatic, MultistepStaticStatic, SteadyState, SteadyStateOption, PowerSpectralDensity, Combo, BridgeDesing)
    raise_warning('Display tables - options', ret)

    return results, StepNum_unique.tolist()


def get_loadcaseforces(Name_elements_group, SapModel,
                       average_values=True, round_coordinates=True,
                       load_case='DEAD'):
    """
    Function duties:
        This function is very similar to get_modalforces (in fact, both could be
            merged)
    Input:
        Name_elements_group: name of the group containing elements whose
            forces will be retrieved
                Remark: This name is defined in SAP2000 .sdb model!!
        SapModel: SAP2000 model
        average_values: if True, values in elements with the same x value are averaged
            (Set to False when comparing with SAP2000 interface tables)
        round_coordinates: if True, coordinates are rounded to 6 significant digits
            (Otherwise, some values could have very small variations)
    Output:
        results: dictionary containing information for the specific load case as follow:
            load_case: element_1 to element_N:
                element_1: P, M2 and M3 per each x coordinate within the element
    """
    # Select modal case
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput
    raise_warning('Deselect all cases', ret)
    ret = SapModel.Results.Setup.SetCaseSelectedForOutput(load_case)
    raise_warning('Select modal case', ret)

    # Read information from SAP2000
    GroupElm = 2  # change to 2
    NumberResults = 0
    Elm = []
    Obj, ObjSta, ElmSta, LoadCase, StepType, StepNum = [], [], [], [], [], []
    P, V2, V3, T, M2, M3 = [], [], [], [], [], []
    output = SapModel.Results.FrameForce(Name_elements_group, GroupElm, NumberResults,
                                         Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3)
    [NumResults, Element, Station, Mesh, MeshStation, LoadCase,
        StepType, StepNum, P, V2, V3, T, M2, M3, ret] = output
    raise_warning('Get element forces', ret)

    StepNum = list(StepNum)
    Element = list(Element)
    if round_coordinates:
        Station = list([outils.round_6_sign_digits(i) for i in Station])
        MeshStation = list([outils.round_6_sign_digits(i)
                           for i in MeshStation])
    P, M2, M3 = list(P), list(M2), list(M3)

    # Save information into a dictionary
    StepNum_unique = list(set(StepNum))
    StepNum_unique.sort()
    Element_unique = outils.sort_list_string(list(set(Element)))

    results = dict()
    for mode in StepNum_unique:
        if mode > 0:
            continue
        mode_label = load_case
        results[mode_label] = dict()
        for element in Element_unique:
            element_label = 'Element_' + str(int(element))
            results[mode_label][element_label] = dict()

            # Bool list for for specific load case and element
            id_mode_element = [el == element and StepNum[i]
                               == mode and LoadCase[i] == load_case for i, el in enumerate(Element)]

            # Coordinates and forces retrievement
            station_mode_element = [st for i, st in enumerate(
                Station) if id_mode_element[i]]
            p = [force for i, force in enumerate(P) if id_mode_element[i]]
            m2 = [force for i, force in enumerate(M2) if id_mode_element[i]]
            m3 = [force for i, force in enumerate(M3) if id_mode_element[i]]

            if average_values:
                # values in elements with the same x value are averaged
                x = outils.sort_list_string(list(set(station_mode_element)))
                p_def, m2_def, m3_def = list(), list(), list()
                for i in x:
                    average_id = [i == x for x in station_mode_element]
                    p_aux = [p[j] for j, _ in enumerate(p) if average_id[j]]
                    m2_aux = [m2[j] for j, _ in enumerate(m2) if average_id[j]]
                    m3_aux = [m3[j] for j, _ in enumerate(m3) if average_id[j]]
                    p_def.append(np.mean(p_aux))
                    m2_def.append(np.mean(m2_aux))
                    m3_def.append(np.mean(m3_aux))
            else:
                # retrieve left-hand value in elements with the same x value (for validating purposes)
                x_coordinate_id = [st != station_mode_element[i-1]
                                   for i, st in enumerate(station_mode_element)]
                x = [x for i, x in enumerate(
                    station_mode_element) if x_coordinate_id[i]]
                p_def = [force for i, force in enumerate(
                    p) if x_coordinate_id[i]]
                m2_def = [force for i, force in enumerate(
                    m2) if x_coordinate_id[i]]
                m3_def = [force for i, force in enumerate(
                    m3) if x_coordinate_id[i]]

            mesh = [f'{element}.{mesh_id}' for mesh_id in range(len(x))]

            # Save into results dictionary
            results[mode_label][element_label]['x'] = x
            results[mode_label][element_label]['P'] = p_def
            results[mode_label][element_label]['M2'] = m2_def
            results[mode_label][element_label]['M3'] = m3_def
            results[mode_label][element_label]['Mesh_id'] = mesh

    return results


def get_elementsections(all_elements, all_elements_stat, SapModel):
    """
    Input:
        all_elements: list with all elements (e.g. ['1', '2'...])
        all_elements_stat: list with all elements stations (e.g. ['1-1', '1-2'...])
        Remark: both come from getnames_point_elements
    Return:
        dictionary giving section (e.g. "FSEC1") associated to each element
    Additional remarks:
        The only OAPI function found to get sections was LineElm.GetProperty
        (which takes element stations, not element, as inputs);
        I am sure this can be optimized; perhaps two sub-elements cannot have
        different sections;
        !! Substituted by get_frame_sections (we found that function)
    """
    # Iteration over element
    element_section = dict()
    for element in outils.sort_list_string(all_elements):
        # it is not a station (non-meshed elements)
        if element in all_elements_stat:
            start_str = element
            id_element = [start_str ==
                          element_stat for element_stat in all_elements_stat]
        else:  # it is a station (meshed elements)
            start_str = element + '-'
            id_element = [element_stat.startswith(
                start_str) for element_stat in all_elements_stat]
        elements_stat_element = [stat for i, stat in enumerate(
            all_elements_stat) if id_element[i]]
        aux_sections = list()
        # Get section for each sub-element of element
        for ElemName in elements_stat_element:

            PropName = ""
            ObjType = 0
            Var, sVarTotalLength, sVarRelStartLoc = 0, 0, 0

            output = SapModel.LineElm.GetProperty(
                ElemName, "", ObjType, Var, sVarTotalLength, sVarRelStartLoc)
            [PropName, ObjType, Var, sVarTotalLength, sVarRelStartLoc, ret] = output
            raise_warning("Get element's sections", ret)

            aux_sections.append(PropName)

        # If all sections are the same -> element has that section
        if len(set(aux_sections)) == 1:
            element_section[element] = aux_sections[0]
        else:  # e.g. "1-1 has "FSEC1" and "1-2" has "FSEC2"
            element_section[element] = dict()
            element_section[element]['stations'] = elements_stat_element
            element_section[element]['section'] = aux_sections
            message = 'More than one section assigned to an element'
            warnings.warn(message, UserWarning)

    return element_section


def get_frame_sections(SapModel, return_list_sections=True):
    """
    Retrieves the section (property) and auto-select list assigned to each frame object.

    Parameters:
        SapModel: COM object from SAP2000 API
        return_list_sections (bool): If True, returns a list of all unique sections.

    Returns:
        frame_sections (dict): Dictionary mapping each frame object to a tuple:
            (section_name, auto_select_list_name or '')
        all_sections (list): List of all unique section names if return_list_sections is True.

    Raises:
        RuntimeError: If retrieval of frame data fails
    """
    # Get all frame object names
    NumberNames = 0
    FrameNames = []
    output = SapModel.FrameObj.GetNameList()
    NumberNames, FrameNames, ret = output
    raise_warning('Get frame object names', ret)

    # Collect section and auto-select list for each frame
    frame_sections = {}
    for frame_name in outils.sort_list_string(FrameNames):
        PropName = ""
        SAuto = ""
        output = SapModel.FrameObj.GetSection(frame_name, PropName, SAuto)
        PropName, SAuto, ret = output
        raise_warning(f'Get section for frame "{frame_name}"', ret)
        frame_sections[frame_name] = PropName

    if return_list_sections:
        all_sections = list(set([frame_sections[i] for i in list(frame_sections)]))
        return frame_sections, all_sections
    else:
        return frame_sections


def get_area_sections(SapModel, return_list_sections=True):
    """
    Retrieves the section (property) assigned to each area object.

    Parameters:
        SapModel: COM object from SAP2000 API
        return_list_sections (bool): If True, returns a list of all unique section names.

    Returns:
        area_sections (dict): Dictionary mapping each area object to its section name.
        all_sections (list): List of unique section names if return_list_sections is True.

    Raises:
        RuntimeError: If retrieval of area data fails.
    """
    # Get all area object names
    NumberNames = 0
    AreaNames = []
    output = SapModel.AreaObj.GetNameList()
    NumberNames, AreaNames, ret = output
    raise_warning('Get area object names', ret)

    # Collect section for each area object
    area_sections = {}
    for area_name in outils.sort_list_string(AreaNames):
        PropName = ""
        output = SapModel.AreaObj.GetProperty(area_name, PropName)
        PropName, ret = output
        raise_warning(f'Get section for area "{area_name}"', ret)
        area_sections[area_name] = PropName

    if return_list_sections:
        all_sections = list(set(area_sections.values()))
        return area_sections, all_sections
    else:
        return area_sections


# def get_areasections(all_areas, SapModel):
#     """
#     Retrieves the section (property) assigned to each area object.

#     Parameters:
#         all_areas: List of all area object names (e.g. ['1', '2', ...])
#         SapModel: SAP2000 model object

#     Returns:
#         Dictionary mapping each area object to its section name.
#         Example:
#             {
#                 '1': 'WALL1',
#                 '2': 'SLAB2',
#                 ...
#             }
#     """
#     area_section = dict()

#     for area_name in outils.sort_list_string(all_areas):
#         PropName = ""
#         output = SapModel.AreaObj.GetProperty(area_name, PropName)
#         [PropName, ret] = output  # unpack to mimic structure

#         raise_warning(f"Get property for area {area_name}", ret)

#         area_section[area_name] = PropName

#     return area_section


def get_section_information(all_sections, SapModel):
    """
    Function Duties:
        Gets geometry and material related to each section of all_sections
    """
    # A) Section properties
    section_properties = get_sectionproperties(all_sections, SapModel)

    # B) Material assigned in each section
    section_material = get_material_section(all_sections, SapModel)

    # C) Material properties
    all_materials = list(set([section_material[i]
                         for i in list(section_material)]))
    material_properties = get_material_properties(all_materials, SapModel)

    # D) Section properties and material (merge all)
    section_properties_material = outils.get_sectionproperties_material(section_properties,
                                                                        section_material,
                                                                        material_properties)
    return section_properties_material


def get_areasection_information(all_area_sections, SapModel):
    """
    Function Duties:
        Gets geometry and material related to each area of all_area_sections
    """
    # A) Section properties
    area_section_properties = get_areasectionproperties(
        all_area_sections, SapModel)

    # B) Material assigned in each section
    section_material = outils.get_material_from_areas(area_section_properties)

    # C) Material properties
    all_materials = list(set([section_material[i]
                         for i in list(section_material)]))
    material_properties = get_material_properties(all_materials, SapModel)

    # D) Section properties and material (merge all)
    area_properties_material = outils.get_areaproperties_material(area_section_properties,
                                                                        material_properties)

    return area_properties_material




def get_areasectionproperties(all_area_sections, SapModel):
    """
    Retrieves geometric/material properties of area sections (shell or plane types).

    Input:
        all_area_sections: list of area section names (e.g. ['WALL_01', 'SLAB_A', ...])
        SapModel: reference to the SAP2000 model

    Output:
        Dictionary mapping section name → section property data

    Keys returned (depending on type):
        - Common: 'Type', 'MatProp', 'MatAng', 'Thickness'
        - Shell: 'ShellType', 'Bending', 'IncludeDrillingDOF'
        - Plane: 'PlaneType', 'Incompatible'
    """
    section_properties = dict()

    for section in all_area_sections:
        # Try Shell first
        ShellType, IncludeDrillingDOF = 0, False
        MatProp, MatAng = "", 0.0
        Thickness, Bending = 0.0, 0.0
        Color, Notes, GUID = 0, "", ""

        shell_output = SapModel.PropArea.GetShell_1(
            section,
            ShellType, IncludeDrillingDOF,
            MatProp, MatAng, Thickness, Bending,
            Color, Notes, GUID
        )

        [ShellType, IncludeDrillingDOF, MatProp, MatAng, Thickness,
         Bending, Color, Notes, GUID, ret_shell] = shell_output

        if ret_shell == 0:
            section_properties[section] = {
                'Type': 'Shell',
                'ShellType': ShellType,
                'IncludeDrillingDOF': IncludeDrillingDOF,
                'MatProp': MatProp,
                'MatAng': MatAng,
                'Thickness': Thickness,
                'Bending': Bending
            }
            continue

        # If shell fails, try Plane
        MyType, Incompatible = 0, False
        MatProp, MatAng = "", 0.0
        Thickness = 0.0
        Color, Notes, GUID = 0, "", ""

        plane_output = SapModel.PropArea.GetPlane(
            section,
            MyType, MatProp, MatAng, Thickness,
            Incompatible, Color, Notes, GUID
        )

        [MyType, MatProp, MatAng, Thickness,
         Incompatible, Color, Notes, GUID, ret_plane] = plane_output

        if ret_plane == 0:
            section_properties[section] = {
                'Type': 'Plane',
                'PlaneType': MyType,
                'Incompatible': Incompatible,
                'MatProp': MatProp,
                'MatAng': MatAng,
                'Thickness': Thickness
            }
            continue

        # If both fail, raise a warning
        raise_warning(f"Unable to retrieve section data for: {section}", 1)

    return section_properties


def get_sectionproperties(all_sections, SapModel):
    """
    Function duties:
        Return a dictionary containing some section properties from a list of sections
    Remark: variables definition
        Area: cross-section (axial) area
        As2: Shear area in 2 direction
        As3: Shear area in 3 direction
        Torsion: Torsional Constant
        I22: Moment of Inertia about 2 axis
        I33: Moment of Inertia about 3 axis
        S22: Section modulus about 2 axis
        S33: Section modulus about 3 axis
        Z22: Plastic modulus about 2 axis
        Z33: Plastic modulus about 3 axis
        R22: Radious of Gyration about 2 axis
        R33: Radious of Gyration about 3 axis
    """
    section_properties = dict()
    for section in all_sections:
        Area, As2, As3, Torsion, I22, I33 = 0, 0, 0, 0, 0, 0
        S22, S33, Z22, Z33, R22, R33 = 0, 0, 0, 0, 0, 0
        output = SapModel.PropFrame.GetSectProps(
            section, Area, As2, As3, Torsion, I22, I33, S22, S33, Z22, Z33, R22, R33)

        [Area, As2, As3, Torsion, I22, I33, S22,
            S33, Z22, Z33, R22, R33, ret] = output
        raise_warning("Section properties", ret)
        section_properties[section] = {
            "Area": Area, "I22": I22, "I33": I33, "S22": S22, "S33": S33}

    return section_properties


def get_material_I_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is I section
    Input:
        SectionName: name of a section
    Return:
        is_I: bool (True if SectionName is I section)
        MatProp: name of the material (if section is I)
    """
    FileName, MatProp = "", ""
    t3, t2, tf, tw, t2b, tfb, Color = 0, 0, 0, 0, 0, 0, 0
    Notes, GUID = "", ""
    output = SapModel.PropFrame.GetISection(
        SectionName, FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_I = True
    else:
        is_I = False

    return is_I, MatProp


def get_material_rectangular_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is rectangular
    Input:
        SectionName: name of a section
    Return:
        is_rectangular: bool (True if SectionName is rectangular)
        MatProp: name of the material (if section is rectangular)
    """
    FileName, MatProp = "", ""
    t3, t2, Color = 0, 0, 0
    Notes, GUID = "", ""
    output = SapModel.PropFrame.GetRectangle(
        SectionName, FileName, MatProp, t3, t2, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_rectangular = True
    else:
        is_rectangular = False

    return is_rectangular, MatProp


def get_material_SD_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_SD: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    MyType: Main of them are (see OAPI doc for other types):
        1 = I-section
        2 = Channel
        3 = Tee
        4 = Angle
        5 = Double Angle
        6 = Box
        7 = Pipe
        8 = Plate
    DesignType:
        0 = No design
        1 = Design as general steel section
        2 = Design as a concrete column; check the reinforcing
        3 = Design as a concrete column; design the reinforcing
    """
    MatProp, ShapeName, Notes, GUID = "", "", "", ""
    NumberItems, Color, DesignType = 0, 0, 0
    MyType = []
    output = SapModel.PropFrame.GetSDSection(
        SectionName, MatProp, NumberItems, ShapeName, MyType, DesignType, Color, Notes, GUID)
    [MatProp, NumberItems, ShapeName, MyType,
        DesignType, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_SD = True
    else:
        is_SD = False

    return is_SD, MatProp


def get_material_angle_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_angle: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, t2, tf, tw = 0.0, 0.0, 0.0, 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetAngle(SectionName, FileName, MatProp, t3, t2, tf, tw,
                                         Color, Notes, GUID)
    [SectionName, MatProp, t3, t2, tf, tw, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_angle = True
    else:
        is_angle = False

    return is_angle, MatProp


def get_material_circle_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_circle: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp, t3 = "", "", 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetCircle(SectionName, FileName, MatProp, t3,
                                          Color, Notes, GUID)
    [FileName, MatProp, t3, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_circle = True
    else:
        is_circle = False

    return is_circle, MatProp


def get_material_tube_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_tube: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, t2, tf, tw = 0.0, 0.0, 0.0, 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetTube(SectionName, FileName, MatProp, t3, t2, tf,
                                        tw, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, tf, tw, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_tube = True
    else:
        is_tube = False

    return is_tube, MatProp


def get_material_general_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_general: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, t2, Area, As2, As3 = 0.0, 0.0, 0.0, 0.0, 0.0
    Torsion, I22, I33, S22, S33 = 0.0, 0.0, 0.0, 0.0, 0.0
    Z22, Z33, R22, R33 = 0.0, 0.0, 0.0, 0.0
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetGeneral(SectionName, FileName, MatProp, t3, t2,
                                           Area, As2, As3, Torsion, I22, I33, S22,
                                           S33, Z22, Z33, R22, R33, Color, Notes, GUID)
    [FileName, MatProp, t3, t2, Area, As2, As3, Torsion, I22, I33,
     S22, S33, Z22, Z33, R22, R33, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_general = True
    else:
        is_general = False

    return is_general, MatProp


def get_material_pipe_section(SectionName, SapModel):
    """
    Function duties:
        Gets the material for SectionName section if SectionName is a
            "Section Design" section
    Input:
        SectionName: name of a section
    Return:
        is_pipe: bool (True if SectionName is defined in Section Designer)
        MatProp: name of the material (if section is Section Designer)
    """
    FileName, MatProp = "", ""
    t3, tw = 0.0, 0.0  # Outside diameter and wall thickness
    Color, Notes, GUID = 0, "", ""
    output = SapModel.PropFrame.GetPipe(
        SectionName, FileName, MatProp, t3, tw, Color, Notes, GUID)

    [FileName, MatProp, t3, tw, Color, Notes, GUID, ret] = output

    if ret == 0:
        is_pipe = True
    else:
        is_pipe = False

    return is_pipe, MatProp


def get_material_section(all_sections, SapModel):
    """
    Function Duties:
        Retrieves the material for each of the sections defined in all_sections
    Input:
        all_sections: list of section names
        SapModel: Sap model
    Return:
        section_material: dictionary containing the name of each material
            assigned to each section
    Remark:
        Only some sections have defined (I, SD, Rectangular...). Add more functions
            if other types are required (e.g. define get_material_channel_section)
            [for that, check all types of "Get" functions in OAPI by searching
            "MatProp"]
    """
    section_material = dict()

    for SectionName in all_sections:

        is_I, MatProp_I = get_material_I_section(SectionName, SapModel)
        is_SD, MatProp_SD = get_material_SD_section(SectionName, SapModel)
        is_rectangular, MatProp_rectangular = get_material_rectangular_section(
            SectionName, SapModel)
        is_angle, MatProp_angle = get_material_angle_section(
            SectionName, SapModel)
        is_circle, MatProp_circle = get_material_circle_section(
            SectionName, SapModel)
        is_tube, MatProp_tube = get_material_tube_section(
            SectionName, SapModel)
        is_general, MatProp_general = get_material_general_section(
            SectionName, SapModel)
        is_pipe, MatProp_pipe = get_material_pipe_section(
            SectionName, SapModel)
        # ... --> Define other functions if required (e.g. get_material_channel_section...)

        if is_I:
            MatProp = MatProp_I
        elif is_SD:
            MatProp = MatProp_SD
        elif is_rectangular:
            MatProp = MatProp_rectangular
        elif is_angle:
            MatProp = MatProp_angle
        elif is_circle:
            MatProp = MatProp_circle
        elif is_tube:
            MatProp = MatProp_tube
        elif is_general:
            MatProp = MatProp_general
        elif is_pipe:
            MatProp = MatProp_pipe
        # add other functions if required (e.g. for channel, etc. sections)
        else:
            MatProp = 'Not found'
            warnings.warn(
                f'No material found for {SectionName} section', UserWarning)

        section_material[SectionName] = MatProp

    return section_material


def get_modal_response_disp(input_data, SapModel):
    """
    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        displacement mode shapes
    """
    # A) Unlock model and set IS units
    unlock_model(SapModel)
    set_ISunits(SapModel)

    # B) Set new material properties
    if 'material' in input_data:
        material_dict = input_data['material']
        set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'joint_springs' in input_data:
        joint_dict = input_data['joint_springs']
        set_jointsprings(joint_dict, SapModel)

    # D) Modify partial fixity
    if 'frame_releases' in input_data:
        frame_releases_dict = input_data['frame_releases']
        set_framereleases(frame_releases_dict, SapModel)

    # E) Run Analysis
    run_analysis(SapModel)

    # F) Read results and save into a dictionary
    frequencies = get_modalfrequencies(SapModel)
    Name_points_group = "modeshape_points"
    disp_modeshapes = get_displmodeshapes(Name_points_group, SapModel)
    all_dict = {'frequencies': frequencies,
                'disp_modeshapes': disp_modeshapes}
    modal_results = outils.merge_all_dict(all_dict)

    return modal_results


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


def get_modal_response_strain(input_data, SapModel):
    """
    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        strain mode shapes
    """
    # A) Unlock model and set IS units
    unlock_model(SapModel)
    set_ISunits(SapModel)

    # B) Set new material properties
    if 'material' in input_data:
        material_dict = input_data['material']
        set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'joint_springs' in input_data:
        joint_dict = input_data['joint_springs']
        set_jointsprings(joint_dict, SapModel)

    # D) Modify partial fixity
    if 'frame_releases' in input_data:
        frame_releases_dict = input_data['frame_releases']
        set_framereleases(frame_releases_dict, SapModel)

    # E) Run Analysis
    run_analysis(SapModel)

    # F) Read results and save into a dictionary
    frequencies = get_modalfrequencies(SapModel)

    Name_points_group, Name_elements_group = "allpoints", "allframes"
    all_points, all_elements, all_elements_stat = getnames_point_elements(Name_points_group,
                                                                          Name_elements_group,
                                                                          SapModel)
    element_section = get_elementsections(
        all_elements, all_elements_stat, SapModel)
    all_sections = list(set([element_section[i]
                        for i in list(element_section)]))
    section_properties_material = get_section_information(
        all_sections, SapModel)
    # Modal Forces and strain modeshapes
    Name_elements_group = 'modeshape_frames'
    modal_forces = get_modalforces(Name_elements_group, SapModel)
    strain_modeshapes = outils.get_strainmodeshapes(
        modal_forces, element_section, section_properties_material)

    # Save results
    all_dict = {'frequencies': frequencies,
                'strain_modeshapes': strain_modeshapes}
    modal_results = outils.merge_all_dict(all_dict)

    return modal_results


def get_modal_response_BI(input_data, SapModel, get_disp=True, get_strain=True):
    """
    Function Duties:
        Modifies sap model with data in input_data,
        runs analysis and return modal results with
        strain mode shapes
    Remark:
        Specifically for the BI project
    """
    # A) Unlock model and set IS units
    unlock_model(SapModel)
    set_ISunits(SapModel)

    # B) Set new material properties
    if 'material' in input_data:
        material_dict = input_data['material']
        set_materials(material_dict, SapModel)

    # C) Modify spring supports
    if 'joint_springs' in input_data:
        joint_dict = input_data['joint_springs']
        set_jointsprings(joint_dict, SapModel)

    # D) Modify partial fixity
    if 'frame_releases' in input_data:
        frame_releases_dict = input_data['frame_releases']
        set_framereleases(frame_releases_dict, SapModel)

    # E) Run Analysis
    run_analysis(SapModel)

    # F) Read results and save into a dictionary
    frequencies = get_modalfrequencies(SapModel)

    if get_disp:
        Name_points_group = "modeshape_points"
        disp_modeshapes = get_displmodeshapes(Name_points_group, SapModel)
    else:
        disp_modeshapes = None

    if get_strain:
        Name_points_group, Name_elements_group = "allpoints", "allframes"
        all_points, all_elements, all_elements_stat = getnames_point_elements(Name_points_group,
                                                                              Name_elements_group,
                                                                              SapModel)
        element_section = get_elementsections(
            all_elements, all_elements_stat, SapModel)
        all_sections = list(set([element_section[i]
                            for i in list(element_section)]))
        section_properties_material = get_section_information(
            all_sections, SapModel)
        # Modal Forces and strain modeshapes
        Name_elements_group = 'modeshape_frames'
        modal_forces = get_modalforces(Name_elements_group, SapModel)
        strain_modeshapes = outils.get_strainmodeshapes(
            modal_forces, element_section, section_properties_material)
    else:
        strain_modeshapes = None

    # Save results
    all_dict = {'frequencies': frequencies,
                'disp_modeshapes': disp_modeshapes,
                'strain_modeshapes': strain_modeshapes}
    modal_results = outils.merge_all_dict(all_dict)

    return modal_results


def get_pointrestraints(point_list, SapModel):
    """
    Input:
        List of points (point_list[i] = str)
    Output:
        Dictionary containing a list of boolean values for
        each point.
        Remark:
            len(restraints[i]) = num_dof
            restraints[i][j] = True if there is restriction in dof j
                for point i
    """
    restraints = dict()
    for i in point_list:
        Value = []
        Value, ret = SapModel.PointObj.GetRestraint(str(i), Value)
        restraints[i] = list(Value)

    raise_warning("Get joint restraints", ret)

    return restraints


def plot_accelerometers_as_forces(Phi_id, n_modes_considered, SapModel) -> None:
    """
    Function Duties:
        Plot sensors locations (given by Phi_id) as forces
        in the SAP2000 model
    Input:
        Phi_id: list of strings with the joint names followed by
            the DOF (e.g. ['1_U1', '3_U2'])
        n_modes_considered: number of modes aimed to find with EfI
        SapModel: SAP2000 model
    Remark:
        After executing the function, navigate through SAP2000 interface
        and click on:
        Display -> Show Object Load Assigns -> Joint ->
            -> "OSP_{n_sensors}_sensors" Load Case
    """
    # Initial variables
    ref_force = 10**5  # magnitude of the plotted force
    # always like that (SAP2000)
    all_dofs = ['U1', 'U2', 'U3', 'R1', 'R2', 'R3']
    n_sensors = len(Phi_id)

    # A) Unlock model
    unlock_model(SapModel, lock=False)

    # B) Define load pattern
    LoadPatName = f"OSP_{n_modes_considered}_modes_{n_sensors}_ACCELEROMETERS"

    # if the load pattern already exists, delete it:
    ret = SapModel.LoadPatterns.Delete(LoadPatName)

    MyType = 3  # 3: Live
    ret = SapModel.LoadPatterns.Add(LoadPatName, MyType, 0, False)
    raise_warning("Create load pattern", ret)

    # C) Plot forces
    # C.1: regrouping Phi_id
    joint_dict = dict()
    for i in Phi_id:
        joint = i.split('_')[0]
        dof = i.split('_')[1]
        if joint not in joint_dict:
            joint_dict[joint] = list()
        joint_dict[joint].append(dof)

    # C.2: plotting forces
    for joint in list(joint_dict):
        dof = joint_dict[joint]
        active = [i in dof for i in all_dofs]
        Value = np.array([0, 0, 0, 0, 0, 0])
        Value[active] = ref_force
        Value = list(Value)
        _, ret = SapModel.PointObj.SetLoadForce(
            joint, LoadPatName, Value, False)
        raise_warning(f"Add load for joint: {joint}", ret)

    # Refresh view for all windows
    ret = SapModel.View.RefreshView()
    raise_warning("Refresh view", ret)


def plot_SG_as_forces(Psi_id, points_for_plotting, aux_loadpat_name, SapModel) -> None:
    """
    Function Duties:
        Plot sensors locations (given by Phi_id) as forces
        in the SAP2000 model
    Input:
        Psi_id: list of strings with the element names with the station
            followed by the location (e.g. ['2.0_up', '6.1_right'])
        points_for_plotting: dictionary coming from prepare_dict_plotting_SG
        aux_loadpat_name: used for naming the LoadPattern; 2 opts:
            -> integer: in this case, LoadPatName = f"OSP_{aux_loadpat_name}_modes_{n_sensors}_SG"
            -> string: in this case, LoadPatName = aux_loadpat_name
        SapModel: SAP2000 model
    Remark:
        After executing the function, navigate through SAP2000 interface
        and click on:
        Display -> Show Object Load Assigns -> Frame ->
            -> "OSP_{n_sensors}_sensors" Load Case
        see "SAP2000_sign_convention" in "docs" folder for understanding eps
            right, up, left, down...
    """
    # Initial variables
    ref_force = 10**5  # magnitude of the plotted force
    n_sensors = len(Psi_id)

    # A) Unlock model
    unlock_model(SapModel, lock=False)

    # B) Define load pattern
    if isinstance(aux_loadpat_name, str):
        LoadPatName = aux_loadpat_name
    else:
        LoadPatName = f"OSP_{aux_loadpat_name}_modes_{n_sensors}_SG"

    # if the load pattern already exists, delete it:
    ret = SapModel.LoadPatterns.Delete(LoadPatName)

    MyType = 3  # 3: Live
    ret = SapModel.LoadPatterns.Add(LoadPatName, MyType, 0, False)

    LoadPat = LoadPatName
    MyType = 1  # 1: Force
    CSys = 'Local'
    RelDist, Replace = False, False
    ItemType = 0  # 0: Object
    for element in list(points_for_plotting):
        Name = element.split('_')[1]
        for Dist, location in zip(points_for_plotting[element]['x'], points_for_plotting[element]['location']):
            if location in ('up', 'down'):
                Dir = 2
                Sense = -1 if location == 'up' else 1
            elif location in ('left', 'right'):
                Dir = 3
                Sense = -1 if location == 'right' else 1
            Val = ref_force * Sense
            ret = SapModel.FrameObj.SetLoadPoint(
                Name, LoadPat, MyType, Dir, Dist, Val, CSys, RelDist, Replace, ItemType)
            raise_warning(f"Plotting {element} at distance {Dist}", ret)

    # Refresh view for all windows
    ret = SapModel.View.RefreshView()
    raise_warning("Refresh view", ret)


# def old_get_material_I_section(all_sections, SapModel):
#     """
#     Function duties:
#         Gets the material for each of sections given in all_sections
#     Input:
#         all_sections: list of sections
#     Remark:
#         Function to be used with I sections (for other sections, i.e. rectangle)
#         use another sap function
#     """
#     section_material = dict()
#     for SectionName in all_sections:
#         FileName, MatProp = "", ""
#         t3, t2, tf, tw, t2b, tfb, Color = 0, 0, 0, 0, 0, 0, 0
#         Notes, GUID = "", ""
#         output = SapModel.PropFrame.GetISection(SectionName, FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID)
#         [FileName, MatProp, t3, t2, tf, tw, t2b, tfb, Color, Notes, GUID, ret] = output
#         raise_warning(f'Get material from {SectionName} section', ret)

#         section_material[SectionName] = MatProp

#     return section_material


def get_material_properties(all_materials, SapModel):
    """
    Function duties:
        Gets material properties from a list of materials
    Remark:
        e: Modulus of Elasticity, E
        u: Poisson
        a: Coefficient of Thermal Expansion
        g: Shear modulus
    """
    material_dict = dict()

    for MatName in all_materials:
        e, u, a, g = 0, 0, 0, 0
        output = SapModel.PropMaterial.GetMPIsotropic(MatName, e, u, a, g)
        [e, u, a, g, ret] = output
        raise_warning("Get material properties", ret)

        material_dict[MatName] = {"E": e, "u": u, "a": a}

    return material_dict


def set_solver(SapModel, SolverType=2, SolverProcessType=2,
               NumberParallelRuns=0, ResponseFileSizeMaxMB=-1, NumberAnalysisThreads=-1,
               StiffCase="MODAL") -> None:
    """
    Analysis options:
    SolverType:
        0 = Standard solver
        1 = Advanced solver
        2 = Multi-threaded solver
    SolverProcessType:
        0 = Auto (program determined)
        1 = GUI process
        2 = Separate process (out of GUI)
    NumberParallelRuns: integer between -8 and 8, inclusive, not including -1
        -8 to -2 = The negative of the program determined value when the assigned value is
        0 = Auto parallel (use up to all physical cores - max 8).
        1 = Serial.
        2 to 8 = User defined parallel (use up to this fixed number of cores - max 8).
    ResponseFileSizeMaxMB: The maximum size of a response file in MB before a new response file
        is created. Positive if user specified, negative if program determined.
    NumberAnalysisThreads: Number of threads that the analysis can use. Positive if user specified,
        negative if program determined.
        IMPORTANT: Check the number in the computer (Task manager -> CPU -> Number of logical processors)
            If the process needs to be solved very quickly, set to logical processors - 1, or
            logical processors - 2 (to leave some for the computer to work).
    StiffCase: The name of the load case used when outputting the mass and stiffness matrices to
        text files If this item is blank, no matrices are output.
    """
    # Set the solver options
    ret = SapModel.Analyze.SetSolverOption_3(
        SolverType, SolverProcessType, NumberParallelRuns, ResponseFileSizeMaxMB, NumberAnalysisThreads, StiffCase)
    raise_warning("Set solver options", ret)


def set_solver_v2(SapModel, options: dict) -> None:
    """
    Sets the solver options for the model using a dictionary of parameters.

    Args:
        SapModel: The SAP2000 model object.
        options (dict): Dictionary containing the solver options with the following keys:
            - SolverType (int): 0 = Standard solver, 1 = Advanced solver, 2 = Multi-threaded solver.
            - SolverProcessType (int): 0 = Auto (program determined), 1 = GUI process, 2 = Separate process.
            - NumberParallelRuns (int): Between -8 and 8, excluding -1 and 0. Defines the number of parallel runs.
            - ResponseFileSizeMaxMB (int): Maximum response file size in MB before a new file is created.
            - NumberAnalysisThreads (int): Number of threads available for analysis (positive = user specified, negative = program determined).
            - StiffCase (str): Name of the load case used when outputting stiffness and mass matrices to text files.
    """
    # Extract values from dictionary, provide defaults if missing
    SolverType = options.get("SolverType", 2)
    SolverProcessType = options.get("SolverProcessType", 2)
    NumberParallelRuns = options.get("NumberParallelRuns", 0)
    ResponseFileSizeMaxMB = options.get("ResponseFileSizeMaxMB", -1)
    NumberAnalysisThreads = options.get("NumberAnalysisThreads", -1)
    StiffCase = options.get("StiffCase", "")

    # API call
    ret = SapModel.Analyze.SetSolverOption_3(
        SolverType,
        SolverProcessType,
        NumberParallelRuns,
        ResponseFileSizeMaxMB,
        NumberAnalysisThreads,
        StiffCase
    )

    # Error handling
    raise_warning("Set solver options", ret)


def save_model(SapModel, file_path: str = "") -> None:
    """
    Saves the current SAP2000 model to the specified file path.

    Parameters:
    ----------
    SapModel : comtypes.client.gen.CSI_SAP2000.cSapModel
        The SAP2000 model object.
    file_path : str, optional
        The full path where the model should be saved. Must include .sdb extension.
        If not specified, saves to the currently assigned file name.
    """
    ret = SapModel.File.Save(file_path)
    raise_warning('Error saving model', ret)


def set_shell_area_property_1(SapModel, Name, ShellType, IncludeDrillingDOF,
                              MatProp, MatAng, Thickness, Bending,
                              Color=-1, Notes="", GUID="") -> None:
    """
    Sets or modifies a shell-type area property using SetShell_1 (recommended SAP2000 API).

    Parameters:
        SapModel: SAP2000 model object
        Name: str, area property name (existing = modifies, new = creates)
        ShellType: int, 1-6 (see SAP2000 doc)
        IncludeDrillingDOF: bool, include drilling DOF? (ignored for types 3, 4, 6)
        MatProp: str, material name (ignored for type 6)
        MatAng: float, material angle [deg] (ignored for type 6)
        Thickness: float, membrane thickness (ignored for type 6)
        Bending: float, bending thickness (ignored for type 6)
        Color: int, color code (-1 = auto assigned)
        Notes: str, optional notes
        GUID: str, optional GUID ("" = auto assigned)

    """
    ret = SapModel.PropArea.SetShell_1(
        Name,
        ShellType,
        IncludeDrillingDOF,
        MatProp,
        MatAng,
        Thickness,
        Bending,
        Color,
        Notes,
        GUID
    )
    raise_warning(f"Set shell area property '{Name}'", ret)


def set_plane_area_property(SapModel, Name, PlaneType, MatProp, MatAng,
                            Thickness, Incompatible, Color=-1, Notes="", GUID="") -> None:
    """
    Sets or modifies a plane-type area property using SetPlane.

    Parameters:
        SapModel: SAP2000 model object
        Name: str, area property name (existing = modifies, new = creates)
        PlaneType: int, 1 = Plane-stress, 2 = Plane-strain
        MatProp: str, material name
        MatAng: float, material angle [deg]
        Thickness: float, thickness
        Incompatible: bool, include incompatible bending modes
        Color: int, color code (-1 = auto assigned)
        Notes: str, optional notes
        GUID: str, optional GUID ("" = auto assigned)
    """
    ret = SapModel.PropArea.SetPlane(
        Name,
        PlaneType,
        MatProp,
        MatAng,
        Thickness,
        Incompatible,
        Color,
        Notes,
        GUID
    )
    raise_warning(f"Set plane area property '{Name}'", ret)


def set_asolid_area_property(SapModel, Name, MatProp, MatAng, Arc,
                             Incompatible, CSys="Global", Color=-1, Notes="", GUID="") -> None:
    """
    Sets or modifies an asolid-type area property using SetAsolid.

    Parameters:
        SapModel: SAP2000 model object
        Name: str, area property name (existing = modifies, new = creates)
        MatProp: str, material name
        MatAng: float, material angle [deg]
        Arc: float, arc angle [deg] (0 means 1 radian ~57.3 deg)
        Incompatible: bool, include incompatible bending modes
        CSys: str, coordinate system ("Global" default)
        Color: int, color code (-1 = auto assigned)
        Notes: str, optional notes
        GUID: str, optional GUID ("" = auto assigned)
    """
    ret = SapModel.PropArea.SetAsolid(
        Name,
        MatProp,
        MatAng,
        Arc,
        Incompatible,
        CSys,
        Color,
        Notes,
        GUID
    )
    raise_warning(f"Set asolid area property '{Name}'", ret)


def get_shell_area_property_1(SapModel, Name):
    """
    Retrieves properties of a shell-type area section using GetShell_1.

    Parameters:
        SapModel: SAP2000 model object
        Name: str, name of the area property

    Returns:
        A dictionary containing the area property data, or None if retrieval fails.
    """
    ShellType = 0
    IncludeDrillingDOF = False
    MatProp = ""
    MatAng = 0.0
    Thickness = 0.0
    Bending = 0.0
    Color = -1
    Notes = ""
    GUID = ""

    output = SapModel.PropArea.GetShell_1(
        Name,
        ShellType,
        IncludeDrillingDOF,
        MatProp,
        MatAng,
        Thickness,
        Bending,
        Color,
        Notes,
        GUID
    )

    [ShellType, IncludeDrillingDOF, MatProp, MatAng,
     Thickness, Bending, Color, Notes, GUID, ret] = output

    raise_warning(f"Get shell area property '{Name}'", ret)

    if ret != 0:
        return None

    output = {
        "Name": Name,
        "ShellType": ShellType,
        "IncludeDrillingDOF": IncludeDrillingDOF,
        "MatProp": MatProp,
        "MatAng": MatAng,
        "Thickness": Thickness,
        "Bending": Bending,
        "Color": Color,
        "Notes": Notes,
        "GUID": GUID
    }

    return output


def get_plane_area_property(SapModel, Name):
    """
    Retrieves properties of a plane-type area section using GetPlane.

    Parameters:
        SapModel: SAP2000 model object
        Name: str, name of the area property

    Returns:
        A dictionary containing the area property data, or None if retrieval fails.
    """
    MyType = 0
    MatProp = ""
    MatAng = 0.0
    Thickness = 0.0
    Incompatible = False
    Color = -1
    Notes = ""
    GUID = ""

    output = SapModel.PropArea.GetPlane(
        Name,
        MyType,
        MatProp,
        MatAng,
        Thickness,
        Incompatible,
        Color,
        Notes,
        GUID
    )

    [MyType, MatProp, MatAng, Thickness,
     Incompatible, Color, Notes, GUID, ret] = output

    raise_warning(f"Get plane area property '{Name}'", ret)

    if ret != 0:
        return None

    output = {
        "Name": Name,
        "PlaneType": MyType,
        "MatProp": MatProp,
        "MatAng": MatAng,
        "Thickness": Thickness,
        "Incompatible": Incompatible,
        "Color": Color,
        "Notes": Notes,
        "GUID": GUID
    }
    return output


def get_asolid_area_property(SapModel, Name):
    """
    Retrieves properties of an asolid-type area section (template).

    Parameters:
        SapModel: SAP2000 model object
        Name: str, name of the area property

    Returns:
        A dictionary containing the area property data, or None if retrieval fails.
    """
    # These are placeholder variables — update with actual API signature and fields
    MatProp = ""
    MatAng = 0.0
    Arc = False
    Incompatible = False
    CSys = ""
    Color = -1
    Notes = ""
    GUID = ""
    
    output = SapModel.PropArea.GetAsolid(
        Name,
        MatProp,
        MatAng,
        Arc,
        Incompatible,
        CSys,
        Color,
        Notes,
        GUID
    )

    [MatProp, MatAng, Arc, Incompatible, CSys,
        Color, Notes, GUID, ret] = output

    raise_warning(f"Get asolid area property '{Name}'", ret)

    if ret != 0:
        return None

    output = {
        "Name": Name,
        "MatProp": MatProp,
        "MatAng": MatAng,
        "Arc": Arc,
        "Incompatible": Incompatible,
        "CSys": CSys,
        "Color": Color,
        "Notes": Notes,
        "GUID": GUID
    }

    return output


def set_areaproperty(area_dict, SapModel):
    """
    Updates an area property in SAP2000 using the appropriate setter
    based on type (Shell, Plane, Asolid).

    Parameters:
        SapModel: SAP2000 model object
        area_dict: dict
            A dictionary where each key is an area property name, and each value is a
            dict of property overrides to apply. The structure should match that
            returned by get_area_property, possibly updated.
    """
    for Name in area_dict:
        area_dict_i = outils.get_area_property(SapModel, Name)
        area_dict_i.update(area_dict[Name])  # Update with new values

        if area_dict_i.get('is_shell'):
            # Extract fields with fallbacks
            ShellType = area_dict_i.get('ShellType')
            IncludeDrillingDOF = area_dict_i.get('IncludeDrillingDOF')
            MatProp = area_dict_i.get('MatProp')
            MatAng = area_dict_i.get('MatAng')
            Thickness = area_dict_i.get('Thickness')
            Bending = area_dict_i.get('Bending')
            Color = area_dict_i.get('Color', -1)
            Notes = area_dict_i.get('Notes', "")
            GUID = area_dict_i.get('GUID', "")

            set_shell_area_property_1(
                SapModel,
                Name,
                ShellType,
                IncludeDrillingDOF,
                MatProp,
                MatAng,
                Thickness,
                Bending,
                Color,
                Notes,
                GUID
            )

        elif area_dict_i.get('is_plane'):
            PlaneType = area_dict_i.get('PlaneType')
            MatProp = area_dict_i.get('MatProp')
            MatAng = area_dict_i.get('MatAng')
            Thickness = area_dict_i.get('Thickness')
            Incompatible = area_dict_i.get('Incompatible')
            Color = area_dict_i.get('Color', -1)
            Notes = area_dict_i.get('Notes', "")
            GUID = area_dict_i.get('GUID', "")

            set_plane_area_property(
                SapModel,
                Name,
                PlaneType,
                MatProp,
                MatAng,
                Thickness,
                Incompatible,
                Color,
                Notes,
                GUID
            )

        elif area_dict_i.get('is_asolid'):
            MatProp = area_dict_i.get('MatProp')
            MatAng = area_dict_i.get('MatAng')
            Arc = area_dict_i.get('Arc')
            Incompatible = area_dict_i.get('Incompatible')
            CSys = area_dict_i.get('CSys')
            Color = area_dict_i.get('Color', -1)
            Notes = area_dict_i.get('Notes', "")
            GUID = area_dict_i.get('GUID', "")

            set_asolid_area_property(
                SapModel,
                Name,
                MatProp,
                MatAng,
                Arc,
                Incompatible,
                CSys,
                Color,
                Notes,
                GUID
            )
        else:
            raise_warning(f"Cannot set area property '{Name}': Type not identified.", 1)


def get_frame_points(SapModel, FrameName):
    """
    Retrieves the names of the I-End and J-End points of a specified frame (line) object in SAP2000.

    Parameters:
        SapModel : SAP2000 model object
            The active SAP2000 model.
        FrameName : str
            The name of the frame (line) object.

    Returns:
        tuple (PointI, PointJ)
            The names of the I-End and J-End points if successful.
        None
            If the API call fails.
    """
    PointI = ""
    PointJ = ""

    output = SapModel.FrameObj.GetPoints(FrameName, PointI, PointJ)
    PointI, PointJ, ret = output

    raise_warning(f"Get points for frame '{FrameName}'", ret)

    return PointI, PointJ


def add_frame_by_point(SapModel, point_i, point_j,
                       prop_name="Default", user_name=""):
    """
    Adds a frame object by point names in SAP2000.

    Parameters:
        SapModel : SAP2000 model object
        point_i : str
            Name of the point at the I-end.
        point_j : str
            Name of the point at the J-end.
        prop_name : str, optional
            Frame section property (default = "Default")
        user_name : str, optional
            User-specified frame name (default = "")

    Returns:
        name : str
            The name assigned to the created frame object
    """
    name = ""
    output = SapModel.FrameObj.AddByPoint(
        point_i,
        point_j,
        name,
        prop_name,
        user_name
    )
    name, ret = output
    """
    Warning omitted: in some SAP2000 OAPI bindings,
    return code may not behave consistently — verify if needed.
    """
    # raise_warning(f"Add frame from points {point_i} to {point_j}", ret)

    return name



def add_frame_by_coord(SapModel, coord_i, coord_j, 
                       prop_name="Default", user_name="", csys="Global"):
    """
    Adds a frame object by coordinates in SAP2000.

    Parameters:
        SapModel : SAP2000 model object
        coord_i : dict with 'x', 'y', 'z' (I-end coordinates)
        coord_j : dict with 'x', 'y', 'z' (J-end coordinates)
        prop_name : str, optional
            Frame section property (default = "Default")
        user_name : str, optional
            User-specified frame name (default = "")
        csys : str, optional
            Coordinate system (default = "Global")

    Returns:
        name : str
            The name assigned to the created frame object
    """
    name = ""
    output = SapModel.FrameObj.AddByCoord(
        coord_i['x'], coord_i['y'], coord_i['z'],
        coord_j['x'], coord_j['y'], coord_j['z'],
        name,
        prop_name,
        user_name,
        csys
    )
    name, ret = output
    """
    Warning omitted: it seems to be a bug; even if the element
    is properly created, return is 1; check further...
    """
    # raise_warning(f"Add frame from {coord_i} to {coord_j}", ret)

    return name


def extend_frame(SapModel, frame_name, i_end, j_end, item1, item2="") -> None:
    """
    Extends a straight frame object at its I-end, J-end, or both using other frame objects as extension lines.

    Parameters:
        SapModel : SAP2000 model object
        frame_name : str
            Name of the frame object to extend
        i_end : bool
            Whether to extend the I-end
        j_end : bool
            Whether to extend the J-end
        item1 : str
            Name of first frame object to use as extension line
        item2 : str, optional
            Name of second frame object to use as extension line (default = "")
    """
    ret = SapModel.EditFrame.Extend(
        frame_name,
        i_end,
        j_end,
        item1,
        item2
    )
    raise_warning(f"Extend frame '{frame_name}' using '{item1}' and '{item2}'", ret)


def add_point_cartesian(SapModel, coord, user_name="",
                        csys="Global", merge_off=False, merge_number=0):
    """
    Adds a point object to the model at given coordinates using AddCartesian.

    Parameters:
        SapModel : SAP2000 model object
        coord : dict with 'x', 'y', 'z'
            Coordinates of the point (in csys)
        user_name : str, optional
            User-specified name for the point (default "")
        csys : str, optional
            Coordinate system name (default "Global")
        merge_off : bool, optional
            If True, prevents merging with existing points at the same location (default False)
        merge_number : int, optional
            Merge number (default 0)

    Returns:
        point_name : str
            The name assigned (or merged into)
    """
    point_name = ""
    output = SapModel.PointObj.AddCartesian(
        coord['x'], coord['y'], coord['z'],
        point_name,
        user_name,
        csys,
        merge_off,
        merge_number
    )
    point_name, ret = output
    raise_warning(f"Add point at {coord} (UserName='{user_name}')", ret)
    return point_name


def get_frame_section(SapModel, frame_name):
    """
    Retrieves the section property (and auto select list, if any) assigned to a frame object.

    Parameters:
        SapModel : SAP2000 model object
        frame_name : str
            The name of the frame object

    Returns:
        dict containing:
            - 'PropName': section property name or None if not assigned
            - 'SAuto': auto select list name ("" if not assigned)
        or None if error
    """
    PropName = ""
    SAuto = ""

    output = SapModel.FrameObj.GetSection(
        frame_name,
        PropName,
        SAuto
    )
    PropName, SAuto, ret = output
    raise_warning(f"Get section for frame '{frame_name}'", ret)

    return PropName, SAuto


def divide_frame_by_ratio(SapModel, frame_name, num, ratio):
    """
    Divides a straight frame object into segments based on Last/First length ratio.

    Parameters:
        SapModel : SAP2000 model object
        frame_name : str
            Name of the frame object to divide
        num : int
            Number of frame objects to create
        ratio : float
            Last/First length ratio for the new frame objects

    Returns:
        new_names : list of str
            Names of the new frame objects created
        ret : int
            0 if success, nonzero if error
    """
    new_names = []
    output = SapModel.EditFrame.DivideByRatio(
        frame_name,
        num,
        ratio,
        new_names
    )
    frame_name_1, frame_name_2 = output[0]
    ret = output[1]

    raise_warning(
        f"Divide frame '{frame_name}' into {num} segments with Last/First ratio {ratio}",
        ret
    )
    return frame_name_1, frame_name_2


def divide_frame_at_distance(SapModel, frame_name, dist, i_end=True):
    """
    DOES NOT WORK PROPERLY (BUG IN OAPI?)

    Divides a straight frame object into two segments at a specified distance from one end.

    Parameters:
    ----------
    SapModel : object
        The SAP2000 model API object.
    frame_name : str
        The name of the frame object to divide.
    dist : float
        Distance from the specified end (I-end or J-end) where the division occurs.
    i_end : bool, default=True
        If True, distance is measured from the I-end; if False, from the J-end.

    Returns:
    -------
    new_names : list of str
        Names of the two new frame objects created after division.
    ret : int
        0 if success, nonzero if error.
    """
    new_names = ["", ""]
    num = 2
    output = SapModel.EditFrame.DivideAtDistance(
        frame_name,
        dist,
        i_end,
        num,
        new_names
    )

    # Unpack output: [NewName array, ret]
    new_names, ret = output

    raise_warning(
        f"Divide frame '{frame_name}' at distance {dist} from {'I-end' if i_end else 'J-end'}",
        ret
    )

    return new_names


def get_group_assignments(SapModel, group_name):
    """
    Retrieves and categorizes object assignments for a specified SAP2000 group.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model API object.
    group_name : str
        The name of the group whose assignments are to be retrieved.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'PointObj' : list of point object names, or None if none assigned
        - 'FrameObj' : list of frame object names, or None if none assigned
        - 'CableObj' : list of cable object names, or None if none assigned
        - 'TendonObj' : list of tendon object names, or None if none assigned
        - 'AreaObj' : list of area object names, or None if none assigned
        - 'SolidObj' : list of solid object names, or None if none assigned
        - 'LinkObj' : list of link object names, or None if none assigned
        
    Notes
    -----
    If the group has no assignments of a particular object type, that entry
    in the dictionary will be None. The function calls `raise_warning`
    internally to handle and report SAP2000 API return codes.
    """
    NumberItems = 0
    ObjectTypes = []
    ObjectNames = []

    output = SapModel.GroupDef.GetAssignments(
        group_name,
        NumberItems,
        ObjectTypes,
        ObjectNames
    )
    NumberItems, ObjectTypes, ObjectNames, ret = output
    raise_warning(f"Get assignments for group '{group_name}'", ret)

    type_map = {
        1: 'PointObj',
        2: 'FrameObj',
        3: 'CableObj',
        4: 'TendonObj',
        5: 'AreaObj',
        6: 'SolidObj',
        7: 'LinkObj'
    }

    result = {v: [] for v in type_map.values()}

    for obj_type, obj_name in zip(ObjectTypes, ObjectNames):
        obj_type_name = type_map.get(obj_type)
        if obj_type_name:
            result[obj_type_name].append(obj_name)

    # Convert empty lists to None
    for k in result:
        if not result[k]:
            result[k] = None

    return result


def select_group_objects(SapModel, group_name) -> None:
    """
    Selects all objects assigned to a specified group in SAP2000.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model API object.
    group_name : str
        The name of the group to select.
    """
    # First clear current selection
    SapModel.SelectObj.ClearSelection()

    # Get group assignments
    NumberItems = 0
    ObjectTypes = []
    ObjectNames = []
    output = SapModel.GroupDef.GetAssignments(
        group_name, NumberItems, ObjectTypes, ObjectNames
    )
    NumberItems, ObjectTypes, ObjectNames, ret = output
    raise_warning(f"Get assignments for group '{group_name}'", ret)

    # Select each object by type
    for obj_type, obj_name in zip(ObjectTypes, ObjectNames):
        if obj_type == 1:
            SapModel.PointObj.SetSelected(obj_name, True)
        elif obj_type == 2:
            SapModel.FrameObj.SetSelected(obj_name, True)
        elif obj_type == 3:
            SapModel.CableObj.SetSelected(obj_name, True)
        elif obj_type == 4:
            SapModel.TendonObj.SetSelected(obj_name, True)
        elif obj_type == 5:
            SapModel.AreaObj.SetSelected(obj_name, True)
        elif obj_type == 6:
            SapModel.SolidObj.SetSelected(obj_name, True)
        elif obj_type == 7:
            SapModel.LinkObj.SetSelected(obj_name, True)


def move_elements_in_group(SapModel, group_name, dx=0.0, dy=0.0, dz=0.0) -> None:
    """
    Moves currently selected objects in SAP2000 by specified offsets.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model API object.
    dx : float, optional
        Offset in the X-direction (default is 0.0).
    dy : float, optional
        Offset in the Y-direction (default is 0.0).
    dz : float, optional
        Offset in the Z-direction (default is 0.0).

    Notes
    -----
    This function applies to all currently selected objects:
    points, frames, cables, tendons, areas, solids, and links.
    """
    select_group_objects(SapModel, group_name)

    ret = SapModel.EditGeneral.Move(dx, dy, dz)
    raise_warning(f"Move selected elements by (dx={dx}, dy={dy}, dz={dz})", ret)


def delete_frame(SapModel, name, item_type=0) -> None:
    """
    Deletes frame objects in SAP2000.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model API object.
    name : str
        The name of the frame object or group, depending on item_type.
        Ignored if item_type is 2 (SelectedObjects).
    item_type : int, optional
        Specifies what to delete:
        0 = Object (default): Deletes the specified frame object.
        1 = Group: Deletes all frame objects in the specified group.
        2 = SelectedObjects: Deletes all selected frame objects.
    """
    ret = SapModel.FrameObj.Delete(name, item_type)
    raise_warning(f"Delete frame(s): name='{name}', item_type={item_type}", ret)


def modify_frame_length_group(SapModel, group_name, length_factor, origin_fixed=True,
                              round_coordinates=True,
                              extension_moved_elem_group='-moved_elements') -> None:
    """
    Extends or trims frames in a SAP2000 group and moves associated elements if applicable.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model API object.
    group_name : str
        The name of the group containing frames to extend or trim.
    length_factor : float
        Factor by which to extend (>1) or trim (<1) the frame length.
    origin_fixed : bool, default=True
        If True, the I-end is fixed and the J-end is extended/trimmed. Otherwise, vice versa.
    round_coordinates : bool, default=True
        If True, frame lengths are rounded to mitigate floating-point precision issues.
    extension_moved_elem_group : str, default='-moved_elements'
        Name suffix for group containing elements to move along with frame extension.
        If no elements to be moved, let the default value (if the group does not exist, nothing happens)
        Remark: it is expected that these elements are in the end point of the frame if origin_fixed.

    Notes
    -----
    It is recommended that original elements have rounded lengths for accurate extension/trimming.

    The most appropriate way would have been to use extend/trim functions.
    This approach was attempted with the following workflow:
        - if ii: coord_k = outils.compute_point_in_perpendicular_plane(coord_0, coord_i)
        - elif jj: coord_k = outils.compute_point_in_perpendicular_plane(coord_f, coord_i)
        - Point_k = add_point_cartesian(SapModel, coord_k)
        - frame_added = add_frame_by_point(SapModel, Point_i, Point_k, prop_name=prop_name)
        - extend_frame(SapModel, frame, ii, jj, item1="", item2=frame_added)
    Nevertheless, it was found that the added points had a small error because
    of rounding, and the extend/trim functions did not work as expected.

    Another approach was to divide the frame (when length_factor<1) but it required rounding
    for working properly (so it is commented)
    """
    group_name_moved = group_name + extension_moved_elem_group
    if origin_fixed:
        ii, jj = False, True
    else:
        ii, jj = True, False

    elements = get_group_assignments(SapModel, group_name)
    frames = elements['FrameObj']

    frames_to_delete, distance_moved = list(), list()

    for frame in frames:
        # A) Get frame length
        Point_0, Point_f = get_frame_points(SapModel, frame)
        coord_0 = get_pointcoordinates([Point_0], SapModel)[Point_0]
        coord_f = get_pointcoordinates([Point_f], SapModel)[Point_f]
        frame_length = np.sqrt(np.sum([(coord_0[i] - coord_f[i])**2 for i in coord_0]))
        if round_coordinates:
            frame_length = outils.round_6_sign_digits(frame_length)
        final_length = frame_length * length_factor
        if round_coordinates:
            final_length = outils.round_3_sign_digits(final_length)

        coord_i = outils.compute_point_on_line(coord_0, coord_f, final_length)
        if round_coordinates:
            coord_i = {key: outils.round_6_sign_digits(coord_i[key]) for key in coord_i}
        if ii:
            dx, dy, dz = [coord_i[key] - coord_0[key] for key in coord_0]
            distance_moved.append(
                np.sqrt(np.sum([(coord_0[i] - coord_i[i])**2 for i in coord_0])))
        elif jj:
            dx, dy, dz = [coord_i[key] - coord_f[key] for key in coord_f]
            distance_moved.append(
                np.sqrt(np.sum([(coord_f[i] - coord_i[i])**2 for i in coord_f])))

        if round_coordinates:
            dx, dy, dz = [outils.round_6_sign_digits(val) for val in (dx, dy, dz)]

        # B) Get the final position
        groups = get_group_frame_obj(SapModel, frame)
        """
        Procedure:
            - Add a point (Point_i) at the new position
            - Add a frame from the static frame point (Point_0 or Point_f) to Point_i
            - Assign groups to that frame
            - Store the frame to delete later
        """
        Point_i = add_point_cartesian(SapModel, coord_i)
        prop_name, _ = get_frame_section(SapModel, frame)
        if ii:
            frame_added = add_frame_by_point(SapModel, Point_i, Point_f, prop_name=prop_name)
        elif jj:
            frame_added = add_frame_by_point(SapModel, Point_0, Point_i, prop_name=prop_name)
        for group in groups:
            if group == 'ALL' or group == 'All':
                continue
            set_group_frame_obj(SapModel, frame_added, group, remove=False)
        frames_to_delete.append(frame)

    # Move elements in group_name_moved (if any)
    # Remark: these elements should be in the beginning of the frames if ii, else in the end if jj
    if len(set(distance_moved)) == 1:
        move_elements_in_group(SapModel, group_name_moved, dx=dx, dy=dy, dz=dz)
    elif len(set(distance_moved)) > 1:
        warnings.warn(
            f"Elements in group '{group_name}' have different distances moved: {distance_moved}. "
            "Elements are not moved.")

    # Remove frames that were trimmed (if any)
    for frame in frames_to_delete:
        delete_frame(SapModel, frame, item_type=0)

    # if final_length < frame_length:  # OLD PROCEDURE
    #     """
    #     Procedure:
    #         - Divide the frame by ratio
    #         - Store the frame to delete later
    #     """
    #     num = 2
    #     if jj:
    #         ratio = (frame_length - final_length) / final_length
    #     elif ii:
    #         ratio = final_length / (final_length - frame_length)

    #     frame, frame_delete = divide_frame_by_ratio(SapModel, frame, num, ratio)
    #     frames_to_delete.append(frame_delete)


def get_group_frame_obj(SapModel, element):
    """
    Returns a list of group names assigned to a frame object in SAP2000.
    """
    num, groups, ret = SapModel.FrameObj.GetGroupAssign(element)

    raise_warning(f"Get group assignments for frame '{element}'", ret)

    return list(groups)


def set_group_frame_obj(SapModel, element, group_name, remove=False) -> None:
    """
    Assigns or removes a frame object from a group in SAP2000.

    Parameters
    ----------
    SapModel : object
        The cSapModel object from the SAP2000 API.

    element : str
        Name of the frame object (e.g., "F1", "7", etc.).

    group_name : str
        Name of the existing group in SAP2000.

    remove : bool, optional
        If True, removes the frame from the group.
        If False (default), adds the frame to the group.
    """
    ret = SapModel.FrameObj.SetGroupAssign(element, group_name, remove, 0)

    raise_warning(f"Get group assignments for frame '{element}'", ret)


def get_group_names(SapModel):
    """
    Retrieves the names of all groups defined in the SAP2000 model.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model object.

    Returns
    -------
    list
        A list of group names.
    """
    output = SapModel.GroupDef.GetNameList()
    num, group_names, ret = output
    raise_warning("Group names", ret)

    return list(group_names)


def group_contains_object_type(SapModel, group_name, obj_type):
    """
    Function Duties:
        Checks if a group contains objects of a specific type.
    Input:
        SapModel: SAP2000 model object.
        group_name: Name of the group to check.
        obj_type: Type of object to check for (e.g., 'point', 'frame', 'cable', etc.).
    Output:
        Returns True if the group contains objects of the specified type, False otherwise.
    """
    type_map = {
        "point": 1,
        "frame": 2,
        "cable": 3,
        "tendon": 4,
        "area": 5,
        "solid": 6,
        "link": 7,
    }

    if obj_type.lower() not in type_map:
        raise ValueError(f"Object type '{obj_type}' is not valid. Allowed types: {list(type_map.keys())}")

    target_code = type_map[obj_type.lower()]

    output = SapModel.GroupDef.GetAssignments(group_name)
    number, obj_types, obj_names, ret = output
    
    raise_warning(f'Group  contains object type {obj_type}, with code {target_code}', ret)

    group_contains = any(t == target_code for t in obj_types)

    return group_contains


def get_point_local_axes(SapModel, point_name):
    """
    Retrieves the local axes orientation angles for a point object in SAP2000.

    Parameters
    ----------
    SapModel : object
        The SAP2000 model API object.
    point_name : str
        Name of the point object.

    Returns
    -------
    dict
        Dictionary with:
        - 'a': rotation about local 3 axis (degrees)
        - 'b': rotation about resulting local 2 axis (degrees)
        - 'c': rotation about resulting local 1 axis (degrees)
        - 'advanced': bool, True if obtained using advanced local axes parameters
    """
    a, b, c, advanced = 0.0, 0.0, 0.0, False
    output = SapModel.PointObj.GetLocalAxes(point_name, a, b, c, advanced)
    a, b, c, advanced, ret = output
    raise_warning(f"Get local axes for point '{point_name}'", ret)

    return {
        "a": a,
        "b": b,
        "c": c,
        "advanced": advanced
    }


def check_local_axes_channels(SapModel, point_list) -> None:
    """
    Check that local axes are the same as global axes for the point objects
    in point_list
    """
    local_axes = dict()
    for point in point_list:
        local_axes[point] = get_point_local_axes(SapModel, point)

    non_aligned_points = [
        pt for pt, axes in local_axes.items()
        if not (axes['a'] == 0 and axes['b'] == 0 and axes['c'] == 0)
    ]

    if non_aligned_points:
        warnings.warn(
            f"Local axes are not aligned with global axes for points: {', '.join(non_aligned_points)}. "
            "Errors can be encountered when reading mode shapes"
        )


def add_load_pattern(SapModel,
                     name: str,
                     pattern_type=3,
                     selfwt_multiplier: float = 0.0,
                     add_load_case: bool = True) -> None:
    """
    Creates a new load pattern in SAP2000.

    Parameters
    ----------
    SapModel : object
        Active SAP2000 model object.
    name : str
        Name for the new load pattern.
    pattern_type : int, optional
        eLoadPatternType code (1=DEAD, 3=LIVE, 6=WIND, 7=SNOW, 8=OTHER, ...).
        Defaults to 3 (LIVE).
    selfwt_multiplier : float, optional
        Self-weight multiplier. Use 1.0 to include self-weight in this pattern.
        Defaults to 0.0.
    add_load_case : bool, optional
        If True, also creates the corresponding linear static load case.
        Defaults to True.

    Raises
    ------
    Warning
        If SAP2000 returns a nonzero code (pattern not created).
    """
    unlock_model(SapModel)
    ret = SapModel.LoadPatterns.Add(name, pattern_type,
                                    selfwt_multiplier, add_load_case)
    raise_warning(f"Add load pattern '{name}'", ret)


def set_point_force_load(
                        SapModel,
                        name: str,
                        load_pattern: str,
                        values,
                        replace: bool = False,
                        csys: str = "Global",
                        item_type: int = 0,
                    ) -> None:
    """
    Assigns a point load (forces and moments) to a point object or group in SAP2000.

    Parameters
    ----------
    SapModel : object
        Active SAP2000 model object (cSapModel).
    name : str
        Name of the point object or group, depending on `item_type`.
    load_pattern : str
        Name of the load pattern to which the point load will be assigned.
    values : sequence of float
        Iterable with 6 load components, in this exact order:
            values[0] = F1  [Force]
            values[1] = F2  [Force]
            values[2] = F3  [Force]
            values[3] = M1  [Force·Length]
            values[4] = M2  [Force·Length]
            values[5] = M3  [Force·Length]
    replace : bool, optional
        If True, all previous point loads (for this pattern and these objects)
        are deleted before making the new assignment. Defaults to False.
    csys : str, optional
        Coordinate system name for the load ("Global", "Local", or a defined CSys).
        Defaults to "Global".
    item_type : int, optional
        How `name` is interpreted:
            0 → Object (single point object named `name`)
            1 → Group (all point objects in group `name`)
            2 → SelectedObjects (all currently selected points; `name` is ignored)
        Defaults to 0.

    Raises
    ------
    Warning
        If SAP2000 returns a nonzero code.
    """
    # basic safety: make sure we always send 6 values
    if len(values) != 6:
        raise ValueError("values must be an iterable with exactly 6 numbers: F1, F2, F3, M1, M2, M3")

    ret = SapModel.PointObj.SetLoadForce(
        name,
        load_pattern,
        list(values),
        replace,
        csys,
        item_type,
    )
    raise_warning(f"Set point force load on '{name}' (pattern='{load_pattern}')", ret)


def set_time_history_function_from_file(
                                        SapModel,
                                        name: str,
                                        file_name: str,
                                        head_lines: int,
                                        pre_chars: int,
                                        points_per_line: int,
                                        value_type: int,
                                        free_format: bool,
                                        number_fixed: int = 10,
                                        dt: float = 0.02,
                                        ) -> None:
    """
    Defines or updates a time-history function in SAP2000 from a text file,
    using Func.FuncTH.SetFromFile_1.

    Parameters
    ----------
    SapModel : object
        Active SAP2000 model object (cSapModel).
    name : str
        Name of the time-history function. If it exists, it is modified;
        otherwise a new one is created.
    file_name : str
        Full path to the text file containing the function data.
    head_lines : int
        Number of header lines to skip before reading data.
    pre_chars : int
        Number of leading characters to skip on each data line.
    points_per_line : int
        Number of function points contained in each line of the file.
    value_type : int
        1 → values at equal time intervals (DT is used)
        2 → (time, value) pairs
    free_format : bool
        True if data is in free format; False if in fixed format.
    number_fixed : int, optional
        Characters per item when `free_format` is False. Default is 10.
    dt : float, optional
        Time step between points when `value_type` is 1. Default is 0.02.

    Raises
    ------
    Warning
        If SAP2000 returns a nonzero code.
    """
    ret = SapModel.Func.FuncTH.SetFromFile_1(
        name,
        file_name,
        head_lines,
        pre_chars,
        points_per_line,
        value_type,
        free_format,
        number_fixed,
        dt,
    )
    raise_warning(f"Set time-history function '{name}' from file '{file_name}'", ret)


def create_linear_modal_history_case(
    SapModel,
    name: str,
    *,
    modal_case: str = "MODAL",
    damping: float | None = None,
    n_steps: int = 200,
    dt: float = 0.02,
    loads: list[dict] | None = None,
) -> None:
    """
    Creates (or resets) a *Linear Modal History* load case and configures it.

    Parameters
    ----------
    SapModel : object
        Active SAP2000 model (cSapModel).
    name : str
        Name of the load case. If it already exists, it is reset to defaults.
    modal_case : str, optional
        Name of the existing modal case this history case will use.
        Default is "MODAL".
    damping : float or None, optional
        Constant modal damping (0 ≤ damping < 1). If None, it is not set.
    n_steps : int, optional
        Number of output time steps. Default 200.
    dt : float, optional
        Output time step size. Default 0.02.
    loads : list of dict, optional
        Each dict describes one load to be added to the case.
        Expected keys (lowercase):
            - "type": "Load" or "Accel"
            - "name": load pattern name (if "Load") or DOF ("U1".."U3","R1".."R3") if "Accel"
            - "func": time history function name
            - "sf": scale factor (float)
            - "tf": time scale factor (float), default 1.0
            - "at": arrival time (float), default 0.0
            - "csys": coord system (str), default "Global" (only for "Accel")
            - "ang": angle (float, deg), default 0.0 (only for "Accel")

    Raises
    ------
    Warning
        If any of the SAP2000 calls returns a nonzero code.
    """
    # 1) init / reset the case
    ret = SapModel.LoadCases.ModHistLinear.SetCase(name)
    raise_warning(f"Create/reset linear modal history case '{name}'", ret)

    # 2) set modal case
    ret = SapModel.LoadCases.ModHistLinear.SetModalCase(name, modal_case)
    raise_warning(f"Set modal case '{modal_case}' for '{name}'", ret)

    # 3) damping (optional)
    if damping is not None:
        ret = SapModel.LoadCases.ModHistLinear.SetDampConstant(name, damping)
        raise_warning(f"Set damping={damping} for '{name}'", ret)

    # 4) time step data
    ret = SapModel.LoadCases.ModHistLinear.SetTimeStep(name, n_steps, dt)
    raise_warning(f"Set time step for '{name}' (n={n_steps}, dt={dt})", ret)

    # 5) loads (optional)
    if loads:
        n = len(loads)
        load_type = []
        load_name = []
        func = []
        sf = []
        tf_ = []
        at = []
        csys = []
        ang = []

        for ld in loads:
            lt = ld["type"]                 # "Load" or "Accel"
            ln = ld["name"]
            fn = ld["func"]
            sfv = ld.get("sf", 1.0)
            tfv = ld.get("tf", 1.0)
            atv = ld.get("at", 0.0)
            csv = ld.get("csys", "Global") if lt == "Accel" else ""
            angv = ld.get("ang", 0.0) if lt == "Accel" else 0.0

            load_type.append(lt)
            load_name.append(ln)
            func.append(fn)
            sf.append(sfv)
            tf_.append(tfv)
            at.append(atv)
            csys.append(csv)
            ang.append(angv)

        ret = SapModel.LoadCases.ModHistLinear.SetLoads(
            name,
            n,
            load_type,
            load_name,
            func,
            sf,
            tf_,
            at,
            csys,
            ang,
        )
        raise_warning(f"Set {n} load(s) for '{name}'", ret)
