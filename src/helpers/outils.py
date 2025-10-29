
import os
import shutil

import numpy as np
import pandas as pd
import sympy as sym

import helpers.sap2000 as sap2000

from scipy.linalg import eigh
from typing import Any, List, Sequence

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


def beam_stiffness(E, I, L):
    """B-E beam element stiffness matrix for bending in a single plane."""
    k = (E * I) / (L**3)
    K = k * np.array([
        [12,    6*L,   -12,    6*L],
        [6*L,  4*L**2, -6*L,  2*L**2],
        [-12,  -6*L,    12,   -6*L],
        [6*L,  2*L**2, -6*L,  4*L**2]
    ], dtype=float)
    return K


def _is_symbolic(*vals) -> bool:
    return any(getattr(v, "free_symbols", set()) for v in vals)


def k_beam(E, I, Le):
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


def column_lateral_stiffness(E, I, L, k_lower, k_upper):
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
    k_col_x = column_lateral_stiffness(E, I2, Lc, ky1, ky2)  # governs Ux
    k_col_y = column_lateral_stiffness(E, I1, Lc, kx1, kx2)  # governs Uy

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
