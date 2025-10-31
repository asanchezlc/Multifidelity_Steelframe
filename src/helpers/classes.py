
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
import sympy as sym

import helpers.outils as outils


@dataclass
class StoryLevelRigidBeam:
    """
    Class to define a story level with rigid diaphragm behavior in its plane
    (infinite flexural stiffness of beams within diaphragm plane).

    Uses equations from 1_1_1_equations_Fh.py and 1_1_2_equations_lumped_model.py.
    Inputs (can be numeric or sympy symbols)
    """
    E: Any
    rho: Any
    B: Any
    H: Any
    t: Any
    Lc: Any
    kx1: Any
    kx2: Any
    ky1: Any
    ky2: Any
    b: Any                 # column section base
    h: Any                 # column section height
    A_b_B: Any                 # beam area (m^2) for corner mass from beam along B
    A_b_H: Any                 # beam area (m^2) for corner mass from beam along H
    is_highest_level: bool
    m_center_extra: Any = 0  # e.g., equipment at center (kg)
    m_corner_extra: Any = 0  # e.g., equipment at center (kg)

    # Derived / Outputs    # Derived (not passed at init)
    A_col: Any = field(init=False)
    I1: Any = field(init=False)
    I2: Any = field(init=False)

    # Outputs
    K_level: Any = None
    M_level: Any = None

    def __post_init__(self) -> None:
        self.A_col = self.b * self.h
        self.I1 = self.b * self.h**3 / 12.0   # strong axis (y-bending -> Uy)
        self.I2 = self.h * self.b**3 / 12.0   # weak axis (x-bending -> Ux)

    def _masses_corner(self) -> Any:
        """Returns total corner mass per corner."""
        m_corner_b_B = outils.lumped_mass_from_beam(self.rho, self.A_b_B, self.B)
        m_corner_b_H = outils.lumped_mass_from_beam(self.rho, self.A_b_H, self.H)
        m_half_col = outils.lumped_mass_from_beam(
            self.rho, self.A_col, self.Lc)
        m_extra = self.m_corner_extra
        if self.is_highest_level:
            m_corner = m_corner_b_B + m_corner_b_H + m_half_col + m_extra
        else:
            m_corner = m_corner_b_B + m_corner_b_H + 2 * m_half_col + m_extra
        return m_corner

    def build_stiffness(self):
        """Compute K_level (3x3) for DOFs [Ux, Uy, Thetaz]."""
        self.K_level = outils.stiffness_matrix_level(self.E, self.I1, self.I2, self.Lc,
                                                     self.kx1, self.kx2, self.ky1, self.ky2,
                                                     self.B, self.H)
        return self.K_level

    def build_mass(self):
        """Compute M_level (3x3) for DOFs [Ux, Uy, Thetaz], origin at geometric center."""
        m_corner = self._masses_corner()
        m_center = self.m_center_extra
        self.M_level = outils.mass_matrix_level(self.rho, self.t, self.B, self.H,
                                                m_corner=m_corner, m_center=m_center)
        return self.M_level

    def level_diagonals(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Return (kx, ky, kt, mx, my, mt) for assembling the interstory global K and M
        (between this level and the one below)."""
        # If K_level is already computed and numeric/symbolic, extract diagonals; otherwise, recompute quickly
        if self.K_level is None:
            self.build_stiffness()
        if self.M_level is None:
            self.build_mass()
        if isinstance(self.K_level, sym.MatrixBase) or isinstance(self.K_level, sym.BlockDiagMatrix):
            output_K = [self.K_level[0, 0],
                        self.K_level[1, 1], self.K_level[2, 2]]
            output_M = [self.M_level[0, 0],
                        self.M_level[1, 1], self.M_level[2, 2]]
        else:
            K = np.array(self.K_level.tolist(), dtype=float)
            M = np.array(self.M_level.tolist(), dtype=float)
            output_K = [K[0, 0], K[1, 1], K[2, 2]]
            output_M = [M[0, 0], M[1, 1], M[2, 2]]
        return output_K + output_M


@dataclass
class StoryLevelGeneralBeam:
    """
    Class to define a story level with diaphragm behavior in its plane
    and a beam which has no infinite flexural stiffness.

    Uses equations from 2_1_1_equations_frame_beam_rotational_k.py.
    Inputs (can be numeric or sympy symbols)
    """
    E: Any
    rho: Any
    B: Any
    H: Any
    t: Any
    Lc: Any
    kx1: Any
    kx2: Any
    ky1: Any
    ky2: Any
    Ac: Any
    Ic_y: Any
    Ic_x: Any
    A_b_B: Any                 # beam area (m^2) for corner mass from beam along B
    A_b_H: Any                 # beam area (m^2) for corner mass from beam along H
    Ib_y_B: Any
    Ib_y_H: Any
    is_highest_level: bool
    m_center_extra: Any = 0  # e.g., equipment at center (kg)
    m_corner_extra: Any = 0  # e.g., equipment at the corner (kg)

    # Outputs
    K_level: Any = None
    M_level: Any = None

    def _masses_corner(self) -> Any:
        """Returns total corner mass per corner."""
        m_corner_b_B = outils.lumped_mass_from_beam(self.rho, self.A_b_B, self.B)
        m_corner_b_H = outils.lumped_mass_from_beam(self.rho, self.A_b_H, self.H)
        m_half_col = outils.lumped_mass_from_beam(
            self.rho, self.Ac, self.Lc)
        m_extra = self.m_corner_extra
        if self.is_highest_level:
            m_corner = m_corner_b_B + m_corner_b_H + m_half_col + m_extra
        else:
            m_corner = m_corner_b_B + m_corner_b_H + 2 * m_half_col + m_extra
        return m_corner

    def build_stiffness(self):
        """Compute K_level (3x3) for DOFs [Ux, Uy, Thetaz]."""
        self.K_level = outils.stiffness_matrix_level_beam_in_flexion(self.E, self.Ic_y, self.Ic_x, self.Lc, self.Ib_y_B,
                                                                     self.Ib_y_H, self.kx1, self.kx2, self.ky1, self.ky2,
                                                                     self.B, self.H)
        return self.K_level

    def build_mass(self):
        """Compute M_level (3x3) for DOFs [Ux, Uy, Thetaz], origin at geometric center."""
        m_corner = self._masses_corner()
        m_center = self.m_center_extra
        self.M_level = outils.mass_matrix_level(self.rho, self.t, self.B, self.H,
                                                m_corner=m_corner, m_center=m_center)
        return self.M_level

    def level_diagonals(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Return (kx, ky, kt, mx, my, mt) for assembling the interstory global K and M
        (between this level and the one below)."""
        # If K_level is already computed and numeric/symbolic, extract diagonals; otherwise, recompute quickly
        if self.K_level is None:
            self.build_stiffness()
        if self.M_level is None:
            self.build_mass()
        if isinstance(self.K_level, sym.MatrixBase) or isinstance(self.K_level, sym.BlockDiagMatrix):
            output_K = [self.K_level[0, 0],
                        self.K_level[1, 1], self.K_level[2, 2]]
            output_M = [self.M_level[0, 0],
                        self.M_level[1, 1], self.M_level[2, 2]]
        else:
            K = np.array(self.K_level.tolist(), dtype=float)
            M = np.array(self.M_level.tolist(), dtype=float)
            output_K = [K[0, 0], K[1, 1], K[2, 2]]
            output_M = [M[0, 0], M[1, 1], M[2, 2]]
        return output_K + output_M
    
@dataclass
class StoryLevelRigidUnions:
    """
    Class to define a story level with rigid unions between
    beam and columns and rigid diaphragm behavior in its plane
    """
    E: Any
    rho: Any
    B: Any
    H: Any
    t: Any
    Lc: Any
    Ac: Any
    Ic_y: Any
    Ic_x: Any
    A_b_B: Any                 # beam area (m^2) for corner mass from beam along B
    A_b_H: Any                 # beam area (m^2) for corner mass from beam along H
    is_highest_level: bool
    m_center_extra: Any = 0  # e.g., equipment at center (kg)
    m_corner_extra: Any = 0  # e.g., equipment at the corner (kg)

    # Outputs
    K_level: Any = None
    M_level: Any = None

    def _masses_corner(self) -> Any:
        """Returns total corner mass per corner."""
        m_corner_b_B = outils.lumped_mass_from_beam(self.rho, self.A_b_B, self.B)
        m_corner_b_H = outils.lumped_mass_from_beam(self.rho, self.A_b_H, self.H)
        m_half_col = outils.lumped_mass_from_beam(
            self.rho, self.Ac, self.Lc)
        m_extra = self.m_corner_extra
        if self.is_highest_level:
            m_corner = m_corner_b_B + m_corner_b_H + m_half_col + m_extra
        else:
            m_corner = m_corner_b_B + m_corner_b_H + 2 * m_half_col + m_extra
        return m_corner

    def build_stiffness(self):
        """Compute K_level (3x3) for DOFs [Ux, Uy, Thetaz]."""
        self.K_level = outils.stiffness_matrix_level_rigid_frame(self.E, self.Ic_x, self.Ic_y, self.Lc, self.B, self.H)
        return self.K_level

    def build_mass(self):
        """Compute M_level (3x3) for DOFs [Ux, Uy, Thetaz], origin at geometric center."""
        m_corner = self._masses_corner()
        m_center = self.m_center_extra
        self.M_level = outils.mass_matrix_level(self.rho, self.t, self.B, self.H,
                                                m_corner=m_corner, m_center=m_center)
        return self.M_level

    def level_diagonals(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Return (kx, ky, kt, mx, my, mt) for assembling the interstory global K and M
        (between this level and the one below)."""
        # If K_level is already computed and numeric/symbolic, extract diagonals; otherwise, recompute quickly
        if self.K_level is None:
            self.build_stiffness()
        if self.M_level is None:
            self.build_mass()
        if isinstance(self.K_level, sym.MatrixBase) or isinstance(self.K_level, sym.BlockDiagMatrix):
            output_K = [self.K_level[0, 0],
                        self.K_level[1, 1], self.K_level[2, 2]]
            output_M = [self.M_level[0, 0],
                        self.M_level[1, 1], self.M_level[2, 2]]
        else:
            K = np.array(self.K_level.tolist(), dtype=float)
            M = np.array(self.M_level.tolist(), dtype=float)
            output_K = [K[0, 0], K[1, 1], K[2, 2]]
            output_M = [M[0, 0], M[1, 1], M[2, 2]]
        return output_K + output_M


@dataclass
class StoryLevelRigidX:
    """
    Class to define a story level with rigid unions between
    beam and columns and rigid diaphragm behavior in its plane
    """
    E: Any
    rho: Any
    B: Any
    H: Any
    t: Any
    Lc: Any
    Ac: Any
    Ic_y: Any
    Ic_x: Any
    Ib: Any
    A_b_B: Any                 # beam area (m^2) for corner mass from beam along B
    A_b_H: Any                 # beam area (m^2) for corner mass from beam along H
    is_highest_level: bool
    m_center_extra: Any = 0  # e.g., equipment at center (kg)
    m_corner_extra: Any = 0  # e.g., equipment at the corner (kg)

    # Outputs
    K_level: Any = None
    M_level: Any = None

    def _masses_corner(self) -> Any:
        """Returns total corner mass per corner."""
        m_corner_b_B = outils.lumped_mass_from_beam(self.rho, self.A_b_B, self.B)
        m_corner_b_H = outils.lumped_mass_from_beam(self.rho, self.A_b_H, self.H)
        m_half_col = outils.lumped_mass_from_beam(
            self.rho, self.Ac, self.Lc)
        m_extra = self.m_corner_extra
        if self.is_highest_level:
            m_corner = m_corner_b_B + m_corner_b_H + m_half_col + m_extra
        else:
            m_corner = m_corner_b_B + m_corner_b_H + 2 * m_half_col + m_extra
        return m_corner

    def build_stiffness(self):
        """Compute K_level (3x3) for DOFs [Ux, Uy, Thetaz]."""
        self.K_level = outils.stiffness_matrix_level_rigid_X_frame(self.E, self.Ic_x, self.Ic_y, self.Lc, self.Ib, self.B, self.H)
        return self.K_level

    def build_mass(self):
        """Compute M_level (3x3) for DOFs [Ux, Uy, Thetaz], origin at geometric center."""
        m_corner = self._masses_corner()
        m_center = self.m_center_extra
        self.M_level = outils.mass_matrix_level(self.rho, self.t, self.B, self.H,
                                                m_corner=m_corner, m_center=m_center)
        return self.M_level

    def level_diagonals(self) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """Return (kx, ky, kt, mx, my, mt) for assembling the interstory global K and M
        (between this level and the one below)."""
        # If K_level is already computed and numeric/symbolic, extract diagonals; otherwise, recompute quickly
        if self.K_level is None:
            self.build_stiffness()
        if self.M_level is None:
            self.build_mass()
        if isinstance(self.K_level, sym.MatrixBase) or isinstance(self.K_level, sym.BlockDiagMatrix):
            output_K = [self.K_level[0, 0],
                        self.K_level[1, 1], self.K_level[2, 2]]
            output_M = [self.M_level[0, 0],
                        self.M_level[1, 1], self.M_level[2, 2]]
        else:
            K = np.array(self.K_level.tolist(), dtype=float)
            M = np.array(self.M_level.tolist(), dtype=float)
            output_K = [K[0, 0], K[1, 1], K[2, 2]]
            output_M = [M[0, 0], M[1, 1], M[2, 2]]
        return output_K + output_M
