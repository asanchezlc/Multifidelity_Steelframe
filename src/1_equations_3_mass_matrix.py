

# -*- coding: utf-8 -*-
# Rigid diaphragm (in-plane) 3-DOF story mass matrix:
# DOFs: [Ux, Uy, Thetaz]
# Slab: rho (volumetric), thickness t, plan B x H
# Point masses: m_corner at each of 4 corners, m_center at the center

import sympy as sp
import numpy as np




# --- Example (numeric) ---
if __name__ == "__main__":
    rho = 2500.0     # kg/m^3 (e.g., concrete)
    t   = 0.20       # m
    B   = 6.0        # m
    H   = 8.0        # m
    m_corner = 150.0 # kg at each corner
    m_center = 500.0 # kg at center

    # Mass matrix at geometric center (diagonal expected)
    M = mass_matrix_level(rho, t, B, H, m_corner, m_center, x0=0.0, y0=0.0)
    print("M @ center [Ux, Uy, Thetaz]:")
    print(M)

    # Mass matrix referenced at a shifted point (introduces coupling terms)
    M_shift = mass_matrix_level(rho, t, B, H, m_corner, m_center, x0=1.0, y0=0.5)
    print("\nM @ shifted origin (x0=1.0 m, y0=0.5 m):")
    print(M_shift)
