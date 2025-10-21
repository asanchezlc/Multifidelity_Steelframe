

from helpers.classes import StoryLevel

import sympy as sp
import numpy as np

import helpers.outils as outils


# Example
n_floors = 2
E = 210e9
rho = 7849.0476
L_c = 0.25
H = 0.25
B = 0.50
b, h = 2.0 * 1e-3, 30.0 * 1e-3
t = 2e-3

levels = list()
for i in range(1, n_floors + 1):

    ky1, ky2 = 1000.0, 5000.0
    kx1, kx2 = 2000.0, 8000.0
    if i != n_floors:
        A_b = 200e-6
        m_motor = 0
    else:
        A_b = 200e-6
        m_motor = 100/9.80665  # kg

    levels.append(
        StoryLevel(
            E=E, rho=rho,
            B=B, H=H, t=t,
            Lc=L_c,
            kx1=kx1, kx2=kx2,
            ky1=ky1, ky2=ky2,
            b=b, h=h,
            A_b=A_b, L_b=B,
            is_highest_level=(i == n_floors),
            m_center_extra=m_motor
        )
    )


K, M, dofs = outils.assemble_interstory_global_from_levels(levels)

print('h')
