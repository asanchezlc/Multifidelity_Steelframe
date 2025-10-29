
import sympy as sym

"""
Symbolic derivation of the horizontal force at node 2 needed to enforce u2 = 1
for a vertical Euler-Bernoulli beam with rotational springs at both ends.

Remark: used for frame with rigid diaprhagm and beams infinitely rigid in flexion.
"""

# --- Symbols ---
E, I, L = sym.symbols('E I L', positive=True, finite=True)
k1, k2 = sym.symbols('k1 k2', nonnegative=True,
                    finite=True)  # rotational springs
EI = sym.Symbol('EI', positive=True)

# --- Local beam stiffness matrix (ordering: [u1, th1, u2, th2]) ---
# Standard Euler-Bernoulli prismatic beam element:
kb = (E*I/L**3) * sym.Matrix([
    [12,   6*L,  -12,   6*L],
    [6*L, 4*L**2, -6*L, 2*L**2],
    [-12,  -6*L,   12,  -6*L],
    [6*L, 2*L**2, -6*L, 4*L**2]
])

# --- Add rotational springs at both ends ---
# Rotational DOFs are th1 (index 1) and th2 (index 3)
K = kb.copy()
K[1, 1] += k1
K[3, 3] += k2

# --- Impose boundary condition u1 = 0 (constrained) ---
# Keep free DOFs q_f = [th1, u2, th2] in that order
free_idx = [1, 2, 3]  # (th1, u2, th2)
Kf = K.extract(free_idx, free_idx)

# --- Static condensation to eliminate rotations (th1, th2) ---
# Partition Kf as:
# [ K_rr   K_ru ]
# [ K_ur   K_uu ], with r = [th1, th2], u = [u2]
# We will keep u = [u2] and condense out r
r_idx = [0, 2]  # indices of th1 and th2 inside Kf
u_idx = [1]     # index of u2 inside Kf

K_rr = Kf.extract(r_idx, r_idx)  # 2x2
K_ru = Kf.extract(r_idx, u_idx)  # 2x1
K_ur = Kf.extract(u_idx, r_idx)  # 1x2
K_uu = Kf.extract(u_idx, u_idx)  # 1x1

# Schur complement: Keff = K_uu - K_ur * inv(K_rr) * K_ru
Keff = (K_uu - K_ur * K_rr.inv() * K_ru)[0, 0]
Keff_simplified = sym.simplify(sym.factor(Keff))

# The required horizontal force for u2 = 1 is F = Keff
F_expr = sym.simplify(Keff_simplified)

# --- Adimensional form with alpha_i = k_i*L/(E*I) ---
alpha1, alpha2 = sym.symbols('alpha1 alpha2', nonnegative=True)
subs_dimless = {
    k1: alpha1 * (E*I)/L,
    k2: alpha2 * (E*I)/L
}
F_dimless = sym.simplify(sym.factor(sym.together(
    F_expr.subs(subs_dimless) / ((E*I)/L**3))))
