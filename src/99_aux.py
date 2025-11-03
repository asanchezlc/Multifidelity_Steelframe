
import helpers.outils as outils
import numpy as np
np.random.seed(1)                     # opcional, para reproducibilidad
t = np.arange(0, 901, 1)              # 0, 1, 2, ..., 900
f = np.random.uniform(-5, 5, len(t))  # fuerzas entre -5 y 5
f[-1] = 0 

filename = "random_load.txt"                           # fuerza final en cero
np.savetxt(filename, np.column_stack([t, f]), fmt="%.3f %.2f")

outils.replace_dots_with_commas_in_file('', filename)
