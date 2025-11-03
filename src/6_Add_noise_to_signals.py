
import json
import matplotlib.pyplot as plt
import numpy as np
import os

"""
File Duties:

Add noise to the time series data of accelerations
"""
sf = 4  # we scale by 4 as there were forces applied only to one dof (in lumped model they were in 4 dofs)
applied_force_in_KN = True  # in SAP2000, force was in kN
filename = 'DA_uncalibrated_v7_updated_with_TH_TimeHistory_accelerations_2.txt'
filepath = r"C:\Users\User\Documents\DOCTORADO\Multifidelity_Steelframe\PythonData\output"
filename_json = filename.replace('.txt', '_details.json')

af = 1  # same as in load functions generation
SNR = 10  # same as in lumped model
rng = np.random.RandomState(12345)  # Set the seed

accelerations = np.loadtxt(os.path.join(filepath, filename))
if applied_force_in_KN:
    accelerations /= 1000  # Convert to N; x 4 as forces where on
if sf:
    accelerations *= sf

with open(os.path.join(filepath, filename_json), 'r') as f:
    info = json.load(f)
fs = info['fs']
# t = np.arange(0, len(accelerations)) / fs
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(t, accelerations, label='Original Signal')

# Add noise to signals
ar = af / (10 ** (SNR / 20))  # Noise amplitude

acc_noise = accelerations + ar * rng.standard_normal(size=accelerations.shape)  # adding noise

outname = filename.replace('.txt', '_with_noise.txt')
np.savetxt(os.path.join(filepath, outname), acc_noise)