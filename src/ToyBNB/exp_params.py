import numpy as np

# CONSTANTS
THRESHOLD = {'miniboone': 0.03, 'microboone': 0.01}
ANGLE_MAX = {'miniboone': 13., 'microboone': 5.}
ENU_MIN = {'miniboone': 0.14, 'microboone': 0.14}
ENU_MAX = {'miniboone': 1.5, 'microboone': 1.5}
EVIS_MIN = {'miniboone': 0.14, 'microboone': 0.14}
EVIS_MAX = {'miniboone': 3., 'microboone': 1.4}

# Resolutions based on https://iopscience.iop.org/article/10.1088/1742-6596/120/5/052003/pdf
ANGULAR = {'miniboone': 3.*np.pi/180.0, 'microboone': 2.*np.pi/180.0}
STOCHASTIC = {'miniboone': 0.12, 'microboone': 0.12}
NOISE = {'miniboone': 0.01, 'microboone': 0.01}

Q2 = {'miniboone': 1.e10, 'microboone': None}
ANALYSIS_TH = {'miniboone': 0.02, 'microboone': None}