import numpy as np
from fastbnb import fit_functions as ff

import slurm_tools

CUT = "circ1"  # "circ0" or "circ1" or "diag1" or "invmass"
PRINT_SPECTRA = 1  # 0 -- do not print spectrum file or 1 -- print spectrum files (Enu, Evis, Cotheta, and invmass spectra.)
NEVAL = int(1e5)
POINTS = 40


def sub3p2(SCAN_TYPE, FIXED_MASS, EPSILON_CASE=1e-2):
    kwargs = {
        "Umu4": np.sqrt(1.0e-12),
        "Umu5": np.sqrt(1.0e-12),
        "UD4": 1.0 / np.sqrt(2.0),
        "UD5": 1.0 / np.sqrt(2.0),
        "gD": 2.0,
        "epsilon": EPSILON_CASE,
        "neval": NEVAL,
        "noHF": True,
        "HNLtype": "dirac",
        "pandas": False,
        "parquet": False,
        "loglevel": "ERROR",
        "sparse": 2,
    }

    if SCAN_TYPE == "delta":
        kwargs["mzprime"] = FIXED_MASS
        RUN_PATH = f"results/3p2_general_{SCAN_TYPE}_mz_{FIXED_MASS}_eps_{EPSILON_CASE:.2g}/"  # Path for this type of scan
        unique_path = slurm_tools.get_unique_path(RUN_PATH)
    elif SCAN_TYPE == "mzprime":
        kwargs["m4"] = FIXED_MASS
        RUN_PATH = f"results/3p2_general_{SCAN_TYPE}_m4_{FIXED_MASS}_eps_{EPSILON_CASE:.2g}/"  # Path for this type of scan
        unique_path = slurm_tools.get_unique_path(RUN_PATH)

    cols = [
        "mzprime",
        "m5",
        "m4",
        "delta",
        "sum_w_post_smearing",
        "v_mu5",
        "v_54",
        "epsilon",
        "u_mu5/u_mu4",
        "chi2",
        "decay_length",
        "N_events",
        "eff_geometry",
        "eff_selection",
        "eff_final",
    ]
    with open(unique_path + "chi2.dat", "w") as f:
        # Header
        f.write("# " + " ".join(cols) + "\n")

    ####################################################
    """
        We create the grid and then print all tuples of m4,mzprime to a file.
        This file will be read by the fitting script based on a row index.
    """

    points = POINTS  # NOTE: JobArray max size is 1001.
    if SCAN_TYPE == "delta":
        yvals = ff.round_sig(np.geomspace(0.05, 10, points), 4)
        M5_min = 0.01
    elif SCAN_TYPE == "mzprime":
        yvals = ff.round_sig(np.geomspace(0.02, 2.0, points), 4)
        M5_min = kwargs["m4"] + 0.005

    m5vals = ff.round_sig(np.geomspace(M5_min, 2.0, points), 4)
    XGRID, YGRID = np.meshgrid(m5vals, yvals)

    slurm_tools.slurm_submit(
        fit_script="fit_3p2_general.py",
        path=unique_path,
        XGRID=XGRID,
        YGRID=YGRID,
        input_kwargs=kwargs,
        jobname="dn_3p2",
        queue="defq",
        timeout="1-00:00:00",
        optional_args=f"--cut {CUT} --print_spectra {PRINT_SPECTRA} --scan_type {SCAN_TYPE}",
    )


def sub3p2_coupling(EPSILON_CASE, DELTA_CASE):
    RUN_PATH = f"results/3p2_coupling_{EPSILON_CASE}_delta_{DELTA_CASE:.2f}/"  # Path for this type of scan
    unique_path = slurm_tools.get_unique_path(RUN_PATH)

    ####################################################
    """
        Now some parameters for the event generation. 
        This is saved to file and then loaded by the fitting script.
    """
    kwargs = {
        "Umu4": np.sqrt(1.0e-12),
        "Umu5": np.sqrt(1.0e-12),
        "UD4": 1.0 / np.sqrt(2.0),
        "UD5": 1.0 / np.sqrt(2.0),
        "gD": 2.0,
        "epsilon": EPSILON_CASE,
        "neval": NEVAL,
        "noHF": True,
        "HNLtype": "dirac",
        "pandas": False,
        "parquet": False,
        "loglevel": "ERROR",
        "sparse": 2,
    }

    # We are adding this delta to kwargs, but it will be pop'ed by the fit function before being passed to DarkNews
    kwargs["delta"] = DELTA_CASE

    cols = [
        "mzprime",
        "m5",
        "m4",
        "delta",
        "v_mu5",
        "epsilon",
        "decay_length",
        "chi2_mb_fhc",
        "chi2_mb_rhc",
        "chi2_mb_comb",
        "mb_fhc_n_events",
        "mb_rhc_n_events",
        "sbnd_n_events",
        "microb_n_events",
        "icarus_n_events",
    ]

    with open(unique_path + "chi2.dat", "w") as f:
        # Header
        f.write("# " + " ".join(cols) + "\n")

    ####################################################
    """
        We create the grid and then print all tuples of m4,mzprime to a file.
        This file will be read by the fitting script based on a row index.
    """
    points = POINTS  # NOTE: JobArray max size is 1001.
    mzvals = np.array(
        [0.03, 0.06, 0.1, 0.2, 0.5, 0.8, 1.25]
    )  # ---> This is fixed -- run for multiple plots
    m5vals = ff.round_sig(np.geomspace(0.01, 1.0, points), 4)
    XGRID, YGRID = np.meshgrid(m5vals, mzvals)

    ####################################################

    slurm_tools.slurm_submit(
        fit_script="fit_3p2_coupling.py",
        path=unique_path,
        XGRID=XGRID,
        YGRID=YGRID,
        input_kwargs=kwargs,
        jobname="dn_3p2",
        queue="defq",
        timeout="1-00:00:00",
        optional_args=f"--cut {CUT} --print_spectra {PRINT_SPECTRA}",
    )


# # sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.015, EPSILON_CASE=1e-4)
# # sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.150, EPSILON_CASE=1e-4)
# # sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.250, EPSILON_CASE=1e-4)

#sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.015, EPSILON_CASE=8e-4)
#sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.150, EPSILON_CASE=8e-4)
#sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.250, EPSILON_CASE=8e-4)

#sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.015, EPSILON_CASE=1e-2)
#sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.150, EPSILON_CASE=1e-2)
#sub3p2(SCAN_TYPE="mzprime", FIXED_MASS=0.250, EPSILON_CASE=1e-2)

#sub3p2(SCAN_TYPE="delta", FIXED_MASS=1.25, EPSILON_CASE=1e-2)
#sub3p2(SCAN_TYPE="delta", FIXED_MASS=1.25, EPSILON_CASE=8e-4)
# sub3p2(SCAN_TYPE="delta", FIXED_MASS=1.25, EPSILON_CASE=1e-4)


### Couplings
# mzprimearray = [0.2, 0.5, 1.25]
# deltaarray = [0.5, 3.0]

#sub3p2_coupling(EPSILON_CASE=8e-4, DELTA_CASE=0.1)
#sub3p2_coupling(EPSILON_CASE=8e-4, DELTA_CASE=0.5)
#sub3p2_coupling(EPSILON_CASE=8e-4, DELTA_CASE=3)

sub3p2_coupling(EPSILON_CASE=1e-2, DELTA_CASE=2)
#sub3p2_coupling(EPSILON_CASE=1e-2, DELTA_CASE=7.5)
#sub3p2_coupling(EPSILON_CASE=1e-2, DELTA_CASE=10)
