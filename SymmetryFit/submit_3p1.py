import numpy as np
from fastbnb import fit_functions as ff

import slurm_tools

##########################
"""
    Some global variables for the grid fitting procedure
"""
CUT = "circ1"  # "circ0" or "circ1" or "diag1" or "invmass"
PRINT_SPECTRA = 1  # 0 -- do not print spectrum file or 1 -- print spectrum files (Enu, Evis, Cotheta, and invmass spectra.)
POINTS = 40
NEVAL = int(1e5)


def sub3p1(EPSILON_CASE, RATIO_CASE=0):
    RUN_PATH = f"results/3p1_general_{EPSILON_CASE}/"  # Path for this type of scan
    unique_path = slurm_tools.get_unique_path(RUN_PATH)

    ####################################################
    """
        Now some parameters for the event generation. 
        This is saved to file and then loaded by the fitting script.
    """
    kwargs = {
        "Umu4": np.sqrt(1.0e-12),
        "UD4": 1.0 / np.sqrt(2.0),
        "gD": 2.0,
        "epsilon": EPSILON_CASE,
        "neval": NEVAL,
        "HNLtype": "dirac",
        "noHF": True,
        "pandas": False,
        "parquet": False,
        "loglevel": "ERROR",
        "sparse": 2,
        "ratio_tau_to_mu": RATIO_CASE,
    }

    cols = [
        "mzprime",
        "m4",
        "sum_w_post_smearing",
        "v_mu4",
        "v_4i",
        "epsilon",
        "u_mu4",
        "chi2",
        "decay_length",
        "MB_N_events",
        "eff_geometry",
        "eff_selection",
        "eff_total",
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
    mzvals = ff.round_sig(np.geomspace(0.02, 2.0, points), 4)
    m4vals = ff.round_sig(np.geomspace(0.01, 1.0, points), 4)
    XGRID, YGRID = np.meshgrid(m4vals, mzvals)

    ####################################################

    slurm_tools.slurm_submit(
        fit_script="fit_3p1_general.py",
        path=unique_path,
        XGRID=XGRID,
        YGRID=YGRID,
        input_kwargs=kwargs,
        jobname="dn_3p1",
        queue="defq",
        timeout="1-00:00:00",
        optional_args=f"--cut {CUT} --print_spectra {PRINT_SPECTRA}",
    )


def sub3p1_coupling(EPSILON_CASE):
    RUN_PATH = f"results/3p1_coupling_{EPSILON_CASE}/"  # Path for this type of scan
    unique_path = slurm_tools.get_unique_path(RUN_PATH)

    ####################################################
    """
        Now some parameters for the event generation. 
        This is saved to file and then loaded by the fitting script.
    """
    kwargs = {
        "Umu4": np.sqrt(1.0e-12),
        "UD4": 1.0 / np.sqrt(2.0),
        "gD": 2.0,
        "epsilon": EPSILON_CASE,
        "neval": NEVAL,
        "HNLtype": "dirac",
        "noHF": True,
        "pandas": False,
        "parquet": False,
        "loglevel": "ERROR",
        "sparse": 2,
    }

    cols = [
        "mzprime",
        "m4",
        "v_mu4",
        "v_4i",
        "epsilon",
        "u_mu4",
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
    m4vals = ff.round_sig(np.geomspace(0.01, 1.0, points), 4)
    XGRID, YGRID = np.meshgrid(m4vals, mzvals)

    ####################################################

    slurm_tools.slurm_submit(
        fit_script="fit_3p1_coupling.py",
        path=unique_path,
        XGRID=XGRID,
        YGRID=YGRID,
        input_kwargs=kwargs,
        jobname="dn_3p1",
        queue="defq",
        timeout="1-00:00:00",
        optional_args=f"--cut {CUT} --print_spectra {PRINT_SPECTRA}",
    )


sub3p1(EPSILON_CASE=1e-2, RATIO_CASE=10)
# sub3p1(EPSILON_CASE=1e-2)
# sub3p1(EPSILON_CASE=0.0008)

# sub3p1_coupling(EPSILON_CASE=1e-2)
# sub3p1_coupling(EPSILON_CASE=8e-4)
