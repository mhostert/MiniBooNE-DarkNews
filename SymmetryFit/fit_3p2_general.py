import numpy as np
import pandas as pd
import scipy
import sys
import argparse

from filelock import FileLock

from DarkNews.GenLauncher import GenLauncher

from fastbnb import fit_functions as ff
from fastbnb import grid_fit as gf
from fastbnb import analysis
from fastbnb import decayer


def fit_point_3p2(
    i_line,
    path,
    cut="circ1",
    print_spectra=True,
    scan_type="delta",
    **kwargs,
):
    # Prob dont need to lock it, but just in case:
    path_input = path + "input_scan.dat"
    lock = FileLock(path_input + ".lock")
    try:
        with lock:
            with open(path_input, "r") as f:
                lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File {path_input} not found. Input file should have been created in submission script."
        )

    try:
        m5s, y = [np.float32(x) for x in lines[i_line].split()]
    except IndexError:
        raise IndexError(f"No input for line {i_line} in {path_input}.")

    # Now input kwargs for generation inputs
    if not kwargs:
        try:
            kwargs = np.load(path + "input_kwargs.npy", allow_pickle=True).item()
        except FileNotFoundError:
            print(
                "No input parameters passed or file found. Please provide kwargs with generation inputs."
            )
            return 0

    if scan_type == "delta":
        mzprimes = kwargs["mzprime"]
        deltas = y
        m4s = m5s / (deltas + 1)
    elif scan_type == "mzprime":
        m4s = kwargs.pop("m4")
        deltas = m5s / m4s - 1
        mzprimes = y
        kwargs["mzprime"] = mzprimes  # still need to pass mzprime

    if m5s - m4s < 2.05 * 511e-6:
        print("m5 - m4 < 2*m_e. Skipping this point.")
        return 0

    # 1. PERFORM AND READ THE DATA FROM THE SIMULATION
    # Run the generation for MiniBooNE
    df = GenLauncher(m5=m5s, m4=m4s, experiment="miniboone_fhc_dirt", **kwargs).run()
    decay_l = df.attrs["N5_ctau0"]

    # Run the generation for MiniBooNE Dirt
    df2 = GenLauncher(m5=m5s, m4=m4s, experiment="miniboone_fhc_dirt", **kwargs).run()

    df_tot = pd.concat([df, df2])

    ###########################
    # 3. DEFINE FUNCTIONS WITH/WITHOUT DIRT TO FIT
    df_tot = decayer.decay_selection(
        df_tot, decay_l, "miniboone", weights="w_event_rate"
    )
    df_tot = analysis.reco_nueCCQElike_Enu(df_tot, cut=cut, clean_df=True)
    sum_w_post_smearing = df_tot["reco_w"].sum()

    v54 = kwargs["gD"] * kwargs["UD5"] * kwargs["UD4"]
    vmu5_def = (
        kwargs["gD"]
        * kwargs["UD5"]
        * (kwargs["Umu4"] * kwargs["UD4"] + kwargs["Umu5"] * kwargs["UD5"])
        / np.sqrt(1 - kwargs["Umu4"] ** 2 - kwargs["Umu5"] ** 2)
    )

    umui_max = 1e-3
    v_cut = (
        kwargs["gD"]
        * kwargs["UD5"]
        * (umui_max * kwargs["UD4"] + umui_max * kwargs["UD5"])
        / np.sqrt(1 - umui_max**2 - umui_max**2)
    )

    # 4. DO THE FITTING
    if sum_w_post_smearing != 0:
        guess0 = np.sqrt(np.sqrt(560.0 / sum_w_post_smearing) * vmu5_def / v_cut)
    else:
        guess0 = 1
    guess = np.min([guess0, 1])
    theta_guess = np.arccos(guess)
    init_guess = np.array([theta_guess])

    chi2 = lambda theta: ff.chi2_MiniBooNE_2020_3p2(
        df_tot,
        rate_rescale=((v_cut * np.cos(theta) ** 2) / vmu5_def) ** 2,
    )

    res = scipy.optimize.minimize(chi2, init_guess, bounds=[(0, 2 * np.pi)])
    vmu5_bf = v_cut * np.cos(res.x[0]) ** 2

    # 5. COMPUTE NUMBERS FROM THE BEST FIT
    rescale = (vmu5_bf / vmu5_def) ** 2
    df_tot["w_event_rate"] *= rescale
    df_tot["reco_w"] *= rescale

    ##############
    # 6. MAIN OUTPUT -- TOTAL RATE AND CHI2

    # Chi2 and total event rate
    output = f"{mzprimes:.6e} {m5s:.6e} {m4s:.6e} {deltas:.6e} {sum_w_post_smearing:.6e} {vmu5_bf:.6e} {v54:.6e} {kwargs['epsilon']:.6e} {vmu5_bf / 2.:.6e} {res.fun:.6e} {decay_l:.6e} {(vmu5_bf / vmu5_def)**2 * sum_w_post_smearing:.6e}\n"
    gf.save_to_locked_file(path + "chi2.dat", output)

    ##############
    # 7. OTHER OUTPUT -- COMPUTE SPECTRA FROM THE BEST FIT
    if print_spectra:
        # Enu
        line = " ".join([f"{i:.6e}" for i in ff.get_MiniBooNE_Enu_bin_vals(df_tot)])
        gf.save_to_locked_file(
            filename=path + "enu_spectrum.dat",
            output=f"{i_line} " + line + "\n",
        )
        # Evis
        line = " ".join([f"{i:.6e}" for i in ff.get_MiniBooNE_Evis_bin_vals(df_tot)])
        gf.save_to_locked_file(
            filename=path + "evis_spectrum.dat",
            output=f"{i_line} " + line + "\n",
        )
        # Cos(Theta)
        line = " ".join(
            [f"{i:.6e}" for i in ff.get_MiniBooNE_Costheta_bin_vals(df_tot)]
        )
        gf.save_to_locked_file(
            filename=path + "costheta_spectrum.dat",
            output=f"{i_line} " + line + "\n",
        )

        # Now for the invariant mass we perform the pi0 selection
        df_pi0 = analysis.reco_pi0like_invmass(df_tot, cut=cut)
        line = " ".join([f"{i:.6e}" for i in ff.get_MiniBooNE_Invmass_bin_vals(df_pi0)])
        gf.save_to_locked_file(
            filename=path + "invmass_spectrum.dat",
            output=f"{i_line} " + line + "\n",
        )

    return 0


######## Input from command line ########
parser = argparse.ArgumentParser()
parser.add_argument("--i_line", type=int, help="line of the input file to read")
parser.add_argument("--path", type=str, help="path to the simulation results")
parser.add_argument(
    "--cut", type=str, default="circ1", help="cut analysis to be used at MiniBooNE"
)
parser.add_argument(
    "--print_spectra", type=bool, default=True, help="path to the simulation results"
)
parser.add_argument(
    "--scan_type",
    type=str,
    default="delta",
    help="what scan to use: delta or mzprime",
)
args = parser.parse_args()

######## Running fit function ########
fit_point_3p2(
    args.i_line,
    args.path,
    cut=args.cut,
    scan_type=args.scan_type,
    print_spectra=args.print_spectra,
)
