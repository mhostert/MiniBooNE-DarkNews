import numpy as np
import pandas as pd
import argparse

from DarkNews.GenLauncher import GenLauncher
from fastbnb import fit_functions as ff
from fastbnb import grid_fit as gf
from fastbnb import analysis
from fastbnb import decayer

v_cut = 1e-3
n = 400


def fit_point_3p1(
    i_line,
    path,
    cut="circ1",
    print_spectra=True,
    experiment="miniboone_fhc",
    **kwargs,
):
    m4s, mzprimes = gf.get_line_from_input(path, i_line)
    if not kwargs:
        kwargs = gf.get_kwargs_from_input(path)

    # def fit_point_3p1(m4s, mz):
    umu4 = np.geomspace(1e-5, 5e-1, n)
    vmu4 = kwargs["gD"] * kwargs["UD4"] * kwargs["UD4"] * umu4 / np.sqrt(1 - umu4**2)
    v4i = kwargs["gD"] * kwargs["UD4"] * umu4

    # 1. PERFORM AND READ THE DATA FROM THE SIMULATION
    # Run the generation for MiniBooNE
    df = GenLauncher(
        mzprime=mzprimes,
        m4=m4s,
        experiment=experiment,
        **kwargs,
    ).run()
    dl = df.attrs["N4_ctau0"]

    # Run the generation for MiniBooNE Dirt
    df2 = GenLauncher(
        mzprime=mzprimes,
        m4=m4s,
        experiment=experiment+"_dirt",
        **kwargs,
    ).run()

    df = pd.concat([df, df2])

    # 3. CYCLE OVER DIFFERENT VALUES OF U_MU4
    v4i_def = kwargs["gD"] * kwargs["UD4"] * kwargs["Umu4"]
    vmu4_def = (
        kwargs["gD"]
        * kwargs["UD4"]
        * kwargs["UD4"]
        * kwargs["Umu4"]
        / np.sqrt(1 - kwargs["Umu4"] ** 2)
    )

    for j in range(n):
        decay_l = dl / (v4i[j] / v4i_def) ** 2

        if experiment == "miniboone_fhc":
            # analyze the dataframe
            df_temp = decayer.decay_selection(
                df, decay_l, "miniboone", weights="w_event_rate"
            )
            df_temp = analysis.reco_nueCCQElike_Enu(df_temp, cut=cut)
            df_temp = df_temp[df_temp.reco_w > 0]
            df_temp["w_event_rate"] *= (vmu4[j] / vmu4_def) ** 2
            df_temp["reco_w"] *= (vmu4[j] / vmu4_def) ** 2

            sum_w_post_smearing = np.abs(np.sum(df_temp["reco_w"]))

            # computing chi2
            hist = np.histogram(
                df_temp["reco_Enu"],
                weights=df_temp["reco_w"],
                bins=ff.bin_enu_def,
                density=False,
            )[0]
            n_events = sum_w_post_smearing
            chi2 = ff.chi2_MiniBooNE_2020(hist, n_events)

            ##############
            # 6. MAIN OUTPUT -- TOTAL RATE AND CHI2

            # Chi2 and total event rate
            output = f"{mzprimes:.6e} {m4s:.6e} {sum_w_post_smearing:.6e} {vmu4[j]:.6e} {v4i[j]:.6e} {kwargs['epsilon']:.6e} {umu4[j]:.6e} {chi2:.6e} {decay_l:.6e} {n_events:.6e}\n"
            gf.save_to_locked_file(path + f"chi2_{i_line}.dat", output)

            ##############
            # 7. OTHER OUTPUT -- COMPUTE SPECTRA FROM THE BEST FIT
            if print_spectra:
                # Enu
                line = " ".join(
                    [f"{i:.6e}" for i in ff.get_MiniBooNE_Enu_bin_vals(df_temp)]
                )
                gf.save_to_locked_file(
                    filename=path + f"enu_spectrum_{i_line}.dat",
                    output=f"{i_line} " + line + "\n",
                )
                # Evis
                line = " ".join(
                    [f"{i:.6e}" for i in ff.get_MiniBooNE_Evis_bin_vals(df_temp)]
                )
                gf.save_to_locked_file(
                    filename=path + f"evis_spectrum_{i_line}.dat",
                    output=f"{i_line} " + line + "\n",
                )
                # Cos(Theta)
                line = " ".join(
                    [f"{i:.6e}" for i in ff.get_MiniBooNE_Costheta_bin_vals(df_temp)]
                )
                gf.save_to_locked_file(
                    filename=path + f"costheta_spectrum_{i_line}.dat",
                    output=f"{i_line} " + line + "\n",
                )

                # Now for the invariant mass we perform the pi0 selection
                df_pi0 = analysis.reco_pi0like_invmass(df_temp, cut=cut)
                line = " ".join(
                    [f"{i:.6e}" for i in ff.get_MiniBooNE_Invmass_bin_vals(df_pi0)]
                )
                gf.save_to_locked_file(
                    filename=path + f"invmass_spectrum_{i_line}.dat",
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
    "--experiment", type=str, default="miniboone_fhc", help="what experiment to use"
)
args = parser.parse_args()

######## Running fit function ########
fit_point_3p1(
    args.i_line,
    args.path,
    cut=args.cut,
    print_spectra=args.print_spectra,
    experiment=args.experiment,
)
