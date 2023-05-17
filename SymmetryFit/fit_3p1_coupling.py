import numpy as np
import pandas as pd
import argparse

from DarkNews.GenLauncher import GenLauncher
from fastbnb import fit_functions as ff
from fastbnb import grid_fit as gf
from fastbnb import analysis
from fastbnb import decayer

n = 500
SBN_experiments = ["microboone", "icarus", "sbnd"]
SBN_experiments_dirt = ["microboone_dirt", "icarus_dirt", "sbnd_dirt_cone"]


def fit_point_3p1(
    i_line,
    path,
    cut="circ1",
    print_spectra=True,
    **kwargs,
):
    # 0. INPUTS
    if not kwargs:
        kwargs = gf.get_kwargs_from_input(path)
    m4s, mzprimes = gf.get_line_from_input(path, i_line)
    kwargs["m4"] = m4s
    kwargs["mzprime"] = mzprimes

    umu4 = np.geomspace(1e-5, 0.5, n)
    vmu4 = kwargs["gD"] * kwargs["UD4"] * kwargs["UD4"] * umu4 / np.sqrt(1 - umu4**2)
    v4i = kwargs["gD"] * kwargs["UD4"] * umu4

    # 1. SIMULATIONS
    MB_fhc_df = GenLauncher(
        experiment="miniboone_fhc",
        **kwargs,
    ).run()
    MB_fhc_df_dirt = GenLauncher(
        experiment="miniboone_fhc_dirt",
        **kwargs,
    ).run()
    # 2. LIFETIME OF HNL
    dl = MB_fhc_df.attrs["N4_ctau0"]
    MB_fhc_df = pd.concat([MB_fhc_df, MB_fhc_df_dirt])

    MB_rhc_df = GenLauncher(
        experiment="miniboone_rhc",
        **kwargs,
    ).run()
    MB_rhc_df_dirt = GenLauncher(
        experiment="miniboone_rhc_dirt",
        **kwargs,
    ).run()
    MB_rhc_df = pd.concat([MB_rhc_df, MB_rhc_df_dirt])

    # sbn experiments
    sbn_dfs = []
    for exp, exp_dirt in zip(SBN_experiments, SBN_experiments_dirt):
        # simulation for SBN and SBN dirt
        sbn_df = GenLauncher(
            experiment=exp,
            **kwargs,
        ).run()

        # Run the generation for MiniBooNE Dirt
        sbn_df_dirt = GenLauncher(
            experiment=exp_dirt,
            **kwargs,
        ).run()

        sbn_dfs.append(pd.concat([sbn_df, sbn_df_dirt]))

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

        ##############
        # 6. MAIN OUTPUT -- TOTAL RATE AND CHI2
        chi2s, events = ff.miniboone_rates_and_chi2(
            MB_fhc_df,
            MB_rhc_df,
            rescale=(vmu4[j] / vmu4_def) ** 2,
            cut=cut,
            decay_l=decay_l,
        )

        # output = f"{mzprimes:.6e} {m4s:.6e} {sum_w_post_smearing:.6e} {vmu4[j]:.6e} {v4i[j]:.6e} {kwargs['epsilon']:.6e} {umu4[j]:.6e} {chi2:.6e} {decay_l:.6e} {MB_n_events:.6e}"
        output = f"{mzprimes:.6e} {m4s:.6e} {vmu4[j]:.6e} {v4i[j]:.6e} {kwargs['epsilon']:.6e} {umu4[j]:.6e} {decay_l:.6e}"

        # Chi2
        output += " " + " ".join([f"{c:.6e}" for c in chi2s])
        # Event rate
        output += " " + " ".join([f"{ev:.6e}" for ev in events])

        for exp, df in zip(SBN_experiments, sbn_dfs):
            # analyze the dataframe
            df_temp = decayer.decay_selection(
                df, decay_l, experiment=exp, weights="w_event_rate"
            )
            n_events = (
                np.abs(np.sum(df_temp["w_event_rate"])) * (vmu4[j] / vmu4_def) ** 2
            )
            output += f" {n_events:.6e}"
        output += "\n"

        gf.save_to_locked_file(path + f"chi2_{i_line}.dat", output)

        # ##############
        # # 7. OTHER OUTPUT -- COMPUTE SPECTRA FROM THE BEST FIT
        # if print_spectra:
        #     # Enu
        #     line = " ".join(
        #         [f"{i:.6e}" for i in ff.get_MiniBooNE_Enu_bin_vals(df_temp)]
        #     )
        #     gf.save_to_locked_file(
        #         filename=path + f"enu_spectrum_{i_line}.dat",
        #         output=f"{i_line} " + line + "\n",
        #     )
        #     # Evis
        #     line = " ".join(
        #         [f"{i:.6e}" for i in ff.get_MiniBooNE_Evis_bin_vals(df_temp)]
        #     )
        #     gf.save_to_locked_file(
        #         filename=path + f"evis_spectrum_{i_line}.dat",
        #         output=f"{i_line} " + line + "\n",
        #     )
        #     # Cos(Theta)
        #     line = " ".join(
        #         [f"{i:.6e}" for i in ff.get_MiniBooNE_Costheta_bin_vals(df_temp)]
        #     )
        #     gf.save_to_locked_file(
        #         filename=path + f"costheta_spectrum_{i_line}.dat",
        #         output=f"{i_line} " + line + "\n",
        #     )

        #     # Now for the invariant mass we perform the pi0 selection
        #     df_pi0 = analysis.reco_pi0like_invmass(df_temp, cut=cut)
        #     line = " ".join(
        #         [f"{i:.6e}" for i in ff.get_MiniBooNE_Invmass_bin_vals(df_pi0)]
        #     )
        #     gf.save_to_locked_file(
        #         filename=path + f"invmass_spectrum_{i_line}.dat",
        #         output=f"{i_line} " + line + "\n",
        #     )

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
args = parser.parse_args()

######## Running fit function ########
fit_point_3p1(
    args.i_line,
    args.path,
    cut=args.cut,
    print_spectra=args.print_spectra,
)
