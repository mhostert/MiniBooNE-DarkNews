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


def fit_point_3p2(
    i_line,
    path,
    cut="circ1",
    print_spectra=True,
    **kwargs,
):
    # 0. INPUTS
    if not kwargs:
        kwargs = gf.get_kwargs_from_input(path)
    m5s, mzprimes = gf.get_line_from_input(path, i_line)
    kwargs["m5"] = m5s
    kwargs["mzprime"] = mzprimes

    if "delta" in kwargs.keys():
        delta = kwargs.pop("delta")
        m4s = m5s / (delta + 1)
        kwargs["m4"] = m4s
    else:
        try:
            m4s = kwargs["m4"]
            delta = m5s / m4s - 1
        except KeyError:
            print("No delta or m4 passed. Skipping this point.")
            return 0

    if m5s - m4s < 2.05 * 511e-6:
        print("m5 - m4 < 2*m_e. Skipping this point.")
        return 0

    # Assuming Umu5 = Umu4 and UD4 = UD5.
    umu5 = np.geomspace(1e-6, 0.5, n)
    vmu5 = (
        kwargs["gD"]
        * kwargs["UD5"]
        * (kwargs["UD4"] * umu5 + kwargs["UD5"] * umu5)
        / np.sqrt(1 - 2 * umu5**2)
    )
    v54_def = kwargs["gD"] * kwargs["UD5"] * kwargs["UD4"]
    vmu5_def = (
        kwargs["gD"]
        * kwargs["UD5"]
        * (kwargs["Umu4"] * kwargs["UD4"] + kwargs["Umu5"] * kwargs["UD5"])
        / np.sqrt(1 - kwargs["Umu4"] ** 2 - kwargs["Umu5"] ** 2)
    )

    # 1. SIMULATIONS
    MB_fhc_df = GenLauncher(
        experiment="miniboone_fhc",
        nu_flavors=["nu_mu"],
        **kwargs,
    ).run()
    MB_fhc_df_dirt = GenLauncher(
        experiment="miniboone_fhc_dirt",
        nu_flavors=["nu_mu"],
        **kwargs,
    ).run()
    # 2. LIFETIME OF HNL
    decay_l = MB_fhc_df.attrs["N5_ctau0"]
    MB_fhc_df = pd.concat([MB_fhc_df, MB_fhc_df_dirt])

    MB_rhc_df = GenLauncher(
        experiment="miniboone_rhc",
        nu_flavors=["nu_mu_bar", "nu_mu"],
        **kwargs,
    ).run()
    MB_rhc_df_dirt = GenLauncher(
        experiment="miniboone_rhc_dirt",
        nu_flavors=["nu_mu_bar", "nu_mu"],
        **kwargs,
    ).run()
    MB_rhc_df = pd.concat([MB_rhc_df, MB_rhc_df_dirt])

    # sbn experiments
    sbn_dfs = []
    for exp, exp_dirt in zip(SBN_experiments, SBN_experiments_dirt):
        # simulation for SBN and SBN dirt
        sbn_df = GenLauncher(
            experiment=exp,
            nu_flavors=["nu_mu"],
            **kwargs,
        ).run()

        # Run the generation for MiniBooNE Dirt
        sbn_df_dirt = GenLauncher(
            experiment=exp_dirt,
            nu_flavors=["nu_mu"],
            **kwargs,
        ).run()

        sbn_dfs.append(pd.concat([sbn_df, sbn_df_dirt]))

    # 3. CYCLE OVER DIFFERENT VALUES OF U_MU4 and U_MU5
    for j in range(n):
        ##############
        # 6. MAIN OUTPUT -- TOTAL RATE AND CHI2
        chi2s, events = ff.miniboone_rates_and_chi2(
            MB_fhc_df,
            MB_rhc_df,
            rescale=(vmu5[j] / vmu5_def) ** 2,
            cut=cut,
            decay_l=decay_l,
        )

        # 6.1. OUTPUT
        output = f"{mzprimes:.6e} {m5s:.6e}  {m4s:.6e}  {delta:.6e} {vmu5[j]:.6e} {kwargs['epsilon']:.6e} {decay_l:.6e}"

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
                np.abs(np.sum(df_temp["w_event_rate"])) * (vmu5[j] / vmu5_def) ** 2
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
fit_point_3p2(
    args.i_line,
    args.path,
    cut=args.cut,
    print_spectra=args.print_spectra,
)
