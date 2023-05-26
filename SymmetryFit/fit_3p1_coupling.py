import numpy as np
import argparse

import DarkNews as dn
from DarkNews.GenLauncher import GenLauncher
from fastbnb import fit_functions as ff
from fastbnb import grid_fit as gf
from fastbnb import decayer

NCOUPLING_POINTS = 200
UMU4_RANGES = {
    0.03: (1e-11, 1e-4),
    0.06: (1e-11, 1e-3),
    0.1: (1e-11, 1e-5),
    0.2: (1e-10, 1e-4),
    0.5: (1e-8, 1e-2),
    0.8: (1e-8, 1e-2),
    1.25: (1e-8, 1e-1),
}
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

    if "ratio_tau_to_mu" in kwargs.keys():
        ratio_tau_to_mu = kwargs.pop("ratio_tau_to_mu")
    else:
        ratio_tau_to_mu = 0
    kwargs["Utau4"] = ratio_tau_to_mu * kwargs["Umu4"]

    rescale = 8e-4 / kwargs["epsilon"]
    umu4 = np.geomspace(
        np.sqrt(UMU4_RANGES[mzprimes][0]) * rescale,
        min(np.sqrt(UMU4_RANGES[mzprimes][1]) * rescale / (1 + ratio_tau_to_mu), 0.25),
        NCOUPLING_POINTS,
    )
    vmu4 = (
        kwargs["gD"]
        * kwargs["UD4"]
        * kwargs["UD4"]
        * umu4
        / np.sqrt(1 - umu4**2 * (1 + ratio_tau_to_mu**2))
    )
    v4i = kwargs["gD"] * kwargs["UD4"] * np.sqrt(umu4**2 * (1 + ratio_tau_to_mu**2))

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
    dl = MB_fhc_df.attrs["N4_ctau0"]
    MB_fhc_df = dn.MC.get_merged_MC_output(MB_fhc_df, MB_fhc_df_dirt)

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
    MB_rhc_df = dn.MC.get_merged_MC_output(MB_rhc_df, MB_rhc_df_dirt)

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

        sbn_dfs.append(dn.MC.get_merged_MC_output(sbn_df, sbn_df_dirt))

    # 3. CYCLE OVER DIFFERENT VALUES OF U_MU4
    v4i_def = (
        kwargs["gD"]
        * kwargs["UD4"]
        * np.sqrt(kwargs["Umu4"] ** 2 * (1 + ratio_tau_to_mu**2))
    )
    vmu4_def = (
        kwargs["gD"]
        * kwargs["UD4"]
        * kwargs["UD4"]
        * kwargs["Umu4"]
        / np.sqrt(1 - kwargs["Umu4"] ** 2 * (1 + ratio_tau_to_mu**2))
    )

    for j in range(NCOUPLING_POINTS):
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
