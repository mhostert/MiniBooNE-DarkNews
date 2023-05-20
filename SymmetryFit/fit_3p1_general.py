import numpy as np
import scipy

# from scipy.optimize import dual_annealing
import argparse

from DarkNews.MC import get_merged_MC_output
from DarkNews.GenLauncher import GenLauncher

from fastbnb import fit_functions as ff
from fastbnb import grid_fit as gf
from fastbnb import analysis
from fastbnb import decayer


def fit_point_3p1(
    i_line,
    path,
    cut="circ1",
    print_spectra=True,
    umu4_max=1e-2,
    **kwargs,
):
    m4s, mzprimes = gf.get_line_from_input(path, i_line)
    if not kwargs:
        kwargs = gf.get_kwargs_from_input(path)

    if "ratio_tau_to_mu" in kwargs.keys():
        ratio_tau_to_mu = kwargs.pop("ratio_tau_to_mu")
        umu4_max = umu4_max / ratio_tau_to_mu
    else:
        ratio_tau_to_mu = 0

    kwargs["Utau4"] = ratio_tau_to_mu * kwargs["Umu4"]

    # 1. PERFORM AND READ THE DATA FROM THE SIMULATION
    # Run the generation for MiniBooNE
    MB_df = GenLauncher(
        mzprime=mzprimes, m4=m4s, experiment="miniboone_fhc", **kwargs
    ).run()

    # Run the generation for MiniBooNE Dirt
    MB_df_dirt = GenLauncher(
        mzprime=mzprimes, m4=m4s, experiment="miniboone_fhc_dirt", **kwargs
    ).run()
    df_tot = get_merged_MC_output(MB_df, MB_df_dirt)
    decay_l = df_tot.attrs["N4_ctau0"]

    ###########################
    # 3. DEFINE FUNCTIONS WITH/WITHOUT DIRT TO FIT
    # The coupling vertex based on the parameters of the simulation
    v4i_def = kwargs["gD"] * kwargs["UD4"] * kwargs["Umu4"] * (1 + ratio_tau_to_mu)
    vmu4_def = (
        kwargs["gD"]
        * kwargs["UD4"]
        * kwargs["UD4"]
        * kwargs["Umu4"]
        / np.sqrt(1 - kwargs["Umu4"] ** 2 * (1 + ratio_tau_to_mu**2))
    )

    vmu4_f = (
        lambda umu4: kwargs["gD"]
        * kwargs["UD4"] ** 2
        * umu4
        / np.sqrt(1 - umu4**2 * (1 + ratio_tau_to_mu**2))
    )

    v4i_f = (
        lambda umu4: kwargs["gD"]
        * kwargs["UD4"]
        * np.sqrt((1 + ratio_tau_to_mu**2) * umu4**2)
    )

    ###################################################################################################
    df_tot_postanalysis = analysis.reco_nueCCQElike_Enu(
        df_tot, exp="miniboone", cut=cut, clean_df=False
    )

    # if decay_l > 100 * 10e2:  # much greater than 10 meters
    #     test_umu4 = np.geomspace(umu4_max * 1e-6, umu4_max, 500)
    # else:
    #     test_umu4 = np.geomspace(umu4_max * 1e-6, umu4_max, 10)

    # chi2_test = np.array(
    #     [
    #         ff.chi2_MiniBooNE_2020_3p1(
    #             df_tot_postanalysis,
    #             cut=cut,
    #             rate_rescale=(vmu4_f(test) / vmu4_def) ** 2,
    #             decay_rescale=(v4i_f(test) / v4i_def) ** 2,
    #             l_decay_proper_cm=decay_l,
    #             post_analysis=True,
    #         )
    #         for test in test_umu4
    #     ]
    # )
    # chi2_test[chi2_test < 0] = 1e10
    # theta_init_guess = -np.log10(test_umu4[np.argmin(chi2_test)] / umu4_max)

    chi2_func = lambda theta: ff.chi2_MiniBooNE_2020_3p1(
        df_tot_postanalysis,
        cut=cut,
        rate_rescale=(vmu4_f(umu4_max * 10 ** (-theta)) / vmu4_def) ** 2,
        decay_rescale=(v4i_f(umu4_max * 10 ** (-theta)) / v4i_def) ** 2,
        l_decay_proper_cm=decay_l,
        post_analysis=True,
    )

    # 4. DO THE FITTING
    x_minima = []
    y_minima = []
    guesses = [
        5,
        4,
        2,
        1,
        0,
    ]
    for guess in guesses:
        # Gradient descent/greedy algo are deterministic, but with low stats, the minimum jumps around.
        res = scipy.optimize.minimize(
            chi2_func, guess, method="Nelder-Mead", bounds=[(0, 6)]
        )
        x_minima.append(res.x[0])
        y_minima.append(res.fun)

    # Get minimum of the multiple minimizations above
    x_min = x_minima[np.argmin(y_minima)]
    y_min = np.min(y_minima)

    umu4_bf = umu4_max * 10 ** (-x_min)
    vmu4_bf = vmu4_f(umu4_bf)

    ###################################################################################################
    ### Jaime's Corner
    # test_guess = np.concatenate(
    #     [
    #         np.arange(1, 10) * 1e-4,
    #         np.arange(1, 10) * 1e-3,
    #         np.arange(1, 10) * 1e-2,
    #         np.arange(1, 11) * 1e-1,
    #     ]
    # )
    # n_test = len(test_guess)
    # test_theta = np.sqrt(np.log(1.0 / test_guess))
    # test_umu4 = umu4_max * np.exp(-(test_theta**2))
    # chi2_test = np.array(
    #     [
    #         ff.chi2_MiniBooNE_2020_3p1(
    #             df_tot,
    #             test_umu4[k_test],
    #             on_shell=on_shell,
    #             v4i_f=v4i_f,
    #             v4i_def=v4i_def,
    #             vmu4_f=vmu4_f,
    #             vmu4_def=vmu4_def,
    #             l_decay_proper_cm=decay_l,
    #         )
    #         for k_test in range(n_test)
    #     ]
    # )
    # mask = chi2_test < 0
    # chi2_test[mask] = 1e10
    # theta_guess = test_theta[chi2_test == chi2_test.min()].mean()
    # init_guess = np.array([theta_guess])

    # chi2 = lambda theta: ff.chi2_MiniBooNE_2020_3p1(
    #     df_tot,
    #     umu4_max * np.exp(-(theta**2)),
    #     on_shell=on_shell,
    #     v4i_f=v4i_f,
    #     v4i_def=v4i_def,
    #     vmu4_f=vmu4_f,
    #     vmu4_def=vmu4_def,
    #     l_decay_proper_cm=decay_l,
    # )

    # # 4. DO THE FITTING
    # res = scipy.optimize.minimize(chi2, init_guess)
    # umu4_bf = umu4_max * np.exp(-res.x[0] ** 2)
    # vmu4_bf = vmu4_f(umu4_bf)
    ###################################################################################################

    # 5. COMPUTE NUMBERS FROM THE BEST FIT
    v4i_bf = v4i_f(umu4_bf)
    rescale = (v4i_bf / v4i_def) ** 2
    decay_l_bf = decay_l / rescale

    # Perform selection with detector events
    df_tot_bf = decayer.decay_selection(
        df_tot_postanalysis,
        decay_l_bf,
        "miniboone",
        weights="reco_w",
    )

    # Perform selection with detector events
    temp_df = decayer.decay_selection(
        df_tot,
        decay_l_bf,
        "miniboone",
        weights="w_event_rate",
    )

    # df_bf = analysis.reco_nueCCQElike_Enu(df_bf, cut=cut)
    # sum_w_post_smearing_det = np.abs(np.sum(df_bf["reco_w"]))

    # Perform selection with dirt events
    # df_bf2 = decayer.decay_selection(
    #     MB_df_dirt, decay_l_bf, "miniboone", weights="w_event_rate"
    # )
    # df_bf2 = analysis.reco_nueCCQElike_Enu(df_bf2, cut=cut)
    # sum_w_post_smearing = sum_w_post_smearing_det + np.abs(np.sum(df_bf2["reco_w"]))

    # df_tot_bf = pd.concat([df_bf, df_bf2])
    sum_w_post_smearing = df_tot_bf["reco_w"].sum()
    ##############
    # 6. MAIN OUTPUT -- TOTAL RATE AND CHI2

    # Efficiency selection before geometry
    eff_selection = df_tot_postanalysis["reco_w"].sum() / df_tot["w_event_rate"].sum()

    # Efficiency geometry before selection
    eff_geometry = temp_df["w_event_rate"].sum() / temp_df["w_pre_decay"].sum()

    # final effieicny selection + geometry
    eff_final = df_tot_bf["reco_w"].sum() / df_tot["w_event_rate"].sum()

    # Chi2 and total event rate
    output = f"{mzprimes:.6e} {m4s:.6e} {sum_w_post_smearing:.6e} {vmu4_bf:.6e} {v4i_bf:.6e} {kwargs['epsilon']:.6e} {umu4_bf:.6e} {y_min:.6e} {decay_l_bf:.6e} {(vmu4_bf/vmu4_def)**2 * sum_w_post_smearing:.6e} {eff_geometry:.6e} {eff_selection:.6e} {eff_final:.6e}\n"
    gf.save_to_locked_file(path + f"chi2_{i_line}.dat", output)

    ##############
    # 7. OTHER OUTPUT -- COMPUTE SPECTRA FROM THE BEST FIT
    if print_spectra:
        # Enu
        line = " ".join([f"{i:.6e}" for i in ff.get_MiniBooNE_Enu_bin_vals(df_tot_bf)])
        gf.save_to_locked_file(
            filename=path + f"enu_spectrum_{i_line}.dat",
            output=f"{i_line} " + line + "\n",
        )
        # Evis
        line = " ".join([f"{i:.6e}" for i in ff.get_MiniBooNE_Evis_bin_vals(df_tot_bf)])
        gf.save_to_locked_file(
            filename=path + f"evis_spectrum_{i_line}.dat",
            output=f"{i_line} " + line + "\n",
        )
        # Cos(Theta)
        line = " ".join(
            [f"{i:.6e}" for i in ff.get_MiniBooNE_Costheta_bin_vals(df_tot_bf)]
        )
        gf.save_to_locked_file(
            filename=path + f"costheta_spectrum_{i_line}.dat",
            output=f"{i_line} " + line + "\n",
        )

        # Now for the invariant mass we perform the pi0 selection
        df_pi0 = analysis.reco_pi0like_invmass(df_tot_bf, cut=cut)
        line = " ".join([f"{i:.6e}" for i in ff.get_MiniBooNE_Invmass_bin_vals(df_pi0)])
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
args = parser.parse_args()

######## Running fit function ########
fit_point_3p1(
    args.i_line,
    args.path,
    cut=args.cut,
    print_spectra=args.print_spectra,
)
