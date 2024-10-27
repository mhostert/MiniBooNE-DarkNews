import numpy as np
import scipy

from DarkNews import const
from fastbnb import decayer
from fastbnb import analysis
from fastbnb import grid_fit


from importlib.resources import open_text

# default value of couplings
ud4_def = 1.0 / np.sqrt(2.0)
ud5_def = 1.0 / np.sqrt(2.0)
gD_def = 2.0
umu4_def = np.sqrt(1.0e-12)
umu5_def = np.sqrt(1.0e-12)
epsilon_def = 1e-4

vmu5_def = gD_def * ud5_def * (umu4_def * ud4_def + umu5_def * ud5_def) / np.sqrt(1 - umu4_def**2 - umu5_def**2)
v4i_def = gD_def * ud4_def * umu4_def
vmu4_def = gD_def * ud4_def * ud4_def * umu4_def / np.sqrt(1 - umu4_def**2)


# couplings functions for simulation
vmu4_f = lambda umu4: gD_def * ud4_def * ud4_def * umu4 / np.sqrt(1 - umu4**2)
v4i_f = lambda umu4: gD_def * ud4_def * umu4
epsilon = 1e-4
r_eps = epsilon / epsilon_def

# bins for fast histogram preparation
bin_e_def = np.array([0.2, 0.3, 0.375, 0.475, 0.55, 0.675, 0.8, 0.95, 1.1, 1.3, 1.5, 3.0])
bin_enu_def = np.genfromtxt(open_text("fastbnb.include.miniboone_2020", "Enu_bin_edges.dat"))
bin_costheta_def = np.genfromtxt(open_text("fastbnb.include.miniboone_2020", "Costheta_bin_edges.dat"))
bin_evis_def = np.genfromtxt(open_text("fastbnb.include.miniboone_2020", "Evis_bin_edges.dat"))
bin_invmass_def = np.genfromtxt(open_text("fastbnb.include.pi0_tools", "Invmass_bin_edges.dat"))

# data for plots
plotvars = {"3+1": ["mzprime", "m4"], "3+2": ["m5", "delta"]}
plotaxes = {
    "3+1": [r"$m_{Z\prime} [\mathrm{GeV}]$", r"$m_{4} [\mathrm{GeV}]$"],
    "3+2": [r"$m_{5} [\mathrm{GeV}]$", r"$\Delta$"],
}

# Location
# loc = 'fastbnb/data'
loc = "data"
# obtain data from MB for the fitting
data_MB_source = {
    "Enu": grid_fit.get_data_MB(varplot="reco_Enu"),
    "angle": grid_fit.get_data_MB(varplot="reco_angle"),
}

# Normalization (temporal variable)
NORMALIZATION = 1


def round_sig(x, sig=1):
    if isinstance(x, float) | isinstance(x, int):
        z = int(np.floor(np.log10(np.abs(x))))
        return round(x, sig - z - 1)
    else:
        n = len(x)
        y = np.floor(np.log10(np.abs(x)))
        z = np.array([int(i) for i in y])
        return np.array([round(x[i], sig - z[i] - 1) for i in range(n)])


def get_decay_length(df, coupling_factor=1.0):
    # get momenta and decay length for decay_N
    pN = df.P_decay_N_parent.values
    l_decay_proper_cm = const.get_decay_rate_in_cm(np.sum(df.w_decay_rate_0)) / coupling_factor**2

    # compute the position of decay
    x, y, z = decayer.decay_position(pN, l_decay_proper_cm)[1:]

    return np.sqrt(x * x + y * y + z * z).mean()


def chi2_test(NP_MC, NPevents, D, sys=[0.1, 0.1]):
    NP_MC = (NP_MC / np.sum(NP_MC)) * NPevents
    sys = sys[0]
    chi2 = ((NP_MC - D) ** 2) / (D + sys**2)

    return chi2.sum()


def safe_log(di, xi):
    mask = di * xi > 0
    d = np.empty_like(di * xi)
    d[mask] = di[mask] * np.log(di[mask] / xi[mask])
    # d[~mask] = di[~mask]*1e100
    d[~mask] = di[~mask] * 0.0
    return d


def chi2_binned_rate(NP_MC, NPevents, back_MC, D, sys=[0.1, 0.1]):
    err_flux = sys[0]
    err_back = sys[1]

    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC) != 0:
        NP_MC = (NP_MC / np.sum(NP_MC)) * NPevents

    dpoints = len(D)

    def chi2bin(nuis):
        alpha = nuis[:dpoints]
        beta = nuis[dpoints:]

        mu = NP_MC * (1 + alpha) + back_MC * (1 + beta)

        return 2 * np.sum(mu - D + safe_log(D, mu)) + np.sum(alpha**2 / (err_flux**2)) + np.sum(beta**2 / (err_back**2))

    cons = {"type": "ineq", "fun": lambda x: x}

    res = scipy.optimize.minimize(chi2bin, np.zeros(dpoints * 2), constraints=cons)

    return chi2bin(res.x)


def chi2_MiniBooNE_2020(NP_MC, NPevents=None, mode="fhc"):
    """chi2_MiniBooNE_2020 Get MiniBOoNE chi2 from data release in 2020 for a given mode (FHC, RHC)

    Parameters
    ----------
    NP_MC : np.array
        Monte Carlo prediction for the signal rate from DarkNews -- shape of the histrogram.
    NPevents : np.float
        Total number of signal events to normalize the NP_MC prediction.

    Returns
    -------
    np.float
        the MiniBooNE chi2 value (non-zero)
    """

    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC) != 0 and NPevents is not None:
        NP_MC = (NP_MC / np.sum(NP_MC)) * NPevents

    mode = mode.lower()
    bar = "bar" if mode == "rhc" else ""

    nue_data = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}mode",
            f"miniboone_nue{bar}data_lowe.txt",
        )
    )
    numu_data = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}mode",
            f"miniboone_numu{bar}data.txt",
        )
    )

    nue_bkg = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}mode",
            f"miniboone_nue{bar}bgr_lowe.txt",
        )
    )
    numu_bkg = np.genfromtxt(open_text(f"fastbnb.include.MB_data_release.{mode}mode", f"miniboone_numu{bar}.txt"))

    fract_covariance = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}mode",
            f"miniboone_full_fractcovmatrix_nu{bar}_lowe.txt",
        )
    )

    # energy bins -- same for nu and nubar
    bin_e = np.genfromtxt(
        open_text(
            "fastbnb.include.MB_data_release.combined",
            "miniboone_binboundaries_nue_lowe.txt",
        )
    )

    NP_diag_matrix = np.diag(np.concatenate([NP_MC, nue_bkg * 0.0, numu_bkg * 0.0]))
    tot_diag_matrix = np.diag(np.concatenate([NP_MC, nue_bkg, numu_bkg]))

    rescaled_covariance = np.dot(tot_diag_matrix, np.dot(fract_covariance, tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix  # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_numu = len(numu_bkg)

    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal + n_numu, n_signal + n_numu])
    error_matrix[0:n_signal, 0:n_signal] = (
        rescaled_covariance[0:n_signal, 0:n_signal]
        + rescaled_covariance[n_signal : 2 * n_signal, 0:n_signal]
        + rescaled_covariance[0:n_signal, n_signal : 2 * n_signal]
        + rescaled_covariance[n_signal : 2 * n_signal, n_signal : 2 * n_signal]
    )
    error_matrix[n_signal : (n_signal + n_numu), 0:n_signal] = (
        rescaled_covariance[2 * n_signal : (2 * n_signal + n_numu), 0:n_signal]
        + rescaled_covariance[2 * n_signal : (2 * n_signal + n_numu), n_signal : 2 * n_signal]
    )
    error_matrix[0:n_signal, n_signal : (n_signal + n_numu)] = (
        rescaled_covariance[0:n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
        + rescaled_covariance[n_signal : 2 * n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
    )
    error_matrix[n_signal : (n_signal + n_numu), n_signal : (n_signal + n_numu)] = rescaled_covariance[
        2 * n_signal : 2 * n_signal + n_numu, 2 * n_signal : (2 * n_signal + n_numu)
    ]

    # assert(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3)
    # if not (np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.0e-3):
    #     return -1

    # compute residuals
    residuals = np.concatenate([nue_data - (NP_MC + nue_bkg), (numu_data - numu_bkg)])

    inv_cov = np.linalg.inv(error_matrix)

    # calculate chi^2
    chi2 = np.dot(residuals, np.dot(inv_cov, residuals))  # + np.log(np.linalg.det(error_matrix))

    if chi2 >= 0:
        return chi2
    else:
        return 1e10


def chi2_MiniBooNE_2020_combined(NP_MC, NP_MC_BAR, NPevents=None, NPevents_BAR=None):
    """chi2_MiniBooNE_2020 Get MiniBOoNE chi2 from data release in 2020 for a given mode (FHC, RHC)

    Parameters
    ----------
    NP_MC : np.array
        Monte Carlo prediction for the FHC signal rate from DarkNews
    NP_MC : np.array
        Monte Carlo prediction for the RHC signal rate from DarkNews
    NPevents : np.float, optional
        Total number of signal events to normalize the NP_MC prediction.
    NPevents_BAR : np.float, optional
        Total number of signal events to normalize the NP_MC_BAR prediction.

    Returns
    -------
    np.float
        the MiniBooNE chi2 value (non-zero)
    """

    # shape of new physics prediction normalized to NPevents
    if np.sum(NP_MC) != 0 and NPevents is not None:
        NP_MC = (NP_MC / np.sum(NP_MC)) * NPevents
    if np.sum(NP_MC_BAR) != 0 and NPevents_BAR is not None:
        NP_MC_BAR = (NP_MC_BAR / np.sum(NP_MC_BAR)) * NPevents_BAR

    mode = "combined"

    ##########################################
    # Load neutrino data
    bar = ""
    nue_data = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_nue{bar}data_lowe.txt",
        )
    )
    numu_data = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_numu{bar}data.txt",
        )
    )

    nue_bkg = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_nue{bar}bgr_lowe.txt",
        )
    )
    numu_bkg = np.genfromtxt(open_text(f"fastbnb.include.MB_data_release.{mode}", f"miniboone_numu{bar}.txt"))

    ##########################################
    # Load antineutrino data
    bar = "bar"
    nue_data_bar = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_nue{bar}data_lowe.txt",
        )
    )
    numu_data_bar = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_numu{bar}data.txt",
        )
    )

    nue_bkg_bar = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_nue{bar}bgr_lowe.txt",
        )
    )
    numu_bkg_bar = np.genfromtxt(open_text(f"fastbnb.include.MB_data_release.{mode}", f"miniboone_numu{bar}.txt"))

    ##########################################
    # Load covariance matrix
    fract_covariance = np.genfromtxt(
        open_text(
            f"fastbnb.include.MB_data_release.{mode}",
            f"miniboone_full_fractcovmatrix_combined_lowe.txt",
        )
    )

    NP_diag_matrix = np.diag(
        np.concatenate(
            [
                NP_MC,
                nue_bkg * 0.0,
                numu_bkg * 0.0,
                NP_MC_BAR,
                nue_bkg_bar * 0.0,
                numu_bkg_bar * 0.0,
            ]
        )
    )
    tot_diag_matrix = np.diag(np.concatenate([NP_MC, nue_bkg, numu_bkg, NP_MC_BAR, nue_bkg_bar, numu_bkg_bar]))

    rescaled_covariance = np.dot(tot_diag_matrix, np.dot(fract_covariance, tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix  # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = len(NP_MC)
    n_numu = len(numu_bkg)
    error_matrix = MassageCovarianceMatrix(rescaled_covariance, n_signal, n_numu)

    # # procedure described by MiniBooNE itself
    # error_matrix = np.zeros([2 * (n_signal + n_numu), 2 * (n_signal + n_numu)])

    # # Upper left block
    # error_matrix[0:n_signal, 0:n_signal] = (
    #     rescaled_covariance[0:n_signal, 0:n_signal]
    #     + rescaled_covariance[n_signal : 2 * n_signal, 0:n_signal]
    #     + rescaled_covariance[0:n_signal, n_signal : 2 * n_signal]
    #     + rescaled_covariance[n_signal : 2 * n_signal, n_signal : 2 * n_signal]
    # )

    # # Lower left block
    # error_matrix[n_signal : (n_signal + n_numu), 0:n_signal] = (
    #     rescaled_covariance[2 * n_signal : (2 * n_signal + n_numu), 0:n_signal]
    #     + rescaled_covariance[
    #         2 * n_signal : (2 * n_signal + n_numu), n_signal : 2 * n_signal
    #     ]
    # )

    # # Upper right block
    # error_matrix[0:n_signal, n_signal : (n_signal + n_numu)] = (
    #     rescaled_covariance[0:n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
    #     + rescaled_covariance[
    #         n_signal : 2 * n_signal, 2 * n_signal : (2 * n_signal + n_numu)
    #     ]
    # )

    # # Lower right block
    # error_matrix[
    #     n_signal : (n_signal + n_numu), n_signal : (n_signal + n_numu)
    # ] = rescaled_covariance[
    #     2 * n_signal : 2 * n_signal + n_numu, 2 * n_signal : (2 * n_signal + n_numu)
    # ]

    # assert(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3)
    # if not (np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.0e-3):
    #     return -1

    # compute residuals
    residuals = np.concatenate(
        [
            nue_data - (NP_MC + nue_bkg),
            (numu_data - numu_bkg),
            nue_data_bar - (NP_MC_BAR + nue_bkg_bar),
            (numu_data_bar - numu_bkg_bar),
        ]
    )

    inv_cov = np.linalg.inv(error_matrix)

    # calculate chi^2
    chi2 = np.dot(residuals, np.dot(inv_cov, residuals))  # + np.log(np.linalg.det(error_matrix))

    if chi2 >= 0:
        return chi2
    else:
        return 1e10


def StackCovarianceMatrix(big_covariance, n_signal, n_numu):
    covariance = np.zeros([n_signal + n_numu, n_signal + n_numu])

    covariance[0:n_signal, 0:n_signal] = (
        big_covariance[0:n_signal, 0:n_signal]
        + big_covariance[n_signal : 2 * n_signal, 0:n_signal]
        + big_covariance[0:n_signal, n_signal : 2 * n_signal]
        + big_covariance[n_signal : 2 * n_signal, n_signal : 2 * n_signal]
    )
    covariance[n_signal : (n_signal + n_numu), 0:n_signal] = (
        big_covariance[2 * n_signal : (2 * n_signal + n_numu), 0:n_signal] + big_covariance[2 * n_signal : (2 * n_signal + n_numu), n_signal : 2 * n_signal]
    )
    covariance[0:n_signal, n_signal : (n_signal + n_numu)] = (
        big_covariance[0:n_signal, 2 * n_signal : (2 * n_signal + n_numu)] + big_covariance[n_signal : 2 * n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
    )
    covariance[n_signal : (n_signal + n_numu), n_signal : (n_signal + n_numu)] = big_covariance[
        2 * n_signal : 2 * n_signal + n_numu, 2 * n_signal : (2 * n_signal + n_numu)
    ]

    # assert np.abs(np.sum(covariance) - np.sum(big_covariance)) < 1.0e-3

    return covariance


def MassageCovarianceMatrix(big_covariance, n_signal, n_numu):
    n_total = n_signal + n_numu
    n_total_big = n_signal * 2 + n_numu

    covariance = np.zeros([n_total * 2, n_total * 2])

    covariance[0:n_total, 0:n_total] = StackCovarianceMatrix(big_covariance[0:n_total_big, 0:n_total_big], n_signal, n_numu)
    covariance[n_total : (2 * n_total), 0:n_total] = StackCovarianceMatrix(big_covariance[n_total_big : (2 * n_total_big), 0:n_total_big], n_signal, n_numu)
    covariance[0:n_total, n_total : (2 * n_total)] = StackCovarianceMatrix(big_covariance[0:n_total_big, n_total_big : (2 * n_total_big)], n_signal, n_numu)
    covariance[n_total : (2 * n_total), n_total : (2 * n_total)] = StackCovarianceMatrix(
        big_covariance[n_total_big : (2 * n_total_big), n_total_big : (2 * n_total_big)],
        n_signal,
        n_numu,
    )
    # assert np.abs(np.sum(covariance) - np.sum(big_covariance)) < 1.0e-3
    return covariance


def cov_matrix_MB():
    # shape of new physics prediction normalized to NPevents
    # using __init__ path definition
    bin_e = np.genfromtxt(
        open_text(
            "fastbnb.include.MB_data_release.fhcmode",
            "miniboone_binboundaries_nue_lowe.txt",
        )
    )
    bin_w = -bin_e[:-1] + bin_e[1:]

    nue_data = np.genfromtxt(open_text("fastbnb.include.MB_data_release.fhcmode", "miniboone_nuedata_lowe.txt"))
    numu_data = np.genfromtxt(open_text("fastbnb.include.MB_data_release.fhcmode", "miniboone_numudata.txt"))

    nue_bkg = np.genfromtxt(open_text("fastbnb.include.MB_data_release.fhcmode", "miniboone_nuebgr_lowe.txt"))
    numu_bkg = np.genfromtxt(open_text("fastbnb.include.MB_data_release.fhcmode", "miniboone_numu.txt"))

    fract_covariance = np.genfromtxt(
        open_text(
            "fastbnb.include.MB_data_release.fhcmode",
            "miniboone_full_fractcovmatrix_nu_lowe.txt",
        )
    )

    NP_diag_matrix = np.diag(np.concatenate([nue_data - nue_bkg, nue_bkg * 0.0, numu_bkg * 0.0]))
    tot_diag_matrix = np.diag(np.concatenate([nue_data - nue_bkg, nue_bkg, numu_bkg]))

    rescaled_covariance = np.dot(tot_diag_matrix, np.dot(fract_covariance, tot_diag_matrix))
    rescaled_covariance += NP_diag_matrix  # this adds the statistical error on data

    # collapse background part of the covariance
    n_signal = 11
    n_background = 11
    n_numu = 8

    # procedure described by MiniBooNE itself
    error_matrix = np.zeros([n_signal + n_numu, n_signal + n_numu])
    error_matrix[0:n_signal, 0:n_signal] = (
        rescaled_covariance[0:n_signal, 0:n_signal]
        + rescaled_covariance[n_signal : 2 * n_signal, 0:n_signal]
        + rescaled_covariance[0:n_signal, n_signal : 2 * n_signal]
        + rescaled_covariance[n_signal : 2 * n_signal, n_signal : 2 * n_signal]
    )
    error_matrix[n_signal : (n_signal + n_numu), 0:n_signal] = (
        rescaled_covariance[2 * n_signal : (2 * n_signal + n_numu), 0:n_signal]
        + rescaled_covariance[2 * n_signal : (2 * n_signal + n_numu), n_signal : 2 * n_signal]
    )
    error_matrix[0:n_signal, n_signal : (n_signal + n_numu)] = (
        rescaled_covariance[0:n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
        + rescaled_covariance[n_signal : 2 * n_signal, 2 * n_signal : (2 * n_signal + n_numu)]
    )
    error_matrix[n_signal : (n_signal + n_numu), n_signal : (n_signal + n_numu)] = rescaled_covariance[
        2 * n_signal : 2 * n_signal + n_numu, 2 * n_signal : (2 * n_signal + n_numu)
    ]

    # assert(np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.e-3)

    if not (np.abs(np.sum(error_matrix) - np.sum(rescaled_covariance)) < 1.0e-3):
        return -1

    # inv_cov = np.linalg.inv(error_matrix)

    return error_matrix


def chi2_MiniBooNE_2020_3p1(
    df,
    cut="circ1",
    decay_rescale=1.0,
    rate_rescale=1.0,
    l_decay_proper_cm=1,
    mode="fhc",
    post_analysis=False,
):
    """chi2_MiniBooNE_2020_3p1 This compute MiniBooNE chi2 from raw DarkNews dataframes -- straight from generation.

    Parameters
    ----------
    df : pd.DataFrame
        The DarkNews events dataframe after reco selection
    umu4 : np.float
        The Umu4 parameter to rescale the total number of events
    cut : str, optional
        What kind of cut to run the reco selection on, by default "circ1"
    decay_rescale : float, optional
        The rescaling factor for the decay length, by default 1.0
    rate_rescale : float, optional
        The rescaling factor for the total number of events, by default 1.0
    l_decay_proper_cm : int, optional
        _description_, by default 1

    Returns
    -------
    np.float
        chi2 of MiniBoonE fit (FHC mode)
    """

    df = df.copy(deep=True)

    decay_l = l_decay_proper_cm / decay_rescale

    if post_analysis:
        df_decay = decayer.decay_selection(df, l_decay_proper_cm=decay_l, experiment="miniboone", weights="reco_w")
    else:
        df_decay = decayer.decay_selection(
            df,
            l_decay_proper_cm=decay_l,
            experiment="miniboone",
            weights="w_event_rate",
        )
        df_decay = analysis.reco_nueCCQElike_Enu(df_decay, cut=cut, clean_df=True)

    return chi2_MiniBooNE_2020(get_MiniBooNE_Enu_bin_vals(df_decay) * rate_rescale, mode=mode)


def chi2_MiniBooNE_2020_3p2(df, rate_rescale=1.0, mode="fhc"):
    """chi2_MiniBooNE_2020_3p2 This compute MiniBooNE chi2 from processed DarkNews dataframes -- i.e., needs to be decayer.decay_selection and analysis.reco_nueCCQElike_Enu.

    Parameters
    ----------
    df : pd.DataFrame
        The DarkNews dataframe after reco selection
    rate_rescale :
        rescale factor for event rate, by default 1.0
    Returns
    -------
    np.float
        chi2 of MiniBoonE fit (FHC mode)
    """

    # No decay length rescale here as xsec and decay are decoupled.

    df_decay = df.copy(deep=True)
    return chi2_MiniBooNE_2020(get_MiniBooNE_Enu_bin_vals(df_decay) * rate_rescale, mode=mode)


def get_MiniBooNE_Enu_bin_vals(df):
    hist = np.histogram(df["reco_Enu"], weights=df["reco_w"], bins=bin_e_def, density=False)
    return hist[0]


def get_MiniBooNE_Evis_bin_vals(df):
    hist = np.histogram(df["reco_Evis"], weights=df["reco_w"], bins=bin_evis_def, density=False)
    return hist[0]


def get_MiniBooNE_Costheta_bin_vals(df):
    hist = np.histogram(
        df["reco_costheta_beam"],
        weights=df["reco_w"],
        bins=bin_costheta_def,
        density=False,
    )
    return hist[0]


def get_MiniBooNE_Invmass_bin_vals(df):
    hist = np.histogram(
        df["reco_mgg"],
        weights=df["reco_w"],
        bins=bin_invmass_def,
        density=False,
    )
    return hist[0]


def miniboone_rates_and_chi2(df_fhc, df_rhc, rescale=1, cut="circ1", decay_l=None):
    if decay_l is None:
        decay_l = df_fhc.attrs["N4_ctau0"]

    # Neutrino mode
    df_fhc_temp = decayer.decay_selection(df_fhc, decay_l, "miniboone", weights="w_event_rate")
    df_fhc_temp = analysis.reco_nueCCQElike_Enu(df_fhc_temp, cut=cut, clean_df=True)
    df_fhc_temp["w_event_rate"] *= rescale
    df_fhc_temp["reco_w"] *= rescale

    # Antineutrino mode
    df_rhc_temp = decayer.decay_selection(df_rhc, decay_l, "miniboone", weights="w_event_rate")
    df_rhc_temp = analysis.reco_nueCCQElike_Enu(df_rhc_temp, cut=cut, clean_df=True)
    df_rhc_temp["w_event_rate"] *= rescale
    df_rhc_temp["reco_w"] *= rescale

    # computing chi2
    hist_fhc = get_MiniBooNE_Enu_bin_vals(df_fhc_temp)
    hist_rhc = get_MiniBooNE_Enu_bin_vals(df_rhc_temp)

    chi2s = []
    chi2s.append(chi2_MiniBooNE_2020(hist_fhc, mode="fhc"))
    chi2s.append(chi2_MiniBooNE_2020(hist_rhc, mode="rhc"))
    chi2s.append(chi2_MiniBooNE_2020_combined(hist_fhc, hist_rhc))

    events = [df_fhc_temp["reco_w"].sum(), df_rhc_temp["reco_w"].sum()]

    return chi2s, events
