import numpy as np
from scipy.interpolate import interp1d

from fastbnb import fastmc, toy_logger

from DarkNews import const
from DarkNews import Cfourvec as Cfv


def miniboone_reco_eff_func():
    # Single photon efficiencies from data release
    eff = np.array(
        [
            0.0,
            0.089,
            0.135,
            0.139,
            0.131,
            0.123,
            0.116,
            0.106,
            0.102,
            0.095,
            0.089,
            0.082,
            0.073,
            0.067,
            0.052,
            0.026,
        ]
    )
    enu = np.array(
        [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.3,
            1.5,
            1.7,
            1.9,
            2.1,
        ]
    )
    enu_c = enu[:-1] + (enu[1:] - enu[:-1]) / 2
    return interp1d(enu_c, eff, fill_value=(eff[0], eff[-1]), bounds_error=False, kind="nearest")


# NOTE: This function is not used in the paper.
def microboone_reco_eff_func():
    # Total nueCCQE efficiency -- NuMI flux averaged xsec paper.
    E_numi_eff_edge = np.array(
        [
            0,
            2.97415218e-1,
            4.70796471e-1,
            7.01196105e-1,
            9.88922361e-1,
            1.43183725e0,
            3.00810009e0,
            6.00428361e0,
        ]
    )
    nueCCQE_numi_eff_edge = np.array(
        [
            6.13255034e-2,
            1.44127517e-1,
            2.12332215e-1,
            2.64681208e-1,
            2.76761745e-1,
            2.97902685e-1,
            2.57885906e-1,
            2.60151007e-1,
        ]
    )
    E_numi_eff = (E_numi_eff_edge[1:] - E_numi_eff_edge[:-1]) / 2 + E_numi_eff_edge[:-1]
    nueCCQE_numi_eff = nueCCQE_numi_eff_edge[:-1]
    return interp1d(
        E_numi_eff,
        nueCCQE_numi_eff,
        fill_value=(nueCCQE_numi_eff[0], nueCCQE_numi_eff[-1]),
        bounds_error=False,
        kind="nearest",
    )


def apply_reco_efficiencies(Energy, w, exp="miniboone"):
    """reco_efficiencies apply reconstruction efficiencies as a function of energy (vis or Enu)

    Parameters
    ----------
    Energy: np.array
        The visible, as assumed in the reco efficiency functions.
    w: np.array
        The event weights to be reweighted with the reco efficiencies.
    exp : str, optional
        experiment name, by default 'miniboone'.

    Returns
    -------
    np.array
        the reweighted event weights.
    """
    if exp == "miniboone":
        eff_func = miniboone_reco_eff_func()
    elif exp == "microboone":
        eff_func = microboone_reco_eff_func()

    return eff_func(Energy) * w


# Full analysis to get reconstructed energy spectrum for BNB experiments
def reco_nueCCQElike_Enu(df, exp="miniboone", cut="circ1", clean_df=False):
    """compute_spectrum _summary_

    Parameters
    ----------
    df : pd.DatagFrame
        DarkNews events (preferrably after selecting events inside detector)
    exp : str, optional
        what experiment to use, by default 'miniboone' but can also be 'microboone'
    cut : str, optional
        what kind of "mis-identificatin" pre-selection cut to be used:
            for photons:
                'photon' assumes this is a photon and therefore always a single shower
            for lepton pairs:
                circ0, circ1, diag -- corresponding to Kelly&Kopp criteria
                invmass -- criterion imposed by hand. Corresponding to invmass criteria from Patterson
                byhand -- criterion imposed by hand. No mee cut.
    clean_df : bool, optional
        whether to remove rejected events from the dataframe, by default False

    Returns
    -------
    pd.DatagFrame
        A new dataframe with additional columns containing weights of the selected events.
    """

    df = df.copy(deep=True)

    # Initial weigths
    w = df["w_event_rate"].values  # typically already selected for fiducial volume

    if "P_decay_photon" in df.columns:
        # Smear photon
        pgamma = fastmc.smear_samples(df["P_decay_photon"], mass=0.0, exp=exp)

        # Reco quantities for nueCCQE-like events
        Evis = pgamma[:, 0]
        costheta = Cfv.get_cosTheta(pgamma)

        # no pre-selection needed in this case
        w_preselection = w

    elif "P_decay_ell_minus" in df.columns:
        # Smear e+ and e-
        pep = fastmc.smear_samples(df["P_decay_ell_plus"], exp=exp)
        pem = fastmc.smear_samples(df["P_decay_ell_minus"], exp=exp)

        # kinetic energy of e+e-
        Evis = pep[:, 0] + pem[:, 0] - 2 * const.m_e

        # angle wrt the beam direction using the sum of e+e-
        costheta = Cfv.get_cosTheta(pep + pem)

        # Evis, theta_beam, w, eff_s = signal_events(pep, pem, Delta_costheta, costhetaep, costhetaem, w, threshold=ep.THRESHOLD[exp], angle_max=ep.ANGLE_MAX[exp], event_type=event_type)
        w_preselection = apply_pre_selection(w, pep, pem, kind="nueCCQElike", cut=cut)

    elif "P_decay_photon_1" in df.columns:
        # Smear two photons
        pep = fastmc.smear_samples(df["P_decay_photon_1"], exp=exp)
        pem = fastmc.smear_samples(df["P_decay_photon_2"], exp=exp)

        # kinetic energy of gamma-gamma pair
        Evis = pep[:, 0] + pem[:, 0]

        # angle wrt the beam direction using the sum of gamma-gamma
        costheta = Cfv.get_cosTheta(pep + pem)

        # Evis, theta_beam, w, eff_s = signal_events(pep, pem, Delta_costheta, costhetaep, costhetaem, w, threshold=ep.THRESHOLD[exp], angle_max=ep.ANGLE_MAX[exp], event_type=event_type)
        w_preselection = apply_pre_selection(w, pep, pem, kind="nueCCQElike", cut=cut)
    else:
        toy_logger.error(f"Could not find pre-selection for the events in DataFrame columns: {df.columns}")

    # Applies analysis cuts on the surviving LEE candidate events
    # skipping the Evis cut as it is included in the reco efficiencies provided by the collaboration
    w_selection = w_preselection  # apply_final_LEEselection(Evis, costheta, w_preselection, exp=exp)

    # Compute reconsructed neutrino energy
    reco_enu = fastmc.reco_EnuCCQE(Evis, costheta)

    FIDUCIAL_MASS_CORRECTION = 1 / 0.55  # Undo the 55%-efficiency fiducial mass cut already included in the reco effs
    w_selection = apply_reco_efficiencies(Evis, w_selection, exp=exp) * FIDUCIAL_MASS_CORRECTION

    ############################################################################
    # return reco observables of LEE
    df["reco_Enu"] = reco_enu
    df["reco_w"] = w_selection
    df["reco_Evis"] = Evis
    df["reco_theta_beam"] = np.arccos(costheta)
    df["reco_costheta_beam"] = costheta
    df["reco_eff"] = w_selection.sum() / w.sum()

    if clean_df:
        return df[df.reco_w > 0]
    else:
        return df


# Full analysis to get reconstructed energy spectrum for BNB experiments
def reco_pi0like_invmass(df, exp="miniboone", cut="circ1"):
    """compute_spectrum _summary_

    Parameters
    ----------
    df : pd.DatagFrame
        DarkNews events (preferrably after selecting events inside detector)
    exp : str, optional
        what experiment to use, by default 'miniboone' but can also be 'microboone'
    cut : str, optional
        what kind of "mis-identificatin" cut pre-selection to be used:
            for photons:
                'photon' assumes this is a photon and therefore always a single shower
            for lepton or photon pairs:
                circ0, circ1, diag -- corresponding to Kelly&Kopp criteria
                invmass -- corresponding to invmass criteria from Patterson
    Returns
    -------
    pd.DatagFrame
        A new dataframe with additional columns containing weights of the selected events.
    """

    df = df.copy(deep=True)

    # Initial weigths
    w = df["w_event_rate"].values  # typically already selected for fiducial volume

    if "P_decay_photon" in df.columns:
        toy_logger.error("pi0like_invmass not defined for single photon events")

    else:
        # Smear e+ and e-
        pep = fastmc.smear_samples(df["P_decay_ell_plus"], exp=exp)
        pem = fastmc.smear_samples(df["P_decay_ell_minus"], exp=exp)
        Evis = pep[:, 0] + pem[:, 0]

        # Evis, theta_beam, w, eff_s = signal_events(pep, pem, Delta_costheta, costhetaep, costhetaem, w, threshold=ep.THRESHOLD[exp], angle_max=ep.ANGLE_MAX[exp], event_type=event_type)
        w_preselection = apply_pre_selection(w, pep, pem, kind="pi0like", cut=cut)

    # No efficiency in this case?
    w_selection = w_preselection

    ############################################################################
    # return reco observables of LEE
    df["reco_mgg"] = Cfv.inv_mass(pep + pem, pep + pem)
    df["reco_w"] = w_selection
    df["reco_Evis"] = Evis
    df["reco_eff"] = w_selection.sum() / w.sum()

    return df


def apply_pre_selection(w, pep, pem, kind="nueCCQElike", cut="circ1"):
    """apply_pre_selection pre-select events based on single- or two-rings criteria

    Parameters
    ----------
    w : np.array
        weights
    pep : np.array
        positron or photon momenta w/ shape (Nevents, 4)
    pem : np.array
        electron or photon momenta w/ shape (Nevents, 4)
    kind : str, optional
        nueCCQElike (1 rings) or pi0like (2 rings), by default 'nueCCQElike'.
    cut : str, optional
        criteria to be used. By default 'circ1'. Other options are:
        *   'circ1' picks events based on Kelly&Kopp criteria: r = sqrt( (1 - CosTheta)^2/4 + (1 - Emax/Etot)^2 )
        *   'circ0' picks events based on Kelly&Kopp criteria: r = sqrt( (1 + CosTheta)^2/4 + (Emax/Etot)^2 )
        *   'diag' picks events based on Kelly&Kopp criteria: r = 1 - 1/2*((1 + CosTheta)/2 + Emax/Etot)
        *   'invmass' picks events based on overlapping = (theta < 13) and asymmetric = (Evis < 30 MeV). Cuts on true invmass from Patterson.
        *   'byhand' picks events based on overlapping = (theta < 13) and asymmetric = (Evis < 30 MeV). No mee cut implemented.

    Returns
    -------
    np.array
        Reweighted weights with selection criteria applied.
    """
    # some useful kinematics
    emax = np.where(pep[:, 0] >= pem[:, 0], pep[:, 0], pem[:, 0])
    Evis = pep[:, 0] + pem[:, 0]
    Delta_costheta = Cfv.get_cos_opening_angle(pem, pep)

    # apply pre-selection cuts
    if cut == "circ1":
        r = np.sqrt((1 - Delta_costheta) ** 2 / 4 + (1 - emax / Evis) ** 2)
        condition = r < fastmc.get_r_cut_func(cut=cut)(Evis)

    elif cut == "circ0":
        r = np.sqrt((1 + Delta_costheta) ** 2 / 4 + (emax / Evis) ** 2)
        condition = r > fastmc.get_r_cut_func(cut=cut)(Evis)

    elif cut == "diag":
        r = 1 - 1 / 2 * ((1 + Delta_costheta) / 2 + emax / Evis)
        condition = r < fastmc.get_r_cut_func(cut=cut)(Evis)

    elif cut == "invmass":
        mee = Cfv.inv_mass(pep + pem, pep + pem)
        ovl = Delta_costheta < np.cos(13 * np.pi / 180)
        asy = ((pep[:, 0] < 0.03) & (pem[:, 0] > 0.03)) | ((pem[:, 0] < 0.03) & (pep[:, 0] > 0.03))
        OldCut = ovl | asy
        condition = (mee < fastmc.mee_cut_func(Evis)) * OldCut

    elif cut == "byhand":
        ovl = Delta_costheta > np.cos(13 * np.pi / 180)
        asy = ((pep[:, 0] < 0.03) & (pem[:, 0] > 0.03)) | ((pem[:, 0] < 0.03) & (pep[:, 0] > 0.03))
        condition = ovl | asy

    else:
        toy_logger.error(f"Could not identify pre-selection cuts for kind {kind}")

    if kind == "nueCCQElike":
        return w * condition
    elif kind == "pi0like":
        return w * (~condition)
    else:
        toy_logger.error(f"Could not identify pre-selection cuts for kind {kind}")
        raise ValueError(f"Could not identify pre-selection cuts for kind {kind}")


def apply_final_LEEselection(Evis, costheta, w, exp="miniboone"):
    # Cuts
    in_energy_range = (Evis > fastmc.EVIS_MIN[exp]) & (Evis < fastmc.EVIS_MAX[exp])
    # there could be an angular cut, but miniboone's acceptance is assumed to be 4*pi
    return w * in_energy_range


# def signal_events(pep, pem, cosdelta_ee, costheta_ep, costheta_em, w, threshold = 0.03, angle_max = 13.0, event_type = 'both', apply_invariant_mass_cut=False):
#     """signal_events
#         This takes the events and asks for them to be either overlapping or asymmetric

#     Parameters
#     ----------
#     pep : numpy.ndarray[ndim=2]
#         four momenta of the positive lepton
#     pem : numpy.ndarray[ndim=2]
#         four momenta of the negative lepton
#     cosdelta_ee : numpy.ndarray[ndim=1]
#         cosine of the opening angle between lepton
#     costheta_ep : numpy.ndarray[ndim=1]
#         costheta of the positive lepton
#     costheta_em : numpy.ndarray[ndim=1]
#         costheta of the negative lepton
#     w : numpy.ndarray[ndim=1]
#         event weights

#     threshold : float, optional
#          how low energy does Esubleading need to be for event to be asymmetric, by default 0.03
#     angle_max : float, optional
#          how wide opening angle needs to be in order to be overlapping, by default 13.0
#     event_type : str, optional
#         what kind of "mis-identificatin" selection to be used:
#             'asymmetric' picks events where one of the leptons (independent of charge) is below a hard threshold
#             'overlapping' picks events where the two leptons are overlapping
#             'both' for *either* asymmetric or overlapping condition to be true
#             'separated' picks events where both leptons are above threshold and non-overlapping by default 'asymmetric'

#     Returns
#     -------
#     set of np.ndarrays
#         Depending on the final event type, a list of energies and angles
#     """

#     ################### PROCESS KINEMATICS ##################
#     # electron kinematics
#     Eep = pep[:,0]
#     Eem = pem[:,0]

#     # angle of separation between ee
#     theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

#     # two individual angles
#     theta_ep = np.arccos(costheta_ep)*180.0/np.pi
#     theta_em = np.arccos(costheta_em)*180.0/np.pi

#     # this is the angle of the combination of ee with the neutrino beam
#     costheta_comb = Cfv.get_cosTheta(pem + pep)
#     theta_comb = np.arccos(costheta_comb)*180.0/np.pi

#     ########################################
#     if apply_invariant_mass_cut:
#         mee = Cfv.inv_mass(pep, pem)
#         mee_cut = 0.03203 + 0.007417*(Eep + Eem) + 0.02738*(Eep + Eem)**2
#         inv_mass_cut = (mee < mee_cut)
#     else:
#         inv_mass_cut = np.full(len(w), True)

#     asym_p_filter = (Eem - const.m_e < threshold) & (Eep - const.m_e > threshold) & inv_mass_cut
#     asym_m_filter = (Eem - const.m_e > threshold) & (Eep - const.m_e < threshold) & inv_mass_cut
#     asym_filter = (asym_p_filter | asym_m_filter) & inv_mass_cut
#     ovl_filter = (Eep - const.m_e > threshold) & (Eem - const.m_e > threshold) & (theta_ee < angle_max) & inv_mass_cut
#     sep_filter = (Eep - const.m_e > threshold) & (Eem - const.m_e > threshold) & (theta_ee > angle_max) & inv_mass_cut
#     inv_filter = (Eep - const.m_e < threshold) & (Eem - const.m_e < threshold) & inv_mass_cut
#     both_filter = (asym_m_filter | asym_p_filter | ovl_filter)

#     w_asym = w[asym_m_filter | asym_p_filter]
#     w_ovl = w[ovl_filter]
#     w_sep = w[sep_filter]
#     w_inv = w[inv_filter]
#     w_tot = w.sum()

#     eff_asym	= w_asym.sum()/w_tot
#     eff_ovl		= w_ovl.sum()/w_tot
#     eff_sep		= w_sep.sum()/w_tot
#     eff_inv		= w_inv.sum()/w_tot

#     if event_type=='overlapping':

#         Evis = np.full_like(Eep, None)
#         theta_beam = np.full_like(Eep, None)

#         # visible energy
#         Evis[ovl_filter] = (Eep*ovl_filter + Eem*ovl_filter)[ovl_filter]

#         # angle to the beam
#         theta_beam[ovl_filter] = theta_comb[ovl_filter]

#         w[~ovl_filter] *= 0.0

#         return Evis, theta_beam, w, eff_ovl

#     elif event_type=='asymmetric':

#         Evis = np.full_like(Eep, None)
#         theta_beam = np.full_like(Eep, None)

#         # visible energy
#         Evis[asym_filter] = (Eep*asym_p_filter + Eem*asym_m_filter)[asym_filter]

#         # angle to the beam
#         theta_beam[asym_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter)[asym_filter]

#         w[~asym_filter] *= 0.0

#         return Evis, theta_beam, w, eff_asym

#     elif event_type=='both':

#         Evis = np.full_like(Eep, None)
#         theta_beam = np.full_like(Eep, None)

#         # visible energy
#         Evis[both_filter] = (Eep*asym_p_filter + Eem*asym_m_filter + (Eep+Eem)*ovl_filter)[both_filter]
#         # angle to the beam
#         theta_beam[both_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter + theta_comb*ovl_filter)[both_filter]

#         w[~both_filter] *= 0.0

#         return Evis, theta_beam, w, eff_ovl+eff_asym

#     elif event_type=='separated':

#         Eplus = np.full_like(Eep, None)
#         Eminus = np.full_like(Eep, None)
#         theta_beam_plus = np.full_like(Eep, None)
#         theta_beam_minus = np.full_like(Eep, None)

#         # visible energy
#         Eplus[sep_filter] = Eep[sep_filter]
#         Eminus[sep_filter] = Eem[sep_filter]

#         # angle to the beam
#         theta_beam_plus[sep_filter] = theta_ep[sep_filter]
#         theta_beam_minus[sep_filter] = theta_em[sep_filter]
#         theta_ee[sep_filter] = theta_ee[sep_filter]

#         return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_ee, w_sep, eff_sep

#     elif event_type=='invisible':

#         Eplus = np.full_like(Eep, None)
#         Eminus = np.full_like(Eep, None)
#         theta_beam_plus = np.full_like(Eep, None)
#         theta_beam_minus = np.full_like(Eep, None)

#         # visible energy
#         Eplus[inv_filter] = Eep[inv_filter]
#         Eminus[inv_filter] = Eem[inv_filter]

#         # angle to the beam
#         theta_beam_plus[inv_filter] = theta_ep[inv_filter]
#         theta_beam_minus[inv_filter] = theta_em[inv_filter]
#         theta_ee[inv_filter] = theta_ee[inv_filter]

#         return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_ee, w_inv, eff_inv

#     else:
#         print(f"Error! Could not find event type {event_type}.")
#         return
