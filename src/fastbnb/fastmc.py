import numpy as np
import pandas as pd
from scipy.stats import exponnorm
from scipy.interpolate import interp1d
from importlib.resources import open_text

from DarkNews import Cfourvec as Cfv
from DarkNews import const

# CONSTANTS
THRESHOLD = {"miniboone": 0.03, "microboone": 0.01}
ANGLE_MAX = {"miniboone": 13.0, "microboone": 5.0}
ENU_MIN = {"miniboone": 0.14, "microboone": 0.100}
ENU_MAX = {"miniboone": 1.5, "microboone": 3.0}
EVIS_MIN = {
    "miniboone": 0.100,  # to apply eff form 2012 data release needs to start from 100 MeV
    "microboone": 0.100,
}
EVIS_MAX = {"miniboone": 3.0, "microboone": 3.0}

# Resolutions based on https://iopscience.iop.org/article/10.1088/1742-6596/120/5/052003/pdf
ANGULAR = {"miniboone": 3.0 * np.pi / 180.0, "microboone": 2.0 * np.pi / 180.0}
STOCHASTIC = {"miniboone": 0.12, "microboone": 0.12}
NOISE = {"miniboone": 0.01, "microboone": 0.01}

Q2 = {"miniboone": 1.0e10, "microboone": None}
ANALYSIS_TH = {"miniboone": 0.02, "microboone": None}


def reco_EnuCCQE(Evis, costheta):
    # this assumes quasi-elastic scattering to mimmick MiniBooNE's assumption that the underlying events are nueCC.
    num = const.m_neutron * Evis - (const.m_neutron**2 + const.m_e**2 - const.m_proton**2) / 2
    physical = Evis**2 - const.m_e**2 > 0
    den = np.full(len(Evis), np.nan)
    den[physical] = const.m_neutron - Evis[physical] + np.sqrt(Evis[physical] ** 2 - const.m_e**2) * costheta[physical]
    Enu = num / den
    return Enu


def mee_cut_func(Evis):
    """mee_cut_func cut function for invariant mass cut.

            m_ee(Evis = Ee+ + Ee-)

    Returns
    -------
    lambda function of Evis
    """
    return 0.03203 + 0.007417 * (Evis) + 0.02738 * (Evis) ** 2


def get_r_cut_func(cut="circ1", extrapolate=True, uncertainty_case="1cut100samples", version=""):
    """get_r_cut_func get interpolation of the r cut from
            Kelly and Kopp (2022) -- https://arxiv.org/abs/2210.08021

    Parameters
    ----------
    cut : str, optional
        what kind of r parameter to take, by default 'circ1'.
        Options are 'circ0', 'circ1', and 'diag'.
    extrapolate: bool, optional
        If True, extrapolate the r cut to the left and right of Evis range, as a constant.
        Default is True.

    Returns
    -------
    function
        interpolatation of the r cut function as a function of Evis.
    """
    cut_data = np.genfromtxt(
        open_text("fastbnb.include.pi0_tools", f"pi0_circle_cuts{version}.dat"),
        unpack=True,
    )

    bin_l = cut_data[0]
    bin_r = cut_data[1]
    bin_c = bin_l + (bin_r - bin_l) / 2.0
    access_indices = {"diag": 2, "circ0": 6, "circ1": 4}

    access_index = access_indices[cut]
    if uncertainty_case == "100cut1sample":
        access_index += 1
    elif uncertainty_case == "1cut100samples":
        pass
    else:
        print(f"Error: No uncertainty case {uncertainty_case}. Options are '100cut1sample' and '1cut100samples'. Defaulting to 1cut100samples.")
        pass

    cuts = cut_data[access_index]

    try:
        cut_data = cuts
    except KeyError:
        print(f"Error: No Rcut function for {cut} kind. Options are 'circ0', 'circ1', and 'diag'. Defaulting to circ0.")
        cut_data = cut_data[4]

    if extrapolate:
        return interp1d(
            bin_c,
            cut_data,
            kind="linear",
            fill_value=(cut_data[0], cut_data[-1]),
            bounds_error=False,
        )
    else:
        return interp1d(bin_c, cut_data, kind="linear", fill_value=None, bounds_error=False)


def efficiency(samples, weights, xmin, xmax):
    mask = samples >= xmin & samples <= xmax

    weights_detected = weights[mask]

    return weights.sum() / weights_detected.sum()


def gauss_smear_angle(cost, sigma):
    """smear_angle smears the angle of travel of a particle with a Gaussian resolution "sigma"

    Parameters
    ----------
    cost : np.array
        cosine of the angle of particle wrt to the z direction
    sigma : float or np.array
        width of the Gaussian resolution
    """
    nentries = len(cost)
    sint = np.sqrt(1 - cost**2)

    # sample random azimuthal angle around 0
    t_reco = Cfv.random_normal(np.zeros(nentries), sigma * np.ones(nentries))

    # sample random polar angle -- this assumes the detector is fully uniform!
    phi_reco = Cfv.random_generator(nentries, 0, 2 * np.pi)

    # cos and sin
    cost_reco = np.cos(t_reco)
    sint_reco = np.sin(t_reco)
    cosphi_reco = np.cos(phi_reco)

    # now rotate to the direction of travel of the original particle
    reco_cost = sint * sint_reco * cosphi_reco + cost * cost_reco

    return reco_cost


def smear_samples(samples, mass=const.m_e, exp="miniboone"):
    """smear_samples Reconstruct particles in detector with
    gaussian resolutions

    Parameters
    ----------
    samples : np.array
        events to be smeared
    mass : float, optional
        particle's mass, typically an electron, by default electron's mass
    exp : str, optional
        name of the experiment, by default 'miniboone'

    Returns
    -------
    np.array
        The smeared events in the same format as inpute samples
    """
    # compute kinematic quantities
    if type(samples) == pd.DataFrame:
        samples = samples.to_numpy()

    E = samples[:, 0]

    nentries = len(E)

    #
    sigma_E = STOCHASTIC[exp] * np.sqrt(E) + NOISE[exp]
    # fit to Patterson's thesis
    sigma_angle = (np.log(1 / E) + 2.7) * np.pi / 180.0

    # compute kinetic energy and spherical angles
    T = E - mass
    ctheta = Cfv.get_cosTheta(samples)
    phi = np.arctan2(samples[:, 2], samples[:, 1])

    if exp == "miniboone":
        # compute smeared quantities
        T = Cfv.random_normal(T, sigma_E)
        ctheta = gauss_smear_angle(ctheta, sigma_angle)

    elif exp == "microboone":
        # apply exponentially modified gaussian with exponential rate lambda = 1/K = 1 --> K=1
        K = 1
        T = exponnorm.rvs(K, loc=T, scale=sigma_E)
        ctheta = np.cos(exponnorm.rvs(K, loc=np.arccos(ctheta), scale=sigma_angle * np.ones(nentries)))

    # Unfortunately, a Gaussian energy smearing can give non-physical results.
    T[T < 0] = 0  # force smearing to be positive for T

    E = T + mass
    P = np.sqrt(E**2 - mass**2)
    stheta = np.sqrt(1 - ctheta**2)

    # put data in an array and then in a DataFrame
    smeared = np.array([E, P * stheta * np.cos(phi), P * stheta * np.sin(phi), P * ctheta]).T

    return smeared
