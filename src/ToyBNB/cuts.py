import numpy as np
import pandas as pd
from scipy.stats import exponnorm
from scipy.interpolate import interp1d
from importlib.resources import open_text

from ToyBNB import exp_params as ep
from DarkNews import Cfourvec as Cfv


def mee_cut_func(Evis):
    """mee_cut_func 
        cut function for invariant mass cut. 
        m_ee(Evis = Ee+ + Ee-)
    
    Returns
    -------
    lambda function of Evis
    """
    return 0.03203 + 0.007417*(Evis) + 0.02738*(Evis)**2


def get_r_cut_func(kind = 'circ1'):
    """get_r_cut_func get interpolation of the r cut from 
        Kelly_Kopp_2022 -- https://arxiv.org/abs/2210.08021

    Parameters
    ----------
    kind : str, optional
        what kind of r parameter to take, by default 'circ1'.
        Options are 'circ0', 'circ1', and 'diag'.

    Returns
    -------
    _type_
        _description_
    """
    cut_data = np.genfromtxt(open_text("ToyAnalysis.include.Kelly_Kopp_2022", 'pi0_circle_cuts.dat'), unpack=True)

    bin_l = cut_data[0]
    bin_r = cut_data[1]
    bin_c = bin_l + (bin_r-bin_l)/2.

    kinds = {'circ0': cut_data[6], 'circ1': cut_data[4], 'diag': cut_data[2]}
    
    try:
        cut = kinds[kind]
    except KeyError:
        print(f"Error: {kind} is not a valid cut kind. Options are 'circ0', 'circ1', and 'diag'. Defaulting to circ0.")
        cut = kinds['circ0']

    return interp1d(bin_c, cut, kind='linear', fill_value=None, bounds_error=False)


# COMPUTE EFFICIENCY
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
    t_reco = Cfv.random_normal(np.zeros(nentries), sigma*np.ones(nentries))
    
    # sample random polar angle -- this assumes the detector is fully uniform! 
    phi_reco = Cfv.random_generator(nentries, 0, 2*np.pi)
    
    # cos and sin
    cost_reco = np.cos(t_reco)
    sint_reco = np.sin(t_reco)
    cosphi_reco = np.cos(phi_reco)

    # now rotate to the direction of travel of the original particle
    reco_cost = sint * sint_reco * cosphi_reco + cost * cost_reco

    return reco_cost

# SMEARING
def smear_samples(samples, mass = 0.0, exp = 'miniboone'):

    # compute kinematic quantities
    if type(samples) == pd.DataFrame:
        samples = samples.to_numpy()

    E = samples[:,0]

    nentries = len(E)
    
    # compute sigmas
    sigma_E = ep.STOCHASTIC[exp]*np.sqrt(E) + ep.NOISE[exp]
    sigma_angle = ep.ANGULAR[exp]

    # compute kinetic energy and spherical angles
    T = E - mass
    ctheta = Cfv.get_cosTheta(samples)
    phi = np.arctan2(samples[:, 2],samples[:, 1])

    if exp=='miniboone':

        # compute smeared quantities
        T = Cfv.random_normal(T, sigma_E)
        ctheta = gauss_smear_angle(ctheta, sigma_angle*np.ones(nentries))
        phi = Cfv.random_normal(phi, sigma_angle*np.ones(nentries))

    elif exp=='microboone':

        #apply exponentially modified gaussian with exponential rate lambda = 1/K = 1 --> K=1
        K = 1
        T = exponnorm.rvs(K, loc = T, scale = sigma_E)
        ctheta = np.cos(exponnorm.rvs(K, loc = np.arccos(ctheta), scale = sigma_angle*np.ones(nentries)))
        phi = exponnorm.rvs(K, loc = phi, scale = sigma_angle*np.ones(nentries))

    T[T < 0] = 0# force smearing to be positive for T

    E = T + mass
    P = np.sqrt(E**2 - mass**2)
    stheta = np.sqrt(1 - ctheta**2)
    
    # put data in an array and then in a DataFrame
    smeared = np.array([E, P * stheta * np.cos(phi), P * stheta * np.sin(phi), P * ctheta]).T
    
    return smeared