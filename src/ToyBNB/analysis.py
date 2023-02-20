import numpy as np
from scipy.interpolate import interp1d

from ToyBNB import cuts
from ToyBNB import exp_params as ep

from DarkNews import const
from DarkNews import Cfourvec as Cfv

# COMPUTE SPECTRUM FOR ANY EXPERIMENT
def compute_spectrum(df, exp='miniboone', event_type='asymmetric'):
    """compute_spectrum _summary_

    Parameters
    ----------
    df : pd.DatagFrame
        DarkNews events (preferrably after selecting events inside detector)
    exp : str, optional
        what experiment to use, by default 'miniboone' but can also be 'microboone'
    event_type : str, optional
        what kind of "mis-identificatin" selection to be used:
            for photons:
                'photon' assumes this is a photon and therefore always a single shower
            for lepton pairs:
                'asymmetric' picks events where one of the letpons (independent of charge) is below a hard threshold
                'overlapping' picks events where the two leptons are overlapping
                'both' for *either* asymmetric or overlapping condition to be true
                'separated' picks events where both letpons are above threshold and non-overlapping by default 'asymmetric'

    Returns
    -------
    pd.DatagFrame
        A new dataframe with additional columns containing weights of the selected events.
    """

    df = df.copy(deep=True)
    # Initial weigths
    w = df['w_event_rate'].values # typically already selected for fiducial volume

    if event_type=='photon':
        # Smear photon
        pgamma = cuts.smear_samples(df['P_decay_photon'], 0.0, exp=exp)

        # compute some reco kinematical variables from smeared electrons
        costhetagamma = Cfv.get_cosTheta(pgamma)
        Evis = pgamma[0]
        theta_beam = np.arccos(costhetagamma)*180.0/np.pi
    else:
        # Smear e+ and e-
        pep = cuts.smear_samples(df['P_decay_ell_plus'], const.m_e, exp=exp)
        pem = cuts.smear_samples(df['P_decay_ell_minus'], const.m_e, exp=exp)

        # compute some reco kinematical variables from smeared electrons
        costhetaep = Cfv.get_cosTheta(pep)
        costhetaem = Cfv.get_cosTheta(pem)
        Delta_costheta = Cfv.get_cos_opening_angle(pem,pep)

        Evis, theta_beam, w, eff_s = signal_events(pep, pem, Delta_costheta, costhetaep, costhetaem, w, threshold=ep.THRESHOLD[exp], angle_max=ep.ANGLE_MAX[exp], event_type=event_type)

    ############################################################################
    # Applies analysis cuts on the surviving LEE candidate events
    Evis2, theta_beam, w2, eff_c = expcuts(Evis, theta_beam, w, exp=exp, event_type=event_type)

    ############################################################################
    # Compute reconsructed neutrino energy
    # this assumes quasi-elastic scattering to mimmick MiniBooNE's assumption that the underlying events are nueCC.
    df['reco_Enu'] = const.m_proton * (Evis) / ( const.m_proton - (Evis)*(1.0 - np.cos(theta_beam)))

    eff_final, w2 = get_efficiencies(df['reco_Enu'], Evis, w, w2, exp=exp)

    ############################################################################
    # return reco observables of LEE -- regime is still a true quantity...
    df['reco_w'] = w2 * (6.1/5.)**3 # correcting for the fiducial volume cut already in MiniBooNE's efficiencies
    df['reco_Evis'] = Evis2
    df['reco_theta_beam'] = theta_beam
    df['reco_costheta_beam'] = np.cos(theta_beam*np.pi/180)
    df['reco_eff'] = eff_final

    return df


def get_efficiencies(reco_Enu, Evis, w, w2, exp='miniboone'):

    if exp=='miniboone':
        ###########################################################################
        # Now, a trick to get the approximate PMT and particle ID efficiencies
        # I am basically subtracting the event selection efficiency from the overall MiniBooNE-provided efficiency
        eff = np.array([0.0,0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102,0.095,0.089,0.082,0.073,0.067,0.052,0.026])
        enu = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5,1.7,1.9,2.1])
        enu_c =  enu[:-1]+(enu[1:] - enu[:-1])/2
        eff_func = interp1d(enu_c, eff, fill_value=(eff[0],eff[-1]), bounds_error=False, kind='nearest')
        wsum = w.sum()
        
        if wsum==0:
            eff_miniboone = (eff_func(reco_Enu)*w).sum()
        else:
            eff_miniboone = (eff_func(reco_Enu)*w).sum()/w.sum()

        ############################################################################
        # Now, apply efficiency as a function of energy to event weights
        w2 = eff_func(reco_Enu)*w2

        return eff_miniboone, w2

    elif exp=='microboone':
        ############################################################################
        # Total nueCCQE efficiency -- using efficiencies provided by MicroBooNE
        E_numi_eff_edge = np.array([0,2.97415218e-1,4.70796471e-1,7.01196105e-1,9.88922361e-1,1.43183725e+0,3.00810009e+0,6.00428361e+0])
        nueCCQE_numi_eff_edge = np.array([6.13255034e-2,1.44127517e-1,2.12332215e-1,2.64681208e-1,2.76761745e-1,2.97902685e-1,2.57885906e-1,2.60151007e-1])
        E_numi_eff = (E_numi_eff_edge[1:] - E_numi_eff_edge[:-1])/2 + E_numi_eff_edge[:-1]
        nueCCQE_numi_eff = nueCCQE_numi_eff_edge[:-1]
        nueCCQE_numi_eff_func = interp1d(E_numi_eff,nueCCQE_numi_eff,fill_value=(nueCCQE_numi_eff[0],nueCCQE_numi_eff[-1]),bounds_error=False,kind='nearest')

        muB_eff = (w*nueCCQE_numi_eff_func(Evis)).sum()/w.sum()

        w2 = nueCCQE_numi_eff_func(reco_Enu)*w2

        return muB_eff, w2

def signal_events(pep, pem, cosdelta_ee, costheta_ep, costheta_em, w, threshold = 0.03, angle_max = 13.0, event_type = 'both', apply_invariant_mass_cut=False):
    """signal_events _summary_
# This takes the events and asks for them to be either overlapping or asymmetric
        # THRESHOLD --
        # ANGLE_MAX --
    Parameters
    ----------
    pep : numpy.ndarray[ndim=2]
        four momenta of the positive lepton
    pem : numpy.ndarray[ndim=2]
        four momenta of the negative lepton
    cosdelta_ee : numpy.ndarray[ndim=1]
        cosine of the opening angle between lepton
    costheta_ep : numpy.ndarray[ndim=1]
        costheta of the positive lepton
    costheta_em : numpy.ndarray[ndim=1]
        costheta of the negative lepton
    w : numpy.ndarray[ndim=1]
        event weights
    
    threshold : float, optional
         how low energy does Esubleading need to be for event to be asymmetric, by default 0.03
    angle_max : float, optional
         how wide opening angle needs to be in order to be overlapping, by default 13.0
    event_type : str, optional
        what kind of "mis-identificatin" selection to be used:
            'asymmetric' picks events where one of the letpons (independent of charge) is below a hard threshold
            'overlapping' picks events where the two leptons are overlapping
            'both' for *either* asymmetric or overlapping condition to be true
            'separated' picks events where both letpons are above threshold and non-overlapping by default 'asymmetric'

    Returns
    -------
    set of np.ndarrays
        Depending on the final event type, a list of energies and angles
    """

    ################### PROCESS KINEMATICS ##################
    # electron kinematics
    Eep = pep[:,0]
    Eem = pem[:,0]

    # angle of separation between ee
    theta_ee = np.arccos(cosdelta_ee)*180.0/np.pi

    # two individual angles
    theta_ep = np.arccos(costheta_ep)*180.0/np.pi
    theta_em = np.arccos(costheta_em)*180.0/np.pi

    # this is the angle of the combination of ee with the neutrino beam
    costheta_comb = Cfv.get_cosTheta(pem + pep)
    theta_comb = np.arccos(costheta_comb)*180.0/np.pi

    ########################################
    if apply_invariant_mass_cut:
        mee = Cfv.inv_mass(pep, pem)
        mee_cut = 0.03203 + 0.007417*(Eep + Eem) + 0.02738*(Eep + Eem)**2
        inv_mass_cut = (mee < mee_cut)
    else:
        inv_mass_cut = np.full(len(w), True)

    asym_p_filter = (Eem - const.m_e < threshold) & (Eep - const.m_e > threshold) & inv_mass_cut
    asym_m_filter = (Eem - const.m_e > threshold) & (Eep - const.m_e < threshold) & inv_mass_cut
    asym_filter = (asym_p_filter | asym_m_filter) & inv_mass_cut
    ovl_filter = (Eep - const.m_e > threshold) & (Eem - const.m_e > threshold) & (theta_ee < angle_max) & inv_mass_cut
    sep_filter = (Eep - const.m_e > threshold) & (Eem - const.m_e > threshold) & (theta_ee > angle_max) & inv_mass_cut
    inv_filter = (Eep - const.m_e < threshold) & (Eem - const.m_e < threshold) & inv_mass_cut
    both_filter = (asym_m_filter | asym_p_filter | ovl_filter)

    w_asym = w[asym_m_filter | asym_p_filter]
    w_ovl = w[ovl_filter]
    w_sep = w[sep_filter]
    w_inv = w[inv_filter]
    w_tot = w.sum()

    eff_asym	= w_asym.sum()/w_tot	
    eff_ovl		= w_ovl.sum()/w_tot	
    eff_sep		= w_sep.sum()/w_tot	
    eff_inv		= w_inv.sum()/w_tot	

    if event_type=='overlapping':

        Evis = np.full_like(Eep, None)
        theta_beam = np.full_like(Eep, None)

        # visible energy
        Evis[ovl_filter] = (Eep*ovl_filter + Eem*ovl_filter)[ovl_filter]

        # angle to the beam
        theta_beam[ovl_filter] = theta_comb[ovl_filter]

        w[~ovl_filter] *= 0.0

        return Evis, theta_beam, w, eff_ovl

    elif event_type=='asymmetric':

        Evis = np.full_like(Eep, None)
        theta_beam = np.full_like(Eep, None)

        # visible energy
        Evis[asym_filter] = (Eep*asym_p_filter + Eem*asym_m_filter)[asym_filter]

        # angle to the beam
        theta_beam[asym_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter)[asym_filter]

        w[~asym_filter] *= 0.0

        return Evis, theta_beam, w, eff_asym

    elif event_type=='both':

        Evis = np.full_like(Eep, None)
        theta_beam = np.full_like(Eep, None)
        
        # visible energy
        Evis[both_filter] = (Eep*asym_p_filter + Eem*asym_m_filter + (Eep+Eem)*ovl_filter)[both_filter]
        # angle to the beam
        theta_beam[both_filter] = (theta_ep*asym_p_filter + theta_em*asym_m_filter + theta_comb*ovl_filter)[both_filter]

        w[~both_filter] *= 0.0

        return Evis, theta_beam, w, eff_ovl+eff_asym

    elif event_type=='separated':

        Eplus = np.full_like(Eep, None)
        Eminus = np.full_like(Eep, None)
        theta_beam_plus = np.full_like(Eep, None)
        theta_beam_minus = np.full_like(Eep, None)

        # visible energy
        Eplus[sep_filter] = Eep[sep_filter]
        Eminus[sep_filter] = Eem[sep_filter]

        # angle to the beam
        theta_beam_plus[sep_filter] = theta_ep[sep_filter]
        theta_beam_minus[sep_filter] = theta_em[sep_filter]
        theta_ee[sep_filter] = theta_ee[sep_filter]

        return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_ee, w_sep, eff_sep
        
    elif event_type=='invisible':

        Eplus = np.full_like(Eep, None)
        Eminus = np.full_like(Eep, None)
        theta_beam_plus = np.full_like(Eep, None)
        theta_beam_minus = np.full_like(Eep, None)

        # visible energy
        Eplus[inv_filter] = Eep[inv_filter]
        Eminus[inv_filter] = Eem[inv_filter]

        # angle to the beam
        theta_beam_plus[inv_filter] = theta_ep[inv_filter]
        theta_beam_minus[inv_filter] = theta_em[inv_filter]
        theta_ee[inv_filter] = theta_ee[inv_filter]

        return Eplus, Eminus, theta_beam_plus, theta_beam_minus, theta_ee, w_inv, eff_inv

    else:
        print(f"Error! Could not find event type {event_type}.")
        return


def expcuts(Evis, theta_beam, w, exp='miniboone', event_type='overlapping'):
    if exp == 'miniboone':
        return MB_expcuts(Evis, theta_beam, w)
    elif exp == 'microboone':
        return muB_expcuts(Evis, theta_beam, w, event_type=event_type)


def MB_expcuts(Evis, theta, weights):

	## Experimental cuts
	Pe  = np.sqrt(Evis**2 - const.m_e**2)
	Enu = (const.m_neutron*Evis - 0.5*const.m_e**2)/(const.m_neutron - Evis + Pe*np.cos(theta*np.pi/180.0))
	
	# Cuts
	in_energy_range = (Evis > ep.EVIS_MIN['miniboone']) & (Evis < ep.EVIS_MAX['miniboone'])
	# there could be an angular cut, but miniboone's acceptance is assumed to be 4*pi
	final_selected = in_energy_range

	eff = weights[final_selected].sum()/weights.sum()

	Evis[~final_selected] = None
	theta[~final_selected] = None
	weights[~final_selected] *= 0.0

	return Evis, theta, weights, eff

def muB_expcuts(Evis, theta, weights, event_type='overlapping'):

	# Cuts
	in_energy_range = (Evis > ep.EVIS_MIN['microboone']) & (Evis < ep.EVIS_MAX['microboone'])
	
	final_selected = in_energy_range

	weights_fs = weights * final_selected

	eff = weights_fs.sum()/weights.sum()

	return Evis, theta, weights_fs, eff
