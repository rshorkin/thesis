import math


# This file contains all the cuts for W boson analysis
# If cut returns True the event is ignored
def calc_theta(lep_eta):
    return 2 * math.atan(math.exp(-lep_eta))


# high missing Et
def cut_met_et(met_et):
    return not (met_et > 30000)


# Triggered by muon or electron
def cut_trig(trigE, trigM):
    return not (trigE or trigM)


# lepton is tight
def cut_tight(lep_isTightID):
    return not lep_isTightID


# isolation and high pt to remove multijet
def cut_multijet(lep_pt, lep_ptcone30, lep_etcone20):
    return not (lep_pt > 35000 and ((lep_ptcone30 / lep_pt) < 0.1) and ((lep_etcone20 / lep_pt) < 0.1))


# electron selection. exclude candidates in the transition region between the barrel and endcap em-calorimeters
def cut_e_fiducial(lep_type, lep_eta):
    return not (lep_type == 11 and abs(lep_eta) < 2.47 and (abs(lep_eta) < 1.37 or abs(lep_eta) > 1.52))


# longitudinal cuts for electrons
def cut_e_long(lep_type, lep_trackd0pvunbiased, lep_tracksigd0pvunbiased):
    return not (lep_type == 11 and lep_trackd0pvunbiased / lep_tracksigd0pvunbiased <= 5)


def cut_e_long_impact(lep_type, lep_z0, lep_eta):
    lep_theta = calc_theta(lep_eta)
    return not (lep_type == 11 and abs(lep_z0) * math.sin(lep_theta) <= 0.5)


# muon selection. exclude candidates in the transition region between the barrel and endcap em-calorimeters
def cut_mu_fiducial(lep_type, lep_eta):
    return not (lep_type == 13 and abs(lep_eta) < 2.5)


# longitudinal cuts for muons
def cut_mu_long(lep_type, lep_trackd0pvunbiased, lep_tracksigd0pvunbiased):
    return not (lep_type == 13 and (lep_trackd0pvunbiased / lep_tracksigd0pvunbiased) <= 3)


def cut_mu_long_impact(lep_type, lep_z0, lep_eta):
    lep_theta = calc_theta(lep_eta)
    return not (lep_type == 13 and abs(lep_z0) * math.sin(lep_theta) <= 0.5)
