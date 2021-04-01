etmiss = {"bin_width": 5,
          "numbins": 32,
          "xmin": 40,
          "xmax": 200,
          "xlabel": "$E^{miss}_T$, [GeV]",
          "xvariable": "met_et",
          "title": "Missing Transverse Momentum",
          }

mtw = {"bin_width": 5,
       "numbins": 24,
       "xmin": 60,
       "xmax": 180,
       "xlabel": "$M^W_T$, [GeV]",
       "xvariable": "mtw",
       "title": "Transverse Mass"}

mtw_enu = {"bin_width": 5,
           "numbins": 24,
           "xmin": 60,
           "xmax": 180,
           "xlabel": "$M^{W\\rightarrow e\\nu}_T$, [GeV]",
           "xvariable": "mtw_enu",
           "title": "W Transverse Mass (electrons only)"}

mtw_munu = {"bin_width": 5,
            "numbins": 24,
            "xmin": 60,
            "xmax": 180,
            "xlabel": "$M^{W\\rightarrow \mu\\nu}_T$, [GeV]",
            "xvariable": "mtw_munu",
            "title": "W Transverse Mass (muons only)"}

lep_E = {"bin_width": 10,
         "numbins": 27,
         "xmin": 30,
         "xmax": 300,
         "xlabel": "$E^{lep}$, [GeV]",
         "xvariable": "lep_E",
         "title": "Lepton Energy"}

lep_pt = {"bin_width": 5,
          "numbins": 33,
          "xmin": 35,
          "xmax": 200,
          "xlabel": "$p_T^{lep}$, [GeV]",
          "xvariable": "lep_pt",
          "title": "Lepton Transverse Momentum"}

lep_eta = {"bin_width": 0.25,
           "numbins": 20,
           "xmin": -2.5,
           "xmax": 2.5,
           "xlabel": "$\eta^{lep}$",
           "xvariable": "lep_eta",
           "title": "Lepton Pseudorapidity"}

jet_n = {"bin_width": 1,
         "numbins": 7,
         "xmin": -0.5,
         "xmax": 6.5,
         "xlabel": "$N_{jets}$",
         "xvariable": "jet_n",
         "title": "Number of Jets"}

lep_ch = {"bin_width": 0.5,
          "numbins": 5,
          "xmin": -1.25,
          "xmax": 1.25,
          "xlabel": "$Q^{lep}$",
          "xvariable": "lep_charge",
          "title": "Lepton Charge"}

lep_type = {"bin_width": 1,
            "numbins": 3,
            "xmin": 10.5,
            "xmax": 13.5,
            "xlabel": "$|PDG ID|^{lep}$",
            "xvariable": "lep_type",
            "title": "Lepton Absolute PDG ID"}

lep_phi = {"bin_width": .2,
           "numbins": 32,
           "xmin": -3.2,
           "xmax": 3.2,
           "xlabel": "$\phi^{lep}$",
           "xvariable": "lep_phi",
           "title": "Lepton Azimuthal Angle"}

lead_jet_phi = {"bin_width": .2,
                "numbins": 32,
                "xmin": -3.2,
                "xmax": 3.2,
                "xlabel": "$\phi^{jet}$",
                "xvariable": "lead_jet_phi",
                "title": "Leading Jet Azimuthal Angle"}

phi_diff = {"bin_width": .2,
            "numbins": 64,
            "xmin": -6.4,
            "xmax": 6.4,
            "xlabel": "$\phi^{lep} - \phi^{jet}$",
            "xvariable": "phi_diff",
            "title": "Difference in $\phi$ between Lepton and Lead. Jet"}

abs_phi_diff = {"bin_width": .2,
                "numbins": 32,
                "xmin": 0.,
                "xmax": 6.4,
                "xlabel": "|$\phi^{lep} - \phi^{jet}$|",
                "xvariable": "abs_phi_diff",
                "title": "Abs. Difference in $\phi$ between Lepton and Lead. Jet"}

lead_jet_pt = {"bin_width": 5,
               "numbins": 20,
               "xmin": 20,
               "xmax": 120,
               "xlabel": "$p_T^{jet}$, [GeV]",
               "xvariable": "lead_jet_pt",
               "title": "Leading Jet Transverse Momentum"}

lead_jet_eta = {"bin_width": 0.25,
                "numbins": 20,
                "xmin": -2.5,
                "xmax": 2.5,
                "xlabel": "$\eta^{jet}$",
                "xvariable": "lead_jet_eta",
                "title": "Leading Jet Pseudorapidity"}

# Asymmetry stuff

lep_abs_eta = {"bin_width": 0.25,
               "numbins": 10,
               "xmin": 0.,
               "xmax": 2.5,
               "xlabel": "$|\eta|^{lep}$",
               "xvariable": "lep_eta",
               "title": "Lepton Absolute Pseudorapidity"}

lep_asym = {"bin_width": 0.25,
            "numbins": 10,
            "xmin": 0.,
            "xmax": 2.5,
            "xlabel": "$|\eta|^{lep}$",
            "xvariable": "lep_asym",
            "title": "Lepton Charge Asymmetry"}

pos_ele_eta = lep_abs_eta.copy()
pos_ele_eta['xvariable'] = 'pos_ele_eta'
neg_ele_eta = lep_abs_eta.copy()
neg_ele_eta['xvariable'] = 'neg_ele_eta'
pos_mu_eta = lep_abs_eta.copy()
pos_mu_eta['xvariable'] = 'pos_mu_eta'
neg_mu_eta = lep_abs_eta.copy()
neg_mu_eta['xvariable'] = 'neg_mu_eta'

hist_dicts = {"met_et": etmiss, "mtw": mtw, "jet_n": jet_n, "lep_pt": lep_pt,
              "lep_eta": lep_eta, 'mtw_enu': mtw_enu, 'mtw_munu': mtw_munu, 'lep_E': lep_E,
              'lep_phi': lep_phi, 'lead_jet_eta': lead_jet_eta, 'lead_jet_pt': lead_jet_pt,
              'phi_diff': phi_diff, 'lead_jet_phi': lead_jet_phi, 'abs_phi_diff': abs_phi_diff,

              'pos_ele_eta': pos_ele_eta, 'neg_ele_eta': neg_ele_eta,
              'pos_mu_eta': pos_mu_eta, 'neg_mu_eta': neg_mu_eta
              }
