etmiss = { "bin_width": 5,
           "numbins":  32,
           "xmin":     40,
           "xmax":    200,
           "xlabel":  "$E^{miss}_T$, [GeV]",
           "xvariable": "met_et",
           "title": "Missing Transverse Momentum",
}

mtw   =  { "bin_width": 4,
           "numbins":  30,
           "xmin":     60,
           "xmax":    180,
           "xlabel": "$M^W_T$, [GeV]",
           "xvariable": "mtw",
           "title": "Transverse Mass"}

lep_pt = {"bin_width": 5,
           "numbins":  34,
           "xmin":     35,
           "xmax":     205,
           "xlabel": "$p_T^{lep}$, [GeV]",
           "xvariable": "lep_pt",
           "title": "Lepton Transverse Momentum"}

lep_eta = {"bin_width": 0.2,
           "numbins":  26,
           "xmin":     -2.6,
           "xmax":     2.6,
           "xlabel": "$\eta^{lep}$",
           "xvariable": "lep_eta",
           "title": "Lepton Pseudorapidity"}

jet_n =  {"bin_width": 1,
           "numbins":  6,
           "xmin":     0,
           "xmax":     6,
           "xlabel": "$N_{jets}$",
           "xvariable": "jet_n",
           "title": "Number of Jets"}

lep_ch  = {"bin_width": 0.5,
           "numbins":     5,
           "xmin":    -1.25,
           "xmax":     1.25,
           "xlabel": "$Q^{lep}$",
           "xvariable": "lep_charge",
           "title": "Lepton Charge"}

lep_type = {"bin_width": 1,
           "numbins":    3,
           "xmin":      10.5,
           "xmax":      13.5,
           "xlabel": "$|PDG ID|^{lep}$",
           "xvariable": "lep_type",
           "title": "Lepton Absolute PDG ID"}


hist_dicts = {"met_et": etmiss, "mtw": mtw, "jet_n": jet_n, "lep_pt": lep_pt, "lep_eta": lep_eta}
