import ROOT
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import uproot
import csv
import time
import uproot3
from matplotlib.ticker import AutoMinorLocator
import mplhep as hep

import WCuts
import infofile
import WSamples
from WHistograms import hist_dicts


def build_asym(data):

    signal = None
    stack_order = ["diboson", "Z+jets", "ttbar", "single top", "W+jets"]
    hist = hist_dicts["lep_eta"]

    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    x_var = hist["xvariable"]
    h_title = hist["title"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    dict_asym = {}

    charge_queries = {"pos": "lep_charge == 1", "neg": "lep_charge == -1"}
    for key, query in charge_queries.items():
        data_temp = {data_dict_key: data_dict_item.query(query) for data_dict_key, data_dict_item in data.items()}
        data_x, _ = np.histogram(data_temp["data"][x_var].values, bins=bins)
        for i in range(int(len(data_x)/2)):
            temp = data_x[-(i+1)]
            data_x[-(i+1)] = data_x[i] + data_x[-(1+i)]
        dict_asym[key] = data_x[int(len(data_x)/2):]
    asym_x = [(dict_asym["pos"][i]-dict_asym["neg"][i])/(dict_asym["pos"][i]+dict_asym["neg"][i])
              for i in range(len(dict_asym["pos"]))]

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.set_title("Lepton Charge Asymmetry")
    plt.errorbar(bin_centers[int(len(data_x)/2):], asym_x, fmt="ko")
    main_axes.set_xlim(0., 2.6)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig("asym_13TeV.jpg")



