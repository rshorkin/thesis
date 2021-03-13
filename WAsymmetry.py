import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
from matplotlib.ticker import AutoMinorLocator
import mplhep as hep

from WHistograms import hist_dicts


def build_asym(data):

    signal = None
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

    dict_asym = {'data': {}, 'MC': {}}

    charge_queries = {"pos": "lep_charge == 1", "neg": "lep_charge == -1"}
    for key, query in charge_queries.items():
        data_temp = {data_dict_key: data_dict_item.query(query) for data_dict_key, data_dict_item in data.items()}
        data_x, _ = np.histogram(data_temp["data"][x_var].values, bins=bins)

        mc_weights = []
        mc_tot_heights = np.zeros(len(bin_centers))

        for sample, df in data_temp.items():
            if sample != "data":
                mc_weights.append(df.totalWeight.values)
                mc_heights, _ = np.histogram(df[x_var].values, bins=bins,
                                             weights=df.totalWeight.values)
                mc_tot_heights = np.add(mc_tot_heights, mc_heights)

        for i in range(int(len(data_x)/2)):
            data_x[-(i+1)] = data_x[i] + data_x[-(i+1)]
            mc_tot_heights[-(i+1)] = mc_tot_heights[i] + mc_tot_heights[-(i+1)]

        dict_asym['data'][key] = data_x[int(len(data_x)/2):]
        dict_asym['MC'][key] = mc_tot_heights[int(len(mc_tot_heights)/2):]

    asym_x = {}
    for sample, collection in dict_asym.items():
        asym_x[sample] = [(collection["pos"][i] - collection["neg"][i]) / (collection["pos"][i] + collection["neg"][i])
                          for i in range(len(collection["pos"]))]

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.set_title("Lepton Charge Asymmetry")
    main_axes.set_xlabel(r'|$\eta_{ll}$|')
    main_axes.set_ylabel(r'Asymmetry Value')
    plt.errorbar(bin_centers[len(asym_x['data']):], asym_x['data'], xerr=h_bin_width/2, fmt="ko")
    plt.errorbar(bin_centers[len(asym_x['MC']):], asym_x['MC'], xerr=h_bin_width / 2, fmt="bo")
    main_axes.set_xlim(0., 2.6)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    plt.savefig("asym_13TeV_new.jpg")



