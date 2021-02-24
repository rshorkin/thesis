import ROOT
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import uproot
import csv
import time

import WCuts
import infofile
import WSamples

def plot_data(data):
    signal_format = "hist"
    plot_label    = "%W \rightarrow l\\nu"
    signal_label  = "Signal $W$"

    lumi_used = str(lumi*fraction)
    signal = None
    stack_order = ["single top", "diboson", "Z+jets", "W+jets"]

    for hist in hist_dicts:
        h_bin_width = hist["bin_width"]
        h_num_bins  = hist["numbins"]
        h_xmin      = hist["xmin"]
        h_xmax      = hist["xmax"]
        h_xlabel    = hist["xlabel"]
        x_var       = hist["xvariable"]

        bins        = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]

        data_x      = np.histogram(data["data"][x_var].values, bins = bins)

        mc_x  =      []
        mc_weights = []
        mc_colors =  []
        mc_labels =  []

        for sample in stack_order:
            mc_labels.append(sample)
            mc_x.append(data[sample][x_var].values)
            mc_colors.append(WSamples.samples[sample]["color"])
            mc_weights.append(data[sample][x_var].values, bins = bins, weights = data[sample]["totalWeight"].values)



        plt.clf()

        plt.axes([0.1, 0.30, 0.85, 0.65])
        main_axes = plt.gca()
        main_axes.set_title("testing")
        ns, bins, patches = main_axes.hist(mc_x, bins = bins, weights = mc_weights, stacked = True, color = mc_colors, label = mc_labels)
        main_axes.legend(loc = "best")
        main_axes.set_xlim(h_xmin, h_xmax)
        main_axes.xaxis.set_minor_locator(AutoMinorLocator())

        main_axes.set_xticklabels([])

        plt.axes([0.1, 0.1, 0.85, 0.2])
        ratio_axes = plt.gca()
        y = []
        for idx in range(len(ns[1])):
            if ns[1][idx] != 0.:
                y.append(ns[0][idx]/ns[1][idx])
            else:
                y.append(0)
        ratio_axes.errorbar(bins[:-1], y, xerr = .01,  fmt = '.', color = "black" )
        ratio_axes.set_ylim(0, 2.5)
        ratio_axes.set_yticks([0, 1, 2])
        ratio_axes.set_xlim(h_xmin, h_xmax)
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

        main_axes.set_ylabel("Normed Counts")
        ratio_axes.set_ylabel("Ratio\nlow/high")
        ratio_axes.set_xlabel()
        plt.grid("True", axis = "y", color = "black", linestyle = "--")
        plt.show()
        #plt.savefig()
