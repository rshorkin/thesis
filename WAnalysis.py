import ROOT
import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import uproot
import csv
import time
from matplotlib.ticker import AutoMinorLocator
import mplhep as hep
import tensorflow as tf
import zfit

import WCuts
import infofile
import WSamples
from WHistograms import hist_dicts
from WAsymmetry import build_asym
import WMETFit

branches = ["runNumber", "eventNumber", "trigE", "trigM", "lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_n",
            "lep_z0", "lep_charge", "lep_type", "lep_isTightID", "lep_ptcone30", "lep_etcone20",
            "lep_trackd0pvunbiased",
            "lep_tracksigd0pvunbiased", "met_et", "met_phi", "jet_n", "jet_pt", "jet_eta", "jet_phi", "jet_E",
            "jet_jvt",
            "jet_MV2c10", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER", "scaleFactor_PILEUP",
            "mcWeight"
            ]

lumi = 10  # 10 fb-1
fraction = .001
common_path = "/media/roman/Backup Plus/data_13TeV/1lep/"
save_choice = int(input("Save dataframes? 0 for no, 1 for yes\n"))
if save_choice != 1:
    save_file = None
elif save_choice == 1:
    save_file = "csv"


def calc_theta(lep_eta):
    return 2 * math.atan(math.exp(-lep_eta))


def top_weight(x):
    return (x/abs(x))


def calc_mtw(lep_pt, met_et, lep_phi, met_phi):
    return math.sqrt(2 * lep_pt * met_et * (1 - math.cos(lep_phi - met_phi)))


def get_xsec_weight(totalWeight, sample):
    info = infofile.infos[sample]
    weight = (lumi * 1000 * info["xsec"]) / (info["sumw"] * info["red_eff"])  # *1000 to go from fb-1 to pb-1
    weight = totalWeight * weight
    return weight


def to_GeV(x):
    return x / 1000.


def extract_from_vector(x):
    return x[0]


def calc_weight(mcWeight, scaleFactor_ELE, scaleFactor_MUON, scaleFactor_PILEUP, scaleFactor_LepTRIGGER):
    return mcWeight * scaleFactor_ELE * scaleFactor_MUON * scaleFactor_PILEUP * scaleFactor_LepTRIGGER


def read_file(path, sample, branches=branches):
    print("=====")
    print("Processing {0} file".format(sample))
    with uproot.open(path) as file:
        tree = file["mini"]
        numevents = tree.num_entries
        print(numevents)
        df = tree.arrays(branches, library='pd', entry_stop=numevents * fraction)
    df = df[0]

    if "data" not in sample and "single" not in sample:
        df["totalWeight"] = np.vectorize(calc_weight)(df.mcWeight, df.scaleFactor_ELE, df.scaleFactor_MUON,
                                                      df.scaleFactor_PILEUP, df.scaleFactor_LepTRIGGER)
        df["totalWeight"] = np.vectorize(get_xsec_weight)(df.totalWeight, sample)
    elif "data" in sample:
        df["totalWeight"] = [1 for item in range(len(df.index))]
    elif "single" in sample:
        df["totalWeight"] = df["mcWeight"].apply(top_weight)

    df.drop(["mcWeight", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER"], axis=1,
            inplace=True)
    num_before_cuts = len(df.index)
    print("Events before cuts: {0}".format(num_before_cuts))
    df = df.query("met_et > 30000")
    df = df.query("trigE or trigM")

    fail = df[np.vectorize(WCuts.cut_tight)(df.lep_isTightID)].index
    df.drop(fail, inplace=True)

    fail = df[np.vectorize(WCuts.cut_multijet)(df.lep_pt, df.lep_ptcone30, df.lep_etcone20)].index
    df.drop(fail, inplace=True)

    e_df = df.copy()

    fail = e_df[np.vectorize(WCuts.cut_e_fiducial)(e_df.lep_type, e_df.lep_eta)].index
    e_df.drop(fail, inplace=True)

    if len(e_df.lep_type) != 0:  # to account for boson -> munu/mumu

        fail = e_df[np.vectorize(WCuts.cut_e_long)(e_df.lep_type, e_df.lep_trackd0pvunbiased,
                                                   e_df.lep_tracksigd0pvunbiased)].index
        e_df.drop(fail, inplace=True)

        fail = e_df[np.vectorize(WCuts.cut_e_long_impact)(e_df.lep_type, e_df.lep_z0, e_df.lep_eta)].index
        e_df.drop(fail, inplace=True)

    mu_df = df.copy()

    fail = mu_df[np.vectorize(WCuts.cut_mu_fiducial)(mu_df.lep_type, mu_df.lep_eta)].index
    mu_df.drop(fail, inplace=True)

    if len(mu_df.lep_type) != 0:  # to account for boson -> enu/ee

        fail = mu_df[np.vectorize(WCuts.cut_mu_long)(mu_df.lep_type, mu_df.lep_trackd0pvunbiased,
                                                     mu_df.lep_tracksigd0pvunbiased)].index
        mu_df.drop(fail, inplace=True)

        fail = mu_df[np.vectorize(WCuts.cut_mu_long_impact)(mu_df.lep_type, mu_df.lep_z0, mu_df.lep_eta)].index
        mu_df.drop(fail, inplace=True)

    df = pandas.concat([e_df, mu_df])

    df["mtw"] = np.vectorize(calc_mtw)(df.lep_pt, df.met_et, df.lep_phi, df.met_phi)
    df = df.query("mtw > 60000.")

    df = df.sort_values(by="entry")
    df["met_et"] = df["met_et"].apply(to_GeV)
    df["mtw"] = df["mtw"].apply(to_GeV)

    df["lep_pt"] = df["lep_pt"].apply(to_GeV)

    num_after_cuts = len(df.index)
    print("Number of events after cuts: {0}".format(num_after_cuts))
    return df


def read_sample(sample, save_file):
    print("###==========###")
    print("Processing: {0} SAMPLES".format(sample))
    start = time.time()
    frames = []
    for val in WSamples.samples[sample]["list"]:
        if sample == "data":
            prefix = "Data/"
        else:
            prefix = "MC/mc_{0}.".format(infofile.infos[val]["DSID"])
        path = common_path + prefix + val + ".1lep.root"
        if not path == "":
            temp_df = read_file(path, val)
            frames.append(temp_df)
            if save_file == "csv":
                temp_df.to_csv("../WOutput/dataframe_{0}.csv".format(val))
        else:
            raise ValueError("Error! {0} not found!".format(val))
    df_sample = pandas.concat(frames)
    print("###==========###")
    print("Finished processing {0} samples".format(sample))
    print("Time elapsed: {0} seconds".format(time.time() - start))
    return df_sample


def get_data_from_files():
    data = {}
    switch = int(input("What do you want to analyze? 0 for all, 1 for data, 2 for MC\n"))
    if switch == 0:
        samples = ["data", "diboson", "ttbar", "Z+jets", "single top", "W+jets"]
    elif switch == 1:
        samples = ["data"]
    elif switch == 2:
        samples = ["diboson", "ttbar", "Z+jets", "single top", "W+jets"]
    else:
        raise ValueError("Option {0} cannot be processed".format(switch))
    for s in samples:
        data[s] = read_sample(s, save_file)
    return data


def plot_data(data):
    print("###==========####")
    print("Started plotting")

    plot_label = "$W \\rightarrow l\\nu$"
    signal_label = "Signal $W$"

    signal = None
    stack_order = ["diboson", "Z+jets", "ttbar", "single top", "W+jets"]

    for key, hist in hist_dicts.items():
        print("==========")
        print("Plotting {0} histogram".format(key))

        h_bin_width = hist["bin_width"]
        h_num_bins = hist["numbins"]
        h_xmin = hist["xmin"]
        h_xmax = hist["xmax"]
        h_xlabel = hist["xlabel"]
        x_var = hist["xvariable"]
        h_title = hist["title"]

        bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
        bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

        data_x, _ = np.histogram(data["data"][x_var].values, bins=bins)

        mc_x = []
        mc_weights = []
        mc_colors = []
        mc_labels = []
        mc_tot_heights = np.zeros(len(bin_centers))

        for sample in stack_order:
            mc_labels.append(sample)
            mc_x.append(data[sample][x_var].values)
            mc_colors.append(WSamples.samples[sample]["color"])
            mc_weights.append(data[sample].totalWeight.values)
            mc_heights, _ = np.histogram(data[sample][x_var].values, bins=bins, weights=data[sample].totalWeight.values)
            mc_tot_heights = np.add(mc_tot_heights, mc_heights)

        plt.clf()
        plt.style.use(hep.style.ATLAS)
        _ = plt.figure(figsize=(9.5, 9))
        plt.axes([0.1, 0.30, 0.85, 0.65])
        plt.yscale("linear")
        main_axes = plt.gca()
        main_axes.set_title(h_title)
        hep.histplot(main_axes.hist(data["data"][x_var], bins=bins, log=False, facecolor="none"),
                     color="black", yerr=True, histtype="errorbar")
        ns, n_bins, patches = main_axes.hist(mc_x, bins=bins, weights=mc_weights, stacked=True, color=mc_colors,
                                             label=mc_labels)
        handles, labels = main_axes.get_legend_handles_labels()
        main_axes.legend(reversed(handles), reversed(labels), title=plot_label, loc="upper right")
        main_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)

        main_axes.xaxis.set_minor_locator(AutoMinorLocator())

        main_axes.set_xticklabels([])

        plt.axes([0.1, 0.1, 0.85, 0.2])
        plt.yscale("linear")
        ratio_axes = plt.gca()
        ratio_axes.errorbar(bin_centers, data_x / mc_tot_heights, xerr=h_bin_width / 2., fmt='.', color="black")
        ratio_axes.set_ylim(0, 2.5)
        ratio_axes.set_yticks([0, 1, 2])
        ratio_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)
        ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

        main_axes.set_ylabel("Events/bin")
        ratio_axes.set_ylabel("Ratio\nData/MC")
        ratio_axes.set_xlabel(h_xlabel)
        plt.grid("True", axis="y", color="black", linestyle="--")
        # plt.show()
        plt.savefig(f"../Results/{key}_linear.pdf")


data = get_data_from_files()
# plot_data(data)
build_asym(data)

obs = zfit.Space('mtw', limits=(60, 180))

data_for_fit = data["data"]["mtw"]
model = WMETFit.create_initial_model(obs, data_for_fit)
fit_params = WMETFit.initial_fitter(data_for_fit, model, obs)
WMETFit.plot_fit_result({"crystal ball": model}, data_for_fit, fit_params, obs)