import matplotlib.pyplot as plt
import numpy as np
import pandas
import uproot3
import uproot

import math
import WSamples
import WSamples_8
import mplhep as hep
from WHistograms import hist_dicts

from matplotlib.ticker import AutoMinorLocator, LogLocator, LogFormatterSciNotation

from matplotlib import rc

font = {'family': 'Verdana', # для вывода русских букв
        'weight': 'normal'}
rc('font', **font)


class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1,10]:
            return LogFormatterSciNotation.__call__(self,x, pos=None)
        else:
            return "{x:g}".format(x=x)


def read_histogram_from_file(hist_name, path):
    file = uproot3.open(path)
    histogram = file[hist_name]
    return histogram.values


def read_histogram_from_sample(hist_name, sample):
    tot_heights = None
    for val in WSamples.samples[sample]["list"]:
        path = f'../Output/{val}.root'
        if tot_heights is None:
            tot_heights = read_histogram_from_file(hist_name, path)
        else:
            tot_heights = np.add(tot_heights, read_histogram_from_file(hist_name, path))
    return tot_heights


def read_histogram_from_sample_8(hist_name, sample):
    tot_heights = None
    for val in WSamples_8.samples[sample]["list"]:
        path = f'../Output_8TeV/{val}.root'
        if tot_heights is None:
            tot_heights = read_histogram_from_file(hist_name, path)
        else:
            tot_heights = np.add(tot_heights, read_histogram_from_file(hist_name, path))
    return tot_heights


def read_histogram(hist_name):
    histogram_13 = {}
    histogram_8 = {}
    for sample in ["data", "diboson", "ttbar", "Z+jets", "single top", "W+jets"]:
        histogram_13[sample] = read_histogram_from_sample(hist_name, sample)
    for sample in ["data", "diboson", "ttbar", "Z", "single top", "W", 'DrellYan']:
        histogram_8[sample] = read_histogram_from_sample_8(hist_name, sample)
    return histogram_13, histogram_8


def calc_ratio(histogram_13, histogram_8):
    ratio_hist = []
    errors = []
    for bin_13, bin_8 in zip(histogram_13, histogram_8):
        ratio_bin = bin_13 / bin_8
        error = math.sqrt(bin_13 * (1 + ratio_bin)) / bin_8
        ratio_hist.append(ratio_bin)
        errors.append(error)
    ratio_hist = np.array(ratio_hist)
    return ratio_hist, errors


def calc_mean(hist):
    mean = 0
    wsum = None
    sum = 0
    stddev = 0
    jet_n = 0
    for bin in hist:
        if wsum is None:
            wsum = bin * jet_n
        else:
            wsum += bin * jet_n
        sum += bin
        jet_n += 1

    mean = wsum / sum
    jet_n = 0
    for bin in hist:
        stddev += bin * (jet_n - mean) ** 2
        jet_n += 1
    stddev = math.sqrt(stddev/sum)
    err = stddev/math.sqrt(sum)
    return mean, err


def plot_ratio(hist_name='jet_n', scale='linear'):
    stack_order_13 = ["diboson", "Z+jets", "ttbar", "single top", "W+jets"]
    stack_order_8 = ["single top", "diboson", 'DrellYan', "ttbar", "Z", "W"]
    plot_label = "$W \\rightarrow l\\nu$"
    # lumi_used = '10'
    print("==========")
    print("Plotting {0} histogram".format(hist_name))

    hist = hist_dicts[hist_name]

    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    x_var = hist["xvariable"]
    h_title = 'Отношение распределений по числу струй'
    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    hist_heights_13, hist_heights_8 = read_histogram(hist_name)
    print('DATA')
    data_ratio_heights, data_errors = calc_ratio(hist_heights_13['data'], hist_heights_8['data'])
    mean_8, err_8 = calc_mean(hist_heights_8['data'])
    print(f'8 TeV: {mean_8:.5f}')
    mean_13, err_13 = calc_mean(hist_heights_13['data'])
    print(f'13 TeV: {mean_13:.5f}')

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale(scale)
    main_axes = plt.gca()
    main_axes.set_title(h_title, fontsize=18)

    mc_tot_heights_13 = np.zeros(len(bin_centers))
    mc_tot_heights_8 = np.zeros(len(bin_centers))

    for sample in stack_order_13:
        mc_heights = hist_heights_13[sample]
        mc_tot_heights_13 = np.add(mc_tot_heights_13, mc_heights)

    for sample in stack_order_8:
        mc_heights = hist_heights_8[sample]
        mc_tot_heights_8 = np.add(mc_tot_heights_8, mc_heights)

    print('-----\nMC')
    mc_ratio_heights, mc_errors = calc_ratio(mc_tot_heights_13, mc_tot_heights_8)
    mc_errors = np.asarray(mc_errors)
    mean_8_MC, err_8_MC = calc_mean(mc_tot_heights_8)
    mean_13_MC, err_13_MC = calc_mean(mc_tot_heights_13)

    # text with calculated means

    text = r'$\mu_{jets}^{13 ТэВ}$ = ' + f'${mean_13:.5f} \pm {err_13:.5f}$\n' + \
           r'$\mu_{jets}^{8 ТэВ}$ = ' + f'${mean_8:.5f} \pm {err_8:.5f}$\n'
    # I hate matplotlib's step function cause of how it handles start and end >:(

    xs = [h_xmin]
    ys = [mc_ratio_heights[0]]
    for i in range(h_num_bins - 1):
        xs.append(h_xmin + h_bin_width * (1 + i))
        xs.append(h_xmin + h_bin_width * (1 + i))
        ys.append(mc_ratio_heights[i])
        ys.append(mc_ratio_heights[i + 1])
    xs.append(h_xmax)
    ys.append(mc_ratio_heights[-1])

    main_axes.plot(xs, ys, color='green', label='Моделирование МК')
    main_axes.bar(bin_centers, 2 * mc_errors, bottom=mc_ratio_heights - mc_errors, alpha=0.5, color='none', hatch="////",
                  width=h_bin_width, label='Погрешность')

    main_axes.errorbar(x=bin_centers, y=data_ratio_heights,
                       xerr=h_bin_width / 2, yerr=data_errors,
                       fmt='ko', markersize='4', label='Данные')

    handles, labels = main_axes.get_legend_handles_labels()
    main_axes.legend(handles, labels, loc='upper right', bbox_transform=main_axes.transAxes)
    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_xticklabels([])

    factor = 1.25
    main_axes.set_ylim(bottom=0.04,
                       top=(max([np.amax(data_ratio_heights), np.amax(mc_ratio_heights)]) * factor))

    plt.axes([0.1, 0.1, 0.85, 0.2])
    plt.yscale("linear")
    ratio_axes = plt.gca()
    ratio_axes.errorbar(bin_centers, data_ratio_heights / mc_ratio_heights, xerr=h_bin_width / 2.,
                        fmt='.', color="black")
    ratio_axes.bar(bin_centers, 2 * mc_errors / mc_ratio_heights, bottom=1 - mc_errors / mc_ratio_heights,
                   alpha=0.5, color='none',
                   hatch="////", width=h_bin_width)
    ratio_axes.set_ylim(0.5, 1.5)
    ratio_axes.set_yticks([0.75, 1., 1.25])
    ratio_axes.set_xlim(h_xmin, h_xmax)
    ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel("События (13 ТэВ)/События (8 ТэВ)")
    ratio_axes.set_ylabel("Данные/МК")
    ratio_axes.set_xlabel(h_xlabel)
    plt.grid("True", axis="y", color="black", linestyle="--")

    plt.text(0.05, 0.97, 'ATLAS Open Data', ha="left", va="top", family='sans-serif', transform=main_axes.transAxes,
             fontsize=20)
    plt.text(0.05, 0.9, plot_label, ha="left", va="top", family='sans-serif',
             fontsize=14, transform=main_axes.transAxes)
    plt.text(0.05, 0.83, text, ha="left", va="top", family='sans-serif',
             fontsize=14, transform=main_axes.transAxes)

    plt.savefig(f'../Results/njets_ratio.jpeg')
    return None


if __name__ == '__main__':
    plot_ratio()