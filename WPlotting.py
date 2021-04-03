import matplotlib.pyplot as plt
import numpy as np
import pandas
import uproot3
import uproot

import math
import WSamples
import mplhep as hep
from WHistograms import hist_dicts

from matplotlib.ticker import AutoMinorLocator,LogLocator,LogFormatterSciNotation

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


def read_histogram(hist_name):
    histogram = {}
    for sample in ["data", "diboson", "ttbar", "Z+jets", "single top", "W+jets"]:
        histogram[sample] = read_histogram_from_sample(hist_name, sample)
    return histogram


def plot_histogram(hist_name, scale='linear'):
    stack_order = ["diboson", "Z+jets", "ttbar", "single top", "W+jets"]
    mc_labels_ru = {"diboson": 'Два бозона', "Z+jets": 'Z+струи', "ttbar": '$t \\bar{t}$',
                 "single top": 'Одиночный t', "W+jets": 'W+струи'}
    if '0' in hist_name:
        mc_labels_ru['W+jets'] = 'W'
        mc_labels_ru['Z+jets'] = 'Z'
    elif '1' in hist_name:
        mc_labels_ru['W+jets'] = 'Wj'
        mc_labels_ru['Z+jets'] = 'Zj'
    plot_label = "$W \\rightarrow l\\nu$"
    lumi_used = '10'
    print("==========")
    print("Plotting {0} histogram".format(hist_name))

    hist = hist_dicts[hist_name]

    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    x_var = hist["xvariable"]
    h_title = hist["title"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    hist_heights = read_histogram(hist_name)

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale(scale)
    main_axes = plt.gca()
    main_axes.set_title(h_title, fontsize=18)

    main_axes.errorbar(x=bin_centers, y=hist_heights['data'],
                       yerr=np.sqrt(hist_heights['data']), xerr=h_bin_width/2,
                       fmt='ko', markersize='4', label='Данные')
    mc_labels = []
    mc_tot_heights = np.zeros(len(bin_centers))
    mc_x = []
    for sample in stack_order:
        mc_labels.append(sample)
        mc_color = WSamples.samples[sample]["color"]
        mc_heights = hist_heights[sample]
        main_axes.bar(x=bin_centers, height=mc_heights, width=h_bin_width, bottom=mc_tot_heights,
                      color=mc_color, label=mc_labels_ru[sample])
        mc_tot_heights = np.add(mc_tot_heights, mc_heights)

    mc_x_err = np.sqrt(mc_tot_heights)
    main_axes.bar(bin_centers, 2 * mc_x_err, bottom=mc_tot_heights - mc_x_err, alpha=0.5, color='none', hatch="////",
                  width=h_bin_width, label='Стат. погр.')
    handles, labels = main_axes.get_legend_handles_labels()
    legend = main_axes.legend(reversed(handles), reversed(labels), title=plot_label, loc="upper right", frameon=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    main_axes.set_xlim(h_xmin, h_xmax)
    if not scale == 'log':
        if 'phi' in hist_name:
            factor = 1.5
        else:
            factor = 1.25
        main_axes.set_ylim(bottom=0,
                           top=(max([np.amax(hist_heights['data']), np.amax(mc_tot_heights)]) * factor))
    else:
        main_axes.set_yscale('log')
        bottom = min(hist_heights[stack_order[0]])
        top = np.amax(hist_heights['data']) * 100
        main_axes.set_ylim(bottom=bottom, top=top)
        main_axes.yaxis.set_major_formatter(CustomTicker())
        locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
        main_axes.yaxis.set_minor_locator(locmin)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_xticklabels([])
    plt.axes([0.1, 0.1, 0.85, 0.2])
    plt.yscale("linear")
    ratio_axes = plt.gca()
    ratio_axes.errorbar(bin_centers, hist_heights['data'] / mc_tot_heights, xerr=h_bin_width / 2.,
                        fmt='.', color="black")
    ratio_axes.bar(bin_centers, 2 * mc_x_err / mc_tot_heights, bottom=1 - mc_x_err / mc_tot_heights,
                   alpha=0.5, color='none',
                   hatch="////", width=h_bin_width)
    ratio_axes.set_ylim(0, 2.5)
    ratio_axes.set_yticks([0, 1, 2])
    ratio_axes.set_xlim(h_xmin, h_xmax)
    ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

    if len(h_xlabel.split('[')) > 1:
        y_units = ' ' + h_xlabel[h_xlabel.find("[") + 1:h_xlabel.find("]")]
    else:
        y_units = ''

    plt.text(0.05, 0.97, 'ATLAS Open Data', ha="left", va="top", family='sans-serif', transform=main_axes.transAxes,
             fontsize=20)
    plt.text(0.05, 0.9, r'$\sqrt{s}=13\,\mathrm{TeV},\;\int\, L\,dt=$' + lumi_used + '$\,\mathrm{fb}^{-1}$', ha="left",
             va="top", family='sans-serif', fontsize=16, transform=main_axes.transAxes)
    main_axes.set_ylabel(f"События / {h_bin_width} {y_units}")
    ratio_axes.set_ylabel("Отношение\nДанные/МК")
    ratio_axes.set_xlabel(h_xlabel)
    plt.grid("True", axis="y", color="black", linestyle="--")
    plt.savefig(f'../Results/{hist_name}_{scale}.jpeg')
    return None


def plotting_main():
    print('Started plotting')
    for hist_name in hist_dicts.keys():
        if 'neg' in hist_name or 'pos' in hist_name:
            continue
        plot_histogram(hist_name, 'linear')
    print('Plotting finished')


if __name__ == '__main__':
    plotting_main()

