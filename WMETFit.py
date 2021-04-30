import pandas
import uproot
import time
import matplotlib.pyplot as plt
import os
import psutil
import numpy as np

import tensorflow as tf
import zfit
import probfit
import matplotlib.pyplot as plt

from WHistograms import hist_dicts
import WSamples
from WHistograms import hist_dicts
import mplhep as hep
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import chisquare

from matplotlib import rc

font = {'family': 'Verdana',  # для вывода русских букв
        'weight': 'normal'}
rc('font', **font)

branches = ['mtw', 'totalWeight', 'jet_n']

pandas.options.mode.chained_assignment = None

common_path = "../DataForFit/"
fraction = .1
lumi_used = str(10 * fraction)


def read_file(path, sample, branches=branches):
    with uproot.open(path) as file:
        tree = file["FitTree"]
        numevents = tree.num_entries
        df = pandas.DataFrame.from_dict(tree.arrays(branches, library='np', entry_stop=numevents * fraction))
    return df


def read_sample(sample):
    print("###==========###")
    print("Processing: {0} SAMPLES".format(sample))

    start = time.time()
    frames = []
    for val in WSamples.samples[sample]["list"]:
        path = common_path + f'{val}/'
        partial_dfs = []
        if not path == "":
            for filename in os.listdir(path):
                if filename.endswith('.root'):
                    filepath = os.path.join(path, filename)
                    partial_df = read_file(filepath, val)
                    partial_dfs.append(partial_df)
            temp_df = pandas.concat(partial_dfs)
            frames.append(temp_df)
        else:
            raise ValueError("Error! {0} not found!".format(val))
    df_sample = pandas.concat(frames)
    print("###==========###")
    print("Finished processing {0} samples".format(sample))
    print("Time elapsed: {0} seconds".format(time.time() - start))
    return df_sample


def get_data_from_files(switch=1):
    data = {}

    mem = psutil.virtual_memory()
    mem_at_start = mem.available / (1024 ** 2)
    print(f'Available Memory: {mem_at_start:.0f} MB')

    # switch = int(input("What do you want to analyze? 0 for all, 1 for data, 2 for MC\n")) todo
    if switch == 0:
        samples = ["data", "diboson", "ttbar", "Z+jets", "single top", "W+jets"]
    elif switch == 1:
        samples = ["data"]
    elif switch == 2:
        samples = ["diboson", "ttbar", "Z+jets", "single top"]
    elif switch == 3:
        samples = ['W+jets']
    else:
        raise ValueError("Option {0} cannot be processed".format(switch))
    for s in samples:
        data[s] = read_sample(s)
        mem = psutil.virtual_memory()
        actual_mem = mem.available / (1024 ** 2)
        print(f'Current available memory {actual_mem:.0f} MB '
              f'({100 * actual_mem / mem_at_start:.0f}% of what we started with)')
        if actual_mem < 150:
            raise Warning('Out of RAM')
    return data


def format_data(data, obs, sample=None):
    if sample != 'data':
        return zfit.Data.from_numpy(obs, data.mtw.to_numpy(), weights=data.totalWeight.to_numpy())
    else:
        return zfit.Data.from_numpy(obs, data.mtw.to_numpy())


def create_initial_model(obs, sample):
    # Crystal Ball

    mu = zfit.Parameter(f"mu_{sample}", 80., 60., 120.)
    sigma = zfit.Parameter(f'sigma_{sample}', 8., 1., 100.)
    alpha = zfit.Parameter(f'alpha_{sample}', -.5, -10., 0.)
    n = zfit.Parameter(f'n_{sample}', 120., 0.01, 200.)
    model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)

    return model


def sum_func(*args):
    return sum(i for i in args)


def initial_fitter(data, sample, initial_parameters, obs):
    print('==========')
    print(f'Fitting {sample} sample')
    df = data[sample]
    bgr_yield = df['totalWeight'].sum()
    print(f'Total number of events: {bgr_yield}')

    mu = zfit.Parameter(f"mu_{sample}", initial_parameters['mu'], 77., 180.)
    sigma = zfit.Parameter(f'sigma_{sample}', initial_parameters['sigma'], 1., 150.)

    alphal = zfit.Parameter(f'alphal_{sample}', initial_parameters['alphal'], 0., 10.)
    alphar = zfit.Parameter(f'alphar_{sample}', initial_parameters['alphar'], 0., 10.)
    nl = zfit.Parameter(f'nl_{sample}', initial_parameters['nl'], 0.01, 30.)
    nr = zfit.Parameter(f'nr_{sample}', initial_parameters['nr'], 0.01, 30.)
    n_bgr = zfit.Parameter(f'yield_DCB_{sample}', bgr_yield, 0., int(1.3 * bgr_yield), step_size=1)

    DCB = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)
    DCB = DCB.create_extended(n_bgr)

    mu_g = zfit.Parameter(f"mu_CB_{sample}", 40., 10., 77.)
    sigma_g = zfit.Parameter(f'sigma_CB_{sample}', 80., 1., 200.)
    ad_yield = zfit.Parameter(f'yield_CB_{sample}', int(0.15 * bgr_yield), 0., int(1.3 * bgr_yield), step_size=1)
    alphal = zfit.Parameter(f'alpha_CB_{sample}', 2.6, 0.001, 20.)
    nl = zfit.Parameter(f'n_CB_{sample}', 13.4, 0.001, 30.)
    alphar = zfit.Parameter(f'alphar_CB_{sample}', 0.6, 0.001, 20.)
    nr = zfit.Parameter(f'nr_CB_{sample}', 2.4, 0.001, 30.)

    gauss = zfit.pdf.DoubleCB(mu=mu_g, sigma=sigma_g, alphal=alphal, nl=nl, alphar=alphar, nr=nr, obs=obs)
    gauss = gauss.create_extended(ad_yield)

    model = zfit.pdf.SumPDF([DCB, gauss])

    bgr_data = format_data(df, obs)
    # Create NLL
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=bgr_data)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
    result = minimizer.minimize(nll)
    if result.valid:
        print("Result is valid")
        print("Converged:", result.converged)
        # param_errors = result.hesse()
        print(result.params)
        if not model.is_extended:
            raise Warning('MODEL NOT EXTENDED')
        return model
    else:
        print('Minimization failed')
        print(result.params)
        return model


# Plotting

def plot_fit_result(models, data, obs, sample='data'):
    plt_name = "mtw"
    print(f'Plotting {sample}')

    lower, upper = obs.limits
    mc_labels_ru = {"diboson": 'Два бозона', "Z+jets": 'Z+струи', "ttbar": '$t \\bar{t}$',
                    "single top": 'Одиночный t', "W+jets": 'W+струи', 'data': 'Данные'}
    if '0' in sample:
        plt_title = f"Аппроксимация поперечной массы W (0 струй)"
        mc_labels_ru['W+jets'] = 'W'
        mc_labels_ru['Z+jets'] = 'Z'
    elif '1' in sample:
        plt_title = f"Аппроксимация поперечной массы W (1 струя)"
        mc_labels_ru['W+jets'] = 'Wj'
        mc_labels_ru['Z+jets'] = 'Zj'
    elif '2' in sample:
        plt_title = f"Аппроксимация поперечной массы W (много струй)"
    else:
        plt_title = f"Аппроксимация поперечной массы W"

    h_bin_width = 5
    h_num_bins = 24
    h_xmin = 60
    h_xmax = 180
    h_xlabel = hist_dicts[plt_name]["xlabel"]
    plt_label = "$W \\rightarrow l\\nu$"

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data.mtw.values, bins=bins, weights=data.totalWeight.values)
    data_sum = data_x.sum()
    plot_scale = data_sum * obs.area() / h_num_bins

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.10, 0.85, 0.85])
    main_axes = plt.gca()
    if sample != 'data':
        hist_label = mc_labels_ru[sample[:-3]]
    else:
        hist_label = mc_labels_ru[sample]
    hep.histplot(main_axes.hist(data.mtw, bins=bins, log=False, facecolor="none", weights=data.totalWeight.values),
                 color="black", yerr=True, histtype="errorbar", label=hist_label)

    main_axes.set_xlim(lower[-1][0], upper[0][0])
    main_axes.set_ylim(0., 1.4 * max(data_x))

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_xlabel(h_xlabel)
    main_axes.set_title(plt_title, fontsize=18)
    main_axes.set_ylabel(f"События/{h_bin_width} ГэВ")
    plt.text(0.05, 0.97, 'ATLAS Open Data', ha="left", va="top", family='sans-serif', transform=main_axes.transAxes,
             fontsize=20)
    plt.text(0.05, 0.9, r'$\sqrt{s}=13\,\mathrm{TeV},\;\int\, L\,dt=$' + lumi_used + '$\,\mathrm{fb}^{-1}$', ha="left",
             va="top", family='sans-serif', fontsize=16, transform=main_axes.transAxes)
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    data_bins = np.asarray(bin_centers)
    print('Data bin centers:')
    print(data_bins)
    print(f'A total of {h_num_bins} bin centers')
    for model_name, model in models.items():
        if model.is_extended:
            if 'combo' not in model_name and 'sum' not in model_name:
                main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins,
                               label=mc_labels_ru[model_name[:-3]] + ' модель')
                chisq, p_val = chisquare(data_x, model.ext_pdf(data_bins) * obs.area() / h_num_bins)
            elif 'j' not in model_name:
                main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins,
                               label='Суммарная модель', color='black')
                chisq, p_val = chisquare(data_x, model.ext_pdf(data_bins) * obs.area() / h_num_bins)
            elif '0' in model_name:
                main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins,
                               label='Вклад без струй', color='green')
            elif '1' in model_name:
                main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins,
                               label='Вклад с 1 струёй', color='blue')
            elif '2' in model_name:
                main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins,
                               label='Вклад с многими струями', color='red')
            print(f'-----\nData Values:\n{data_x}')
            print(f'-----\nModel Values:\n{model.ext_pdf(data_bins) * obs.area() / h_num_bins}')
        else:
            main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, label=model_name)
    main_axes.legend(title=plt_label, loc="best")
    text = f'$\chi^2$ = {chisq}\np = {p_val}'
    plt.text(0.7, 0.6, text, transform=main_axes.transAxes)
    plt.savefig(f"../FitResults/{sample}_fit_{plt_name}.jpg")
    plt.close()


def plot_component(dfs, component):
    print("###==========####")
    print("Started plotting")

    plot_label = "$W \\rightarrow l\\nu$"
    hist = hist_dicts['mtw']

    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    x_var = hist["xvariable"]
    h_title = hist["title"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(dfs[component][x_var].values, bins=bins)

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale("linear")
    main_axes = plt.gca()
    main_axes.set_title(h_title)
    hep.histplot(main_axes.hist(dfs[component][x_var], bins=bins, log=False, facecolor="none"),
                 color="black", yerr=True, histtype="errorbar", label='Данные')

    main_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_ylabel(f"События/{h_bin_width} ГэВ")
    plt.savefig(f"../Results/{component}_mtw.jpg")


obs = zfit.Space('mtw', limits=(60, 180))
initial_parameters = {'diboson': {'mu': 81., 'sigma': 15., 'nl': 26., 'alphal': 5., 'nr': 20., 'alphar': 1.5},
                      'ttbar': {'mu': 81., 'sigma': 23., 'nl': 350., 'alphal': 5.6, 'nr': 17., 'alphar': 0.8},
                      'single top': {'mu': 80., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 20., 'alphar': 1.},
                      'Z+jets': {'mu': 81., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 20., 'alphar': 1.},

                      'W+jets': {'mu': 81., 'sigma': 16., 'nl': 14., 'alphal': 1.5, 'nr': 10., 'alphar': 1.3},
                      'W + 0 jets': {'mu': 81., 'sigma': 6., 'nl': 10., 'alphal': 5.5, 'nr': 12., 'alphar': 0.5},
                      'W + 1 jet': {'mu': 82., 'sigma': 13., 'nl': 10., 'alphal': 5.5, 'nr': 20., 'alphar': 1.},
                      'W + multi jets': {'mu': 81., 'sigma': 17., 'nl': 10., 'alphal': 5.5, 'nr': 20., 'alphar': 0.9},

                      'data_0': {'mu': 81., 'sigma': 6., 'nl': 10., 'alphal': 5.5, 'nr': 12., 'alphar': 0.5},
                      'data_1': {'mu': 82., 'sigma': 13., 'nl': 10., 'alphal': 5.5, 'nr': 20., 'alphar': 1.},
                      'data_2': {'mu': 81., 'sigma': 17., 'nl': 10., 'alphal': 5.5, 'nr': 20., 'alphar': 0.9}
                      }


def MainFit():
    cats = {'_0j': 'jet_n == 0', '_1j': 'jet_n == 1', '_2j': 'jet_n > 1'}
    data = get_data_from_files(switch=0)
    models = {}
    for cat_name, cat_cut in cats.items():
        cat_models = {}
        for sample in ["diboson", "ttbar", "Z+jets", "single top", 'W+jets']:
            data[sample + cat_name] = data[sample].query(cat_cut)
            model = initial_fitter(data, sample + cat_name, initial_parameters[sample], obs)
            plot_fit_result({sample + cat_name: model}, data[sample + cat_name], obs, sample=sample + cat_name)
            cat_models[sample + cat_name] = model
            del data[sample + cat_name]
        cat_model = zfit.pdf.SumPDF([cat_models[key] for key in cat_models.keys() if cat_name in key])
        cat_models['combo_model'] = cat_model
        plot_fit_result(cat_models, data['data'].query(cat_cut), obs, sample='data' + cat_name)
        models['sum_model' + cat_name] = cat_model
    sum_model = zfit.pdf.SumPDF([model for model in models.values()])
    models['sum_model_tot'] = sum_model
    plot_fit_result(models, data['data'], obs, sample='data')


MainFit()


