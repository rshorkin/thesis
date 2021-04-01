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

branches = ['mtw', 'totalWeight', 'jet_n']

pandas.options.mode.chained_assignment = None

lumi = 1  # 10 fb-1
common_path = "../DataForFit/"
fraction = .3


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


def get_data_from_files(switch = 1):
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
    n = zfit.Parameter(f'n_{sample}', 120., 0.01, 500.)
    model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)

    return model


def sum_func(*args):
    return sum(i for i in args)


def initial_fitter(data, sample, initial_parameters, obs):
    print('==========')
    print(f'Fitting {sample} sample')
    df = data[sample]
    bgr_yield = len(df.index)
    print(f'Total number of events: {bgr_yield}')

    mu = zfit.Parameter(f"mu_{sample}", initial_parameters[sample]['mu'], 40., 100.)
    sigma = zfit.Parameter(f'sigma_{sample}', initial_parameters[sample]['sigma'], 1., 100.)
    alphal = zfit.Parameter(f'alphal_{sample}', initial_parameters[sample]['alphal'], 0., 10.)
    alphar = zfit.Parameter(f'alphar_{sample}', initial_parameters[sample]['alphar'], 0., 10.)
    nl = zfit.Parameter(f'nl_{sample}', initial_parameters[sample]['nl'], 0.01, 500.)
    nr = zfit.Parameter(f'nr_{sample}', initial_parameters[sample]['nr'], 0.01, 500.)
    n_bgr = zfit.Parameter(f'yield_DCB_{sample}', bgr_yield, 0., int(1.3 * bgr_yield), step_size=1)

    DCB = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)
    DCB = DCB.create_extended(n_bgr)

    mu_g = zfit.Parameter(f"mu_CB_{sample}", 42., 40., 100.)
    sigma_g = zfit.Parameter(f'sigma_CB_{sample}', 80., 1., 100.)
    ad_yield = zfit.Parameter(f'yield_CB_{sample}', int(0.15 * bgr_yield), 0., int(1.3 * bgr_yield), step_size=1)
    alphal = zfit.Parameter(f'alpha_CB_{sample}', 2.6, 0.001, 100.)
    nl = zfit.Parameter(f'n_CB_{sample}', 13.4, 0.001, 400.)
    alphar = zfit.Parameter(f'alphar_CB_{sample}', 0.6, 0.001, 100.)
    nr = zfit.Parameter(f'nr_CB_{sample}', 2.4, 0.001, 400.)

    gauss = zfit.pdf.DoubleCB(mu=mu_g, sigma=sigma_g, alphal=alphal, nl=nl, alphar=alphar, nr=nr, obs=obs)
    gauss = gauss.create_extended(ad_yield)
    if 'W' not in sample:
        model = zfit.pdf.SumPDF([DCB, gauss])
    else:
        model = DCB

    bgr_data = format_data(df, obs)
    # Create NLL
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=bgr_data)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True, tolerance=0.01)
    result = minimizer.minimize(nll)
    if result.valid:
        print("Result is valid")
        print("Converged:", result.converged)
        # param_errors = result.hesse()
        print(result.params)
        mu1 = zfit.run(mu)
        mu2 = zfit.run(mu_g)
        weight1 = zfit.run(bgr_yield)
        weight2 = zfit.run(ad_yield)
        mean_mu = (mu1 * weight1 + mu2 * weight2) / (weight2 + weight1)
        avg_mu = (mu1 + mu2) / 2
        print(mean_mu)
        if not model.is_extended:
            raise Warning('MODEL NOT EXTENDED')
        return model, [mean_mu, avg_mu, mu1, mu2]
    else:
        print('Minimization failed')
        print(result.params)
        mu1 = zfit.run(mu)
        mu2 = zfit.run(mu_g)
        weight1 = zfit.run(bgr_yield)
        weight2 = zfit.run(ad_yield)
        mean_mu = (mu1 * weight1 + mu2 * weight2) / (weight2 + weight1)
        avg_mu = (mu1 + mu2) / 2
        print(mean_mu)
        return model, [mean_mu, avg_mu, mu1, mu2]


# Plotting

def plot_fit_result(models, data, obs, sample='data', x=[80.]):
    plt_name = "mtw"
    print(f'Plotting {sample}')

    lower, upper = obs.limits

    h_bin_width = 5
    h_num_bins = 18
    h_xmin = 60
    h_xmax = 150
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
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    hep.histplot(main_axes.hist(data.mtw, bins=bins, log=False, facecolor="none", weights=data.totalWeight.values),
                 color="black", yerr=True, histtype="errorbar", label=sample)

    main_axes.set_xlim(lower[-1][0], upper[0][0])
    main_axes.set_ylim(0., 1.4 * max(data_x))
    # for point in x:
       # plt.axvline(point)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_xlabel(h_xlabel)
    main_axes.set_title("W Transverse Mass Fit")
    main_axes.set_ylabel("Events/4 GeV")
    # main_axes.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2]) todo

    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    for model_name, model in models.items():
        if model.is_extended:
            print('Model is extended')
            main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins, label=model_name)
        else:
            main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, label=model_name)
            print('Model is not extended')
        y_plot = model.ext_pdf(x_plot)
        y_plot = y_plot.numpy().tolist()
        x_index = y_plot.index(max(y_plot))
        # plt.axvline(x_plot[x_index], color='red')
    main_axes.legend(title=plt_label, loc="best")
    plt.savefig(f"../Results/{sample}_fit_{plt_name}_Complex.jpg")
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
    hep.histplot(main_axes.hist(data[component][x_var], bins=bins, log=False, facecolor="none"),
                 color="black", yerr=True, histtype="errorbar", label='data')

    main_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_ylabel(f"Events/{h_bin_width}")
    plt.savefig(f"../Results/{component}_mtw.jpg")


obs = zfit.Space('mtw', limits=(60, 150))
initial_parameters = {'diboson': {'mu': 81., 'sigma': 15., 'nl': 26., 'alphal': 10., 'nr': 85., 'alphar': 1.5},
                      'ttbar': {'mu': 77., 'sigma': 23., 'nl': 350., 'alphal': 5.6, 'nr': 17., 'alphar': 0.8},
                      'single top': {'mu': 80., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.},
                      'Z+jets': {'mu': 73., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.},

                      'W': {'mu': 77., 'sigma': 16., 'nl': 14., 'alphal': 1.5, 'nr': 10., 'alphar': 1.3},
                      'W + 0 jets': {'mu': 77., 'sigma': 6., 'nl': 10., 'alphal': 5.5, 'nr': 12., 'alphar': 0.5},
                      'W + 1 jet': {'mu': 82., 'sigma': 13., 'nl': 10., 'alphal': 5.5, 'nr': 105., 'alphar': 1.},
                      'W + multi jets': {'mu': 81., 'sigma': 17., 'nl': 10., 'alphal': 5.5, 'nr': 135., 'alphar': 0.9},

                      'data_0': {'mu': 77., 'sigma': 6., 'nl': 10., 'alphal': 5.5, 'nr': 12., 'alphar': 0.5},
                      'data_1': {'mu': 82., 'sigma': 13., 'nl': 10., 'alphal': 5.5, 'nr': 105., 'alphar': 1.},
                      'data_2': {'mu': 81., 'sigma': 17., 'nl': 10., 'alphal': 5.5, 'nr': 135., 'alphar': 0.9}
                      }

# {'mu': 78., 'sigma': 7., 'nl': 2., 'alphal': 8.6, 'nr': 120., 'alphar': 0.5}
models = {}
data = get_data_from_files(switch=2)
for sample in ["diboson", "ttbar", "Z+jets", "single top"]:
    model, x = initial_fitter(data, sample, initial_parameters, obs)
    plot_fit_result({sample: model}, data[sample], obs, sample=sample, x=x)
    models[sample] = model
    del data[sample]

data = get_data_from_files(switch=3)
cats = {'W + 0 jets': 'jet_n == 0', 'W + 1 jet': 'jet_n == 1', 'W + multi jets': 'jet_n > 1'}

for cat_name, cat_cut in cats.items():
    data[cat_name] = data['W+jets'].query(cat_cut)
    model, x = initial_fitter(data, cat_name, initial_parameters, obs)
    plot_fit_result({sample: model}, data[cat_name], obs, sample=cat_name, x=x)
    models[cat_name] = model
    del data[cat_name]

data = get_data_from_files(switch=1)
final_model = zfit.pdf.SumPDF([model for model in models.values()])
models['combined model'] = final_model
plot_fit_result(models, data['data'], obs, sample='data', x=x)


