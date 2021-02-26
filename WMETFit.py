import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import csv

from WHistograms import hist_dicts


def format_data(data, obs):
    return zfit.Data.from_numpy(obs, data.to_numpy())


def create_initial_model(obs, data):
    # Crystal Ball
    num_events = len(data.index)
    mu = zfit.Parameter("mu", 80., 60., 120.)
    sigma = zfit.Parameter('sigma', 10, 1., 100.)
    alpha = zfit.Parameter('alpha', -1., -10., 0.)
    n = zfit.Parameter('n', 100., 0.01, 500.)
    model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)

    lambd = zfit.Parameter("lambda", -1., -10., 0.)
    model_bgr = zfit.pdf.Exponential(lambd, obs=obs)

    mu_add = zfit.Parameter("mu_add", 100., 60., 140.)
    sigma_add = zfit.Parameter("sigma_add", 50., 1., 200.)
    model_gauss = zfit.pdf.Gauss(mu=mu_add, sigma=sigma_add, obs=obs)

    n_add = zfit.Parameter("n_add",
                           int(num_events * 0.99), int(num_events * 0.2), int(num_events * 1.2), step_size=1)
    n_sig = zfit.Parameter("n_signal",
                           int(num_events * 0.99), int(num_events * 0.5), int(num_events * 1.2), step_size=1)
    n_bgr = zfit.Parameter("n_bgr",
                           int(num_events * 0.01), 0, int(num_events * 0.5), step_size=1)

    model_extended = model.create_extended(n_sig)
    model_bgr_extended = model_bgr.create_extended(n_bgr)
    model_gauss_extended = model_gauss.create_extended(n_add)

    model = zfit.pdf.SumPDF([model_extended, model_bgr_extended])
    return model


# Minimizing the J/Psi model

def initial_fitter(data, model, obs):
    data = format_data(data, obs)
    # Create NLL
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
    result = minimizer.minimize(nll)
    if result.valid:
        print("Result is valid")
        print("Converged:", result.converged)
        param_errors = result.hesse()
        params = result.params
        print(params)
        if not result.valid:
            print("Error calculation failed \nResult is not valid")
            return None
        else:
            return {param[0].name: {"value": param[1]['value'], "error": err[1]['error']}
                    for param, err in zip(result.params.items(), param_errors.items())}
    else:
        print('Minimization failed \nResult: \n{0}'.format(result))
        return None


# Plotting

def plot_fit_result(models, data, p_params, obs):

    plt_name = "mtw"

    mu_v = p_params["mu"]["value"]
    mu_err = p_params["mu"]["error"]
    sigma_v = p_params["sigma"]["value"]
    sigma_err = p_params["sigma"]["error"]
    n_v = p_params["n"]["value"]
    n_err = p_params["n"]["error"]
    alpha_v = p_params["alpha"]["value"]
    alpha_err = p_params["alpha"]["error"]

    text = f"$\mu = {mu_v:.2f} \pm {mu_err:.2f}$\n" \
           f"$\sigma = {sigma_v:.2g} \pm {sigma_err:.2g}$\n" \
           r"$\alpha$" + f"$ = {alpha_v:.2g} \pm {alpha_err:.2g}$\n" \
           f"$n = {n_v:.2g} \pm {n_err:.2g}$\n"

    lower, upper = obs.limits

    h_bin_width = hist_dicts[plt_name]["bin_width"]
    h_num_bins = hist_dicts[plt_name]["numbins"]
    h_xmin = hist_dicts[plt_name]["xmin"]
    h_xmax = hist_dicts[plt_name]["xmax"]
    h_xlabel = hist_dicts[plt_name]["xlabel"]
    plt_label = "$W \\rightarrow l\\nu$"

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data.values, bins=bins)
    data_sum = data_x.sum()
    plot_scale = data_sum * obs.area() / h_num_bins

    plt.clf()
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.errorbar(bin_centers, data_x, xerr=h_bin_width / 2, fmt="ok", label="met_et")

    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())


    main_axes.set_xlabel(h_xlabel)

    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    for model_name, model in models.items():
        if type(model) is zfit.models.functor.SumPDF:
            main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins, label=model_name)
        else:
            main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, label=model_name)
    main_axes.legend(title=plt_label, loc="best")
    plt.text(0.65, 0.50, text, transform=main_axes.transAxes)
    plt.savefig("../Results/fit_plot_{0}_experiment.pdf".format(plt_name))
    plt.close()