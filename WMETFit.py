import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import mplhep as hep

from WHistograms import hist_dicts


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


# Minimizing the J/Psi model

def initial_fitter(data, obs):
    num_events = len(data['data'].index)
    bgr_yields = []
    bgr_models = []
    for sample, df in data.items():
        if sample not in ('data', 'W+jets', 'ttbar'):
            print('==========')
            print(f'Fitting {sample} background')
            bgr_yield = len(df.index)
            mu = zfit.Parameter(f"mu_{sample}", 80., 60., 120.)
            sigma = zfit.Parameter(f'sigma_{sample}', 8., 1., 100.)
            alpha = zfit.Parameter(f'alpha_{sample}', -.5, -10., 0.)
            n = zfit.Parameter(f'n_{sample}', 120., 0.01, 500.)
            n_bgr = zfit.Parameter(f'yield_{sample}', bgr_yield, 0., int(1.3 * bgr_yield), step_size=1)
            bgr_frag_model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)
            bgr_frag_model = bgr_frag_model.create_extended(n_bgr)
            bgr_models.append(bgr_frag_model)
            bgr_yields.append(n_bgr)
            bgr_data = format_data(df, obs)
            # Create NLL
            nll = zfit.loss.ExtendedUnbinnedNLL(model=bgr_frag_model, data=bgr_data)
            # Create minimizer
            minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
            result = minimizer.minimize(nll)

            if result.valid:
                print("Result is valid")
                print("Converged:", result.converged)
                # param_errors = result.hesse()
                params = result.params
                print(params)
                if not bgr_frag_model.is_extended:
                    raise Warning('MODEL NOT EXTENDED')
            else:
                raise Warning(f'Background {sample} fit failed')

        if sample == 'ttbar':
            print('==========')
            print(f'Fitting {sample} background')
            bgr_yield = len(df.index)

            mu = zfit.Parameter(f"mu_{sample}", 80., 60., 120.)
            sigma = zfit.Parameter(f'sigma_{sample}', 8., 1., 100.)
            alpha = zfit.Parameter(f'alpha_{sample}', -.5, -10., 0.)
            n = zfit.Parameter(f'n_{sample}', 120., 0.01, 500.)
            n_bgr = zfit.Parameter(f'yield_{sample}', int(0.5*bgr_yield), 0., int(1.3 * bgr_yield), step_size=1)

            bgr_frag_model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)
            bgr_frag_model = bgr_frag_model.create_extended(n_bgr)

            ad_mu = zfit.Parameter(f'ad_mu_{sample}', 80., 60., 120.)
            ad_sigma = zfit.Parameter(f'ad_sigma_{sample}', 8., 1., 100.)
            ad_yield = zfit.Parameter(f'n_ad_{sample}', int(0.5*bgr_yield), 0., int(1.3 * bgr_yield), step_size=1)

            gauss = zfit.pdf.Gauss(mu=ad_mu, sigma=ad_sigma, obs=obs)
            gauss = gauss.create_extended(ad_yield)

            ttbar_model = zfit.pdf.SumPDF([bgr_frag_model, gauss])

            bgr_models.append(ttbar_model)
            bgr_yields.append(n_bgr)
            bgr_data = format_data(df, obs)
            # Create NLL
            nll = zfit.loss.ExtendedUnbinnedNLL(model=ttbar_model, data=bgr_data)
            # Create minimizer
            minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
            result = minimizer.minimize(nll)

            if result.valid:
                print("Result is valid")
                print("Converged:", result.converged)
                # param_errors = result.hesse()
                params = result.params
                print(params)
                if not bgr_frag_model.is_extended:
                    raise Warning('MODEL NOT EXTENDED')
            else:
                print(result)
                print(result.params)
                raise Warning(f'Background {sample} fit failed')

        if sample == 'W+jets':
            print('==========')
            print(f'Fitting {sample} signal')
            sig_yield = len(df.index)
            mu = zfit.Parameter(f"mu_{sample}", 80., 60., 120.)
            sigma = zfit.Parameter(f'sigma_{sample}', 8., 1., 100.)
            alpha = zfit.Parameter(f'alpha_{sample}', -.5, -10., 0.)
            n = zfit.Parameter(f'n_{sample}', 120., 0.01, 500.)
            n_sig = zfit.Parameter(f'yield_{sample}', int(0.7*sig_yield), 0., int(1.4 * sig_yield), step_size=1)
            signal_model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, n=n, alpha=alpha)
            signal_model = signal_model.create_extended(n_sig)

            ad_mu = zfit.Parameter(f'ad_mu_{sample}', 80., 60., 120.)
            ad_sigma = zfit.Parameter(f'ad_sigma_{sample}', 8., 1., 100.)
            ad_yield = zfit.Parameter(f'n_ad_{sample}', int(0.4 * sig_yield), 0., int(1.3 * sig_yield), step_size=1)

            gauss = zfit.pdf.Gauss(mu=ad_mu, sigma=ad_sigma, obs=obs)
            gauss = gauss.create_extended(ad_yield)

            Wjets_model = zfit.pdf.SumPDF([signal_model, gauss])

            signal_data = format_data(df, obs)
            # Create NLL
            nll = zfit.loss.ExtendedUnbinnedNLL(model=signal_model, data=signal_data)
            # Create minimizer
            minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True, tolerance=0.01)
            result = minimizer.minimize(nll)

            if result.valid:
                print("Result is valid")
                print("Converged:", result.converged)
                # param_errors = result.hesse()
                params = result.params
                print(params)
                if not signal_model.is_extended:
                    raise Warning('MODEL NOT EXTENDED')
                sig_parameters = {param[0].name: param[1]['value'] for param in result.params.items()}
            else:
                print('Minimization failed')
               # raise ValueError('Signal fit failed')

    mu = zfit.Parameter('data_mu', sig_parameters['mu_W+jets'], 60., 120.)
    sigma = zfit.Parameter('data_sigma',
                           sig_parameters['sigma_W+jets'],
                           1., 100.)
    alpha = zfit.Parameter('data_alpha',
                           -.5,
                           -10., 0.)
    n = zfit.Parameter('data_n',
                       100.,
                       0., 10000.)
    n_sig = zfit.Parameter('sig_yield', int(0.9 * num_events), 0., int(1.1*num_events), step_size=1)

    data_model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)
    data_model = data_model.create_extended(n_sig)
    bgr_models.append(signal_model)
    models = bgr_models
    for model in models:
        if not model.is_extended:
            raise Warning(f'A MODEL {model} IS NOT EXTENDED')
    data_to_fit = format_data(data['data'], obs, 'data')
    # Create NLL
    nll = zfit.loss.ExtendedUnbinnedNLL(model=data_model, data=data_to_fit)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
    result = minimizer.minimize(nll)

    if result.valid:
        print("Result is valid")
        print("Converged:", result.converged)
        param_errors = result.hesse()
        params = result.params
        print(params)
        return data_model, models
    if not result.valid:
        print("Error calculation failed \nResult is not valid")
        return None


# Plotting

def plot_fit_result(models, data, p_params, obs, sample='data'):
    plt_name = "mtw"
    print(f'Plotting {sample}')

    lower, upper = obs.limits

    h_bin_width = hist_dicts[plt_name]["bin_width"]
    h_num_bins = hist_dicts[plt_name]["numbins"]
    h_xmin = hist_dicts[plt_name]["xmin"]
    h_xmax = hist_dicts[plt_name]["xmax"]
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
    print('weights')
    print(data.totalWeight.values)

    main_axes.set_xlim(h_xmin, h_xmax)
    main_axes.set_ylim(0., 1.4*max(data_x))
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
    main_axes.legend(title=plt_label, loc="best")
    plt.savefig(f"../Results/{sample}_plot_{plt_name}_Complex.pdf")
    plt.close()
