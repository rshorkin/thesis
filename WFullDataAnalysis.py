import ROOT
import numpy as np
import pandas
import math
import uproot
import time

import concurrent.futures
from multiprocessing import Process

import os
import psutil
import gc
import uproot3

import WCuts
import infofile
import WSamples
from WHistograms import hist_dicts

import types
import uproot3_methods.classes.TH1


branches = ["eventNumber", "trigE", "trigM", "lep_pt", "lep_eta", "lep_phi", "lep_E", "lep_n",
            "lep_z0", "lep_charge", "lep_type", "lep_isTightID", "lep_ptcone30", "lep_etcone20",
            "lep_trackd0pvunbiased",
            "lep_tracksigd0pvunbiased", "met_et", "met_phi", "jet_n", "jet_pt", "jet_eta", "jet_phi",
            "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_LepTRIGGER", "scaleFactor_PILEUP",
            "mcWeight"
            ]
pandas.options.mode.chained_assignment = None

lumi = 10  # 10 fb-1
fraction = 1.
common_path = "/media/sf_Shared/data_13TeV/1lep/"
# save_choice = int(input("Save dataframes? 0 for no, 1 for yes\n")) todo
save_choice = 0
if save_choice != 1:
    save_file = None
elif save_choice == 1:
    save_file = "csv"


def calc_theta(lep_eta):
    return 2 * math.atan(math.exp(-lep_eta))


def top_weight(x):
    return x / abs(x)


def abs_value(x):
    return abs(x)


def calc_mtw(lep_pt, met_et, lep_phi, met_phi):
    return math.sqrt(2 * lep_pt * met_et * (1 - math.cos(lep_phi - met_phi)))


def get_xsec_weight(totalWeight, sample):
    info = infofile.infos[sample]
    weight = (lumi * 1000 * info["xsec"]) / (info["sumw"] * info["red_eff"])  # *1000 to go from fb-1 to pb-1
    weight = totalWeight * weight
    return weight


def to_GeV(x):
    return x / 1000.


def subtract(x, y):
    return x - y


def find_lead_jet(pt, y):
    max_pt = max(pt)
    index = np.where(pt == max_pt)
    if not len(y[index]) > 1:
        return y[index]
    else:
        return y[index][0]


def extract_from_vector(x):
    return x[0]


def calc_weight(mcWeight, scaleFactor_ELE, scaleFactor_MUON, scaleFactor_PILEUP, scaleFactor_LepTRIGGER):
    return mcWeight * scaleFactor_ELE * scaleFactor_MUON * scaleFactor_PILEUP * scaleFactor_LepTRIGGER


def read_file(path, sample, branches=branches):
#    print("=====")
#    print("Processing {0} file".format(sample))
    mem = psutil.virtual_memory()
    mem_at_start = mem.available / (1024 ** 2)
#    print(f'{sample}: Available Memory: {mem_at_start:.0f} MB')
    count = 0
    hists = {}
    start = time.time()
    batch_num = 0
    with uproot.open(path) as file:
        tree = file['mini']
        numevents = tree.num_entries
#        print(f'{sample}: Total number of events in file: {numevents}')

        for batch in tree.iterate(branches, step_size='30 MB', library='np',
                                  entry_stop=numevents*fraction):
#            print('==============')
            df = pandas.DataFrame.from_dict(batch)
            del batch
            num_before_cuts = len(df.index)
            count += num_before_cuts
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
 #           print("Events before cuts: {0}".format(num_before_cuts))
            df = df.query("met_et > 30000")
            df = df.query("trigE or trigM")

            fail = df[np.vectorize(WCuts.cut_tight)(df.lep_isTightID)].index
            df.drop(fail, inplace=True)

            fail = df[np.vectorize(WCuts.cut_multijet)(df.lep_pt, df.lep_ptcone30, df.lep_etcone20)].index
            df.drop(fail, inplace=True)

            df["mtw"] = np.vectorize(calc_mtw)(df.lep_pt, df.met_et, df.lep_phi, df.met_phi)
            df = df.query("mtw > 60000.")
            df["mtw"] = df["mtw"] / 1000

            e_df = df.copy()

            fail = e_df[np.vectorize(WCuts.cut_e_fiducial)(e_df.lep_type, e_df.lep_eta)].index
            e_df.drop(fail, inplace=True)

            if len(e_df.lep_type) != 0:  # to account for boson -> munu/mumu

                fail = e_df[np.vectorize(WCuts.cut_e_long)(e_df.lep_type, e_df.lep_trackd0pvunbiased,
                                                           e_df.lep_tracksigd0pvunbiased)].index
                e_df.drop(fail, inplace=True)

                fail = e_df[np.vectorize(WCuts.cut_e_long_impact)(e_df.lep_type, e_df.lep_z0, e_df.lep_eta)].index
                e_df.drop(fail, inplace=True)

                e_df['mtw_enu'] = e_df['mtw']

            mu_df = df.copy()

            fail = mu_df[np.vectorize(WCuts.cut_mu_fiducial)(mu_df.lep_type, mu_df.lep_eta)].index
            mu_df.drop(fail, inplace=True)

            if len(mu_df.lep_type) != 0:  # to account for boson -> enu/ee

                fail = mu_df[np.vectorize(WCuts.cut_mu_long)(mu_df.lep_type, mu_df.lep_trackd0pvunbiased,
                                                             mu_df.lep_tracksigd0pvunbiased)].index
                mu_df.drop(fail, inplace=True)

                fail = mu_df[np.vectorize(WCuts.cut_mu_long_impact)(mu_df.lep_type, mu_df.lep_z0, mu_df.lep_eta)].index
                mu_df.drop(fail, inplace=True)

                mu_df['mtw_munu'] = mu_df['mtw']

            df = pandas.concat([e_df, mu_df])
            del mu_df
            del e_df
            gc.collect()

            # df = df.sort_values(by="entry")

            df["met_et"] = df["met_et"] / 1000
            df['lep_E'] = df['lep_E'] / 1000
            df["lep_pt"] = df["lep_pt"] / 1000

            # Asymmetry related histograms
            df['pos_ele_eta'] = df.query('lep_type == 11 and lep_charge == 1')['lep_eta']
            df['pos_ele_eta'] = np.vectorize(abs_value)(df.pos_ele_eta)

            df['neg_ele_eta'] = df.query('lep_type == 11 and lep_charge == -1')['lep_eta']
            df['neg_ele_eta'] = np.vectorize(abs_value)(df.neg_ele_eta)

            df['pos_mu_eta'] = df.query('lep_type == 13 and lep_charge == 1')['lep_eta']
            df['pos_mu_eta'] = np.vectorize(abs_value)(df.pos_mu_eta)

            df['neg_mu_eta'] = df.query('lep_type == 13 and lep_charge == -1')['lep_eta']
            df['neg_mu_eta'] = np.vectorize(abs_value)(df.neg_mu_eta)

            df['lep_pt_j0'] = df.query('jet_n == 0')['lep_pt']
            df['lep_pt_j1'] = df.query('jet_n == 1')['lep_pt']
            df['lep_pt_j2'] = df.query('jet_n > 1')['lep_pt']

            df['mtw_j0'] = df.query('jet_n == 0')['mtw']
            df['mtw_j1'] = df.query('jet_n == 1')['mtw']
            df['mtw_j2'] = df.query('jet_n > 1')['mtw']

            df['met_et_j0'] = df.query('jet_n == 0')['met_et']
            df['met_et_j1'] = df.query('jet_n == 1')['met_et']
            df['met_et_j2'] = df.query('jet_n > 1')['met_et']

            df['lep_eta_j0'] = df.query('jet_n == 0')['lep_eta']
            df['lep_eta_j1'] = df.query('jet_n == 1')['lep_eta']
            df['lep_eta_j2'] = df.query('jet_n > 1')['lep_eta']

            if len(df.loc[df['jet_n'] > 0].index) > 0:
                temp_df = pandas.DataFrame()
                temp_df['eventNumber'] = df.loc[df['jet_n'] > 0]['eventNumber']
                for column in df.columns:
                    if 'jet' in column and column != 'jet_n':
                        temp_df[f'lead_{column}'] = np.vectorize(find_lead_jet)(df.loc[df['jet_n'] > 0]['jet_pt'],
                                                                                df.loc[df['jet_n'] > 0][column])
                temp_df['lead_jet_pt'] = temp_df['lead_jet_pt'] / 1000.
                temp_df['phi_diff'] = np.vectorize(subtract)(df.loc[df['jet_n'] > 0]['lep_phi'],
                                                             temp_df['lead_jet_phi'])
                temp_df['abs_phi_diff'] = np.vectorize(abs_value)(temp_df.phi_diff)
                df = pandas.merge(left=df, right=temp_df, left_on='eventNumber', right_on='eventNumber', how='left')

            num_after_cuts = len(df.index)
#            print(f"{sample}: Number of events after cuts: {num_after_cuts}")
            print(f'{sample}: Currently at {(count * 100 / numevents):.0f}% of events ({count}/{numevents})')

            for key, hist in hist_dicts.items():
                h_bin_width = hist["bin_width"]
                h_num_bins = hist["numbins"]
                h_xmin = hist["xmin"]
                x_var = hist["xvariable"]
                bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
                if x_var in df:
                    data_x, binning = np.histogram(df.loc[df[x_var].notnull()][x_var].values, bins=bins,
                                                   weights=df.loc[df[x_var].notnull()]['totalWeight'].values)
                else:
                    data_x = np.asarray([0 for bin in range(h_num_bins)])
                    binning = np.asarray(bins)
                data_x = data_x.astype('float64')
                histo = uproot3_methods.classes.TH1.from_numpy((data_x, binning))
                if key not in hists.keys():
                    hists[key] = histo
                else:
                    for i in range(len(hists[key])):
                        hists[key][i] += histo[i]
            if not os.path.exists(f'../DataForFit/{sample}/'):
                os.mkdir(f'../DataForFit/{sample}')
            f = uproot3.recreate(f'../DataForFit/{sample}/{sample}_{batch_num}.root')

            f['FitTree'] = uproot3.newtree({'mtw': uproot3.newbranch(np.float64, 'mtw'),
                                            'jet_n': uproot3.newbranch(np.int32, 'jet_n'),
                                            'totalWeight': uproot3.newbranch(np.float64, 'totalWeight')})

            f['FitTree'].extend({'mtw': df['mtw'].to_numpy(dtype=np.float64),
                                 'jet_n': df['jet_n'].to_numpy(dtype=np.int32),
                                 'totalWeight': df['totalWeight'].to_numpy(dtype=np.float64)})
            f.close()
            batch_num += 1
            del df
            gc.collect()
            # diagnostics
            mem = psutil.virtual_memory()
            actual_mem = mem.available/(1024 ** 2)
 #           print(f'{sample}: Current available memory {actual_mem:.0f} MB '
 #                 f'({100*actual_mem/mem_at_start:.0f}%)')

    file = uproot3.recreate(f'../Output/{sample}.root', uproot3.ZLIB(4))

    for key, hist in hists.items():

        file[key] = hist
 #       print(f'{key} histogram')
  #      file[key].show()

    file.close()

    mem = psutil.virtual_memory()
    actual_mem = mem.available / (1024 ** 2)
 #   print(f'Current available memory {actual_mem:.0f} MB '
  #        f'({100 * actual_mem / mem_at_start:.0f}% of what we started with)')
    print(f'{sample}: Finished!')
    print(f'{sample}: Time elapsed: {time.time() - start} seconds')
    return None


def read_sample(sample):
 #   print("###==========###")
    print("Processing: {0} SAMPLES".format(sample))
    start = time.time()
    frames = []
    for val in WSamples.samples[sample]["list"]:
        if "data" in sample:
            prefix = "Data/"
        else:
            prefix = "MC/mc_{0}.".format(infofile.infos[val]["DSID"])
        path = common_path + prefix + val + ".1lep.root"
        if not path == "":
            read_file(path, val)
        else:
            raise ValueError("Error! {0} not found!".format(val))
#    print("###==========###")
    print("Finished processing {0} samples".format(sample))
    print("Time elapsed: {0} seconds".format(time.time() - start))
    return None


def getting_data_main(switch=0):
    if switch == 0:
        samples = ["data_1", "diboson"]
    elif switch == 1:
        samples = ["data_2", 'ttbar']
    elif switch == 2:
        samples = ['W+jets_1', 'Z+jets']
    elif switch == 3:
        samples = ['W+jets_2', "single top"]
    else:
        raise ValueError("Option {0} cannot be processed".format(switch))
    for s in samples:
        read_sample(s)
    return None


def runInParallel(*fns):
    proc = []
    i = 0
    for fn in fns:
        p = Process(target=fn, args=(i,))
        p.start()
        proc.append(p)
        i += 1
    for p in proc:
        p.join()


runInParallel(getting_data_main, getting_data_main, getting_data_main, getting_data_main)
