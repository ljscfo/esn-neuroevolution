"""
Instantiates echo state networks (implemented in esn_cell.py (name still?))
 their performance on MC, MMSE and NARMA benchmarks is measured
 and the lyapunov exponent for each network is evaluated

 the experiment's parameters are part of experiment class:
    n_reservoirs: amount of esns to be evaluated
    reservoir_size: the esns' amount of reservoir neurons
    spectral_radius_from, spectral_radius_to: set uniformly distributed range of spectral radii of the experiment's reservoirs
    input_range: uniform random range from which the input's timeseries is generated ((-1,1) for MC/MMSE, (0,0.5) for NARMA)
    resultfile: file in which the experiment's results are stored
    append_results: whether results are stored in a new file (resultfile) or appended to an existing file (resultfile)
"""

import numpy as np
import random
import dill

import sys
sys.path.append(sys.path[0] + "/..") # Adds higher directory to python modules path in order to get files from there imported
from benchmark_esn import esn

#Creates n_esns esns with spectral radii in range [spectral_radius_from, spectral_radius_to]
# and calls run_esn()
# if append_results: appends results to existing file with results of previous experiment append results from run_esn() to resultfile
def experiment(n_reservoirs = 100, reservoir_size = 150, weights_std_range = (0.1,0.3), input_range = (-1,1), resultfile = "random_esn_scores.pickle", append_results = False):
    res_units = reservoir_size
    in_units = 1
    out_units = 301
    ESN_arch = [in_units, res_units, out_units]

    if (append_results):
        with open(resultfile, "rb") as input_file:
            scores = dill.load(input_file)
    else:
        scores = {"fitness": [],"lyapunov": [], "mc": [], "std_mc":[], "mmse": [], "std_mmse": [] ,"narma": [], "std_narma":[], "spectral_radius": [], "weights_std": []}

    for i_esn in range(n_reservoirs):

        #spectral_radius = random.uniform(0.22,0.26)#np.random.lognormal(0,3)
        weights_std = random.uniform(weights_std_range[0], weights_std_range[1])
        score = run_esn(ESN_arch, weights_std, input_range)

        scores["fitness"].append(score['fitness'])
        scores["mc"].append(score['mc'])
        scores["std_mc"].append(score['std_mc'])
        scores["mmse"].append(score['mmse'])
        scores["std_mmse"].append(score['std_mmse'])
        scores["narma"].append(score['narma'])
        scores["std_narma"].append(score['std_narma'])
        scores["lyapunov"].append(score['lyapunov'])
        scores["spectral_radius"].append(score['spectral_radius'])
        scores["weights_std"].append(weights_std)

        with open(resultfile, "wb") as output_file:
            dill.dump(scores, output_file)

        print(i_esn+1,"/",n_reservoirs,"done; spectral radius:",score['spectral_radius'],"; weights_std:,", weights_std, "; score:",score)#fitness:",score['fitness'])


#Create n_repetitions esns with node amounts defined in ESN_arch and spectral_radius
#Training and evaluation code is defined in benchmark_esn.py
#MC, MMSE and NARMA scores are averaged over n_repetitions and returned alongside with standard deviation
# ESN implementation is in esn_cell.py (still??)
def run_esn(ESN_arch, weights_std, input_range, n_repetitions = 3):

    mcs = []
    mmses=[]
    narmas=[]

    #todo: look at variance of repetitions against score, by that seeing what may cause not-convergence
    #Because of the randomness involved, training of same ESN results in different readout weights
    # Taking the mean fitness value of multiple ESN-runs and -trainings to get steadier results
    for run in range(n_repetitions):
        #set up ESN with weight matrices from NEAT (input, bias, reservoir)-weights; output weights are trained in ESN
        esn_instance = esn(ESN_arch, input_range, weights_std = weights_std)
        #Run task in ESN and get performances on narma and mmse
        mc, mmse, narma = esn_instance.calc_esn()
        mcs.append(mc)
        mmses.append(mmse)
        narmas.append(narma)

    mc = np.mean(mcs)
    mmse = np.mean(mmses)
    narma = np.mean(narmas)

    #calculate standart deviations
    std_mc = np.std(mcs)
    std_mmse = np.std(mmses)
    std_narma = np.std(narmas)

    score = 2-mmse-narma

    lyapunov = esn_instance.calc_lyapunov()

    return {'fitness':score, 'mc':mc, 'std_mc':std_mc,'mmse':mmse, 'std_mmse':std_mmse, 'narma':narma, 'std_narma':std_narma, 'lyapunov': lyapunov, 'spectral_radius':esn_instance.get_spectral_radius()}


experiment(n_reservoirs = 300, reservoir_size = 150, weights_std_range = (0,2), input_range = (0,0.5), resultfile = "random_esn_scores_narma.pickle", append_results = True)
