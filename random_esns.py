"""
Generates random echo state networks; their performance on NARMA and MMSE tasks is measured
 and the lyapunov exponent for each network is evaluated
 writes results to file
"""

import narma_mmse_esn
import numpy as np
import random

import dill

#Creates n_esns esns with spectral radii in range [spectral_radius_from, spectral_radius_to]
# and calls run_esn()
# if append_results: appends results to existing file with results of previous experiment append results from run_esn() to resultfile
def experiment(n_esns = 100, spectral_radius_from = 0.5, spectral_radius_to = 1.2, resultfile = "random_esn_scores.pickle", append_results = False):
    res_units = 150
    in_units = 1
    out_units = 301
    ESN_arch = [in_units, res_units, out_units]

    if (append_results):
        with open(resultfile, "rb") as input_file:
            scores = dill.load(input_file)
    else:
        scores = {"fitness": [],"lyapunov": [], "mc": [], "std_mc":[], "mmse": [], "std_mmse": [] ,"narma": [], "std_narma":[], "spectral_radius": []}

    for i_esn in range(n_esns):

        spectral_radius = random.uniform(spectral_radius_from, spectral_radius_to)
        score = run_esn(ESN_arch, spectral_radius)

        scores["fitness"].append(score['fitness'])
        scores["mc"].append(score['mc'])
        scores["std_mc"].append(score['std_mc'])
        scores["mmse"].append(score['mmse'])
        scores["std_mmse"].append(score['std_mmse'])
        scores["narma"].append(score['narma'])
        scores["std_narma"].append(score['std_narma'])
        scores["lyapunov"].append(score['lyapunov'])
        scores["spectral_radius"].append(spectral_radius)

        with open(resultfile, "wb") as output_file:
            dill.dump(scores, output_file)

        print(i_esn+1,"/",n_esns,"done; spectral radius:",spectral_radius,";",score)#fitness:",score['fitness'])


#Create n_iterations esns with node amounts defined in ESN_arch and spectral_radius
#Training and evaluation code is defined in narma_mmse_esn.py
#MMSE and NARMA scores are averaged over n_iterations and returned
# ESN implementation is in ESN_CELL
def run_esn(ESN_arch, spectral_radius = 1, n_iterations = 3):

    mcs = []
    mmses=[]
    narmas=[]

    #todo: look at variance of repetitions against score, by that seeing what may cause not-convergence
    #Because of the randomness involved, training of same ESN results in different readout weights
    # Taking the mean fitness value of multiple ESN-runs and -trainings to get steadier results
    for run in range(n_iterations):
        #set up ESN with weight matrices from NEAT (input, bias, reservoir)-weights; output weights are trained in ESN
        esn_instance = narma_mmse_esn.esn(ESN_arch, spectral_radius = spectral_radius)
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

    return {'fitness':score, 'mc':mc, 'std_mc':std_mc,'mmse':mmse, 'std_mmse':std_mmse, 'narma':narma, 'std_narma':std_narma, 'lyapunov': lyapunov}

experiment(200, resultfile = "random_esn_scores.pickle", append_results = True)
