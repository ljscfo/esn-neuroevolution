"""
Visualize results gained during random reservoirs and neuroevolution-experiment
Call visualize() as done at the bottom of this file
    parameters:
        resultfile: pickle file with results from random reservoirs- or neuroevolutionexperiment
        amount_y_axis: amount of y axis for second/last generated diagram (e.g. display standard deviation with different scale on second axis)
        plot_progress: whether or not plot experiment's progress (mostly just for neuroevolution experiment)
        make_relative_standard_deviations: standard deviations in resultfiles are absolute, set make_relative_standard_deviations to True to make them relative
"""

import dill
import matplotlib.pyplot as plt
import numpy as np


#visualizes scores aquired neat algorithm, stored in resultfile
# see head of file for parameter description
def visualize(resultfile, amount_y_axis = 2, plot_progress = False, make_relative_standard_deviations = True):

    with open(resultfile, "rb") as input_file:
        scores = dill.load(input_file)

    with open(resultfile, "wb") as output_file:
        dill.dump(scores, output_file)


    #Add calculated standard deviation of the fitness
    scores['std_fitness'] = [std_narma + std_mmse for std_mmse,std_narma,mmse,narma in zip(scores['std_mmse'],scores['std_narma'],scores['mmse'],scores['narma'])]

    if make_relative_standard_deviations:
        #make standard deviations relative (except std_fitness)
        scores['std_mc'] = [std/mc for std,mc in zip(scores['std_mc'],scores['mc'])]
        scores['std_mmse'] = [std/mmse for std,mmse in zip(scores['std_mmse'],scores['mmse'])]
        scores['std_narma'] = [std/narma for std,narma in zip(scores['std_narma'],scores['narma'])]

    if plot_progress:
        plt.plot(scores["fitness"], label = "fitness", linewidth = 1)
        plt.plot(scores["lyapunov"], label = "lyapunov", linewidth = 1)
        #plt.fill_between(range(len(scores["fitness"])), np.array(scores["fitness"])- np.array(scores["std_fitness"]), np.array(scores["fitness"])+np.array(scores["std_fitness"]), color = (1,0.3,0.3,0.5))
        plt.xlabel("generation")
        plt.grid()
        plt.legend()
        plt.tight_layout(pad=0.4)
        plt.show()

    #assert False


    #Second diagram
    fig = plt.figure(figsize=(3.1+amount_y_axis*0.6,2.3))

    #first y-axis of diagram
    ax1 = fig.add_subplot(111)

    ax1.plot(scores["lyapunov"],scores["mc"],marker = '.', markersize = 3, linewidth = 0, color = 'black', zorder = 2)
    #ax1.plot(scores["spectral_radius"],scores["std_mc"],marker = '.', markersize = 2, linewidth = 0, color = 'red')

    ax1.set_ylabel('fitness', color="black")
    for tl in ax1.get_yticklabels():
        tl.set_color('black')
    ax1.set_xlabel('spectral radius')
    ax1.grid(linestyle='--', linewidth=0.5)

    ax1.margins(0.165)

    #second y-axis
    if (amount_y_axis == 2):
        ax2 = ax1.twinx()

        ax2.plot(scores["lyapunov"],scores["std_mc"],marker = '.', markersize = 1.3, linewidth = 0, color = 'red', zorder = 1)

        ax2.set_ylabel('relative std.dev.', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax2.margins(0.165)

    plt.tight_layout(pad=0.3)
    plt.show()

#visualize("Neuroevolution_Experiment/neat_progress.pickle", plot_progress = True, amount_y_axis = 2)
visualize("Random_Experiment/random_esn_scores_mc_mmse.pickle", plot_progress = False, amount_y_axis = 2)
