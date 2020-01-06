"""
Visualize results gained during random- and neuroevolution-experiment
Call visualize(file_with_results [,parameters]) as done at the bottom of this file
"""

import dill
import matplotlib.pyplot as plt
import numpy as np

def postprocess(scores):
    #Filter data, like remove uninteresting areas of spectral_radii
    to_delete = []

    for idx, element in enumerate(scores["spectral_radius"]):
        if element >10:
            to_delete.append(idx)

    for idx in sorted(list(set(to_delete)), reverse = True):
        scores = del_index(scores, idx)

    return scores

def del_index(scores, index):
    for key in scores:
        scores[key].pop(index)
    return scores


#visualizes scores aquired neat algorithm, stored in resultfile
def visualize(resultfile, amount_y_axis = 2, plot_progress = False, make_relative_standard_deviations = True):

    with open(resultfile, "rb") as input_file:
        scores = dill.load(input_file)

    #Add calculated standard deviation of the fitness
    scores['std_fitness'] = [std_narma + std_mmse for std_mmse,std_narma,mmse,narma in zip(scores['std_mmse'],scores['std_narma'],scores['mmse'],scores['narma'])]

    if make_relative_standard_deviations:
        #make standard deviations relative (except std_fitness)
        scores['std_mc'] = [std/mc for std,mc in zip(scores['std_mc'],scores['mc'])]
        scores['std_mmse'] = [std/mmse for std,mmse in zip(scores['std_mmse'],scores['mmse'])]
        scores['std_narma'] = [std/narma for std,narma in zip(scores['std_narma'],scores['narma'])]

    if plot_progress:
        #Scores["error"]=[1/i for i in Scores["error"]]
        plt.plot(scores["fitness"], label = "fitness", linewidth = 1)
        plt.plot(scores["lyapunov"], label = "lyapunov", linewidth = 1)
        #plt.fill_between(range(len(scores["fitness"])), np.array(scores["fitness"])- np.array(scores["std_fitness"]), np.array(scores["fitness"])+np.array(scores["std_fitness"]), color = (1,0.3,0.3,0.5))
        plt.xlabel("generation")
        #plt.ylabel("")
        plt.grid()
        plt.legend()
        plt.tight_layout(pad=0.4)
        plt.show()

    #assert False

    fig = plt.figure(figsize=(3.1+amount_y_axis*0.6,2.3))

    ax1 = fig.add_subplot(111)

    ax1.plot(scores["lyapunov"],scores["mc"],marker = '.', markersize = 3, linewidth = 0, color = 'black', zorder = 2)
    #ax1.plot(scores["spectral_radius"],scores["std_mc"],marker = '.', markersize = 2, linewidth = 0, color = 'red')

    ax1.set_ylabel('fitness', color="black")
    for tl in ax1.get_yticklabels():
        tl.set_color('black')
    ax1.set_xlabel('spectral radius')
    ax1.grid(linestyle='--', linewidth=0.5)
    #ax1.set_xlim([-0.1,3.15])
    #ax1.set_xlim([-0.18,0.11])
    #ax1.set_ylim([0.87,1.4])
    #ax1.set_ylim([0.94,1.17])
    ax1.margins(0.165)

    if (amount_y_axis == 2):
        ax2 = ax1.twinx()

        ax2.plot(scores["lyapunov"],scores["std_mc"],marker = '.', markersize = 1.3, linewidth = 0, color = 'red', zorder = 1)
        #ax2.set_ylim([-0.015,0.2])

        ax2.set_ylabel('relative std.dev.', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax2.margins(0.165)

    plt.tight_layout(pad=0.3)
    plt.show()

#TODO: file as command line argument
#visualize("Neuroevolution_Experiment/neat_progress.pickle", plot_progress = True, amount_y_axis = 1)
#visualize("Neuroevolution_Experiment/neat_progress_2nd_longrun_from_ordered.pickle", plot_progress = True, amount_y_axis = 2)
visualize("Random_Experiment/random_esn_scores_lognormnobias3.pickle", plot_progress = False, amount_y_axis = 2)
#visualize("Random_Experiment/random_esn_scores_testing_lyapfunc1.pickle", plot_progress = True, amount_y_axis = 1)
