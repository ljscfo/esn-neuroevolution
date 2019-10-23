"""
Visualize results gained during random- and neuroevolution-experiment
Call visualize(file_with_results [,parameters]) as done at the bottom of this files
"""

import dill
import matplotlib.pyplot as plt

def postprocess(scores):
    #Filter data, like remove uninteresting areas of spectral_radii
    to_delete = []

    for idx, element in enumerate(scores["spectral_radius"]):
        if element > 20:
            to_delete.append(idx)

    for idx in sorted(list(set(to_delete)), reverse = True):
        scores = del_index(scores, idx)

    """
    #Replace resultfile with postprocessed one
    with open(resultfile, "wb") as output_file:
        dill.dump(scores, output_file)
    """

    return scores

def del_index(scores, index):
    for key in scores:
        scores[key].pop(index)
    return scores


#visualizes scores aquired neat algorithm, stored in resultfile
def visualize(resultfile, amount_y_axis = 2, plot_progress = False, make_relative_standard_deviations = True):

    with open(resultfile, "rb") as input_file:
        scores = dill.load(input_file)

    #scores = postprocess(scores)

    if make_relative_standard_deviations:
        #make standard deviations relative
        scores['std_mc'] = [std/mc for std,mc in zip(scores['std_mc'],scores['mc'])]
        scores['std_mmse'] = [std/mmse for std,mmse in zip(scores['std_mmse'],scores['mmse'])]
        scores['std_narma'] = [std/narma for std,narma in zip(scores['std_narma'],scores['narma'])]

    if plot_progress:
        #Scores["error"]=[1/i for i in Scores["error"]]
        plt.plot(scores["fitness"], label = "Fitness")
        plt.plot(scores["lyapunov"], label = "Lyapunov")
        plt.grid()
        plt.legend()
        plt.show()


    fig = plt.figure(figsize=(3.1+amount_y_axis*0.6,2.3))

    ax1 = fig.add_subplot(111)

    ax1.plot(scores["lyapunov"],scores["mmse"],marker = '.', markersize = 3, linewidth = 0, color = 'black', zorder = 2)
    #ax1.plot(scores["spectral_radius"],scores["std_mc"],marker = '.', markersize = 2, linewidth = 0, color = 'red')

    ax1.set_ylabel('narma', color="black")
    for tl in ax1.get_yticklabels():
        tl.set_color('black')
    ax1.set_xlabel('lyapunov')
    ax1.grid(linestyle='--', linewidth=0.5)
    #ax1.set_xlim([0.1,2.7])#[-0.17,0.12])
    #ax1.set_ylim([0.87,1.4])
    #ax1.margins(0.165)

    if (amount_y_axis == 2):
        ax2 = ax1.twinx()

        ax2.plot(scores["lyapunov"],scores["mc"],marker = '.', markersize = 1.3, linewidth = 0, color = 'red', zorder = 1)
        #ax2.set_ylim([-0.095,0.7])

        ax2.set_ylabel('mc not relative std.dev.', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        #ax2.margins(0.165)

    #plt.grid(linestyle='--', linewidth=0.5)
    #plt.xlabel("lyapunov exponent")
    #plt.ylabel("MC")
    plt.tight_layout(pad=0.3)
    plt.show()

#TODO: file as command line argument
visualize("neat_progress.pickle")
#visualize("random_esn_scores_narma.pickle")
#visualize("random_esn_scores_small_res.pickle")
