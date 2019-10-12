import dill
import matplotlib.pyplot as plt

def postprocess(scores):


    to_delete = []
    """
    for idx, element in enumerate(scores["spectral_radius"]):
        if element > 20:
            to_delete.append(idx)

    for idx, element in enumerate(scores["mmse"]):
        if element > 8:
            to_delete.append(idx)

    for idx, element in enumerate(scores["narma"]):
        if element > 2.6:
            to_delete.append(idx)
    """

    scores['std_mc'] = [std/mc for std,mc in zip(scores['std_mc'],scores['mc'])]
    scores['std_mmse'] = [std/mc for std,mc in zip(scores['std_mmse'],scores['mmse'])]


    for idx in sorted(list(set(to_delete)), reverse = True):
        scores = del_index(scores, idx)
    return scores

def del_index(scores, index):
    for key in scores:
        scores[key].pop(index)
    return scores


#visualizes scores aquired neat algorithm, stored in resultfile
def visualize(resultfile):

    with open(resultfile, "rb") as input_file:
        scores = dill.load(input_file)

    scores = postprocess(scores)

    with open(resultfile, "wb") as output_file:
        dill.dump(scores, output_file)
    
    #Scores["error"]=[1/i for i in Scores["error"]]

    #plt.plot(Scores["fitness"], label = "Fitness")
    #plt.plot(Scores["lyapunov"], label = "Lyapunov")

    if (False):

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(Scores["fitness"], 'b*' ,label = "fitness")
        ax1.plot(Scores["narma"], 'b+', label = "narma")
        ax1.plot(Scores["mmse"], 'bx', label = "mmse")
        ax1.set_ylabel('fitness', color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(Scores["lyapunov"], "r-", label = "lyapunov")
        ax2.set_ylabel('lyapunov', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        plt.legend()

        #plt.xlabel("Iteration")
        #plt.ylabel("Fitness")
        #plt.legend()
        plt.show()


    amount_axis = 2

    fig = plt.figure(figsize=(3.1+amount_axis*0.6,2.3))


    ax1 = fig.add_subplot(111)

    ax1.plot(scores["spectral_radius"],scores["mc"],marker = '.', markersize = 3, linewidth = 0, color = 'black', zorder = 2)
    #ax1.plot(scores["spectral_radius"],scores["std_mc"],marker = '.', markersize = 2, linewidth = 0, color = 'red')

    ax1.set_ylabel('MC', color="black")
    for tl in ax1.get_yticklabels():
        tl.set_color('black')
    ax1.set_xlabel('spectral radius ')
    ax1.grid(linestyle='--', linewidth=0.5)
    ax1.set_xlim([-0.17,0.12])
    #ax1.set_ylim([0.87,1.4])
    ax1.margins(0.165)

    if (amount_axis == 2):
        ax2 = ax1.twinx()

        ax2.plot(scores["spectral_radius"],scores["std_mc"],marker = '.', markersize = 1.3, linewidth = 0, color = 'red', zorder = 1)
        ax2.set_ylim([-0.095,0.7])

        ax2.set_ylabel('relative std.dev.', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        ax2.margins(0.165)

    #plt.grid(linestyle='--', linewidth=0.5)
    #plt.xlabel("lyapunov exponent")
    #plt.ylabel("MC")
    plt.tight_layout(pad=0.3)
    plt.show()

#TODO: file as command line argument
visualize("random_esn_scores_mc_mmse_2.pickle")
