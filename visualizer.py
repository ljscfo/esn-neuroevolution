import dill
import matplotlib.pyplot as plt

#visualizes scores aquired neat algorithm, stored in resultfile
def visualize(resultfile):

    with open(resultfile, "rb") as input_file:
        Scores = dill.load(input_file)

    #Scores["error"]=[1/i for i in Scores["error"]]

    #plt.plot(Scores["fitness"], label = "Fitness")
    #plt.plot(Scores["lyapunov"], label = "Lyapunov")
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

    plt.plot(Scores["lyapunov"],Scores["mc"],'b.')
    plt.xlabel("lyapunov exponent")
    plt.ylabel("mc")
    plt.show()

#TODO: file as command line argument
visualize("random_esn_scores.pickle")
