import dill
import matplotlib.pyplot as plt

#visualizes scores aquired neat algorithm, stored in statefile
def visualize(statefile):

    with open(statefile, "rb") as input_file:
        Scores = dill.load(input_file)

    #Scores["error"]=[1/i for i in Scores["error"]]

    #plt.plot(Scores["fitness"], label = "Fitness")
    #plt.plot(Scores["lyapunov"], label = "Lyapunov")
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(Scores["error"], label = "error")
    ax1.set_ylabel('error', color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(Scores["lyapunov"], "r-", label = "lyapunov")
    ax2.set_ylabel('lyapunov', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    #plt.xlabel("Iteration")
    #plt.ylabel("Fitness")
    #plt.legend()
    plt.show()

    plt.plot(Scores["lyapunov"],Scores["error"],'bo')
    plt.show()

visualize("neat_progress.pickle")
