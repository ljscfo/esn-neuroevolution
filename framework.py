
import sys, os
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork



"""
in esn:
 get_network?
 set_network

  evaluate_network(network)
   return fitness
"""

"""
def init_neat():
    init_pop()
    pass

def run():
    train_neat_one_step(pop)
    networks = getnetworks(pop)
    for network in networks:
        esn.evaluate(network)
"""

class ESNTask_internal(object):

    mg = np.load('mackey_glass_t17.npy')
    INPUTS  = [(m,) for m in mg[:2000]]
    OUTPUTS = [(m,) for m in mg[1:2001]]
    EPSILON = 1e-100

    def __init__(self):
        self.INPUTS = np.array(self.INPUTS, dtype=float)
        self.OUTPUTS = np.array(self.OUTPUTS, dtype=float)

    def evaluate(self, network, verbose=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)

        if not network.node_types[-1](-1000) < -0.95:
            raise Exception("Network should be able to output value of -1, e.g. using a tanh node.")

        pairs = list(zip(self.INPUTS, self.OUTPUTS))
        #print(pairs[0:10])

        for (i, target) in pairs[:1000]:
            # Feed with bias
            output = network.feed(i)

        graph = [list(),list()]
        rmse = 0.0
        for (i, target) in pairs[1000:2000]:
            # Feed with bias
            output = network.feed(i)

            # Grab the output
            output = output[-len(target):]
            graph[0].append(target)
            graph[1].append(output)

            err = (target - output)
            #err[abs(err) < self.EPSILON] = 0;
            err = (err ** 2).mean()
            # Add error
            #print(("%r -> %r (should: %r) (%.2f)" % (i, output, target, err)))
            rmse += err
        #print("E",rmse)
        score = 1/(1+np.sqrt(rmse / len(pairs[1000:2000])))
        #print("S",score)
        return {'fitness':score,'graph':graph}

    def solve(self, network):
        score = self.evaluate(network)
        return score['fitness'] > 0.9

class ESNTask_external(object):


    def __init__(self):
        pass

    def evaluate(self, network, verbose=False):

        #Network to ESN_CELL
        nodes = network.node_genes   #: [] Tuples of (fforder, type, bias, response, layer)
        conns = network.conn_genes    #: {} Tuples of (innov, from, to, weight, enabled) conn[(i, j)]

        output_layer = np.int64(0) #output layer is max value for int64, but it's checked just in case

        nodes_per_layer = defaultdict(lambda: 0)
        print("TestDD:", nodes_per_layer[42])

        for i, node in enumerate(nodes):
            layer = node[4]
            nodes_per_layer[layer] += 1
            # Output layer has highest layer number
            if layer > output_layer:
                output_layer = layer

        n_res_nodes = 0 #Amount of reservoir nodes
        for layer in nodes_per_layer:
            if layer != 0 and layer != output_layer:
                n_res_nodes += nodes_per_layer[layer]

        weights_in = np.zeros((nodes_per_layer[0],n_res_nodes))

        for conn_key in conns:
            conn = conns[conn_key]
            enabled = conn[4]
            if :
                if conn[]

        graph = [list(),list()]

        #print("E",rmse)
        score = 1/(1+np.sqrt(rmse / len(pairs[1000:2000])))
        #print("S",score)
        return {'fitness':score,'graph':graph}

    def solve(self, network):
        score = self.evaluate(network)
        return score['fitness'] > 0.9

def get_network(network):
    nodes = network.node_genes   #: [] Tuples of (fforder, type, bias, response, layer)
    conns = network.conn_genes    #: {} Tuples of (innov, from, to, weight, enabled) conn[(i, j)]
    print("Nodes:",nodes)
    print("Connections:",conns)

    """
    Nodes: [[0.0, 'tanh', 0.0, 4.924273, 0], [1024.0, 'tanh', 0.0, 4.924273, 9223372036854775807], [512.0, 'tanh', 0.0, 4.924273, 1]]
    Connections: {(0, 1): [0, 0, 1, 0.038451762473330264, False], (0, 2): [1, 0, 2, 1.0, True], (2, 1): [2, 2, 1, 0.038451762473330264, True]}

    Nodes: [[0.0, 'tanh', 0.0, 4.924273, 0], [1024.0, 'tanh', 0.0, 4.924273, 9223372036854775807], [512.0, 'tanh', 0.9850203970958736, 4.924273, 1], [256.0, 'tanh', 0.0, 4.924273, 1], [768.0, 'tanh', 0.0, 4.924273, 2], [384.0, 'tanh', 0.0, 4.924273, 2]]
    Connections: {(0, 1): [0, 0, 1, 0.21498898225512877, True], (0, 2): [1, 0, 2, 1.0, True], (2, 1): [2, 2, 1, 0.03234019480256184, False], (1, 2): [4, 1, 2, 0.05097810092701511, True], (3, 2): [5, 3, 2, 1.0, False], (0, 3): [7, 0, 3, 1.0, True], (4, 1): [8, 4, 1, 0.03234019480256184, True], (3, 5): [10, 3, 5, 1.0, True], (5, 2): [11, 5, 2, 1.0, True], (1, 5): [12, 1, 5, -1.526050757421022, True]}

    """

def init():
    genotype = lambda: NEATGenotype(inputs=1, weight_range=(-1, 1), types=['tanh'], feedforward = False,prob_add_node=0.2,prob_add_conn=0.5)

    # Create a population
    pop = NEATPopulation(genotype, popsize=50)

    # Create a task
    task = ESNTask_internal()

    return task, pop

def run(task, population):
    for i in range(1):
        # Run the evolution, tell it to use the task as an evaluator
        pop.epoch(generations=10, evaluator=task, solution=task)

    get_network(pop.champions[-1])

    final = task.evaluate(pop.champions[-1])

    #print("Final:",final)
    plt.plot(final['graph'][0][:400],label = "Original")
    plt.plot(final['graph'][1][:400], label = "predicted")
    plt.show()

task, pop = init()
run(task, pop)
