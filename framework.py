
"""
Evolve Echo State Networks (ESNs) using NEAT neuroevolution algorithm

ESN code is implemented in ESN_CELL.py and gets instanciated in Testing_ESN.py
  Task for evaluating performance of ESN is timeseries prediction using mackey glass equation

NEAT is used to evolve reservoir nodes and connections which are normally just random
The readout/output weights of ESN are trained in ESN_CELL.py each time Network gets changed by NEAT

Top level parameters for ESN and NEAT are defined in init() method
"""

import sys, os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from ddeint import ddeint

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

from Testing_ESN import *
from ESN_CELL import *


# Testing Neat without ESN code on mackey_glass timeseries, partly copied from neat examples
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

#evaluate a network evolved by neat using ESN with mackey_glass
class ESNTask_external(object):

    def __init__(self, ESN_arch):
        self.ESN_arch = ESN_arch

    def evaluate(self, network, verbose=False):
        matrices = self.get_weight_matrices(network)

        #set up ESN with weight matrices from NEAT
        esn_instance = test_esn(self.ESN_arch, matrices)
        #Run task in ESN and get performance (mean squared error)
        mse = esn_instance.calc_esn()

        score = 1.0/mse

        return {'fitness':score}

    def solve(self, network):
        score = self.evaluate(network)
        return score['fitness'] > 0.9

    def get_weight_matrices(self, network):

        matrix = network.get_network_data()[0]

        #TODO: Check if bias is node or not (default False, but property of NEATGenotype)
        bias = 1

        nodes = network.node_genes   #: [] Tuples of (fforder, type, bias, responsraininge, layer)
        self.ESN_arch=(self.ESN_arch[0], len(nodes)-self.ESN_arch[0]-self.ESN_arch[2],self.ESN_arch[2])
        conns = network.conn_genes    #: {} Tuples of (innov, from, to, weight, enabled) conn[(i, j)]

        output_layer = np.int64(0) #output layer is supposed to be max value for int64, but it's checked just in case

        nodes_per_layer = defaultdict(lambda: 0)
        node_layer = np.zeros(len(nodes) + bias)

        for i, node in enumerate(nodes):
            layer = node[4]
            nodes_per_layer[layer] += 1

            # Output layer has highest layer number
            if layer > output_layer:
                output_layer = layer

        #bias node/layer
        node_layer[0] = 1

        for i, node in enumerate(nodes):
            layer = node[4]
            if layer == 0:
                node_layer[i+bias] = 0 #Input layer
            elif layer == output_layer:
                node_layer[i+bias] = 3 #Output layer
            else:
                node_layer[i+bias] = 2 #reservoir layer

        weights_in = np.rot90(matrix[:,node_layer == 0][node_layer == 2,:])
        weights_bias = np.rot90(matrix[:,node_layer == 1][node_layer == 2,:])
        weights_reservoir = np.rot90(matrix[:,node_layer == 2][node_layer == 2,:])
        weights_out = np.rot90(matrix[:,node_layer == 2][node_layer == 3,:])

        #Pack matrices
        matrices = weights_in, weights_bias, weights_reservoir, weights_out

        for matrix_i, matrix in enumerate(matrices):
            #in earlier version additional output node appeared
            assert not (matrix_i == 3 and matrix.shape[1] > self.ESN_arch[2])

            #Replace NaNs with 0s
            for y_i, y in enumerate(matrix):
                for x_i, x in enumerate(y):
                    if x!=x: #if x is NaN
                        matrices[matrix_i][y_i,x_i] = 0

        return matrices

def random_topology(n_nodes, sparsity):
    #just create tuples of kind (source_node, target_node) in a complicated way

    topology = defaultdict(lambda: 0)

    for i in range(math.floor((n_nodes**2-n_nodes)*sparsity)):
        while True:
            source = np.random.randint(0,n_nodes)
            target = np.random.randint(1,n_nodes)
            #TODO: Connection to self possible or not?
            if (source != target and topology[(source,target)] == 0):
                break
        topology[(source,target)] = 1

    return topology

#Initialize Neat and ESN
def init():
    #Defining node amounts for ESN
    res_units = 50
    in_units = 1
    out_units = 1
    ESN_arch = [in_units, res_units, out_units]
    sparsity = 0.2

    neat_iterations = 50
    neat_population_size = 50

    topology = random_topology(res_units+in_units+out_units, sparsity = sparsity)
    genotype = lambda: NEATGenotype(inputs=1, outputs=1, topology = topology, types=['tanh'], feedforward = False,prob_add_node=0.2,prob_add_conn=0.5)

    # Create a population
    pop = NEATPopulation(genotype, popsize = neat_population_size)

    # Create a task
    task = ESNTask_external(ESN_arch)
    #task = ESNTask_internal() #instead for just testing NEAT

    return task, pop, neat_iterations

#Called after every neat generation, just for saving each generation's fitness
def epoch_callback(self):
    global Scores
    Scores.append(self.champions[-1].stats['fitness'])

def run_neat(task, population, iterations):
    global Scores
    Scores = []

    pop.epoch(generations=iterations, evaluator=task, solution=task, callback=epoch_callback)

    #final = pop.champions[-1]

    plt.plot(Scores, label = "Fitness")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.show()


task, pop, neat_iterations = init()
run_neat(task, pop, neat_iterations)
