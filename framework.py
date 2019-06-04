
import sys, os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork


#from pylab import *
#import pandas as pd
#import seaborn as sns

from ESN_CELL import *
from ddeint import ddeint



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
        #Still needs a lot of work
        matrices = get_weight_matrices()

        res_units = 100
        in_units = 1
        ESN_arch = [in_units, res_units]

        leakrates = [0.1, 0.25, 0.5, 0.75, 0.9]
        activation = np.tanh
        weights_variance = 0.1
        sparsity = 0.1

        esn = ESN(ESN_arch, activation, leakrates[0], weights_variance, sparsity, matrices)

        #TODO: don't do that step for every evaluation but once
        model = lambda X,t,beta,n,tau,gamma : beta*((X(t-tau))/(1 + X(t-tau)**n)) - gamma*X(t)

        X_0 = lambda t:0.5 # history before t=0

        X_ts = {}

        bifurcation_para = [10]

        for i, n in enumerate(bifurcation_para):
            t = np.linspace(0,30,3000)
            X_t = ddeint(model, X_0, t, fargs=(2, n, 2, 1)) #beta=2, n=n, tau=2, gamma=1
            X_ts[n] = X_t

            #length of input series
        len_in = len(X_ts[10])

        esn_input = X_ts[10].reshape([in_units, len_in]) # of shape: [in_units, len_in]

        print("ESN_inp_type",type(esn_input))
        print(esn_input.shape)

        esn.res_states(esn_input, np.zeros((1,res_units)))

        #TODO: Continue working here

        graph = [list(),list()]

        #print("E",rmse)
        score = 1/(1+np.sqrt(rmse / len(pairs[1000:2000])))
        #print("S",score)
        return {'fitness':score,'graph':graph}

    def solve(self, network):
        score = self.evaluate(network)
        return score['fitness'] > 0.9


def get_weight_matrices(network):

    matrix = network.get_network_data()[0]

    #TODO: Check if bias is node or not (default False, but property of NEATGenotype)
    bias = 1

    nodes = network.node_genes   #: [] Tuples of (fforder, type, bias, responsraininge, layer)
    conns = network.conn_genes    #: {} Tuples of (innov, from, to, weight, enabled) conn[(i, j)]

    output_layer = np.int64(0) #output layer is max value for int64, but it's checked just in case

    nodes_per_layer = defaultdict(lambda: 0)
    node_layer = np.zeros(len(nodes) + bias)

    for i, node in enumerate(nodes):
        layer = node[4]
        nodes_per_layer[layer] += 1
        # Output layer has highest layer number
        if layer > output_layer:
            output_layer = layer

    node_layer[0] = 1
    for i, node in enumerate(nodes):
        layer = node[4]
        if layer == 0:
            node_layer[i+bias] = 0
        elif layer == output_layer:
            node_layer[i+bias] = 3
        else:
            node_layer[i+bias] = 2

    weights_in = matrix[:,node_layer == 0][node_layer == 2,:]
    weights_bias = matrix[:,node_layer == 1][1:,:]
    weights_reservoir = matrix[:,node_layer == 2][node_layer == 2,:]
    weights_out = matrix[:,node_layer == 2][node_layer == 3,:]

    return weights_in, weights_bias, weights_reservoir, weights_out

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

def init():
    topology = random_topology(10, 0.5)
    genotype = lambda: NEATGenotype(inputs=1, outputs=1, topology = topology, types=['tanh'], feedforward = False,prob_add_node=0.2,prob_add_conn=0.5)

    # Create a population
    pop = NEATPopulation(genotype, popsize=50)

    # Create a task
    task = ESNTask_internal()
    return task, pop


def run(task, population):
    for i in range(1):
        # Run the evolution, tell it to use the task as an evaluator
        pop.epoch(generations=10, evaluator=task, solution=task)

    print(get_weight_matrices(pop.champions[-1]))

    final = task.evaluate(pop.champions[-1])

    plt.plot(final['graph'][0][:400],label = "Original")
    plt.plot(final['graph'][1][:400], label = "predicted")
    plt.show()

task, pop = init()
run(task, pop)
