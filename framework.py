
"""
Evolve Echo State Networks (ESNs) using NEAT neuroevolution algorithm

ESN code is implemented in ESN_CELL.py and gets instanciated in Testing_ESN.py
  Task for evaluating performance of ESN is timeseries prediction using mackey glass equation

NEAT is used to evolve reservoir nodes and connections which are normally just random
The readout/output weights of ESN are trained in ESN_CELL.py each time Network gets changed by NEAT

Top level parameters for ESN and NEAT are defined via init() method
"""

import sys, os
import random
import numpy as np
import math
import pickle
import dill
from collections import defaultdict
from functools import partial
from ddeint import ddeint
from functools import partialmethod
from functools import partial

sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

import narma_mmse_esn
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

    def __init__(self, ESN_arch, esn_repetitions):
        self.ESN_arch = ESN_arch
        self.esn_repetitions = esn_repetitions

    def evaluate(self, genotype):
        matrices = self.get_weight_matrices(genotype)

        mmses=[]
        narmas=[]

        #Because of the randomness involved, training of same ESN results in different readout weights
        # Taking the mean fitness value of multiple ESN-runs and -trainings to get steadier results
        for run in range(self.esn_repetitions):
            #set up ESN with weight matrices from NEAT (input, bias, reservoir)-weights; output weights are trained in ESN
            esn_instance = narma_mmse_esn.esn(self.ESN_arch, matrices)
            #Run task in ESN and get performances on narma and mmse
            mmse, narma_error = esn_instance.calc_esn()
            mmses.append(mmse)
            narmas.append(narma_error)

        score = 2-np.mean(mmses)-np.mean(narmas)

        return {'fitness':score}

    def calc_lyapunov(self, genotype):
        matrices = self.get_weight_matrices(genotype)
        #set up ESN with weight matrices from NEAT
        esn_instance = narma_mmse_esn.esn(self.ESN_arch, matrices)

        lyapunov = esn_instance.calc_lyapunov()
        return lyapunov

    def solve(self, genotype):
        score = self.evaluate(genotype)
        return score['fitness'] > 2

    def get_weight_matrices(self, genotype):

        matrix = genotype.get_network_data()[0]
        #TODO: node types (activation) (which we find in genotype.get_network_data()[1])
        #TODO: Check if bias is node or not (default False, but property of NEATGenotype)
        #TODO: remove output stuff traces, should get simpler
        bias = 1

        nodes = genotype.node_genes   #: [] Tuples of (fforder, type, bias, responsraininge, layer)
        self.ESN_arch=(self.ESN_arch[0], len(nodes)-self.ESN_arch[0],self.ESN_arch[2])
        conns = genotype.conn_genes    #: {} Tuples of (innov, from, to, weight, enabled) conn[(i, j)]

        #output_layer = np.int64(0) #output layer is supposed to be max value for int64, but it's checked just in case

        nodes_per_layer = defaultdict(lambda: 0)
        node_layer = np.zeros(len(nodes) + bias)

        for i, node in enumerate(nodes):
            layer = node[4]
            nodes_per_layer[layer] += 1

        #bias node/layer
        node_layer[0] = 1

        for i, node in enumerate(nodes):
            layer = node[4]
            if layer == 0:
                node_layer[i+bias] = 0 #Input layer
            else:
                node_layer[i+bias] = 2 #reservoir layer

        weights_in = np.rot90(matrix[:,node_layer == 0][node_layer == 2,:])
        weights_bias = np.rot90(matrix[:,node_layer == 1][node_layer == 2,:])
        weights_reservoir = np.rot90(matrix[:,node_layer == 2][node_layer == 2,:])
        #weights_out = np.rot90(matrix[:,node_layer == 2][node_layer == 3,:])

        #Pack matrices
        matrices = weights_in, weights_bias, weights_reservoir#, weights_out

        for matrix_i, matrix in enumerate(matrices):
            #in earlier versions additional output node appeared
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
def init(n_reservoir_units = 100, reservoir_sparsity = 0.1, neat_iterations = 1000, neat_population_size = 100, esn_repetitions = 3):
    #Defining node amounts for ESN
    res_units = n_reservoir_units
    in_units = 1
    out_units = 101
    ESN_arch = [in_units, res_units, out_units]
    sparsity = reservoir_sparsity

    neat_iterations = neat_iterations
    neat_population_size = neat_population_size

    global Scores
    Scores = {"error": [],"lyapunov": []}

    #TODO: Scaling of weigths regarding spectral radius is hidden for neat and done just in esn, is that the correct way?
    topology = random_topology(res_units+in_units, sparsity = sparsity)
    genotype = lambda: NEATGenotype(inputs=in_units, outputs=0, topology = topology, types=['tanh'], feedforward = False,prob_add_node=0,prob_add_conn=0.1)

    # Create a population
    pop = NEATPopulation(genotype, popsize = neat_population_size, min_elitism_size=5)

    # Create a task
    task = ESNTask_external(ESN_arch, esn_repetitions)
    #task = ESNTask_internal() #instead for just testing NEAT

    return task, pop, neat_iterations

#loads earlier started neuroevolution state as to continue it
def load_neat_state(statefile, neat_iterations = 1000):
    global Scores
    with open(statefile, "rb") as input_file:
        Scores = dill.load(input_file)
        pop = dill.load(input_file)
        task = dill.load(input_file)

    return task, pop, neat_iterations

#Called after every neat generation, just for saving each generation's fitness
def epoch_callback(self, task):
    global Scores
    Scores["error"].append(1/self.champions[-1].stats['fitness'])
    Scores["lyapunov"].append(task.calc_lyapunov(self.champions[-1]))

    # Save Scores, population, task to pickle file, as to enable possibility to interrupt neuroevolution
    with open(r"neat_progress.pickle", "wb") as output_file:
        dill.dump(Scores, output_file)
        dill.dump(self, output_file) #Population
        dill.dump(task, output_file)


start_anew = True #either initialize new neat run or load earlier started one
if start_anew:
    task, population, neat_iterations = init()
else:
    task, population, neat_iterations = load_neat_state("neat_progress.pickle")

population.epoch(generations=neat_iterations, evaluator=task, solution=task, callback=partial(epoch_callback, task=task), reset = start_anew)

    #champion = pop.champions[-1]
    #print("Lypunov Exponent:",task.calc_lyapunov(champion))
