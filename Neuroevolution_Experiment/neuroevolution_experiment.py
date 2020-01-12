
"""
Evolve Echo State Networks (ESNs) using NEAT neuroevolution algorithm
 in this experiment, neuroevolution is applied to initially random connection weights
 output nodes' weights of ESN are trained in esn_cell.py

ESN code is implemented in esn_cell.py and gets instantiated in benchmark_esn.py, where MC, MMSE and NARMA benchmarks are applied

parameters are defined via init() method
    reservoir_size: the esns' amount of reservoir neurons
    reservoir_sparsity: sparsity of the reservoirs
    input_range: uniform random range from which the input's timeseries is generated (we used (0,0.5))
    neat_iterations: number of neuroevolution's generations
    neat_population_size: amount of individuals for the neuroevolution
    esn_repetitions: how often to repeat benchmark calculations for one reservoir (mean is taken)
    spectral_radius: normally None, otherwise reservoirs' spectral radii are always set to that value
    neat_state_file: storage file for experiment's results and progress, in order to able to interrupt and continue an experiment
    start_anew: either start and initialize new neat run or load earlier started one
"""

import sys
import random
import numpy as np
import math
import dill
from collections import defaultdict
from functools import partial
from ddeint import ddeint
from functools import partialmethod
from functools import partial

sys.path.append(sys.path[0] + "/..") # Adds parent directory to python modules path in order to get files from there imported
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

import benchmark_esn


#evaluates a reservoir evolved by neat
class ESNTask_external(object):

    def __init__(self, ESN_arch, input_range, esn_repetitions):
        self.ESN_arch = ESN_arch
        self.esn_repetitions = esn_repetitions
        self.input_range = input_range

    def evaluate(self, genotype):
        #get fitness value and benchmark scores for an individual (NEATGenotype)

        matrices = self.get_weight_matrices(genotype)

        mcs = []
        mmses=[]
        narmas=[]

        #Because of the randomness involved, training of same ESN results in different readout weights
        # Taking the mean fitness value of multiple ESN-runs and -trainings to get steadier results
        for run in range(self.esn_repetitions):
            #set up ESN with weight matrices from NEAT (input, bias, reservoir)-weights; output weights are trained in ESN
            esn_instance = benchmark_esn.esn(self.ESN_arch, input_range = self.input_range, weight_matrices = matrices)
            #Run task in ESN and get performances on mc, mmse and narma
            mc, mmse, narma = esn_instance.calc_esn()
            mcs.append(mc)
            mmses.append(mmse)
            narmas.append(narma)

        mc = np.mean(mcs)
        mmse = np.mean(mmses)
        narma = np.mean(narmas)

        #calculate standart deviations
        std_mc = np.std(mcs)
        std_mmse = np.std(mmses)
        std_narma = np.std(narmas)

        fitness = 2-narma-mmse

        return {'fitness':fitness, 'mc':mc, 'std_mc':std_mc, 'mmse':mmse, 'std_mmse':std_mmse, 'narma':narma, 'std_narma':std_narma, 'spectral_radius':esn_instance.esn.spectral_radius}

    #calculates lyapunov exponent for a given reservoir (genotype)
    def calc_lyapunov(self, genotype):
        matrices = self.get_weight_matrices(genotype)
        #set up ESN with weight matrices from NEAT
        esn_instance = benchmark_esn.esn(self.ESN_arch, self.input_range, weight_matrices = matrices)

        lyapunov = esn_instance.calc_lyapunov()
        return lyapunov

    #necessary function for NEAT but only dummy use here
    def solve(self, genotype):
        score = self.evaluate(genotype)
        return score['fitness'] > 2

    #Converts NEATGenotype network data to weight matrices for ESN code
    def get_weight_matrices(self, genotype):

        bias = 1 #1 if bias is included and not a seperate node, 0 otherwise

        matrix = genotype.get_network_data()[0]

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

        #removed output weigths since they get trained via ridge regression
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

#just creates tuples of kind (source_node, target_node) in a complicated way
def random_topology(n_nodes, sparsity):

    topology = defaultdict(lambda: 0)

    for i in range(math.floor((n_nodes**2-n_nodes)*sparsity)):
        while True:
            source = np.random.randint(0,n_nodes)
            target = np.random.randint(1,n_nodes)
            if (topology[(source,target)] == 0):
                break
        topology[(source,target)] = 1

    return topology

#Initialize Neat and ESN (see start of file for parameter description)
def init(reservoir_size = 100,
        reservoir_sparsity = 0.1,
        input_range = (-1,1),
        neat_iterations = 1000,
        neat_population_size = 100,
        esn_repetitions = 3,
        initial_weights_std = 0.12,
        neat_state_file = "neat_progress.pickle", #neuroevolution state file
        start_anew = True ): #either start and initialize new neat run or load earlier started one

    global Scores
    reset_population = start_anew

    if not start_anew:
        #loads earlier started neuroevolution state as to continue it

        with open(neat_state_file, "rb") as input_file:
            Scores = dill.load(input_file)
            pop = dill.load(input_file)
            task = dill.load(input_file)

    else:

        #Defining node amounts for ESN
        res_units = reservoir_size
        in_units = 1
        out_units = 301
        ESN_arch = [in_units, res_units, out_units]
        sparsity = reservoir_sparsity

        neat_iterations = neat_iterations
        neat_population_size = neat_population_size

        Scores = {"fitness": [],"lyapunov": [], "mc": [], "std_mc":[], "mmse": [], "std_mmse": [] ,"narma": [], "std_narma":[], "spectral_radius":[]}

        topology = random_topology(res_units+in_units, sparsity = sparsity)
        genotype = lambda: NEATGenotype(inputs = in_units,
                                        outputs = 0,
                                        topology = topology,
                                        types=['tanh'],
                                        feedforward = False,
                                        prob_add_node = 0,
                                        prob_add_conn = 0,
                                        #max_nodes = 51,

                                        initial_weight_stdev = initial_weights_std,

                                        prob_mutate_weight = 0.2,
                                        stdev_mutate_weight = 0.02,
                                        prob_reset_weight = 0.008,

                                        prob_mutate_bias=0.1,
                                        stdev_mutate_bias = 0.01,
                                        bias_as_node=False,

                                        prob_reenable_conn=0,
                                        prob_disable_conn=0,
                                        #prob_reenable_parent=0.25,

                                        #weight_range=(-5., 5.),

                                        #distances for speciation
                                        distance_excess=1.0,
                                        distance_disjoint=1.0,
                                        distance_weight=1.0)
        # Create a population
        pop = NEATPopulation(genotype, popsize = neat_population_size, target_species = neat_population_size/neat_population_size, compatibility_threshold = 0.01, compatibility_threshold_delta = 0.001)

        # Create a task
        task = ESNTask_external(ESN_arch, input_range, esn_repetitions)
        #task = ESNTask_internal() #instead for just testing NEAT

    return task, pop, neat_iterations, reset_population, neat_state_file

# Called after every neat generation, just for saving each generation's fitness
def epoch_callback(self, task, state_file):
    global Scores
    Scores["fitness"].append(self.champions[-1].stats['fitness'])
    Scores["mc"].append(self.champions[-1].stats['mc'])
    Scores["std_mc"].append(self.champions[-1].stats['std_mc'])
    Scores["mmse"].append(self.champions[-1].stats['mmse'])
    Scores["std_mmse"].append(self.champions[-1].stats['std_mmse'])
    Scores["narma"].append(self.champions[-1].stats['narma'])
    Scores["std_narma"].append(self.champions[-1].stats['std_narma'])
    Scores["spectral_radius"].append(self.champions[-1].stats['spectral_radius'])

    #calculate lyapunov exponent of champion
    lyapunov = task.calc_lyapunov(self.champions[-1])
    Scores["lyapunov"].append(lyapunov)
    print("champs spectral radius",self.champions[-1].stats['spectral_radius'], "lyapunov exp:",lyapunov)

    # Save Scores, population, task to pickle file, as to enable possibility to interrupt neuroevolution
    with open(state_file, "wb") as output_file:
        dill.dump(Scores, output_file)
        dill.dump(self, output_file) #Population
        dill.dump(task, output_file)


#define experiment's parameters here:
task, population, neat_iterations, reset_population, neat_state_file = init(reservoir_size = 50,
                                                            input_range = (0,0.5),
                                                            neat_population_size = 75,
                                                            initial_weights_std = 1,
                                                            neat_state_file = "neat_progress.pickle",
                                                            start_anew = True)

#Run neuroevolution
population.epoch(generations = neat_iterations, evaluator = task, solution = task, callback = partial(epoch_callback, task = task, state_file = neat_state_file), reset = reset_population)
