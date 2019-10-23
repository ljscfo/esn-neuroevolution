
"""
Evolve Echo State Networks (ESNs) using NEAT neuroevolution algorithm

ESN code is implemented in ESN_CELL.py and gets instanciated in Testing_ESN.py
  Task for evaluating performance of ESN is timeseries prediction using mackey glass equation

NEAT is used to evolve reservoir nodes and connections which are normally just random
The readout/output weights of ESN are trained in ESN_CELL.py each time Network gets changed by NEAT

Top level parameters for ESN and NEAT are defined via init() method
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

sys.path.append(sys.path[0] + "/..") # Adds higher directory to python modules path in order to get files from there imported
#sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..'))
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.networks.rnn import NeuralNetwork

import benchmark_esns


#evaluate a reservoir evolved by neat
class ESNTask_external(object):

    def __init__(self, ESN_arch, esn_repetitions):
        self.ESN_arch = ESN_arch
        self.esn_repetitions = esn_repetitions

    def evaluate(self, genotype):
        matrices = self.get_weight_matrices(genotype)

        mcs = []
        mmses=[]
        narmas=[]

        #todo: look at variance of repetitions against score, by that seeing what may cause not-convergence
        #Because of the randomness involved, training of same ESN results in different readout weights
        # Taking the mean fitness value of multiple ESN-runs and -trainings to get steadier results
        for run in range(self.esn_repetitions):
            #set up ESN with weight matrices from NEAT (input, bias, reservoir)-weights; output weights are trained in ESN
            esn_instance = benchmark_esns.esn(self.ESN_arch, matrices, spectral_radius = None)
            #Run task in ESN and get performances on narma and mmse
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

        score = 2-narma-mmse
        #score = mc

        return {'fitness':score, 'mc':mc, 'std_mc':std_mc, 'mmse':mmse, 'std_mmse':std_mmse, 'narma':narma, 'std_narma':std_narma, 'spectral_radius':esn_instance.esn.spectral_radius}

    def calc_lyapunov(self, genotype):
        matrices = self.get_weight_matrices(genotype)
        #set up ESN with weight matrices from NEAT
        esn_instance = benchmark_esns.esn(self.ESN_arch, matrices, spectral_radius = None)

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
    out_units = 301
    ESN_arch = [in_units, res_units, out_units]
    sparsity = reservoir_sparsity

    neat_iterations = neat_iterations
    neat_population_size = neat_population_size

    global Scores
    Scores = {"fitness": [],"lyapunov": [], "mc": [], "std_mc":[], "mmse": [], "std_mmse": [] ,"narma": [], "std_narma":[], "spectral_radius":[]}

    #TODO: Scaling of weigths regarding spectral radius is hidden for neat and done just in esn, is that the correct way?
    topology = random_topology(res_units+in_units, sparsity = sparsity)
    genotype = lambda: NEATGenotype(inputs=in_units, outputs=0, topology = topology, types=['tanh'], feedforward = False, initial_weight_stdev = 0.04, stdev_mutate_weight = 0.04, stdev_mutate_bias = 0.01, prob_add_node=0, prob_add_conn=0, prob_reset_weight = 0.02)

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
    Scores["fitness"].append(self.champions[-1].stats['fitness'])
    Scores["mc"].append(self.champions[-1].stats['mc'])
    Scores["std_mc"].append(self.champions[-1].stats['std_mc'])
    Scores["mmse"].append(self.champions[-1].stats['mmse'])
    Scores["std_mmse"].append(self.champions[-1].stats['std_mmse'])
    Scores["narma"].append(self.champions[-1].stats['narma'])
    Scores["std_narma"].append(self.champions[-1].stats['std_narma'])
    Scores["spectral_radius"].append(self.champions[-1].stats['spectral_radius'])

    #calculate lyapunov exponent of champion
    Scores["lyapunov"].append(task.calc_lyapunov(self.champions[-1]))
    print("champs spectral r",self.champions[-1].stats['spectral_radius'])

    # Save Scores, population, task to pickle file, as to enable possibility to interrupt neuroevolution
    with open(r"neat_progress.pickle", "wb") as output_file:
        dill.dump(Scores, output_file)
        dill.dump(self, output_file) #Population
        dill.dump(task, output_file)


start_anew = False #either initialize new neat run or load earlier started one
if start_anew:
    task, population, neat_iterations = init(n_reservoir_units = 50, neat_population_size = 10)
else:
    task, population, neat_iterations = load_neat_state("neat_progress.pickle")

population.epoch(generations=neat_iterations, evaluator=task, solution=task, callback=partial(epoch_callback, task=task), reset = start_anew)

    #champion = pop.champions[-1]
    #print("Lypunov Exponent:",task.calc_lyapunov(champion))
