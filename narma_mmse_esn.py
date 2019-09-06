"""
Testing ESN with the NARMA and MMSE task
  implementation of ESN is in ESN_CELL.py
"""

import numpy as np
import matplotlib.pyplot as plt

from ESN_CELL import *
from ddeint import ddeint

#Instanciates, runs and trains an ESN as implemented in ESN_CELL.py
class esn:

    Plotting = True

    def __init__(self, ESN_arch, weight_matrices = None):

        self.ESN_arch = ESN_arch
        in_units = self.ESN_arch[0]
        leakrate = 0.5
        activation = np.tanh
        weights_variance = 0.1
        sparsity = 0.1

        self.esn = ESN(self.ESN_arch, activation, leakrate, weights_variance, sparsity, weight_matrices)

        X_t, Y_t = self.prepare_narma()
        self.X_t = X_t
        self.Y_t = Y_t[0:6000]

        for k in range(1,100,3):
            self.Y_t = np.vstack((self.Y_t,np.hstack((np.random.uniform(0,0.5,100),X_t[100-k:6000-k])))) #Desired output for mmse

        self.Y_t = np.swapaxes(self.Y_t,0,1)

        #split data
        #TODO: work on inconsistent shape
        self.input_pre = self.X_t[0:2000].reshape([in_units, 2000]) #Heat up/Stabilize ESN
        self.input_train = self.X_t[2000:4000].reshape([in_units, 2000]) #Inputs used for training
        self.input_post = self.X_t[4000:6000].reshape([in_units, 2000]) #For testing trained readout's performance
        self.train_targets = self.Y_t[2000:4000,:] # desired output for narma (node 0) and mmse (nodes 1 to last)

        #self.plot_train_data()

    #Plot some data the network is trained on
    # out_node_idx indicates the node for which the target values should be plotted
    def plot_train_data(self, out_node_idx = 0):
        plt.plot(self.X_t[2000:2500], label ="Input")
        plt.plot(self.train_targets[0:500, out_node_idx], label="Target values") #Targets
        plt.legend()
        plt.show()
        #assert False

    #Prepare data for timeseries
    def prepare_narma(self):
        length = 7000
        order = 30

        X_t = np.random.uniform(0,0.5,length+order)
        Y_t = np.zeros(order)

        for _,t in enumerate(range(order,order+length)):
            Y_t = np.append(Y_t, 0.2*Y_t[-1] + 0.004*Y_t[-1]*sum(Y_t[-1:-(order+1):-1]) + 1.5*X_t[t-(order)]*X_t[t-1] + 0.001)

        X_t = X_t[order:]
        Y_t = Y_t[order:]

        return X_t, Y_t

    #Feed input into ESN and make it calculate timesteps as well as training
    #returns mean squared error of trained readout in comparison to desired output
    def calc_esn(self):

        _, res_state, outputs1 = self.esn.res_states(self.input_pre, np.zeros((1,self.ESN_arch[1])), compute_readout = True)
        res_states, res_state, outputs2 = self.esn.res_states(self.input_train, res_state, compute_readout = True)
        new_weights = self.esn.train(res_states, self.train_targets)#np.reshape(targets,2000))
        self.esn.weights_out = new_weights
        _, _, outputs3 = self.esn.res_states(self.input_post, res_state, compute_readout = True)

        narma_error = np.sqrt(np.mean(np.square(outputs3[:,0] - self.Y_t[4000:6000,0])) / np.var(self.Y_t[4000:6000,0]))
        mmse = np.sqrt(np.mean(np.square(self.Y_t[4000:6000,1:]-outputs3[:,1:]))/np.var(self.X_t[0:6000]))

        #mse = np.sum(np.square((np.reshape(outputs3,2000) - self.Y_t[4000:6000])))/len(outputs3)

        if self.Plotting:
            self.plot_network_output(np.vstack((outputs1,outputs2,outputs3)), 10, 0.02)

        return mmse, narma_error

    #Plot output and target values of node with index out_node_idx
    # frequency indicates how often a plot is produced by setting the range of a random generator
    def plot_network_output(self, outputs, out_node_idx = 0, frequency = 1):
        #Plot target values and predicted values
        if np.random.randint(0, 1/frequency) == 0 :
            plt.plot(outputs[5500:6000,10], label ="ESN output") #Predicted
            plt.plot(self.Y_t[5500:6000,10], label="Target values") #Targets
            plt.legend()
            plt.show()
            #assert False

    def calc_lyapunov(self):
        return self.esn.lyapunov_exponent(self.input_pre,np.zeros((1,self.ESN_arch[1])))

def __main__():
    esn_ins = esn((1,8,1))
    esn_ins.calc_esn()
