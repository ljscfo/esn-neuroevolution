"""
Applies MC, MMSE and NARMA benchmark tasks to ESNs
  esn's object's parameter input_range defines the range of the uniformly distributed range of random input ([-1; 1] for MC and MMSE, [0; 0.5] for NARMA)
  implementation of ESN is in esn_cell.py
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from esn_cell import *
from ddeint import ddeint

#Instantiates, runs and trains an ESN as implemented in esn_cell.py
class esn:

    #Plotting some network-output diagrams
    Plotting = False

    def __init__(self, ESN_arch, input_range, weights_std = 0.1, sparsity = 0.1, leakrate = 0.15, activation = np.tanh, weight_matrices = None):

        self.ESN_arch = ESN_arch
        self.input_range = input_range

        in_units = self.ESN_arch[0]

        self.esn = ESN(self.ESN_arch, activation, leakrate, weights_std, bias_std = 0, sparsity = sparsity, weights_external = weight_matrices)

        X_t, Y_t = self.prepare_narma()
        self.X_t = X_t
        self.Y_t = Y_t[0:6000]

        for k in range(1,301):
            self.Y_t = np.vstack((self.Y_t,np.hstack((np.random.uniform(self.input_range[0],self.input_range[1],300),X_t[300-k:6000-k])))) #Desired output for mmse/mc

        self.Y_t = np.swapaxes(self.Y_t,0,1)

        #split data (works for one input neuron only)
        self.input_pre = self.X_t[0:2000].reshape([2000, in_units]) #Heat up/Stabilize ESN
        self.input_train = self.X_t[2000:4000].reshape([2000, in_units]) #Inputs used for training
        self.input_post = self.X_t[4000:6000].reshape([2000, in_units]) #For testing trained readout's performance
        self.train_targets = self.Y_t[2000:4000,:] # desired output for narma (node 0) and mmse (nodes 1 to last)


    #Prepare input and target timeseries for NARMA
    def prepare_narma(self):
        length = 7000
        order = 30

        X_t = np.random.uniform(self.input_range[0],self.input_range[1],length+order)
        Y_t = np.zeros(order)

        for _,t in enumerate(range(order,order+length)):
            Y_t = np.append(Y_t, 0.2*Y_t[-1] + 0.004*Y_t[-1]*sum(Y_t[-1:-(order+1):-1]) + 1.5*X_t[t-(order)]*X_t[t-1] + 0.001)

        X_t = X_t[order:]
        Y_t = Y_t[order:]

        return X_t, Y_t

    #Feeds input into ESN and makes it calculate timesteps as well as training
    #returns benchmark scores of trained output in comparison to target output
    def calc_esn(self):

        _, res_state, outputs1 = self.esn.res_states(self.input_pre, np.zeros((1,self.ESN_arch[1])), compute_readout = True)
        res_states, res_state, outputs2 = self.esn.res_states(self.input_train, res_state, compute_readout = True)
        new_weights = self.esn.train(res_states, self.train_targets)#np.reshape(targets,2000))
        self.esn.weights_out = new_weights
        _, _, outputs3 = self.esn.res_states(self.input_post, res_state, compute_readout = True)

        narma_error = np.sqrt(np.mean(np.square(outputs3[:,0] - self.Y_t[4000:6000,0])) / np.var(self.Y_t[4000:6000,0]))
        mmse = np.sqrt(np.mean(np.square(self.Y_t[4000:6000,1:]-outputs3[:,1:]))/np.var(self.X_t[0:6000]))

        #Memory capacity
        mc = 0
        for delay in range(1,300):
            mc = mc + scipy.stats.pearsonr(self.Y_t[4000:6000,delay],outputs3[:,delay])[0]**2 #squared pearson correlation

        #mse = np.sum(np.square((np.reshape(outputs3,2000) - self.Y_t[4000:6000])))/len(outputs3)

        if self.Plotting:
            self.plot_network_output(np.vstack((outputs1,outputs2,outputs3)), 2, 1)

        return mc, mmse, narma_error

    #Plot output (parameter data) and target values of node with index node_idx
    # frequency indicates how often a plot is produced by setting the range of a random generator
    def plot_network_output(self, data, node_idx = 0, frequency = 1):
        #Plot target values and predicted values
        if np.random.randint(0, 1/frequency) == 0 :
            #plt.plot(self.X_t[5500:5700], marker = '.', markersize = 2.5, color = (0,0,1,0.3),linewidth = 1, label="input") #Targets
            plt.plot(self.X_t[5500:5700], marker = '.', markersize = 2.5, color = (1,0,0,0.6), linewidth = 0.7, label="target sequence") #Targets
            plt.plot(data[5500+2:5700+2, node_idx], color = 'green', label ="delay 2") #Predicted
            plt.plot(data[5500+10:5700+10, node_idx+8], color = 'blue', label ="delay 10") #Predicted
            plt.plot(data[5500+100:5700+100, node_idx+98], color = 'black', label ="delay 100") #Predicted
            plt.xlabel('timestep')
            plt.xticks(np.arange(20, 81, 20), np.arange(5520, 5581, 20))
            plt.xlim(20,80)
            #plt.legend()
            plt.show()
            assert False #End program after diagram is shown

    def calc_lyapunov(self):
        return self.esn.lyapunov_exponent(self.input_pre,np.zeros((1,self.ESN_arch[1])))[0]

    def get_spectral_radius(self):
        return self.esn.spectral_radius

#Testing
def __main__():
    esn_ins = esn((1,8,1))
    esn_ins.calc_esn()
