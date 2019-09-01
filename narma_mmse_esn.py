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

    Plotting = False

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

        for k in range(1,300,3):
            self.Y_t = np.vstack((self.Y_t,X_t[k:k+6000])) #Desired output for mmse

        self.Y_t = np.rot90(self.Y_t)

        #split data
        #TODO: work on inconsistent shape
        self.input_pre = self.X_t[0:2000].reshape([in_units, 2000]) #Heat up/Stabilize ESN
        self.input_train = self.X_t[2000:4000].reshape([in_units, 2000]) #Inputs used for training
        self.input_post = self.X_t[4000:6000].reshape([in_units, 2000]) #For testing trained readout's performance
        self.train_targets = self.Y_t[2000:4000,:] # desired output for narma

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

        #todo: plotting doesn't work right now!
        if (self.Plotting):
            #Plot target values and predicted values
            plt.plot((outputs1+outputs2+outputs3)[0:6000], label ="ESN output") #Predicted
            plt.plot(self.Y_t, label="Target values") #Targets
            plt.legend()
            plt.show()

        return mmse, narma_error

    def calc_lyapunov(self):
        return self.esn.lyapunov_exponent(self.input_pre,np.zeros((1,self.ESN_arch[1])))

def __main__():
    esn_ins = test_esn((1,8,1))
    esn_ins.calc_esn()

#__main__()
"""
#NARMA values plot
length = 50
order = 30

X_t = np.random.uniform(0,0.5,length+order)
Y_t = np.zeros(order)

for _,t in enumerate(range(order,order+length)):
    Y_t = np.append(Y_t, 0.2*Y_t[-1] + 0.004*Y_t[-1]*sum(Y_t[-1:-order:-1]) + 1.5*X_t[t-(order-1)]*X_t[t] + 0.001)

print(Y_t)

X_t = X_t[order:]
Y_t = Y_t[order:]

plt.plot(X_t)
plt.plot(Y_t)
plt.show()
"""
