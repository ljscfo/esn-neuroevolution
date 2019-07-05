"""
Testing ESN with the task of timeseries-prediction using mackey glass equation
  implementation of ESN is in ESN_CELL.py
"""

import numpy as np
import matplotlib.pyplot as plt

from ESN_CELL import *
from ddeint import ddeint

#Instanciates, runs and trains an ESN as implemented in ESN_CELL.py
class test_esn:

    Load_mackey_glass = True #Load value table of mackey glass equation from file to save some time, or calculate and save it to file
    Plotting = False

    def __init__(self, ESN_arch, weight_matrices = None):

        self.ESN_arch = ESN_arch
        in_units = self.ESN_arch[0]
        leakrate = 0.5
        activation = np.tanh
        weights_variance = 0.1
        sparsity = 0.1

        self.target_timegap = 5 #How many timesteps ahead of the input is the network supposed to predict

        self.esn = ESN(self.ESN_arch, activation, leakrate, weights_variance, sparsity, weight_matrices)

        X_t = self.prepare_mackey_glass()
        self.X_t = X_t

        #split data
        #TODO: work on inconsistent shape
        self.input_pre = X_t[0:2000].reshape([in_units, 2000]) #Heat up ESN
        self.input_train = X_t[2000:4000].reshape([in_units, 2000]) #Inputs used for training
        self.input_post = X_t[4000:6000].reshape([in_units, 2000]) #For testing trained readout's performance
        self.train_targets = X_t[2000+self.target_timegap:4000+self.target_timegap].reshape([2000,in_units]) #Desired output while training input is fed (Targets)

    #Prepare data for timeseries
    def prepare_mackey_glass(self):
        length = 6005

        if not self.Load_mackey_glass:
            model = lambda X,t,beta,n,tau,gamma : beta*((X(t-tau))/(1 + X(t-tau)**n)) - gamma*X(t)

            X_0 = lambda t:0.5 # history before t=0

            bifurcation_para = 10
            t = np.linspace(0,100,length)
            X_t = ddeint(model, X_0, t, fargs=(2, bifurcation_para, 2, 1)) #beta=2, n=n, tau=2, gamma=1

            np.save("mackey_glass",X_t)
        else:
            X_t = np.load("mackey_glass.npy", allow_pickle=True)

        return X_t

    #Feed input into ESN and make it calculate timesteps as well as training
    #returns mean squared error of trained readout in comparison to desired output
    def calc_esn(self):

        _, res_state, outputs1 = self.esn.res_states(self.input_pre, np.zeros((1,self.ESN_arch[1])))
        res_states, res_state, outputs2 = self.esn.res_states(self.input_train, res_state)
        new_weights = self.esn.train(res_states, self.train_targets)#np.reshape(targets,2000))
        self.esn.weights_out = new_weights
        _, _, outputs3 = self.esn.res_states(self.input_post, res_state)

        mse = np.sum(np.square((np.reshape(outputs3,2000) - self.X_t[4005:6005])))/len(outputs3)

        if (self.Plotting):
            #Plot target values and predicted values
            plt.plot((outputs1+outputs2+outputs3)[0:6000-self.target_timegap], label ="ESN output") #Predicted
            plt.plot(self.X_t[self.target_timegap:], label="Target values") #Targets
            plt.legend()
            plt.show()

        return mse

def __main__():
    esn_ins = test_esn((1,8,1))
    esn_ins.calc_esn()

#__main__()
