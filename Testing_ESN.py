"""
Testing ESN with the task of timeseries-prediction using mackey glass equation
"""

import numpy as np
import matplotlib.pyplot as plt

from ESN_CELL import *
from ddeint import ddeint

#Defining Reservoir parameters
res_units = 100
in_units = 1
out_units = 1
ESN_arch = [in_units, res_units, out_units]

leakrate = 0.5
activation = np.tanh
weights_variance = 0.1
sparsity = 0.1

esn = ESN(ESN_arch, activation, leakrate, weights_variance, sparsity)

Load_mackey_glass = False #Load value table of mackey glass equation from file to save some time, or calculate and save it to file

if not Load_mackey_glass:
    model = lambda X,t,beta,n,tau,gamma : beta*((X(t-tau))/(1 + X(t-tau)**n)) - gamma*X(t)

    X_0 = lambda t:0.5 # history before t=0

    bifurcation_para = 10
    t = np.linspace(0,100,6000)
    X_t = ddeint(model, X_0, t, fargs=(2, bifurcation_para, 2, 1)) #beta=2, n=n, tau=2, gamma=1

    np.save("mackey_glass",X_t)

else:

    X_t = np.load("mackey_glass.npy", allow_pickle=True)


#split data
input_pre = X_t[0:2000].reshape([in_units, 2000])
input_train = X_t[2000:4000].reshape([in_units, 2000])
input_post = X_t[4000:6000].reshape([in_units, 2000])
targets = X_t[2005:4005].reshape([2000,in_units]) #inconsistent shape

_, res_state, outputs1 = esn.res_states(input_pre, np.zeros((1,res_units)))
res_states, res_state, outputs2 = esn.res_states(input_train, res_state)
new_weights = esn.train(res_states, targets)#np.reshape(targets,2000))
esn.weights_out = new_weights
_, _, outputs3 = esn.res_states(input_post, res_state)

#Plot target values and predicted values
plt.plot((outputs1+outputs2+outputs3)[0:5995], label ="ESN output") #Predicted
plt.plot(X_t[5:], label="Target values") #Targets
plt.legend()
plt.show()
