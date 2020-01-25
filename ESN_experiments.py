import sys
import numpy as np
import matplotlib.pyplot as plt
#from pylab import *
import pandas as pd
import seaborn as sns
import time
import pickle

from esn_cell import *

import os
os.getcwd()


#______________________________TEST PARAMETERS__________________________________
res_units = [100]
in_units = 1

activation = np.tanh
leak_rates = [0.4]
sparsity = [0.10]

# Input of ESN
esn_input = np.random.uniform(0, 0.5, size=[2000, in_units])

# logs_beforeEOC = np.linspace(-3.5,-1.4,20)
# logs_atEOC = np.linspace(-1.3,0,100)
# logs_afterEOC = np.linspace(0.01,1.6,30)

# list_weights_sd = np.exp(np.concatenate((logs_beforeEOC, logs_afterEOC)))

logs = np.linspace(-1.4,0.7,100)
list_weights_sd = np.exp(logs)

len(list_weights_sd)
# print(list_weights_std, '\n', logs_list)

#___________________________ITERATIONS_________________________________________

TE_AIS = pd.DataFrame(columns=['Reservoir_Size', 'Leak_Rate', 'Weights_SD', \
                           'Lyapunov_Exponent', 'Nof_Samples_(%)', 'TE', 'Nonzero_Connections', 'AIS'])

te_results = {}

n = 1

start = time.time()
    
for res_size in res_units:    
    
    ESN_arch = [in_units, res_size, 0]
    init_esn = np.zeros([1, res_size])
    
    for alpha in leak_rates:

        for sd in list_weights_sd:

            s = time.time()
            
            esn = ESN(ESN_arch, activation, alpha, weights_std=sd)

            reservoir_states = esn.res_states(inputs=esn_input, init_state=init_esn)[0]
            
            LE, perc_nof_samples = esn.lyapunov_exponent(res_states=reservoir_states)
            
            _, te, nonzero = esn.transfer_entropy(res_states=reservoir_states)
            
            _, ais = esn.active_info_storage(res_states=reservoir_states)

            TE_AIS.loc[n] = [res_size, alpha, sd, LE, perc_nof_samples, te, nonzero, ais]
            
            with open('TE_AIS.pickle', 'wb') as f:
                pickle.dump(TE_AIS, f)  
            
            print('\n \033[1m Computation for n = ',n ,' completed in: ', (time.time()-s),' \033[0m \n\n\n')
            
#             te_results[n] = results

            n+=1

print(time.time()-start)

