import numpy as np
import pandas as pd
import time
import pickle

from esn_cell import *

#import os
#os.getcwd()


#%%______________________________TEST PARAMETERS__________________________________

res_units = [100]
in_units = 1

activation = np.tanh
leak_rates = [0.4]
sparsity = [0.10]

# Input of ESN
esn_input = np.random.uniform(0, 0.5, size=[2000, in_units])

# logs_beforeEOC = np.linspace(-3.5,-1.4,20) 
logs_atEOC = np.linspace(-1.39,0,100)
# logs_afterEOC = np.linspace(0.01,1.6,30)

# list_weights_sd = np.exp(np.concatenate((logs_beforeEOC, logs_atEOC, logs_afterEOC)))
list_weights_sd = np.exp(logs_atEOC[48:])

len(list_weights_sd)
# print(list_weights_std, '\n', logs_list)

#%%___________________________ITERATIONS_________________________________________

TE = pd.DataFrame(columns=['Reservoir_Size', 'Leak_Rate', 'Weights_SD', \
                           'Lyapunov_Exponent', 'Nof_Samples_(%)', 'TE', 'Nonzero_Connections'])

te_results = {}

n = 0

start = time.time()
    
for res_size in res_units:    
    
    ESN_arch = [in_units, res_size, 0]
    init_esn = np.zeros([1, res_size])
    
    for alpha in leak_rates:

        for sd in list_weights_sd:

            s = time.time()
            
            esn = ESN(ESN_arch, activation, alpha, weights_std=sd)

            reservoir_states = esn.res_states(inputs=esn_input, init_state=init_esn)[0]
            
            LE, perc_nof_samples = esn.lyapunov_exponent(res_states=reservoir_states, inputs=esn_input)
            
            _, te, nonzero = esn.transfer_entropy(res_states=reservoir_states)
            
            #_, ais = esn.active_info_storage(res_states=reservoir_states)

            TE.loc[n] = [res_size, alpha, sd, LE, perc_nof_samples, te, nonzero]
            
            with open('TE.pickle', 'wb') as f:
                pickle.dump(TE, f)  
            
            print('\n \033[1m Computation for n = ',n ,' completed in: ', (time.time()-s),' \033[0m \n\n\n')
            
#             te_results[n] = results

            n+=1

print(time.time()-start)


