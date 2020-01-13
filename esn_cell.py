import numpy as np
import pandas as pd
import copy

from sklearn.linear_model import Ridge
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.active_information_storage import ActiveInformationStorage

class ESN():

    def __init__(self, ESN_arch, activation=np.tanh, leak_rate=1.0, weights_std=0.1, sparsity=0.1, \
                 weights_external=None, use_normalized_weights=False):

        """
        Args:
            ESN_arch: 1-D int array, [no. of input units, no. of reservoir units, no. of output units] .
            activation: Nonlinear activation function.  Default: `np.tanh`.
            leak_rate: float64, (0,1], leaking rate of the reservoir units (alpha)
            weights_std: float64, variance of the normal dist. used in initializing weight matrices.
            sparsity: float64, [0,1], sparseness of the reservoir weight matrix. Default: 0.1.
            weights_external: if None, reservoir weights are created randomly,
              otherwise, a list of length 3 or 4 is expected, entries being weight matrices for input, bias, reservoir and optionally output
        """

        self.ESN_arch = ESN_arch
        self.in_units = ESN_arch[0]
        self.res_units = ESN_arch[1]
        self.out_units = ESN_arch[2]

        self.activation = activation
        self.leak_rate = np.float64(leak_rate)
        self.weights_std = np.float64(weights_std)
        self.sparsity = np.float64(sparsity)

        if weights_external != None:
            #Weights are given and don't need to be initialized randomly

            self.weights_in = weights_external[0]
            self.bias = weights_external[1]
            self.weights_res = weights_external[2]

            if len(weights_external) == 4: #output weights are given
                self.weights_out = weights_external[3]
            else:
                self.weights_out = np.random.normal(size=[self.res_units, self.out_units], scale=self.weights_std)
        
        else:
            
#             self.weights_in = np.random.normal(size=[self.in_units, self.res_units], scale=self.weights_std)
            self.weights_in = np.random.uniform(-0.1, 0.1, size=[self.in_units, self.res_units])
            # dims: [in_units, res_units]

            self.weights_res = np.random.normal(size=[self.res_units, self.res_units], scale=self.weights_std)
            # dims: [res_units, res_units]

            self.bias = np.random.normal(size=[1, self.res_units], scale=1e-40)
            # dims: [1, res_units]

            self.weights_out = np.random.normal(size=[self.res_units, self.out_units], scale=self.weights_std)
            # dims: [res_units, out_units]
            
            self.sparse_mask = np.float64(np.less_equal(np.random.uniform(size=[self.res_units, self.res_units]), \
                                                        self.sparsity))
            # dims: [res_units, res_units]
            
            # W_res is transformed for making it sparse
            self.weights_res = np.multiply(self.weights_res, self.sparse_mask)
        
        # Compute Spectral Radius
        self.spectral_radius = np.abs(np.linalg.eigvals(self.weights_res)).max()
   
        if (use_normalized_weights):
            self.weights_res = np.multiply(self.weights_res, 1/(self.spectral_radius))
            self.normalized_spectral_radius = np.abs(np.linalg.eigvals(self.weights_res)).max()
            

    def res_states(self, inputs, init_state, compute_readout = False):

        """ Runs one feedforward step of ESN.

        Args:
          inputs: shape [<timeserieslength> x self.in_units]
          init_state: shape [1 x self.res_units]
          compute_readout: True if readout of esn should be computed

        Returns:
          res_state = (1 - alpha)*res_state + alpha*activation(weights_in*input
          + weights_res*state + bias).
          res_states: collection of res_state for all input timesteps
          outputs: if compute_readout is set to True: output of esn, shape [<timeserieslength> x self.out_units] else empty list
        """
        res_states = []
        outputs = []
        
        res_state = init_state
 
        len_in = int(np.size(inputs)/self.in_units)
    
        for n in range(len_in):
            
            new_state = self.activation(np.multiply(inputs[n] ,self.weights_in) + \
                                        np.matmul(res_state, self.weights_res) + \
                                        self.bias)
            
            res_state = np.multiply(1-self.leak_rate, res_state) + np.multiply(self.leak_rate, new_state)
            res_states.append(res_state)
            
            if compute_readout:
                output = self.out_states(res_state)
                outputs.append(output)

        res_states = np.array(res_states).reshape([len_in, self.res_units]) # dims: [<timeserieslength>, self.res_units]
        
        return res_states, res_state, np.array(outputs)

    
    #Computes the readout of the esn for one timestep (given the reservoir state and the readout weigths)
    def out_states(self, res_state):
        return np.matmul(res_state,self.weights_out).flatten()

    def train(self, res_states, targets):

        """ Train the readout/output weights using ridge regression

        Args:
          inputs: an array of reservoir_states at different timesteps, shape: [<timeserieslength> x self.res_units]
          targets: target values for the output units, shape: [<timeserieslength> x self.out_units]

        Returns:
            new weights that linarly map res_states to corresponding target values, shape [self.res_units x self.out_units]
        """

        ridge = Ridge(alpha=3e-9, fit_intercept = False)#, fit_intercept = True, normalize = True)
        #packed_targets = [(target) for target in targets]
        ridge.fit(res_states, targets)
        #print(res_states.shape,targets.shape,ridge.coef_.shape)
        new_weights_out = np.swapaxes(ridge.coef_,0,1)#.reshape([self.res_units, self.out_units])

        return new_weights_out

    
    def lyapunov_exponent(self, res_states=None, compute_res_states=False, inputs=None, init_state=None, perturbation_order=1e-12):

        """
        Computes Lyapunov Exponent of the ESN

        Args:
          inputs: shape [<timeserieslength> x self.in_units]
          init_state: shape [1 x self.res_units]
          perturbation_order: the order of perturbation of the state of the ESN

        Returns:
          A tuple (lyapunov_exp, percent_nof_samples)
        """

        pert_ord = perturbation_order
        len_in = int(np.size(inputs)/self.in_units)
        
        if compute_res_states:
            res_states = self.res_states(inputs=inputs, init_state=init_state)[0]

        # Instantiating a copy ESN
        copy_esn = ESN(self.ESN_arch, self.activation, self.leak_rate, self.weights_std, self.sparsity)
        copy_esn.weights_in = self.weights_in
        copy_esn.weights_res = self.weights_res
        copy_esn.bias = self.bias
        
        # length of discarded initial transient
        init_transient = 999
        
        # No. of samples for computing Lyapunov Exponent
        target_nof_samples = (len_in-init_transient-1)*self.res_units

        #df_n_avgdist = pd.DataFrame(columns=['Perturbed_Neuron','Initial_Perturbation','Avg_Dist'])
        ln_d1_d0 = []

        for pert_n in range(self.res_units):

            d_0 = np.random.uniform(pert_ord, 1.1*pert_ord, size=[])
            
            # Extracting Initial State for Copy of ESN
            init_copy_esn = copy.deepcopy(res_states[init_transient].reshape([1, self.res_units]))

            #Perturbing 'pert_n'
            np.put(init_copy_esn, ind=pert_n, v=d_0+init_copy_esn[0][pert_n])

            for step in range(1, len_in-init_transient):

                # res_state of 'esn' after 'step' timesteps of initial transient
                res_esn = res_states[init_transient+step].reshape([1, self.res_units])

                # one timestep input for 'copy_esn'
                input_copy_esn = inputs[init_transient+step] # dims: [1, in_units]

                # res_state of 'copy_esn' after one iteration over input
                res_copy_esn = copy_esn.res_states(inputs=input_copy_esn, init_state=init_copy_esn)[0] # dims: [1, res_units]
                
                # Euclidean distance between esn and copy_esn after 1 timestep
                d_1 = np.linalg.norm(res_copy_esn-res_esn)

                if (d_1==0):
                    init_copy_esn = copy.deepcopy(res_states[init_transient+step].reshape([1, self.res_units]))
                    d_0 = np.random.uniform(pert_ord, 1.1*pert_ord, size=[])
                    np.put(init_copy_esn, ind=pert_n, v=d_0+init_copy_esn[0][pert_n])
                else:
                    ln_d1_d0.append(np.log(d_1/d_0))
                    # Normalizing state of copy_esn to distance d_0 for next iteration over next input timestep
                    init_copy_esn = res_esn + np.multiply((d_0/d_1),np.subtract(res_copy_esn, res_esn))
                    
                    
            #df_n_avgdist.loc[pert_n+1] = [pert_n+1, d_0, np.mean(dist_esns)]

        LE = np.mean(ln_d1_d0)
        
        percent_nof_samples = (len(ln_d1_d0)/target_nof_samples)*100
        
        return LE, percent_nof_samples
    
    
    def transfer_entropy(self, res_states=None, compute_res_states=False, inputs=None, init_state=None):
        
        """
        Computes Transfer Entropy of the ESN using 'IDTxl' library

        Args:
        
        """
        
        sources_list = []
        targets_list = []
        nonzero_connect = 0
        total_TE = 0
        
        for n in range(self.res_units):
            sources = []
            for i in range(self.res_units):
                if self.sparse_mask[n][i] == 1.0:
                    nonzero_connect += 1
                    if i!=n:
                        sources.append(i)
            
            if len(sources)!=0:
                targets_list.append(n)
                sources_list.append(sources)
            
        if compute_res_states:
            res_states = self.res_states(inputs=inputs, init_state=init_state)[0]
        
        # length of discarded initial transient
        init_transient = 999
        
        # Dataset for IDTxl MultivariateTE()
        d = Data(res_states[init_transient:,:], dim_order='sp')
        
        # Settings for MultivariateTE() computation
        settings = {'cmi_estimator': 'JidtKraskovCMI',
                    'max_lag_sources': 2,
                    'min_lag_sources': 1,
                    'n_perm_max_stat': 200,
                    'n_perm_min_stat': 200,
                    'n_perm_omnibus': 400,
                    'n_perm_max_seq': 400,
                    'verbose':False}
        
        TE_results = MultivariateTE().analyse_network(settings=settings, data=d, 
                                                   targets=targets_list, sources=sources_list)
        
        for i in targets_list:
            
            if TE_results.get_single_target(i, fdr=False).omnibus_te != None:
                total_TE += TE_results.get_single_target(i, fdr=False).omnibus_te
            
        avg_TE = total_TE/len(targets_list)
            
        return TE_results, avg_TE, nonzero_connect
    
    
    def active_info_storage(self, res_states=None, compute_res_states=False, inputs=None, init_state=None):
        
        """
        Computes Active Information Storage of the ESN using 'IDTxl' library

        Args:
        
        """
        
        total_AIS = 0
        targets_list = []
        
        for n in range(self.res_units):
            sources = []
            for i in range(self.res_units):
                if self.sparse_mask[n][i] == 1.0:
                    if i!=n:
                        sources.append(i)
            
            if len(sources)!=0:
                targets_list.append(n)
  
            
        if compute_res_states:
            res_states = self.res_states(inputs=inputs, init_state=init_state)[0]
        
        # length of discarded initial transient
        init_transient = 999
        
        # Dataset for IDTxl MultivariateTE()
        d = Data(res_states[init_transient:,:], dim_order='sp')
        
        # Settings for ActiveInformationStorage() computation
        settings = {'cmi_estimator': 'JidtKraskovCMI',
                    'max_lag': 4,
                    'n_perm_max_stat': 200,
                    'n_perm_min_stat': 200,
                    'n_perm_mi': 400,
                    'verbose':False}
        
        AIS_results = ActiveInformationStorage().analyse_network(settings=settings, data=d,
                                                                 processes=targets_list)
        for i in targets_list:
            
            if AIS_results.get_single_process(process=i, fdr=False).ais != None:
                total_AIS += AIS_results.get_single_process(process=i, fdr=False).ais
            
        avg_AIS = total_AIS/len(targets_list)

        return AIS_results, avg_AIS


