import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

#TODO: use consitently either shape [timeserieslength x e.g.in_units] or the other way round but not both
class ESN():

    def __init__(self, ESN_arch, activation=np.tanh, leak_rate=0.0, weights_std=0.1, sparsity=0.1, weights_external = None):

        """
        Args:
            ESN_arch: 1-D int array, [no. of input units, no. of reservoir units, no. of output units] .
            activation: Nonlinear activation function.  Default: `np.tanh`.
            leak_rate: float64, (0,1], leaking rate of the reservoir units (alpha)
            weights_std: float64, variance of the normal dist. used in initializing weight matrices.
            sparsity: float64, [0,1], sparseness of the reservoir weight matrix. Default: 0.1.
            weights_external: if None, reservoir weights are created randomly,
              otherwise, a list of length 4 is expected, entries being weight matrices for input, bias, reservoir, output
        """

        self.ESN_arch = ESN_arch
        self.in_units = ESN_arch[0]
        self.res_units = ESN_arch[1]
        self.out_units = ESN_arch[2]

        self.activation = activation
        self.leak_rate = np.float64(leak_rate)
        self.weights_std = np.float64(weights_std)
        self.sparsity = np.float64(sparsity)

        # All instantiations of the class have the same random seed
        # and hence the same weight initializations
        #np.random.seed(123)

        if weights_external != None:
            #Weights are given and don't need to be initialized randomly

            self.weights_in = weights_external[0]
            self.bias = weights_external[1]
            self.weights_res = weights_external[2]
            self.weights_out = weights_external[3]

        else:
            # Initialize 'W_in'
            self.weights_in = np.random.normal(size=[self.in_units, self.res_units], scale=self.weights_std)
            # dims: [in_units, res_units]

            # Initialize 'W_res'
            self.weights_res = np.random.normal(size=[self.res_units, self.res_units], scale=self.weights_std)
            # dims: [res_units, res_units]

            # Initialize 'bias'
            self.bias = np.random.normal(size=[1, self.res_units], scale=self.weights_std)
            # dims: [1, res_units]

            self.weights_out = np.random.normal(size=[self.res_units, self.out_units], scale=self.weights_std)

            # Compute sparse_mask
            # dims: [res_units, res_units]
            self.sparse_mask = np.float64(np.less_equal(np.random.uniform(size=[self.res_units, self.res_units]), \
                                                        self.sparsity))

            # W_res is transformed for making it sparse
            self.weights_res = np.multiply(self.weights_res, self.sparse_mask)

        # Compute Spectral Radius
        #print(np.linalg.eigvals(self.weights_res))
        self.spectral_radius = np.abs(np.linalg.eigvals(self.weights_res)).max()

        # W_res is normalized wrt spectral_radius
        self.weights_res = np.multiply(self.weights_res, 1/(self.spectral_radius))



    def res_states(self, inputs, init_state, compute_readout = False):

        """ Runs one feedforward step of ESN.

        Args:
          inputs: shape [self.in_units x <timeserieslength>]
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
        len_in = len(inputs[0])

        for n in range(len_in):
            """
            print(res_state.shape,self.weights_res.shape)
            print(np.matmul(res_state, self.weights_res).shape)
            print("some",self.weights_in.shape,self.bias.shape)
            """
            new_state = self.activation(np.multiply(self.weights_in, inputs[0][n]) + \
                                        np.matmul(res_state, self.weights_res) + \
                                        self.bias)

            res_state = np.multiply(1-self.leak_rate, res_state) + np.multiply(self.leak_rate, new_state)
            res_states.append(res_state)

            #TODO: Or output before leak?, different activiation function for output?, how to include activation into ridge
            if compute_readout:
                output = self.out_states(res_state)
                outputs.append(output)

        res_states = np.array(res_states).reshape([len_in, self.res_units]) # dims: [<timeserieslength>, self.res_units]

        return res_states, res_state, outputs

    #Computes the readout of the esn for one timestep (given the reservoir state and the readout weigths)
    def out_states(self, res_state):
        return sum(np.matmul(res_state,self.weights_out))
        
    def train(self, res_states, targets):

        """ Train the readout/output weights using ridge regression

        Args:
          inputs: an array of reservoir_states at different timesteps, shape: [<timeserieslength> x self.res_units]
          targets: target values for the output units, shape: [<timeserieslength> x self.out_units]

        Returns:
            new weights that linarly map res_states to corresponding target values shape [self.res_units x self.out_units]
        """

        ridge = Ridge(alpha=0.1, fit_intercept = False)
        #packed_targets = [(target) for target in targets]
        ridge.fit(res_states, targets)
        new_weights_out = ridge.coef_.reshape([self.res_units, self.out_units])

        return new_weights_out

    def lyapunov_exponent(self, inputs, init_state):

        """
        Computes Lyapunov Exponent of the ESN

        Args:
          inputs: shape [self.in_units x <timeserieslength>]
          init_state: shape [1 x self.res_units]

        Returns:
          A tuple (lyapunov_exp, df_n_avgdist)

        """

        res_states = []
        res_state = init_state
        len_in = len(inputs[0])

        for n in range(len_in):
            new_state = self.activation(np.multiply(self.weights_in, inputs[0][n]) + \
                                        np.matmul(res_state, self.weights_res) + \
                                        self.bias)

            res_state = np.multiply(1-self.leak_rate, res_state) + np.multiply(self.leak_rate, new_state)

            res_states.append(res_state)

        res_states = np.array(res_states).reshape([len_in, self.res_units]) # dims: [<timeserieslength>, self.res_units]



        # Instantiating a copy ESN
        copy_esn = ESN(self.ESN_arch, self.activation, self.leak_rate, self.weights_std, self.sparsity)
        copy_esn.weights_in = self.weights_in
        copy_esn.weights_res = self.weights_res
        copy_esn.bias = self.bias
        copy_esn.spectral_radius = self.spectral_radius

        # length of discarded initial transient
        init_transient = 999

        # Extracting Initial State for a Copy of ESN
        init_copy_esn = res_states[init_transient].reshape([1, self.res_units])

        # d_0 is the perturbation magnitude of order 1e-8
        np.random.seed(1)
        d_0 = np.random.normal(size=[], loc=1e-8, scale=1e-10)


        #df_n_avgdist = pd.DataFrame(columns=['Perturbed_Neuron','Initial_Perturbation','Avg_Dist'])
        ln_d1_d0 = []


        for pert_n in range(self.res_units):

            #Perturbing 'pert_n'
            np.put(init_copy_esn, ind=pert_n, v=d_0)

            for step in range(1, len_in-init_transient):

                # output of esn
                out_esn = res_states[init_transient+step].reshape([1, self.res_units])

                # one timestep Input for copy_esn
                input_copy_esn = inputs[:,init_transient+step].reshape([1,1]) # dims: [1,1]

                # output of copy_esn
                _, out_copy_esn = copy_esn.res_states(inputs=input_copy_esn, init_state=init_copy_esn) # dims: [1,100]

                # Euclidean distance between esn and copy_esn after 1 timestep
                d_1 = np.linalg.norm(out_copy_esn-out_esn)

                #dist_esns.append(d_1)
                ln_d1_d0.append(np.log(d_1/d_0))

                # Normalizing the state of esn_2 to the distance d_0
                # for next iteration over the next input timestep
                init_copy_esn = out_esn + np.multiply((d_0/d_1),np.subtract(out_copy_esn, out_esn))

            #df_n_avgdist.loc[pert_n+1] = [pert_n+1, d_0, np.mean(dist_esns)]

        lyapunov_exp = np.mean(ln_d1_d0)

        return lyapunov_exp #, df_n_avgdist
