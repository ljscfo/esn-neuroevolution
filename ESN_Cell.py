import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl


class ESN(rnn_cell_impl.RNNCell):
    
    def __init__(self, ESN_arch, activation=tf.math.tanh, leak_rate=0.0, weights_std=0.1,\
                 sparsity=0.1, sparseness=True):
        """
        Args:
            ESN_arch: 1-D int array, [no. of input units, no. of reservoir units] .
            activation: Nonlinear activation function.  Default: `tf.tanh`.
            leak_rate: float (0,1], leaking rate of the reservoir units (alpha)
            weights_std: float, variance of the normal dist. used in initializing weight matrices.
            sparsity: float [0,1], sparseness of the reservoir weight matrix. Default: 0.1.
            sparseness: boolean, variable to store if sparsity is needed or not. Default: True
        """
        self.in_units = ESN_arch[0]
        self.res_units = ESN_arch[1]
        
        self.activation = activation
        self.alpha = leak_rate
        self.weights_std = weights_std
        self.sparsity = sparsity
        self.sparseness = sparseness
        
        
        self.weights_in = tf.get_variable("InputWeights", \
                                          initializer=self.init_weights_in(self.weights_std),\
                                          trainable=False)
            # 'weights_in' is: [in_units x res_units]
        
        self.weights_res = self.normalize_weights_res(tf.get_variable("ReservoirWeights", \
                                           initializer=self.init_weights_res(self.weights_std),\
                                           trainable=False))
            # 'weights_res' is: [res_units x res_units]
            
        self.bias = tf.get_variable("Bias", \
                                    initializer=self.init_bias(self.weights_std),\
                                    trainable=False)
            # 'bias' is: [1, res_units]
        
        self.spectral_radius = tf.get_variable("SpectralRadius",\
                                               initializer=self.get_spectral_radius(self.weights_res),\
                                               trainable=False)
        
        if self.sparseness:
            self.weights_res = self.sparse_weights_res(self.weights_res)
        
    @property
    def output_size(self):
        return self.res_units
   
    @property
    def state_size(self):
        return self.res_units
    
    def init_weights_in(self, var):
        
        """
        Initializes the input weight matrix.
        
        Args:
            var: variance of the normal distr. used to draw the weight values from
            
        Returns:
            Weight matrix of shape [in_units, res_units]
        """
        
        return tf.random.normal([self.in_units, self.res_units], stddev=var)
    
    def init_weights_res(self, var):
        
        """
        Initializes the reservoir weight matrix.
        
        Args:
            var: variance of the normal distr. used to draw the weight values from
            
        Returns:
            Weight matrix of shape [res_units, res_units]
        """
        
        return tf.random.normal([self.res_units, self.res_units], stddev=var)
    
    def init_bias(self, var):
        
        """
        Initializes the bias weight matrix.
        
        Args:
            var: variance of the normal distr. used to draw the weight values from
            
        Returns:
            Weight matrix of shape [1, res_units]
        """
        
        return tf.random.normal([1, self.res_units], stddev=var)
    
    def get_spectral_radius(self, W_res):
        
        """
        Computes the spectral radius of normalized reservoir matrix.
        
        Args:
            W_res: matrix of shape [res_units, res_units] with float values,
            reservoir weight matrix
            
        Returns:
            Float, Spectral radius of the normalized reservoir matrix
        """
        
        eigen = tf.linalg.eigh(W_res)
        spectral_radius = tf.math.reduce_max(tf.math.abs(eigen[0]))
        
        return spectral_radius
    
    def normalize_weights_res(self, W_res):
        
        """
        Computes normalized reservoir matrix, normalized with respect to the spectral radius.
        
        Args:
            W_res: matrix of shape [res_units, res_units] with float values,
            reservoir weight matrix
            
        Returns:
            Float entries, Normalized reservoir matrix with shape [res_units, res_units]
        """
        
        eigen = tf.linalg.eigh(W_res)
        spectral_radius = tf.math.reduce_max(tf.math.abs(eigen[0]))
        
        W_res = tf.scalar_mul((1/spectral_radius), W_res) 

        return W_res
    
    def sparse_weights_res(self, W_res):
        
        """
        Computes sparsed reservoir matrix with sparsity defined by __init__.
        Only called if sparseness == True
        
        Args:
            W_res: matrix of shape [res_units, res_units] with float values,
            reservoir weight matrix
            
        Returns:
            Float entries, Sparsed reservoir matrix with shape [res_units, res_units]
        """
        
        sparse_mask = tf.less_equal(tf.random_uniform(W_res.shape, minval=0, maxval=1),\
                                    self.sparsity)
        sparse_mask = tf.cast(sparse_mask, dtype=W_res.dtype)
        
        W_res = tf.multiply(W_res, sparse_mask)

        return W_res
        
    def __call__(self, esn_input, state):
        
        """ Runs one feedforward step of ESN.

        Args:
          esn_input: 3-D Tensor, of shape [batch_size x input_size x in_units]
          state: 2-D Tensor, of shape [batch_size x self.state_size]

        Returns:
          A tuple (output, new_state), computed as:
          output = new_state = (1 - alpha)*state + alpha*activation(weights_in*esn_input 
          + weights_res*state + bias).
          
        """
        
        
        new_state = (1-self.alpha)*state + \
                    self.alpha*self.activation(tf.matmul(tf.ones([1,1]), self.bias) + \
                                               tf.matmul(esn_input, self.weights_in) + \
                                               tf.matmul(state, self.weights_res))
        
        output = new_state
        
        return output, new_state
