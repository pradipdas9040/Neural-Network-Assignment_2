from common import np
import common
from common import np
from typing import Tuple
from constants import Defaults, Activations as act_fn_names, Optimizers as opt_names
from typing import Union, List

class Loss(object):
    def __init__(self): pass

    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:
        pass

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        pass

    @classmethod
    def mean_loss(cls, y1: np.ndarray, y2: np.ndarray) -> float:
        
        a = np.sum(cls.loss(y1=y1, y2=y2)) / len(y1)
        if common.cupy:
            # cupy returns an array instead of
            a = a.tolist()

        return a

class Activation(object):
    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        # apply the activation function
        # returns a matrix of row vectors for each datapoint
        pass

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> float:
        # return the gradients for each datapoint
        # as a matrix of row vectors (gradients)
        # OR
        # return the jacobian for each datapoint
        # as a 3D matrix of 2D matrices (jacobians)
        pass

    
class SquaredError(Loss):
    __name__ = 'squarederror'

    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:

        return 0.5 * np.square(np.linalg.norm(y1 - y2, axis=1, keepdims=True))

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:
        
        return y1 - y2


class CrossEntropy(Loss):
    __name__ = 'cross-entropy'
    @staticmethod
    def loss(y1: np.ndarray, y2: np.ndarray) -> float:

        return -np.sum(y2 * np.log2(y1), axis=1, keepdims=True)

    @staticmethod
    def gradient(y1: np.ndarray, y2: np.ndarray) -> float:

        return - y2 / y1

class Sigmoid(Activation):
    __name__ = 'sigmoid'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        return y * (1-y)


class ReLu(Activation):
    __name__ = 'relu'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        
        indices = np.nonzero(x > 0)  # indices where value is > 0

        out = np.zeros(x.shape)
        out[indices] = x[indices]

        return out

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        indices = np.nonzero(x > 0)  # indices where value is > 0

        out = np.zeros(x.shape)
        out[indices] = 1

        return out


class Softmax(Activation):
    __name__ = 'softmax'

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        
        a = np.exp(x)
        return a / np.sum(a, axis=1, keepdims=True)

    @staticmethod
    def gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        
        diag = y[..., np.newaxis] * np.eye(y.shape[1])
        outer = y[..., np.newaxis] * y[:, np.newaxis]
        jacobians = diag - outer

        return jacobians

    jacobian = gradient

#######
class Momentum:
    def __init__(self, beta1=Defaults.beta1, beta2=Defaults.beta2, learning_rate=Defaults.learning_rate):
        
        self._beta1 = beta1
        self._beta2 = beta2
        self._learning_rate = learning_rate
        self._eps = Defaults.eps

        # for weights
        self._m_w = None
        self._v_w = None

        # for bias
        self._m_b = None
        self._v_b = None

        self._weight_updates = None  # most recent weight update produced
        self._bias_updates = None  # most recent bias update produced

        self._time_step = 1

    @property
    def weight_updates(self):
        # most recent weight update produced
        return self._weight_updates

    @property
    def bias_updates(self):
        # most recent bias update produced
        return self._bias_updates

    def calculate_update(self, gradw: np.ndarray, gradb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        if self._m_w is None:
            self._m_w = np.zeros(gradw.shape)
            self._v_w = np.zeros(gradw.shape)
            self._m_b = np.zeros(gradb.shape)
            self._v_b = np.zeros(gradb.shape)

        # Biased moments
        self._m_w = self._beta1 * self._m_w + (1 - self._beta1) * gradw
        self._v_w = self._beta2 * self._v_w + (1 - self._beta2) * np.power(gradw, 2)

        self._m_b = self._beta1 * self._m_b + (1 - self._beta1) * gradb
        self._v_b = self._beta2 * self._v_b + (1 - self._beta2) * np.power(gradb, 2)

        # Unbiased moments
        mhat_w = self._m_w / (1 - np.power(self._beta1, self._time_step))
        vhat_w = self._v_w / (1 - np.power(self._beta2, self._time_step))

        mhat_b = self._m_b / (1 - np.power(self._beta1, self._time_step))
        vhat_b = self._v_b / (1 - np.power(self._beta2, self._time_step))

        self._weight_updates = self._learning_rate / (np.sqrt(vhat_w) + self._eps) * mhat_w
        self._bias_updates = self._learning_rate / (np.sqrt(vhat_b) + self._eps) * mhat_b

        self._time_step += 1

        return self._weight_updates, self._bias_updates
    
class GradientDescent:
    def __init__(self, learning_rate):
        
        self._learning_rate = learning_rate

        self._weight_updates = None  # most recent weight update produced
        self._bias_updates = None  # most recent bias update produced

    @property
    def weight_updates(self):
        # most recent weight update produced
        return self._weight_updates

    @property
    def bias_updates(self):
        # most recent bias update produced
        return self._bias_updates

    def calculate_update(self, gradw: np.ndarray, gradb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        self._weight_updates = self._learning_rate * gradw
        self._bias_updates = self._learning_rate * gradb

        return self._weight_updates, self._bias_updates
###

class Dense:
    __name__ = 'Dense'

    def __init__(self, input_size: int, n_nodes: int, act_fn: str,
                 weights: np.ndarray = None, biases: Union[List[float], np.ndarray] = None):
        

        self._input_size = input_size
        self._n_nodes = self._output_size = n_nodes
        self._act_fn_name = act_fn
        self._act_fn = self.__select_act_fn(act_fn)
        self._weights: np.ndarray = None  # weight array
        self._biases: np.ndarray = None  # bias row vector
        self.__initialise_weights(weights=weights)
        self.__initialise_biases(biases=biases)
        self.__optimizer = Momentum()

        # stash
        self.input: np.ndarray = None  # stash for input row vectors
        self.preactivated_output: np.ndarray = None  # stash for pre activated output row vectors
        self.output: np.ndarray = None  # stash for output row vectors

        self.upstream_grad: np.ndarray = None  # stash for upstream grad row vectors (dL/dy) (received from next layer)
        self.grad_weight: np.ndarray = None  # stash for most recent weight update matrix (dL/dw)
        # saves the weight updates as received from backprop. These must be subtracted from the current weights
        # (possibly with a learning rate)
        self.weight_updates_history: np.ndarray = None  # stash for previous weight update matrix
        # saves the amount of weight updates made when weights were updated last time
        # (these were subtracted from the weights)
        self.grad_bias: np.ndarray = None  # stash for most recent bias update row vector (dL/db)
        self.bias_updates_history: np.ndarray = None  # stash for previous bias update row vector
        self.downstream_grad: np.array = None  # stash for downstream grad row vectors (dL/dx)

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    def set_optimizer(self, optimizer, learning_rate=Defaults.learning_rate, momentum_factor=Defaults.momentum_factor,
                      beta1=Defaults.beta1, beta2=Defaults.beta2):
        # Changes the optimizer
        # Use with caution. If the optimizer is changed mid-learning.
        self.__optimizer = self.__select_optimizer(optimizer, learning_rate=learning_rate,
                                                   momentum_factor=momentum_factor, beta1=beta1, beta2=beta2)

    @staticmethod
    def __select_act_fn(fn_name: str):
        # TODO: Have an option to take in vectorised function as input so that it can be shared across layers

        if fn_name == act_fn_names.sigmoid:
            return Sigmoid

        if fn_name == act_fn_names.relu:
            return ReLu

        if fn_name == act_fn_names.softmax:
            return Softmax


        raise TypeError(f'Unknown activation function \'{fn_name}\'')

    @staticmethod
    def __select_optimizer(optimizer, learning_rate=Defaults.learning_rate, momentum_factor=Defaults.momentum_factor,
                           beta1=Defaults.beta1, beta2=Defaults.beta2):
        if optimizer == opt_names.gd:
            return GradientDescent(learning_rate=learning_rate)
        if optimizer == opt_names.adam:
            return Momentum(beta1=beta1, beta2=beta2, learning_rate=learning_rate)

        raise TypeError(f'Unknown optimizer function \'{optimizer}\'')

    def reset_weights(self):
        self.__initialise_weights()
        self.__initialise_biases()

    def __initialise_weights(self, weights: np.ndarray = None):
        # initialise weights. If weights is not None use that value
        if weights is not None:
            self._weights = weights
            return

        self._weights = np.random.randn(self._input_size, self._output_size) * np.sqrt(2 / self._input_size)

    def __initialise_biases(self, biases: Union[List[float], np.ndarray] = None):
        # initialise biases. If biases is not None use that value
        # [[b1 b2 ---- b_n_nodes]]
        # bias is a row vector
        if biases is not None:
            self._biases = np.array([biases])
            return

        self._biases = np.zeros((1, self._n_nodes))

    def load_weights(self, weights: np.ndarray, biases: Union[List[float], np.ndarray]):
        # TODO: check sizes of weights and biases
        self._weights = weights
        self._biases = biases

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        
        # bias addition is broadcasted over all inputs

        self.input = x
        self.preactivated_output = x @ self._weights + self._biases
        self.output = self._act_fn.apply(self.preactivated_output)

        return self.output

    def backward_pass(self, gradients: np.ndarray):
        
        self.upstream_grad = gradients  # dL/dy

        # derivative wrt pre activated output
        if self._act_fn_name == act_fn_names.softmax:
            # derivative = upstream_grad @ Jacobian.T (for each datapoint)
            # add axis to make grad 3d from 2d
            # swap 2nd and 3rd axes of jacobian to take transpose for each jacobian
            d = self.upstream_grad[:, np.newaxis, :] @ \
                self._act_fn.jacobian(self.preactivated_output, self.output).swapaxes(1, 2)
            # convert 3D matrix back to 2D
            d = d[:, 0, :]

            # jacobian for softmax is symmetric, so no need of transpose but it has been retained to have generalisation

        elif self._act_fn_name in (act_fn_names.sigmoid, act_fn_names.relu):
            # derivative = gradient_vector * upstream_grad (for each datapoint)
            d = self._act_fn.gradient(self.preactivated_output, self.output) * self.upstream_grad
        else:
            raise TypeError

        # derivative wrt weights = a matrix  (dL/dW)
        self.grad_weight = self.input.T @ d

        # derivative wrt biases = a vector  (dL/db)
        self.grad_bias = np.sum(d, axis=0, keepdims=True)

        # derivative wrt inputs = a vector (dL/dx) (passed to the previous layer as upstream grads)
        self.downstream_grad = d @ self._weights.T

    def update_weights_biases(self):
        
        weight_updates, bias_updates = \
            self.__optimizer.calculate_update(gradw=self.grad_weight, gradb=self.grad_bias)

        self.weight_updates_history = weight_updates
        self.bias_updates_history = bias_updates

        self._weights -= weight_updates
        self._biases -= bias_updates
