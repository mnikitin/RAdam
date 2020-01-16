import mxnet as mx
from mxnet.ndarray import NDArray, zeros, clip, sqrt
from mxnet.optimizer import Optimizer
import math

class Radam(Optimizer):
    """
    Parameters
    ----------
    beta1 : float, optional
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional
        Small value to avoid division by 0
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 **kwargs):
        super(Radam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.rho_inf = 2. / (1. - beta2) - 1.

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        # update m_t and v_t
        m_t, v_t = state
        m_t[:] *= self.beta1
        m_t[:] += (1. - self.beta1) * grad
        v_t[:] *= self.beta2
        v_t[:] += (1. - self.beta2) * grad * grad

        # bias corrected 1st moment
        m_t_prime = m_t / (1. - pow(self.beta1, t))
        
        # compute the length of approximated SMA
        beta2_pow_t = pow(self.beta2, t)
        rho_t = self.rho_inf - 2. * t * beta2_pow_t / (1. - beta2_pow_t)
        # update weights
        if rho_t > 4.:
            # bias-corrected 2nd moment
            v_t_prime = v_t / (1. - beta2_pow_t)
            # variance rectification term
            r_t = math.sqrt( (rho_t - 4.) * (rho_t - 2.) * self.rho_inf / 
                            ((self.rho_inf - 4.) * (self.rho_inf - 2.) * rho_t) )
            weight[:] -= lr * r_t * m_t_prime / (sqrt(v_t_prime) + self.epsilon)
        else:
            weight[:] -= lr * m_t_prime
