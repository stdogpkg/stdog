"""
Trace Functions
==============================

The core module responsible to calc trace functions.


Given a semi-positive definite matrix :math:`A \in \mathbb R^{|V|\\times|V|}`,
which has the set of eigenvalues given by :math:`\{\lambda_i\}` a trace of
a matrix function is given by

.. math:: 

    \mathrm{tr}(f(A)) = \sum\limits_{i=0}^{|V|} f(\lambda_i)

The methods for calculating such traces functions have a
cubic computational complexity lower bound,  :math:`O(|V|^3)`.
Therefore, it is not feasible for  large networks. One way
to overcome such computational complexity it is use stochastic approximations
combined with a mryiad of another methods
to get the results with enough accuracy and with a small computational cost. 
The methods available in this module uses the Sthocastic Lanczos Quadrature, 
a procedure proposed in the work made by Ubaru, S. et.al. [1] (you need to cite them).

References
^^^^^^^^^^

[1] Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via
Stochastic Lanczos Quadrature. SIAM Journal on Matrix Analysis and
Applications, 38(4), 1075-1099.

[2] Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
influence matrix for laplacian smoothing splines. Communications in
Statistics-Simulation and Computation, 19(2), 433-450.
"""
import tensorflow as tf
import numpy as np

from emate.symmetric.slq import pyslq as slq


def estrada_index(L, num_vecs=100, num_steps=50, device="/gpu:0"):
    """ Given the Laplacian matrix :math:`L \in \mathbb R^{|V|\\times |V|}` s.t. for all :math:`v \in \mathbb R^{|V|}`  we have :math:`v^T L v > 0` the Estrada Index
    is given by

    .. math::

      \mathrm{tr}\exp(L) = \sum\limits_{i=0}^{|V|} e^{\lambda_i}


    Parameters
    ----------
        L: sparse matrix
        num_vecs: int
            Number of  random vectors used to approximate the trace
            using the Hutchison's trick [1]
        num_steps: int
            Number of Lanczos steps or Chebyschev's moments
        device: str
            "/cpu:int" our "/gpu:int"
    
    Returns
    -------
        approximated_estrada_index: float

    References 
    ----------

    1 - Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via 
    Stochastic Lanczos Quadrature.  SIAM Journal on Matrix Analysis and Applications, 
    38(4), 1075-1099.
    
    2 - Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
    influence matrix for laplacian smoothing splines. Communications in
    Statistics-Simulation and Computation, 19(2), 433-450.
    """
    def trace_function(eig_vals):
        return tf.exp(eig_vals)

    approximated_estrada_index, _ = slq(
        L, num_vecs, num_steps,  trace_function, device=device)

    return approximated_estrada_index


def entropy(L_sparse, num_vecs=100, num_steps=50, device="/gpu:0"):
    """Compute the spectral entropy

    .. math::

      \sum\limits_{i=0}^{|V|} f(\lambda_i)
      
      f(\lambda) = \\begin{cases}
        -\lambda \log_2\lambda  \ \ if \  \lambda  > 0; \\newline
        0,\  \ otherwise
      \end{cases}

    Parameters
    ----------
        L: sparse matrix
        num_vecs: int
            Number of  random vectors used to approximate the trace
            using the Hutchison's trick [1]
        num_steps: int
            Number of Lanczos steps or Chebyschev's moments
        device: str
            "/cpu:int" our "/gpu:int"
    
    Returns
    -------
        approximated_spectral_entropy: float

    References 
    ----------

    1 - Ubaru, S., Chen, J., & Saad, Y. (2017). Fast Estimation of tr(f(A)) via 
    Stochastic Lanczos Quadrature.  SIAM Journal on Matrix Analysis and Applications, 
    38(4), 1075-1099.
    
    2 - Hutchinson, M. F. (1990). A stochastic estimator of the trace of the
    influence matrix for laplacian smoothing splines. Communications in
    Statistics-Simulation and Computation, 19(2), 433-450.
   """
 
    def trace_function(eig_vals):
        return tf.map_fn(
            lambda val: tf.cond(
                val > 0,
                lambda: -val*tf.log(val),
                lambda: 0.),
            eig_vals)

    approximated_entropy, _ = slq(
        L_sparse, num_vecs, num_steps,  trace_function, device=device)

    return approximated_entropy


__all__ = ["slq", "entropy", "estrada_index"]
