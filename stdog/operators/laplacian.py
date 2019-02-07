from scipy import sparse
import numpy as np


def magnetic(W, charge=1/3., normed=False, return_all=False):
    '''
    '''
    theta = 2.*np.pi*charge
    N = W.shape[0]

    A = W-W.T
    Ws = (W+W.T)/2.

    T = 1j*theta*A.T
    np.exp(T.data, out=T.data)

    T = sparse.csr_matrix(T)

    # pointwise
    L = - Ws.multiply(T)

    degree_list = sparse.csr_matrix.sum(Ws, axis=1).tolist()
    if normed:
        degree_list = np.power(degree_list, -1/2.).flatten()
        degree = sparse.diags(degree_list)

        # usual matrix multiplication
        L = degree*L
        L = L*degree
        L = L + sparse.eye(N)

    else:
        L = L + sparse.diags(np.array(degree_list).flatten())

    if return_all:
        return L, A, T

    return L
