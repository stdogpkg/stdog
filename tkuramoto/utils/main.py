import numpy as np
from tensorflow import SparseTensor
from scipy.sparse import coo_matrix


def ig2sparse(G, transpose=False, attr=None, precision=32):
    if attr:
        source, target, data = zip(*[
            (e.source, e.target, e[attr])
            for e in G.es if not np.isnan(e[attr])
        ])
    else:
        source, target = zip(*[
            (e.source, e.target)
            for e in G.es
        ])
        data = np.ones(len(source)).astype('int').tolist()
    if not G.is_directed():
        source, target = source + target, target + source
        data = data + data
    if precision == 64:
        np_type = np.float64
    elif precision == 32:
        np_type = np.float32

    data = np.array(data, dtype=np_type)
    if transpose:
        L = coo_matrix(
            (data, (target, source)),
            shape=[G.vcount(), G.vcount()]
        )
    else:
        L = coo_matrix(
            (data, (source, target)),
            shape=[G.vcount(), G.vcount()]
        )
    return L


def sparse_matrix2sparse_tensor(matrix):

    coo = matrix.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return SparseTensor(indices, coo.data, coo.shape, )
