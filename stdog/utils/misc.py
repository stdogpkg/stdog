"""
misc
=====
something

"""

"""
Tools
======

Contains tools to help some boring tasks
"""
import numpy as np
from scipy.sparse import coo_matrix


def ig2sparse(G, transpose=False, attr=None, precision=32):
    """Given an igraph instance returns the sparse adjacency matrix
    in CSR format.

    Parameters
    ----------
        G: igraph instance
        transpose : bool 
            If the adjacency matrix should be transposed or not
        attr : str 
            The name of weight attribute
        precision : int
             The precision used to store the weight attributes

    Returns
    --------
        L : csr_matrix


    """

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


__all__ = ["ig2sparse"]
