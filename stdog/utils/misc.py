import numpy as np
from scipy.sparse import coo_matrix


def ig2sparse(G, transpose=False, attr=None, precision=32):
    """Example function with types documented in the docstring.

    Args:
        G (igraph): The first parameter.
        transpose (bool): The second parameter.
        attr (bool): The second parameter.
        precision (int): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.



    .. note:: can be useful to emphasize
        important feature
    .. seealso:: :class:`MainClass2`
    .. warning:: arg2 must be non-zero.
    .. todo:: check that arg2 is non zero.

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
