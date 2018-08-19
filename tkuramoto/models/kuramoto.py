import numpy as np
from scipy.sparse.coo import coo_matrix


class Kuramoto:
    def __init__(
        self,
        adjacency: "A NxN  matrix",
        phases: "List of size N",
        omegas: "List of size N",
        couplings: ""
    ):
        self.is_sparse_matrix = isinstance(
            adjacency,
            coo_matrix
        )

        self.adjacency = adjacency

        self.phases = phases
        self.omegas = omegas
        self.couplings = couplings

    @property
    def num_couplings(self):
        return len(self.couplings)

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, values):

        if self.is_sparse_matrix is False:
            values = np.array(values)

        if values.shape != (values.shape[0], values.shape[0]):
            raise Exception("Adjacency Matrix: dimension mismatch")

        self._adjacency = values
        self.num_oscilators = values.shape[0]

    @property
    def phases(self):
        return self._phases

    @phases.setter
    def phases(self, values):

        if len(values) != self.num_oscilators:
            raise Exception("Phases: dimension mismatch with adjacency matrix")
        self._phases = values

    @property
    def omegas(self):
        return self._omegas

    @omegas.setter
    def omegas(self, values):

        if len(values) != self.num_oscilators:
            raise Exception("Omegas: dimension mismatch with adjacency matrix")
        self._omegas = values

    def __repr__(self):
        str_repr = "Kuramoto:"
        str_repr += "\t\n num. oscilators=%d " % self.num_oscilators
        str_repr += "\t\n num. couplings=%d" % self.num_couplings
        str_repr += "\t\n is_sparse=%d" % int(self.is_sparse_matrix)

        return str_repr
