"""
Heun's CUDA
===========

References
----------

[1] - Thomas Peron, Bruno Messias, Angélica S. Mata, Francisco A. Rodrigues,
and Yamir Moreno. On the onset of synchronization of Kuramoto oscillators in
scale-free networks. arXiv:1905.02256 (2019).


"""
import numpy as np
try:
    import cukuramoto
except ImportError:
    pass


class CUHeuns:
    """Allow efficiently simulating phase oscillators (the Kuramoto model) on
    large heterogeneous networks using the Heun’s method. This class uses a
    pure CUDA implementation of Heun’s method. Therefore, should be faster
    than TensorFlow implementation also provided by stDoG

    Attributes
    ----------
        adjacency : coo matrix
        phases : np.ndarray
        omegas : np.ndarray
        couplings : np.ndarray
        total_time : float
        dt : float
        transient : bool
        order_parameter_list : np.ndarray

    """

    def __init__(
        self,
        adjacency,
        phases,
        omegas,
        couplings,
        total_time,
        dt,
        transient=False,
        block_size=1024,
    ):

        self._adjacency = adjacency
        self._phases = phases.astype("float32")
        self._omegas = omegas.astype("float32")
        self._couplings = couplings.astype("float32")
        self.transient = transient
        self.total_time = total_time
        self.dt = dt

        self.block_size = block_size

        self.order_parameter_list = np.array([])
        self.create_simulation()

    @property
    def num_couplings(self):
        return len(self.couplings)

    @property
    def num_oscilators(self):
        return self.adjacency.shape[0] 

    @property
    def num_temps(self):
        return int(self.total_time/self.dt)

    @property
    def phases(self):
        phases = self.simulation.get_phases().reshape(
            (self.num_couplings, self.num_oscilators)
        )
        self._phases = phases
        return phases

    @property
    def omegas(self):
        return self._omegas

    @omegas.setter
    def omegas(self, omegas):
        self._omegas = omegas.astype("float32") 
        self.create_simulation()

    @property
    def couplings(self):
        return self._couplings

    @couplings.setter
    def couplings(self, couplings):
        self._couplings = couplings.astype("float32")
        self.create_simulation()

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency):
        self._adjacency = adjacency
        self.create_simulation()

    def create_simulation(self):
        """This method method crates the simulation.
        """
        adj = self.adjacency.tocsr()
        ptr, indices = adj.indptr.astype("int32"), adj.indices.astype("int32")

        simulation = cukuramoto.Heuns(
            self.num_oscilators, self.block_size, self.omegas, 
            self._phases.flatten(), self.couplings, indices, ptr)

        self.simulation = simulation

    def run(self):
        """This runs the algorithm and updates the phases.

        If transiet is set to True, then  the order parameters is
        calculated and  the array order_parameter_list is updated.
        """
      
        if self.transient:
            order_parameter_list = self.simulation.get_order_parameter(
                self.num_temps, self.dt)
            self.order_parameter_list = order_parameter_list.reshape(
                (self.num_couplings, self.num_temps)
            )
        else:
            self.simulation.heuns(
                self.num_temps, self.dt)

__all__ = ["CUHeuns"]
