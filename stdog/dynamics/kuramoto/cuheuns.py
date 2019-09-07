import numpy as np
try:
    import cukuramoto
except ImportError:
    pass


class CUHeuns:
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

        self.adjacency = adjacency
        self.__phases = phases.astype("float32")
        self.omegas = omegas.astype("float32")
        self.couplings = couplings.astype("float32")
        self.transient = transient
        self.total_time = total_time
        self.dt = dt
        self.num_couplings = len(self.couplings)
        self.num_oscilators = adjacency.shape[0]

        self.block_size = block_size

        self.order_parameter_list = np.array([])
        self.create_simulation()

    @property
    def num_temps(self):
        return int(self.total_time/self.dt)

    @property
    def phases(self):
        phases = self.simulation.get_phases().reshape(
            (self.num_couplings, self.num_oscilators)
        )
        self.__phases = phases
        return phases

    def create_simulation(self):
        adj = self.adjacency.tocsr()
        ptr, indices = adj.indptr.astype("int32"), adj.indices.astype("int32")

        simulation = cukuramoto.Heuns(
            self.num_oscilators, self.block_size, self.omegas, 
            self.__phases.flatten(), self.couplings, indices, ptr)

        self.simulation = simulation

    def run(self):

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
