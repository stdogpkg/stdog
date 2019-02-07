import numpy as np
import tensorflow as tf

from .tfops.heuns import heuns


class Heuns:
    def __init__(
        self,
        adjacency,
        phases,
        omegas,
        couplings,
        total_time,
        dt,
        transient=False,
        frustration=None,
        precision=32,
        device="/gpu:0",
        log=None,
    ):

        self.device = device
        self.log = log

        self.adjacency = adjacency
        self.phases = phases
        self.omegas = omegas
        self.couplings = couplings
        self.frustration = frustration
        self.transient = transient
        self.total_time = total_time
        self.dt = dt

        self.num_couplings = len(self.couplings)
        self.num_oscilators = adjacency.shape[0]

        if precision == 64:
            self.complex_np_type = np.complex128
            self.complex_tf_type = tf.complex128

            self.real_np_type = np.float64
            self.real_tf_type = tf.float64

            self.int_tf_type = tf.int64

        elif precision == 32:
            self.complex_np_type = np.complex64
            self.complex_tf_type = tf.complex64

            self.real_np_type = np.float32
            self.real_tf_type = tf.float32

            self.int_tf_type = tf.int64

        else:
            raise Exception("Valid options for precision are: 32 or 64")

        self.omegas = np.array(
            self.omegas,
            dtype=self.real_np_type
        )

        self.create_tf_graph()

    @property
    def num_temps(self):
        return int(self.total_time/self.dt)

    @property
    def frustration(self):
        return self.__frustration

    @frustration.setter
    def frustration(self, frustration):

        not_first_call = hasattr(self, "frustration")
        if not_first_call:
            old = self.__frustration is None
            new = frustration is None
            self.__frustration = frustration

            if not_first_call and (old ^ new):
                self.create_tf_graph()
        else:
            self.__frustration = frustration

    @property
    def transient(self):
        return self.__transient

    @transient.setter
    def transient(self, transient):

        not_first_call = hasattr(self, "transient")
        if not_first_call:
            old = self.__transient
            self.__transient = transient

            if not_first_call and (old ^ transient):
                self.create_tf_graph()
        else:
            self.__transient = transient

    def create_tf_graph(self):
        with tf.device(self.device):
            self.graph = tf.Graph()

            with self.graph.as_default():

                initial_phases = tf.placeholder(
                    dtype=self.real_tf_type,
                    shape=[self.num_couplings, self.num_oscilators],
                    name="initial_phases"
                )

                if self.frustration is not None:
                    frustration = tf.placeholder(
                        dtype=self.real_tf_type,
                        shape=[self.num_couplings, self.num_oscilators],
                        name="frustration"
                    )

                else:
                    frustration = None

                omegas = tf.placeholder(
                    shape=self.omegas.shape,
                    dtype=self.real_tf_type,
                    name="omegas"
                )
                couplings = tf.placeholder(
                    shape=self.couplings.shape,
                    dtype=self.real_tf_type,
                    name="couplings"
                )
                dt = tf.placeholder(self.real_tf_type, shape=[], name="dt")

                num_temps = tf.placeholder(tf.int64, name="num_temps")

                sp_indices = tf.placeholder(dtype=tf.int64, name="sp_indices")
                sp_values = tf.placeholder(
                    dtype=self.real_tf_type,
                    name="sp_values"
                )
                adjacency = tf.SparseTensor(
                    sp_indices,
                    sp_values,
                    dense_shape=np.array(self.adjacency.shape, dtype=np.int32)
                )

                self.tf_phases, self.tf_order_parameters = heuns(
                    initial_phases,
                    frustration,
                    adjacency,
                    couplings,
                    omegas,
                    dt,
                    self.num_couplings,
                    num_temps,
                    transient=self.transient,
                    tf_float=self.real_tf_type,
                    tf_complex=self.complex_tf_type
                )

    def run(self):

        coo = self.adjacency.tocoo()

        sp_values = np.array(coo.data, dtype=self.real_np_type)
        sp_indices = np.mat([coo.row, coo.col], dtype=np.int64).transpose()

        feed_dict = {
            "sp_values:0": sp_values,
            "sp_indices:0": sp_indices,
            "initial_phases:0": self.phases,
            "omegas:0": self.omegas,
            "couplings:0": self.couplings,
            "dt:0": self.dt,
            "num_temps:0": self.num_temps,
        }

        if self.frustration is not None:
            feed_dict["frustration:0"] = self.frustration

        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            if self.log is not None:
                tf.summary.FileWriter(self.log, sess.graph)

            phases, order_parameters = sess.run(
                [self.tf_phases, self.tf_order_parameters],
                feed_dict
            )
            self.phases = phases
            if self.transient:
                self.order_parameter_list = order_parameters


__all__ = ["Heuns"]
