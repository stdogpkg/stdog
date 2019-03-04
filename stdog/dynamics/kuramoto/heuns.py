import numpy as np
import tensorflow as tf

from .tfops.heuns import heuns_while, heuns_step
from .tfops.misc import get_order_parameter

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
        use_while=False,
    ):

        self.device = device
        self.log = log
        self.use_while = use_while

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
        self.order_parameter_list = np.array([])
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


                if self.frustration is not None:
                    frustration = tf.placeholder(
                        dtype=self.real_tf_type,
                        shape=[self.num_couplings, self.num_oscilators],
                        name="frustration"
                    )

                else:
                    frustration = None

                initial_phases = tf.placeholder(
                    dtype=self.real_tf_type,
                    shape=[self.num_couplings, self.num_oscilators],
                    name="initial_phases"
                )
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


                pi2 = tf.constant(
                    2*np.pi,
                    dtype=self.real_tf_type
                )
                dtDiv2 = tf.divide(
                    dt, 2.
                )



                omegas= omegas[tf.newaxis:, ]
                omegasDouble = tf.multiply(
                    2., omegas
                )
                couplings = couplings[:, tf.newaxis]

                if self.use_while:
                    num_temps = tf.placeholder(tf.int64, name="num_temps")
                    self.tf_phases, self.tf_order_parameters = heuns_while(
                        initial_phases,
                        frustration,
                        adjacency,
                        couplings,
                        omegas,
                        dt,
                        omegasDouble,
                        dtDiv2,
                        pi2,
                        self.num_couplings,
                        num_temps,
                        transient=self.transient,
                        tf_float=self.real_tf_type,
                        tf_complex=self.complex_tf_type
                    )
                else:
                    phases =  tf.Variable(
                        np.zeros_like(self.phases),
                        dtype=self.real_tf_type,
                        name="phases"
                    )
                    assign_initial_phases = tf.assign(phases, initial_phases, name="assign_initial_phases")

                    new_phases = heuns_step(phases, frustration, adjacency, couplings,
                        omegas, dt, omegasDouble, dtDiv2, pi2)

                    assign_new_phases = tf.assign(phases, new_phases, name="assign_new_phases")

                    if self.transient:
                        with tf.control_dependencies([assign_new_phases]):
                            new_order_parameter = get_order_parameter(
                                phases,
                                tf_complex=self.complex_tf_type
                            )
                            new_order_parameter = tf.reshape(
                                new_order_parameter, (self.num_couplings, 1),
                                name="new_order_parameter"
                            )


    def run(self):

        coo = self.adjacency.tocoo()

        sp_values = np.array(coo.data, dtype=self.real_np_type)
        sp_indices = np.mat([coo.row, coo.col], dtype=np.int64).transpose()



        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            if self.log is not None:
                tf.summary.FileWriter(self.log, sess.graph)

            if self.use_while:
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

                phases, order_parameters = sess.run(
                    [self.tf_phases, self.tf_order_parameters],
                    feed_dict
                )
                self.phases = phases
                if self.transient:
                    order_parameters = np.delete(order_parameters, (0), axis=1)
                    self.order_parameter_list = order_parameters
            else:
                sess.run("assign_initial_phases:0", feed_dict={"initial_phases:0": self.phases})
                feed_dict = {
                    "sp_values:0": sp_values,
                    "sp_indices:0": sp_indices,
                    "omegas:0": self.omegas,
                    "couplings:0": self.couplings,
                    "dt:0": self.dt,
                }
                if self.frustration is not None:
                    feed_dict["frustration:0"] = self.frustration


                for i_temp in range(self.num_temps):
                    if self.transient:
                        order_parameter = sess.run("new_order_parameter:0", feed_dict)
                        if self.order_parameter_list.shape[0] > 0:
                            self.order_parameter_list = np.concatenate([
                                self.order_parameter_list, order_parameter
                            ], axis=1)
                        else:
                            self.order_parameter_list = order_parameter

                    else:
                        sess.run("assign_new_phases:0", feed_dict)

                self.phases = sess.run("phases:0")


__all__ = ["Heuns"]
