import tensorflow as tf
import numpy as np
from tkuramoto.utils import ig2sparse, sparse_matrix2sparse_tensor


class Heuns:
    def __init__(
        self,
        kuramoto: "Kuramoto instance",
        total_time: "Simulation time",
        dt: "Time step",
        time_slice: "Time slice"
    ):

        self.total_time = total_time
        self.time_slice = time_slice
        self.dt = dt
        self.num_temps = int((total_time/dt))
        self.kuramoto = kuramoto

        self.times = np.linspace(0, total_time, self.num_temps)

        omegas = np.array(
            kuramoto.omegas,
            dtype=np.complex128
        )
        self.omegas = omegas[np.newaxis, :]

        couplings = np.array(
            kuramoto.couplings,
            dtype=np.complex128
        )
        self.couplings = couplings[:, np.newaxis]

    def create_sparse_tf(self):

        self.graph = tf.Graph()
        with self.graph.as_default():
            adjacency = sparse_matrix2sparse_tensor(
                self.kuramoto.adjacency
            )
            omegas = tf.constant(
                self.omegas,
            )
            couplings = tf.constant(
                self.couplings,
            )
            dt = tf.constant(
                self.dt,
                dtype=tf.complex128
            )
            phases = tf.Variable(
                np.zeros(
                    (
                        self.kuramoto.num_couplings,
                        self.kuramoto.num_oscilators
                     )
                ),
                dtype=tf.complex128,
                name="phases"
            )
            phases_placeholder = tf.placeholder(
                dtype=tf.complex128,
                shape=[
                    self.kuramoto.num_couplings,
                    self.kuramoto.num_oscilators
                ],
                name="phases_placeholder"
            )

            assign_initial_phases = tf.assign(
                phases,
                phases_placeholder,
                name="assign_initial_phases"
            )

            order_parameter_list = tf.Variable(
                np.zeros(
                    (self.kuramoto.num_couplings, self.time_slice)
                ),
                dtype=tf.float64,
                name="order_parameter_list"
            )

            i_transient = tf.placeholder(
                dtype=tf.int64,
                shape=[],
                name="i_transient"
            )

            v = tf.multiply(
                1j,
                phases
            )
            v = tf.exp(v)

            M = tf.sparse_tensor_dense_matmul(
                adjacency,
                tf.transpose(v),
            )
            M = tf.transpose(M)
#           M_tf = tf.reduce_sum(M_tf, axis=1)
            M = tf.multiply(
                tf.conj(
                    v
                ),
                M
            )
            M = tf.imag(M)
            M = tf.cast(M, tf.complex128)

            M = tf.multiply(
                M,
                couplings
            )

            temporary_phases = tf.add(
                omegas,
                M
            )

            temporary_phases = tf.multiply(
                dt,
                temporary_phases
            )
            temporary_phases = tf.add(
                phases,
                temporary_phases
            )

            v2 = tf.multiply(
                1j,
                temporary_phases
            )
            v2 = tf.exp(v2)

            M2 = tf.sparse_tensor_dense_matmul(
                adjacency,
                tf.transpose(v2),
            )
            M2 = tf.transpose(M2)
#           M_tf = tf.reduce_sum(M_tf, axis=1)
            M2 = tf.multiply(
                tf.conj(
                    v2
                ),
                M2
            )

            M2 = tf.imag(M2)
            M2 = tf.cast(M2, tf.complex128)
            M2 = tf.multiply(
                M2,
                couplings
            )

            new_phases = tf.add(M2, M)
            new_phases = tf.add(omegas, new_phases)
            new_phases = dt*new_phases/2.
            new_phases = tf.add(phases, new_phases)

            assign_phases = tf.assign(phases, new_phases, name="assign_phases")
            with tf.control_dependencies([assign_phases]):
                new_order_parameter = tf.exp(1j*phases)
                new_order_parameter = tf.reduce_mean(
                    new_order_parameter,
                    axis=1
                )
                new_order_parameter = tf.abs(new_order_parameter)
#               new_rtitem = tf.transpose(new_rtitem)
                self.assign_order_parameter_list = tf.assign(
                    order_parameter_list[:, i_transient],
                    new_order_parameter,
                    name="assign_order_parameter_list"
                )

    def launch_sparse(self):
        order_parameter_evolution = []
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            initial_phases = [
                self.kuramoto.phases
                for i_l in range(self.kuramoto.num_couplings)
            ]
            sess.run(
                "assign_initial_phases:0",
                feed_dict={
                    "phases_placeholder:0": initial_phases,
                }
            )

            for j_step in range(self.num_temps//self.time_slice):
                for i in range(self.time_slice):
                    sess.run(
                        self.assign_order_parameter_list,
                        feed_dict={
                            "i_transient:0": i,
                        }
                    )

                order_parameter_evolution.append(
                    [
                        sess.run("order_parameter_list:0")
                    ]
                )

                phases_by_couplings = sess.run("phases:0")
                sess.run(
                    "assign_initial_phases:0",
                    feed_dict={
                        "phases_placeholder:0": phases_by_couplings,
                    }
                )

        sess.close()
        return order_parameter_evolution
