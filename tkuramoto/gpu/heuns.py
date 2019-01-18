"""This module illustrates how to write your docstring in OpenAlea
and other projects related to OpenAlea."""

import tensorflow as tf
import numpy as np
from scipy.sparse.coo import coo_matrix


class Heuns:
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        adjacency: A NxN  matrix
        phases: "List of size N",
        omegas: "List of size N",
        couplings: ""
        total_time (int) : Total time of simulation
        dt (int) :
        precision (int) : Floating Point precision. Should be 64 or 32.
        transient (int):

    .. seealso:: heuns

    .. note::
        There are many other Info fields but they may be redundant:
            * param, parameter, arg, argument, key, keyword: Description of a
              parameter.
            * type: Type of a parameter.
            * raises, raise, except, exception: That (and when) a specific
              exception is raised.
            * var, ivar, cvar: Description of a variable.
            * returns, return: Description of the return value.
            * rtype: Return type.

    .. note::
        There are many other directives such as versionadded, versionchanged,
        rubric, centered, ... See the sphinx documentation for more details.

    Here below is the results of the :func:`function1` docstring.

    """
    def __init__(
        self,
        adjacency,
        phases,
        omegas,
        couplings,
        total_time,
        dt,
        precision,
        transient
    ):

        self.is_sparse_matrix = isinstance(
            adjacency,
            coo_matrix
        )

        self.adjacency = adjacency

        self.phases = phases
        self.omegas = omegas
        self.couplings = couplings
        self.num_couplings = len(self.couplings)
        self.num_oscilators = adjacency.shape[0]
        self.total_time = total_time
        self.dt = dt
        self.num_temps = int((total_time/dt))
        self.transient = transient
        if precision == 64:
            print("precision 64")
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

        omegas = np.array(
            self.omegas,
            dtype=self.real_np_type
        )
        self.omegas = omegas[np.newaxis, :]
        self.phases = self.phases.astype(self.real_np_type)
        couplings = np.array(
            self.couplings,
            dtype=self.real_np_type
        )
        self.couplings = couplings[:, np.newaxis]

    def create_tf_graph(self):
        """Creates tensorflow graph. The graph can be acessed trough
        Heuns.graph.

        Returns:
            tensorflow.python.framework.ops.Graph: TensorFlow Graph.

        Example:
            >>> from tkuramoto.solvers.kuramoto.gpu import Heuns
            >>> heuns_solver = Heuns(
            ... kuramoto=kuramoto_instance,
            ... total_time=2000,
            ... dt=0.01,
            ... precision=32,
            ... transient=1000
            ... )
            >>> heuns_solver.create_tf_graph()
            >>> heuns_solver.graph
            <tensorflow.python.framework.ops.Graph at 0x7f0d084ebb00>


        .. note:: can be useful to emphasize
            important feature
        .. seealso:: :class:`MainClass2`
        .. warning:: arg2 must be non-zero.
        .. todo:: check that arg2 is non zero.

        """

        self.graph = tf.Graph()
        coo = self.adjacency.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose().astype(np.int64)
        with self.graph.as_default():

            imag_number = tf.constant(0+1j, dtype=self.complex_tf_type)

            number2 = tf.constant(
                2,
                dtype=self.real_tf_type
            )

            adjacency = tf.SparseTensor(indices, coo.data, coo.shape)
            omegas = tf.constant(
                self.omegas,
            )

            couplings = tf.constant(
                self.couplings,
            )
            dt = tf.constant(
                self.dt,
                dtype=self.real_tf_type
            )
            omegasDouble = tf.multiply(
                number2, omegas
            )
            dtDiv2 = tf.divide(
                dt, number2
            )
            phases = tf.Variable(
                np.zeros(
                    (
                        self.num_couplings,
                        self.num_oscilators
                     )
                ),
                dtype=self.real_tf_type,
                name="phases"
            )
            phases_placeholder = tf.placeholder(
                dtype=self.real_tf_type,
                shape=[
                    self.num_couplings,
                    self.num_oscilators
                ],
                name="phases_placeholder"
            )

            assign_initial_phases = tf.assign(
                phases,
                phases_placeholder,
                name="assign_initial_phases"
            )

            vs = tf.sin(phases)
            vc = tf.cos(phases)
            vst = tf.transpose(vs)
            vct = tf.transpose(vc)
            Ms = tf.sparse_tensor_dense_matmul(
                adjacency,
                vst,
            )
            Mc = tf.sparse_tensor_dense_matmul(
                adjacency,
                vct,
            )

            Ms = tf.transpose(Ms)
            Mc = tf.transpose(Mc)

            Ms = tf.multiply(
                vc,
                Ms
            )
            Mc = tf.multiply(
                vs,
                Mc
            )
            M = tf.subtract(Ms, Mc)
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

            vs2 = tf.sin(temporary_phases)
            vc2 = tf.cos(temporary_phases)
            vs2t = tf.transpose(vs2)
            vc2t = tf.transpose(vc2)
            Ms2 = tf.sparse_tensor_dense_matmul(
                adjacency,
                vs2t,
            )
            Mc2 = tf.sparse_tensor_dense_matmul(
                adjacency,
                vc2t,
            )

            Ms2 = tf.transpose(Ms2)
            Mc2 = tf.transpose(Mc2)

            Ms2 = tf.multiply(
                vc2,
                Ms2
            )

            Mc2 = tf.multiply(
                vs2,
                Mc2
            )
            M2 = tf.subtract(Ms2, Mc2)
            M2 = tf.multiply(
                M2,
                couplings
            )

            new_phases = tf.add(M2, M)

            new_phases = tf.add(new_phases, omegasDouble)
            new_phases = tf.multiply(dtDiv2, new_phases)

            new_phases = tf.add(phases, new_phases, name="new_phases_end")

            assign_phases = tf.assign(phases, new_phases, name="assign_phases")
            with tf.control_dependencies([assign_phases]):
                new_order_parameter = tf.exp(imag_number*tf.cast(
                    new_phases,
                    self.complex_tf_type
                    )
                 )
                new_order_parameter = tf.reduce_mean(
                    new_order_parameter,
                    axis=1
                )

                new_order_parameter_abs = tf.abs(
                    new_order_parameter,
                    name="new_order_parameter_abs"
                )

    def run_tf_graph(self):
        """Creates tensorflow graph. The graph can be acessed trough
        Heuns.graph.

        Returns:
            numpy.ndarray (float): order parameter evolution

        shape is (num. transient, num. couplings)


        Example:
            >>> from tkuramoto.solvers.kuramoto.gpu import Heuns
            >>> heuns_solver = Heuns(
            ... kuramoto=kuramoto_instance,
            ... total_time=2000,
            ... dt=0.01,
            ... precision=32,
            ... transient=1000
            ... )
            >>> heuns_solver.create_tf_graph()
            >>> heuns_solver.graph
            <tensorflow.python.framework.ops.Graph at 0x7f0d084ebb00>


        .. note:: can be useful to emphasize
            important feature
        .. seealso:: :class:`MainClass2`
        .. warning:: wtf


        """
        order_parameter_evolution = []
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())

            initial_phases = np.array([
                self.phases
                for i_l in range(self.num_couplings)
            ]).astype(self.real_np_type)
            sess.run(
                "assign_initial_phases:0",
                feed_dict={
                    "phases_placeholder:0": initial_phases,
                }
            )

            for i_step in range(self.num_temps):

                if self.num_temps - i_step > self.transient:
                    sess.run("assign_phases")
                else:
                    new_order_parameter0 = sess.run(
                        "new_order_parameter_abs:0"
                    )
                    order_parameter_evolution.append(new_order_parameter0)

        self.order_parameter_evolution = np.array(
            order_parameter_evolution,
            dtype=self.real_np_type
        )

        return self.order_parameter_evolution
