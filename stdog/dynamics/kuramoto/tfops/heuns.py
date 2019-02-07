import numpy as np
import tensorflow as tf

from stdog.utils.tfops.misc import tfmod
from .misc import get_order_parameter


def heuns_evolve_vec(a, b, adjacency):
    with tf.name_scope("heuns_evolve_vec"):
        c = tf.sparse_tensor_dense_matmul(
            adjacency,
            tf.transpose(a),
        )
        c = tf.multiply(
            b,
            tf.transpose(c)
        )

        return c


def heuns_evolve_m(phases, frustration, adjacency, couplings):
    with tf.name_scope("heuns_evolve_m"):
        if frustration is not None:
            phases = tf.add(phases, frustration)

        vs = tf.sin(phases)
        vc = tf.cos(phases)

        Ms = heuns_evolve_vec(vs, vc, adjacency)
        Mc = heuns_evolve_vec(vc, vs, adjacency)

        M = tf.multiply(
            tf.subtract(Ms, Mc),
            couplings
        )

        return M


def heuns_step(phases, frustration, adjacency, couplings, omegas,
    dt, omegasDouble, dtDiv2, pi2, mod_phases=True):
    with tf.name_scope("heuns_step"):
        M = heuns_evolve_m(phases, frustration, adjacency, couplings)
        temporary_phases = tf.add(
            phases,
            tf.multiply(
                dt,
                tf.add(omegas, M)
            )
        )
        # temporary_phases = tfmod(temporary_phases, pi2)

        M2 = heuns_evolve_m(temporary_phases, frustration, adjacency, couplings)

        M = tf.add(M2, M)

        new_phases = tf.add(
            phases,
            tf.multiply(
                dtDiv2,
                tf.add(M, omegasDouble)
            )
        )

        if mod_phases:
            new_phases = tfmod(new_phases, pi2)

        return new_phases


def heuns(phases, frustration, adjacency, couplings, omegas, dt,
        num_couplings, num_temps, transient=False, tf_float=tf.float32,
        tf_complex=tf.complex64):
    with tf.name_scope("heuns"):
        with tf.name_scope("init"):
            i_dt = tf.constant(0, dtype=tf.int64)
            omegas = omegas[tf.newaxis:, ]
            couplings = couplings[:, tf.newaxis]
            order_parameters = tf.zeros(
                (num_couplings, 1),
                dtype=tf_float
            )
            pi2 = tf.constant(
                2*np.pi,
                dtype=tf_float
            )
            dtDiv2 = tf.divide(
                dt, 2.
            )
            omegasDouble = tf.multiply(
                    2., omegas
            )

        def cond(phases, order_parameters, dt, i_dt, num_temps):
            return tf.less(i_dt, num_temps)

        def body(phases, order_parameters, dt, i_dt, num_temps):
            new_phases = heuns_step(phases, frustration, adjacency, couplings,
                omegas, dt, omegasDouble, dtDiv2, pi2)

            if transient:
                new_order_parameter = get_order_parameter(
                    phases,
                    tf_complex=tf_complex
                )
                new_order_parameter = tf.reshape(
                    new_order_parameter, (num_couplings, 1)
                )
                order_parameters = tf.concat(
                    [
                        order_parameters, new_order_parameter
                    ],
                    axis=1
                )
            return [
                new_phases,
                order_parameters,
                dt,
                tf.add(i_dt, 1),
                num_temps
            ]

        phases, order_parameters, dt, i_dt, num_temps = tf.while_loop(
            cond,
            body,
            [
                phases,
                order_parameters,
                dt,
                i_dt,
                num_temps
            ],
            shape_invariants=[
                phases.get_shape(),
                tf.TensorShape([num_couplings, None]),
                dt.get_shape(),
                i_dt.get_shape(),
                num_temps.get_shape()
            ]
        )

        return phases, order_parameters


__all__ = ["heuns_evolve_vec", "heuns_evolve_m", "heuns_step", "heuns"]
