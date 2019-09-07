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


def update_phases(phases, dt, M, omegas):
    return tf.add(
            phases,
            tf.multiply(
                dt,
                tf.add(omegas, M)
            )
        )


def heuns_step(
        phases, frustration, adjacency, couplings, omegas, dt, omegasDouble,
        dtDiv2, pi2):

    with tf.name_scope("heuns_step"):
        M = heuns_evolve_m(phases, frustration, adjacency, couplings)

        temporary_phases = update_phases(phases, dt, M, omegas)

        M2 = heuns_evolve_m(
            temporary_phases, frustration, adjacency, couplings)

        M = tf.add(M2, M)

        new_phases = update_phases(phases, dtDiv2, M, omegasDouble)

        return new_phases


def heuns_while(
        phases, frustration, adjacency, couplings, omegas, dt,
        omegasDouble, dtDiv2, pi2,
        num_couplings, num_temps, transient=False, tf_float=tf.float32,
        tf_complex=tf.complex64):
    with tf.name_scope("heuns"):
        with tf.name_scope("init"):
            i_dt = tf.constant(0, dtype=tf.int64)

            order_parameters = tf.zeros(
                (num_couplings, 1),
                dtype=tf_float
            )

        def cond(phases, order_parameters, dt, i_dt, num_temps):
            return tf.less(i_dt, num_temps)

        def body(phases, order_parameters, dt, i_dt, num_temps):
            new_phases = heuns_step(
                phases, frustration, adjacency, couplings,
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


__all__ = [
    "heuns_evolve_vec", "heuns_evolve_m", "heuns_step", "heuns_while"]
