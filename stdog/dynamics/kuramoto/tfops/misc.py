import tensorflow as tf


def get_order_parameter(phases, tf_complex=tf.complex64):
    with tf.name_scope("get_order_parameter"):
        order_parameter = tf.exp(1j*tf.cast(
            phases,
            tf_complex
            )
         )
        order_parameter = tf.reduce_mean(
            order_parameter,
            axis=1
        )

        order_parameter = tf.abs(
            order_parameter,
        )

        return order_parameter


__all__ = ["get_order_parameter"]
