import tensorflow as tf


def tfmod(vals, mod_int):
    with tf.name_scope("tfmod"):
        mod_vals = tf.subtract(
            vals, tf.multiply(tf.floordiv(vals, mod_int), mod_int)
        )
        return mod_vals


__all__ = ["tfmod"]
