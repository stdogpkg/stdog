import tensorflow as tf
import numpy as np

from emate.symmetric.slq import pyslq as slq


def estrada_index(L_sparse, num_vecs=100, num_steps=50, device="/gpu:0"):
    def trace_function(eig_vals):
        return tf.exp(eig_vals)

    approximated_estrada_index, _ = slq(
        L_sparse, num_vecs, num_steps,  trace_function, device=device)

    return approximated_estrada_index


def entropy(L_sparse, num_vecs=100, num_steps=50, device="/gpu:0"):
    def trace_function(eig_vals):
        return tf.map_fn(
            lambda val: tf.cond(
                val > 0,
                lambda: -val*tf.log(val),
                lambda: 0.),
            eig_vals)

    approximated_entropy, _ = slq(
        L_sparse, num_vecs, num_steps,  trace_function, device=device)

    return approximated_entropy


__all__ = ["slq", "entropy", "estrada_index"]
