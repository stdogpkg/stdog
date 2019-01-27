import tensorflow as tf
from scipy import sparse


def get_bounds(H, ncv=25):
    lmax = float(
        splin.eigsh(
            H,
            k=1,
            which="LA",
            return_eigenvectors=False,
            ncv=ncv
        )
    )
    lmin = float(
        splin.eigsh(
                H,
                k=1,
                which="SA",
                return_eigenvectors=False,
                ncv=ncv
        )
    )

    return lmin, lmax


def kpm(
    H,
    n_moments = 10,
    n_vecs = 10,
    extra_points = 12,
    precision=32,
    lmin=None,
    lmax=None,
    epsilon=0.01,
    device='/gpu:0',
    parallel_iterations_while=1,
    swap_memory_while=False,
    parallel_iterations_map=None,
    swap_memory_map=False,
):

    if (lmin is None) or (lmax is None):
        lmin, lmax = get_bounds(H)

    scale_fact_a = (lmax - lmin) / (2. - epsilon)
    scale_fact_b = (lmax + lmin) / 2


    H_rescaled = (1/scale_fact_a)*(H-1*scale_fact_b*sparse.eye(H.shape[0]))

    if precision == 32:
        tf_float = tf.float32
        tf_complex = tf.complex64
        np_float = np.float32
    elif precision == 64:
        tf_float = tf.float64
        tf_complex = tf.complex128
        np_float = np.float64


    coo = H_rescaled.tocoo()
    data_H = np.array(coo.data, dtype=np_float)
    shape_H = np.array(coo.shape, dtype=np.int32)
    indices_H = np.mat([coo.row, coo.col], dtype=np_float).transpose()

    n_vertices = H.shape[0]

    with tf.device(device):


        tf_n_vecs = tf.constant(n_vecs)
        tf_n_moments = tf.constant(n_moments)
        vecs = tf.constant(list(range(n_vecs)), dtype=tf_float)
        i_moment = tf.constant(0)
        n_iter_moments  = tf.constant(int(n_moments//2-1))
        pi = tf.constant(np.pi, dtype=tf_float)
        phase = tf.constant(np.pi*2, dtype=tf_float)

        tfscale_fact_a = tf.constant(scale_fact_a, dtype=tf_float)
        tfscale_fact_b = tf.constant(scale_fact_b, dtype=tf_float)
        Htf =  tf.SparseTensor(indices_H, data_H, shape_H)

        def body_moments(i_vec):
            random_phases = tf.random.uniform([n_vertices, 1],dtype=tf_float)
            alpha0_sin = tf.sin(
                phase*random_phases
            )
            alpha0_cos = tf.cos(
                phase*random_phases
            )
            alpha0 = tf.add(
                tf.cast(alpha0_cos, dtype=tf_complex),
                1j*tf.cast(alpha0_sin, dtype=tf_complex)
            )

            alpha1_sin = tf.sparse_tensor_dense_matmul(Htf, alpha0_sin )
            alpha1_cos = tf.sparse_tensor_dense_matmul(Htf, alpha0_cos )
            alpha1 = tf.add(
                tf.cast(alpha1_cos, dtype=tf_complex),
                1j*tf.cast(alpha1_sin, dtype=tf_complex)
            )

            mu0 = tf.linalg.matmul(alpha0, alpha0, adjoint_a=True)
            mu1 = tf.linalg.matmul(alpha0, alpha1, adjoint_a=True)

            mu = tf.concat([mu0, mu1], axis=1)

            def cond(mu, mu0, mu1, alpha0, alpha1, i_moment, n_iter_moments):
                return tf.less(i_moment, n_iter_moments)

            def body(mu, mu0, mu1, alpha0, alpha1, i_moment, n_iter_moments):
                alpha1_imag = tf.math.imag(alpha1)
                alpha1_real = tf.math.real(alpha1)

                matrix_mul_real =  2*tf.sparse_tensor_dense_matmul(Htf, alpha1_real)
                matrix_mul_imag =  2*tf.sparse_tensor_dense_matmul(Htf, alpha1_imag)

                matrix_mul = tf.add(
                    tf.cast(matrix_mul_real, tf_complex),
                    1j*tf.cast(matrix_mul_imag, tf_complex),
                )
                alpha2 = matrix_mul-alpha0


                even = 2*tf.linalg.matmul(alpha1, alpha1, adjoint_a=True) - mu0
                odd = 2*tf.linalg.matmul(alpha2, alpha1, adjoint_a=True) - mu1

                new_mu = tf.concat([even, odd], axis=1)

                return [
                    tf.concat(
                        [mu, new_mu],
                        axis=1
                    ),
                    mu0,
                    mu1,
                    alpha1,
                    alpha2,
                    tf.add(i_moment, 1),
                    n_iter_moments
                ]

            result = tf.while_loop(
                cond,
                body,
                [
                    mu,
                    mu0,
                    mu1,
                    alpha0,
                    alpha1,
                    i_moment,
                    n_iter_moments
                ],
                shape_invariants=[
                    tf.TensorShape([1,None]),
                    mu0.get_shape(),
                    mu1.get_shape(),

                    alpha0.get_shape(),
                    alpha1.get_shape(),
                    i_moment.get_shape(),
                    n_iter_moments.get_shape()
                ],
                parallel_iterations=parallel_iterations_while,
                swap_memory=swap_memory_while

            )
            mus = tf.math.real(result[0])
            return mus
        mus = tf.map_fn(
            body_moments,
            vecs,
            dtype=tf_float,
            parallel_iterations=parallel_iterations_map,
            swap_memory=swap_memory_map,
            infer_shape=False
        )
        mus = mus
        mu = tf.reduce_sum(mus, axis=0)
        mu = mu/n_vecs/n_vertices


        moments = tf.constant(list(range(n_moments)), dtype=tf_float)
        moments_phase = pi*moments/(n_moments+1)
        moments = tf.constant(list(range(n_moments)), dtype=tf_float)
        moments_phase = pi*moments/(n_moments+1)

        jackson_kernel = tf.div(
            tf.add(
                (n_moments-moments+1)*tf.cos(moments_phase),
                tf.sin(moments_phase)/tf.tan(pi/(n_moments+1))
            ),
            (n_moments+1)
        )

        mu_ext = tf.concat(
            [
                mu*jackson_kernel,
                tf.zeros((1, extra_points),
                         dtype=tf_float)
            ],
            axis=1
        )
        mu_ext = tf.reshape(mu_ext, [n_moments+extra_points])
        mu_T = tf.spectral.dct( mu_ext ,type =3)
        k = tf.range(0, n_moments+extra_points, dtype=tf_float)
        xk_rescaled = tf.cos(pi*(k+0.5)/(n_moments+extra_points))
        gk = pi*tf.sqrt(1.-xk_rescaled**2)
        xk = xk_rescaled*scale_fact_a + scale_fact_b
        rho = tf.divide ( mu_T ,gk)/( scale_fact_a )

        with tf.Session() as sess:
            xk = sess.run(xk)
            rho = sess.run(rho)

    return rho , xk
