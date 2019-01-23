import scipy
import numpy as np
import scipy.fftpack as fftp


def get_bounds(H, ncv=25):
    lmax = float(
        scipy.sparse.linalg.eigsh(
            H,
            k=1,
            which="LA",
            return_eigenvectors=False,
            ncv=ncv
        )
    )
    lmin = float(
        scipy.sparse.linalg.eigsh(
                H,
                k=1,
                which="SA",
                return_eigenvectors=False,
                ncv=ncv
        )
    )

    return lmin, lmax


def Jackson_kernel(n_moments):
    moments = np.arange(n_moments)
    norm = np.pi/(n_moments+1)
    phase_vec = norm*moments
    kernel = (n_moments-moments+1)*np.cos(phase_vec)
    kernel = kernel + np.sin(phase_vec)/np.tan(norm)
    kernel = kernel/(n_moments + 1)

    return kernel


def get_kpm_moments(
    H_rescaled,
    n_moments,
    np_complex=np.complex128
):

    n_vertices = H_rescaled.shape[0]
    alpha0 = np.exp(1j*2*np.pi*np.random.rand(n_vertices))
    alpha1 = H_rescaled.dot(alpha0)
    mu = np.zeros(n_moments, dtype=np_complex)
    mu[0] = (alpha0.T.conj()).dot(alpha0)
    mu[1] = (alpha0.T.conj()).dot(alpha1)

    for i_moment in range(1, n_moments//2):
        alpha2 = 2*H_rescaled.dot(alpha1)-alpha0
        mu[2*i_moment] = 2*(alpha1.T.conj()).dot(alpha1) - mu[0]
        mu[2*i_moment+1] = 2*(alpha2.T.conj()).dot(alpha1) - mu[1]

        alpha0 = alpha1
        alpha1 = alpha2

    return mu


def rescale(
    H,
    lmin=None,
    lmax=None,
    epsilon=0.01
):
    n_vertices = H.shape[0]
    if (lmin is None) or (lmax is None):
        lmin, lmax = get_bounds(H)

    scale_fact_a = (lmax - lmin) / (2. - epsilon)
    scale_fact_b = (lmax + lmin) / 2

    a = (lmax - lmin) / (2-epsilon)
    b = (lmax + lmin) / 2
    H_rescaled = (1/a)*(H - b*scipy.sparse.eye(n_vertices))

    return H_rescaled, a, b


def kpm_dos(
    mus,
    n_moments,
    n_vecs,
    n_points,
    n_vertices,
    scale_fact_a,
    scale_fact_b

):

    mu = np.sum(mus, axis=0)
    mu = mu.real
    mu = mu/n_vecs/n_vertices

    K = n_points+n_moments
    mu_ext = np.zeros(K)
    mu_ext[0:n_moments] = mu*Jackson_kernel(n_moments)

    mu_T = fftp.dct(mu_ext, type=3)
    k = np.arange(0, K)
    xk_rescaled = np.cos(np.pi*(k+0.5)/K)
    gk = np.pi*np.sqrt(1.-xk_rescaled**2)
    xk = xk_rescaled*scale_fact_a + scale_fact_b
    rho = np.divide(mu_T, gk)/(scale_fact_a)

    idx = xk.argsort()
    xk = xk[idx]
    rho = rho[idx]

    return rho, xk


def kpm(
    H,
    n_moments=10,
    n_vecs=10,
    n_points=12,
    precision=32,
    lmin=None,
    lmax=None,
    epsilon=0.01
):

    n_vertices = H.shape[0]

    H_rescaled, scale_fact_a, scale_fact_b = rescale(H, lmin, lmax, epsilon)

    mus = [
        get_kpm_moments(H_rescaled, n_moments)
        for i in range(n_vecs)
    ]

    rho, xk, mu = kpm_dos(
        mus,
        n_moments,
        n_vecs,
        n_points,
        n_vertices,
        scale_fact_a,
        scale_fact_b

    )

    return rho, xk
