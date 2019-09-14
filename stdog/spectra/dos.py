"""
Spectral Density
================

The Kernel Polynomial Method can estimate the spectral density of large
sparse Hermitan matrices with a computational cost almost linear. This method
combines three key ingredients: the Chebyshev expansion + the stochastic
trace estimator + kernel smoothing.

"""
from emate.hermitian.kpm import pykpm as kpm

__all__ = ["kpm"]