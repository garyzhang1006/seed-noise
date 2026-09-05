"""Seed noise as the nonshared environment of a training run.

The package estimates ``Sigma_E``, the covariance across replicate training runs
that differ only in their data-order seed, and reports it as ``Lambda``, the
factor by which correlated seed noise inflates the noise of a battery average.
"""

__version__ = "0.1.0"

from seednoise.estimator import estimate, k_eff, p_flip, rbar_e, sigma_e
from seednoise.population import ACCURACY, MARGIN, Phenotype, Population

__all__ = ["__version__", "Population", "Phenotype", "MARGIN", "ACCURACY",
           "estimate", "sigma_e", "k_eff", "rbar_e", "p_flip"]
