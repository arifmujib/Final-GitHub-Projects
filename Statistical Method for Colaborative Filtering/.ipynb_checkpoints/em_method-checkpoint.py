"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from tools import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    mu = mixture.mu
    var = mixture.var
    p = mixture.p

    n, d = X.shape
    k, _ = mu.shape

    non_null_index = (X != 0)
    non_null_index = non_null_index.astype('float')
    dim = non_null_index.sum(axis=1)
    ex = np.sum((X[:, None, :] - non_null_index[:, None, :] * mu) ** 2, axis=2)
    con = np.log(p[None, :] + 1e-16) - dim[:, None] / 2 * np.log(2 * np.pi * var[None, :])
    f = con - ex/(2*var)
    fmax = np.max(f, axis=1)
    df = f - fmax[:, None]
    f2 = fmax + logsumexp(df, axis=1)
    #f2 = logsumexp(df, axis=1)
    ans = f - f2[:, None]
    ll = np.sum(np.exp(ans)*f2[:,None])

    return np.exp(ans), ll

    raise NotImplementedError
    



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    mu = mixture.mu
    var = mixture.var
    p = mixture.p

    n, d = X.shape
    k, _ = mu.shape
    non_null_index = (X != 0)
    non_null_index = non_null_index.astype('float')


    p = 1/n * post.sum(axis=0)

    update_condition = np.dot(post.T, non_null_index) + 1e-16
    potential_mu = np.dot( post.T, X*non_null_index)/update_condition

    mu[update_condition>1]=potential_mu[update_condition>1]

    ab = np.sum(np.sum((X[:, None, :] - non_null_index[:, None, :] * mu) ** 2, axis=2)* post, axis=0)
    denom = np.sum(non_null_index.sum(axis=1)[:,None]*post, axis=0)
    var = ab/denom
    for i,j in enumerate(var):
        if j<0.25:
            var[i]= 0.25

    return GaussianMixture(mu=mu, var=var, p=p)
    raise NotImplementedError



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_ll = None
    ll = None
    while (prev_ll is None or ll - prev_ll > 1e-6 * abs(ll)):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll
    raise NotImplementedError
    
    


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    mu = mixture.mu
    var = mixture.var
    p = mixture.p

    n, d = X.shape
    k, _ = mu.shape

    xx = X.copy()

    post, ll = estep(X, mixture)

    for i,j in enumerate(post):
        mean = np.dot(j, mu)
        u = xx[i]
        u[u==0.] = mean[u==0.]
        
    return xx
    raise NotImplementedError
    

