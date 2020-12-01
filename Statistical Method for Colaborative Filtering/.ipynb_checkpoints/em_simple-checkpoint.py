"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from tools import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    k, _ = mixture.mu.shape
    post = np.zeros([n, k])
    ll = 0
    p = mixture.p
    m = mixture.mu
    v = mixture.var

    for i, xx in enumerate(X):
        prob = np.zeros(k)
        for j in range(k):
            ex = - 1/(2*v[j]) * np.linalg.norm( xx - m[j])**2
            prob[j] = p[j]/np.power(2 * np.pi*v[j], d/2) * np.exp(ex)

        ll += np.log(prob.sum())
        prob = prob/prob.sum()
        post[i,:] = prob

    return post, ll
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _ , k = post.shape
    mu = np.zeros((k,d))
    var = np.ones(k)
    n_hat = post.sum(axis=0)
    p = n_hat / n
    px = np.dot(post.T, X)
    for i in range(k):
        mu[i] = px[i] / n_hat[i]

    p = 1/n * np.sum(post, axis=0)
    su = 0
    for i in range(n):
        mu_h = np.zeros_like(var)
        for j in range(k):
            mu_h[j] = np.linalg.norm(X[i]- mu[j])**2
        nu = post[i]* mu_h
        su += nu

    var = su/(d * np.sum(post, axis=0))


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
        mixture = mstep(X, post)


    return mixture, post, ll
    raise NotImplementedError
