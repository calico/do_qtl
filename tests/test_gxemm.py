import unittest

import numpy as np
import scipy.optimize

from lib.models.gxemm import Core

def compute_kinship(G):
    n_markers = G.shape[1]
    return np.dot(G, G.T) / n_markers


def generate_data(n_phenotypes=1,
                  n_mice=20,
                  n_markers=30,
                  n_environments=1,
                  sigma_g=2,
                  sigma_e=[1, 3],
                  sigma_eps=[0, 0.5, 1.5],
                  seed=1):
    np.random.seed(seed)
    G = np.random.choice(2, size=(n_mice, n_markers))
    K = compute_kinship(G)
    X = []
    for _ in np.arange(n_environments):
        x = np.random.normal(size=(n_mice, 1))
        X.extend([(x > 0)*1, (x <= 0)*1])
    X = np.hstack(X)
    beta = np.expand_dims(np.random.normal(size=(2*n_environments)) + np.arange(2*n_environments), axis=-1)
    V = sigma_eps[0] * np.eye(n_mice) + \
        sigma_g * K
    for eps, e, x in zip(sigma_eps[1:], sigma_e, X.T):
        V += eps * np.diag(x) + e * K * (np.expand_dims(x, -1) * x)
    uk, sk, _ = np.linalg.svd(V)
    np.testing.assert_almost_equal(uk.dot(np.diag(sk)).dot(uk.T), V)
    e = uk @ np.diag(np.sqrt(sk)) @ np.random.normal(size=(n_mice, n_phenotypes))
    y = X @ beta + e
    return y, X, G, K, beta, e


def get_H(K, X, h):
    n, _ = X.shape
    H = h[0][0]*np.eye(n) + h[1]*K
    for j, x in enumerate(X.T):
        H += h[0][1+j] * np.diag(x) + h[2][j] * K * (np.expand_dims(x, -1) * x)
    return H


def get_res(y, X, beta):
    return y - X @ beta


def gxemm_log_likelihood_full(y, X, K, beta_est, h_est):
    n, _ = X.shape
    assert K.shape == (n, n)
    assert y.shape[0] == n
    H = get_H(K, X, h_est)
    s, u = np.linalg.eigh(H)
    invH = u.dot(np.diag(1/s)).dot(u.T)
    res = get_res(y, X, beta_est)
    ll = 1/2*(-n * np.log(2*np.pi)
              -np.sum(np.log(s))
              -np.squeeze(res.T @ invH @ res))
    return ll

def make_likelihood_slow_fn(y, X, K):
    def likelihood(h):
        n = y.shape[0]
        H = get_H(K, X, h)
        s, u = np.linalg.eigh(H)
        invH = u.dot(np.diag(1/s)).dot(u.T)
        np.testing.assert_almost_equal(H.dot(invH), np.eye(n))
        beta_est = beta_ml(y, X, invH)
        return gxemm_log_likelihood_full(y, X, K, beta_est, h)
    return likelihood


def beta_ml(y, X, invH):
    L = X.T @ invH @ X
    R = X.T @ invH @ y
    return np.linalg.pinv(L) @ R


class TestGxemm(unittest.TestCase):
    def test_likelihood(self):
        sigma_g = 2.0
        sigma_eps = np.array([0, 0.5, 1.5])
        sigma_e = np.array([1, 3])
        h = (sigma_eps, np.array([sigma_g]), sigma_e)
        y, X, _, K, _, _ = generate_data(sigma_g=sigma_g, sigma_eps=sigma_eps, sigma_e=sigma_e)

        n = y.size
        kinships = [[np.eye(n)] + \
                    [np.diag(x) for x in X.T],
                    [K],
                    [K*(np.expand_dims(x, -1) * x) for x in X.T]]
        core = Core(X, kinships, params=h)
        likelihood_fn = core.likelihood

        likelihood_slow_fn = make_likelihood_slow_fn(y, X, K)

        hs = [tuple([np.log(1+np.exp(np.random.normal(scale=10, size=h_.size)))
                     for h_ in h])
              for i in np.arange(100)]
        lls_full = np.asarray([likelihood_slow_fn(h) for h in hs])
        lls = np.asarray([likelihood_fn(np.hstack(h), y) for h in hs])
        np.testing.assert_almost_equal(lls, lls_full)

    def test_likelihood_gradient(self):
        sigma_g = 2.0
        sigma_eps = [0, 0.5, 1.5]
        sigma_e = [1, 3]
        h = (sigma_eps, [sigma_g], sigma_e)
        y, X, _, K, _, _ = generate_data(sigma_g=sigma_g, sigma_eps=sigma_eps, sigma_e=sigma_e)

        n = y.size
        kinships = [[np.eye(n)] + \
                    [np.diag(x) for x in X.T],
                    [K],
                    [K*(np.expand_dims(x, -1) * x) for x in X.T]]
        core = Core(X, kinships, params=h)
        likelihood_fn = core.likelihood
        gradient_fn = core.gradient

        rand_h = tuple([np.log(1+np.exp(np.random.normal(loc=np.zeros_like(h_),
                                                         scale=10*np.ones_like(h_))))
                        for h_ in h])
        _f = lambda delta: np.squeeze(likelihood_fn(delta, y))
        _g = lambda delta: [gradient_fn(delta, y)]
        scipy.optimize.check_grad(_f, _g, np.hstack(rand_h))
