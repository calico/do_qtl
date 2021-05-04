import itertools
import os
import shutil
import tempfile
import unittest

import numpy as np
from numpy import unravel_index
from numpy.linalg import inv
import pandas as pd
import scipy.optimize
import scipy.stats as stats

from lib.models.emma import Core


def compute_kinship(G):
  n_markers = G.shape[1]
  return np.dot(G, G.T) / n_markers


def generate_data(n_phenotypes=1,
                  n_mice=20,
                  n_markers=30,
                  n_covariates=2,
                  sigma_g=2,
                  sigma_e=0.5,
                  seed=1):
  np.random.seed(seed)
  G = np.random.choice(2, size=(n_mice, n_markers))
  K = compute_kinship(G)
  X = np.random.normal(size=(n_mice, n_covariates))
  beta = np.expand_dims(np.random.normal(size=(n_covariates)) + np.arange(n_covariates), axis=-1)
  uk, sk, _ = np.linalg.svd(K*sigma_g**2.0)
  np.testing.assert_almost_equal(uk.dot(np.diag(sk)).dot(uk.T), K*sigma_g**2.0)
  u = uk @ np.diag(np.sqrt(sk)) @ np.random.normal(size=(n_mice, n_phenotypes))
  e = sigma_e*np.random.normal(size=(n_mice, n_phenotypes))
  y = X @ beta + u + e
  return y, X, G, K, beta, u


def get_H(K, delta):
  n = K.shape[0]
  return K + delta*np.eye(n)


def get_res(y, X, beta):
  return y - X @ beta


def emma_log_likelihood_full(y, X, K, beta_est, delta, sigma_est):
  n, c = X.shape
  assert K.shape == (n, n)
  assert y.shape[0] == n
  H = get_H(K, delta)
  s, u = np.linalg.eigh(H)
  invH = u.dot(np.diag(1/s)).dot(u.T)
  res = get_res(y, X, beta_est)
  ll = 1/2*(-n * np.log(2*np.pi*sigma_est**2.0)
        -np.sum(np.log(s))
        -(1/(sigma_est**2.0))*np.squeeze(res.T @ invH @ res))
  return ll

def make_likelihood_slow_fn(y, X, K):
  def likelihood(delta):
    n = y.shape[0]
    H = get_H(K, delta)
    s, u = np.linalg.eigh(H)
    invH = u.dot(np.diag(1/s)).dot(u.T)
    np.testing.assert_almost_equal(H.dot(invH), np.eye(n))
    beta_est = beta_ml(y, X, invH)
    sigma_est = sigma_g_ml(y, X, invH, beta_est)
    return emma_log_likelihood_full(y, X, K, beta_est, delta, sigma_est)
  return likelihood

def get_delta(sigma_g, sigma_e):
  return (sigma_e**2.0)/(sigma_g**2.0)


def beta_ml(y, X, invH):
  P = inv(X.T @ invH @ X) @ X.T @ invH
  return P @ y


def sigma_g_ml(y, X, invH, beta_est):
  n = y.shape[0]
  res = get_res(y, X, beta_est)
  sigma_ml = (res.T @ invH @ res)/n
  return np.sqrt(np.squeeze(sigma_ml))



def sigma_g_ml_restricted(y, X, invH, beta_est):
  n, q = X.shape
  return sigma_g_ml(y, X, invH, beta_est)*np.sqrt(n/(n-q))


def emma_log_likelihood_restricted(y, X, K, beta_est, delta, sigma_est):
  q = X.shape[1]
  H = get_H(K, delta)
  invH = inv(H)
  sx, _ = np.linalg.eigh(X.T @ X)
  sxihx, _ = np.linalg.eigh(X.T @ invH @ X)
  ll_f = emma_log_likelihood_full(y, X, K, beta_est, delta, sigma_est)
  return ll_f + 1/2*(q*np.log(2*np.pi*sigma_est**2.0)
                     + np.sum(np.log(sx))
                     - np.sum(np.log(sxihx)))


class TestEmma(unittest.TestCase):
  def test_likelihood(self):
    sigma_g = 2.0
    sigma_e = 0.5
    y, X, G, K, _, _ = generate_data(sigma_g=sigma_g, sigma_e=sigma_e)
    Ke, Kv = np.linalg.eigh(K)
    core = Core(X, K)
    likelihood_fn = core.likelihood

    likelihood_slow_fn = make_likelihood_slow_fn(y, X, K)

    deltas = np.arange(0.1,1.0,0.1)
    lls_full = np.asarray([likelihood_slow_fn(delta) for delta in deltas])
    lls = np.asarray([np.squeeze(likelihood_fn(delta, y)) for delta in deltas])
    np.testing.assert_almost_equal(lls, lls_full)

  def test_likelihood_gradient(self):
    sigma_g = 2.0
    sigma_e = 0.5
    y, X, G, K, _, _ = generate_data(sigma_g=sigma_g, sigma_e=sigma_e)
    Ke, Kv = np.linalg.eigh(K)

    core = Core(X, K)
    likelihood_fn = core.likelihood
    gradient_fn = core.gradient

    _f = lambda delta: np.squeeze(likelihood_fn(delta[0], y))
    _g = lambda delta: [gradient_fn(delta[0], y)]
    scipy.optimize.check_grad(_f, _g, [0.1])


