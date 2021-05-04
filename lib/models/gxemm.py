import functools

import numpy as np
import scipy.optimize as opt

from ..models import utils

TOL = 1e-8
MAX_PERM = 1e8
NUM_NULL = 10
NONNEG_VC = True

class Core:

    """
    Model core enables efficient computation of likelihood
    and gradients by maintaining the current parameter state

    Parameters
    ----------
    X : ndarray, shape (n, c)
        Matrix of covariates, where 'n' is the number of samples
        and 'c' is the number of covariates.

    kinships : list of lists, len=3
        List of 3 lists
            - residual multiplier matrices (components of :math:`\Theta`)
            - genetic kinship matrix (:math:`\mathcal{K}`)
            - context-specific kinship matrices (:math:`\mathcal{K} \circ \mathbf{Z} \mathbf{Z}^T`)

    params : tuple of ndarray, len=3
        List of model parameters, sigma_e, rho, and omega.
        See manuscript for details on parameters.
    """

    def __init__(self, X, kinships, params=None):

        # matrix of covariates
        self._X = X
        self.N = X.shape[0]

        self._kinships = [kin for kins in kinships for kin in kins]
        self._sizes = np.array([len(kin) for kin in kinships])
        if np.any(self._sizes == 0):
            raise AssertionError("There should be at least one matrix in each group \
                                  defining the covariance matrix.")

        if params is None:
            # initialize parameters
            params = [np.random.rand(size) for size in self._sizes]
        h = np.hstack(params)

        # given a parameter value, set the state
        # of variables specific to learning
        self._set_state(h)

    def likelihood(self, h, Y):

        """Compute likelihood

        Parameters
        ----------
        h : ndarray, shape (p, )
            values for 'p' model parameters

        Y : ndarray, shape (n, 1)
            phenotype values for 'n' samples
        """

        if not np.allclose(h, self.h):
            self._set_state(h)

        # likelihood of a multivariate normal distribution
        # at the ML estimate of the mean parameter
        L = -0.5*(self.N*np.log(2*np.pi) + \
                  np.sum(np.log(self._He)) + \
                  np.sum(Y.T @ self._P @ Y))

        return L

    def function(self, h, Y):

        """Compute function (- log likelihood) to be minimized

        Parameters
        ----------
        h : ndarray, shape (p, )
            values for 'p' model parameters

        Y : ndarray, shape (n, 1)
            phenotype values for 'n' samples
        """

        return -self.likelihood(h, Y)

    def gradient(self, h, Y):

        """Compute gradient of -log likelihood

        Parameters
        ----------
        h : ndarray, shape (p, )
            values for 'p' model parameters

        Y : ndarray, shape (n, 1)
            phenotype values for 'n' samples
        """

        if not np.allclose(h, self.h):
            self._set_state(h)

        dL = 0.5*np.array([np.trace(self._Hinv @ kinship)
                           - np.sum(Y.T @ self._P.T @ kinship @ self._P @ Y)
                           for kinship in self._kinships])

        return dL

    def _set_state(self, h):

        """Recompute intermediate variables, when the model parameters change

        Parameters
        ----------
        h : ndarray, shape (p, )
            values for 'p' model parameters

        """

        self.h = h.copy()
        self.param = (self.h[:self._sizes[0]],
                      self.h[self._sizes[0]:self._sizes[0]+self._sizes[1]],
                      self.h[self._sizes[0]+self._sizes[1]:])
        self._H = functools.reduce(lambda u, v: u+v,
                                   [x_*k_ for x_, k_ in zip(self.h, self._kinships)])
        self._He, _ = np.linalg.eigh(self._H)
        converge = False
        while not converge:
            try:
                Hinv = np.linalg.pinv(self._H)
                self._P = Hinv - Hinv @ self._X @ np.linalg.pinv(self._X.T @ Hinv @ self._X) @ self._X.T @ Hinv
                self._Hinv = Hinv
                converge = True
            except np.linalg.LinAlgError:
                converge = False

class Gxemm:

    """Genotype x Environment Mixed Model (GxEMM)

    Parameters
    ----------

    kinship : ndarray, shape (n, n)
        Genetic relatedness amongst 'n' samples

    phenotype : ndarray, shape (n, 1)
        Phenotype values across 'n' samples

    covariates : ndarray, shape (n, e)
        Binary matrix specifying which environments 'e' a sample 'n' belongs to

    Attributes
    ----------

    sigma : ndarray, shape (e+1, )
            Variance component for noise terms

    rho : float
          Variance component for environment-independent genetic term

    omega : ndarray, shape (e, )
            Variance component for environment-dependent genetic term

    log_likelihood : float
        Log likelihood at optimal model parameters

    pve : ndarray, shape (e, )
        Environment-dependent PVE

    total_pve : float
        Total PVE

    var : ndarray, shape (e, )
        Environment-dependent expected genetic contribution to phenotypic variance

    total_var : float
        Total expected genetic contribution to phenotypic variance

    pve_serr : ndarray, shape (e, )
        Standard error of environment-dependent PVE

    total_pve_serr : float
        Standard error of total PVE

    var_serr : ndarray, shape (e, )
        Standard error of environment-dependent expected genetic contribution to phenotypic variance

    total_var_serr : float
        Standard error of total expected genetic contribution to phenotypic variance
    """

    def __init__(self, kinship, phenotype, covariates):

        self.kinship = kinship
        self.N = self.kinship.shape[0]

        self.phenotype = phenotype
        self._Y = phenotype

        self.covariates = covariates
        self._X = np.hstack([np.ones((self.N, 1))] + \
                            [covariates.data for covariates in self.covariates
                             if covariates.effect == 'fixed'])
        self.C = self._X.shape[1]

        self._blocks = [block
                        for covariates in self.covariates
                        if covariates.gxe
                        for block in covariates.blocks]
        if np.any([covariates.gxe for covariates in self.covariates]):
            self._x_covariates = np.hstack([covariates.data
                                            for covariates in self.covariates
                                            if covariates.gxe])
        else:
            self._x_covariates = np.zeros((self.N, 0))
        self.E = self._x_covariates.shape[1]

        self._kinships = [[np.eye(self.N)] \
                          + [np.diag(x_cov) for x_cov in self._x_covariates.T],
                          [self.kinship],
                          [self.kinship*block for block in self._blocks]]

        # attributes to be computed, given data
        # model parameters
        self.sigma = None
        self.rho = None
        self.omega = None
        # likelihood
        self.log_likelihood = None
        # estimates of PVE and std err
        self.pve = None
        self.var = None
        self.pve_serr = None
        self.var_serr = None
        self.total_pve = None
        self.total_var = None
        self.total_pve_serr = None
        self.total_var_serr = None

    def fit_variance_components(self, X, Y, init_param=None, indices=None, tol=TOL):

        """Estimate variance component parameters of GxEMM model

        Parameters
        ----------
        X : ndarray, shape (n, c)
            Matrix of covariates

        Y : ndarray, shape (n, 1)
            Matrix of phenotype values

        init_params : tuple of ndarrays, len=3
            (sigma, rho, omega)
            See manuscript for definitions of these parameters.

        indices : ndarray, shape (m, ), m<n
            Subset of samples to limit, when estimating parameters

        tol : float
            Termination criterion for scipy.optimize

        """

        if indices is None:
            X_ = X.copy()
            Y_ = Y.copy()
            kinships = self._kinships
        else:
            X_ = X[indices, :]
            Y_ = Y[indices, :]
            kinships = [[kinship[indices, :][:, indices]
                         for kinship in kinships_]
                        for kinships_ in self._kinships]

        retry = 0
        success = False
        while not success and retry < 3:
            if retry > 0:
                core = Core(X_, kinships)
            else:
                core = Core(X_, kinships, params=init_param)

            xo = core.h.copy()
            args = (Y_, )
            if NONNEG_VC:
                bounds = [(0, np.inf)]*xo.size
            else:
                bounds = [(-np.inf, np.inf)]*xo.size

            result = opt.minimize(core.function, xo, jac=core.gradient, tol=tol,
                                  args=args, method='L-BFGS-B', bounds=bounds)

            if result['success']:
                optimal_param = core.param
                # parameters should be strictly non-negative
                # tiny negative values cause trouble
                if NONNEG_VC:
                    for param in optimal_param:
                        param[param < 0] = 0
                log_likelihood = -result['fun']
                success = True
            else:
                retry += 1

        if success:
            return optimal_param, log_likelihood
        else:
            raise ValueError("Failed to find optimal variance component parameters")

    def fit_effect_size(self, X, Y, params, indices=None, compute_stderr=False):

        """Estimate fixed effects, given variance component
        parameters

        Parameters
        ----------
        X : ndarray, shape (n, c)
            Matrix of covariates

        Y : ndarray, shape (n, 1)
            Matrix of phenotype values

        params : tuple of ndarrays, len=3
            (sigma, rho, omega)
            See manuscript for definitions of these parameters.

        indices : ndarray, shape (m, ), m<n, default=None
            Subset of samples to limit, when estimating parameters

        compute_stderr : bool, default=False
            Flag specifying whether to compute the standard error of the
            parameter estimates, using the Fisher information matrix.

        """

        H = functools.reduce(lambda u, v: u+v,
                             [p_*k_ for param, kinship in zip(params, self._kinships)
                              for p_, k_ in zip(param, kinship)])

        if indices is None:
            Hinv = np.linalg.pinv(H)
            L = X.T @ Hinv @ X
            R = X.T @ Hinv @ Y
        else:
            Hinv = np.linalg.pinv(H[indices, :][:, indices])
            L = X[indices].T @ Hinv @ X[indices]
            R = X[indices].T @ Hinv @ Y[indices]

        beta = np.linalg.pinv(L) @ R

        if compute_stderr:
            s = X.T @ Hinv @ X
            fisher_information = 0.5*(s+s.T)
            S = np.linalg.pinv(fisher_information)
            serr = np.diag(S)**0.5

            return beta, serr
        else:
            return beta

    def compute_pve(self, params, indices=None):

        """Compute the proportion of phenotypic variance 
        explained by genetics, given estimates of model parameters

        Parameters
        ----------

        params : tuple of ndarrays, len=3
            (sigma, rho, omega)
            See manuscript for definitions of these parameters.

        indices : ndarray, shape (m, ), default=None
            Subset of samples to limit, when estimating parameters

        """

        sigma, rho, omega = params
        beta = self.fit_effect_size(self._X, self._Y, params, indices=indices)
        mu = self._X @ beta
        noise_mat = [np.eye(self.N)] + [np.diag(x_cov) for x_cov in self._x_covariates.T]
        noise = functools.reduce(lambda u, v: u+v, [sig*n_ for sig, n_ in zip(sigma, noise_mat)])

        if indices is None:
            # compute PVE using all samples
            ns = np.sum(self._x_covariates, 0)
            den = np.array([np.var(mu[cov == 1]) for cov in self._x_covariates.T])
            den += np.array([np.trace(noise[cov == 1, :][:, cov == 1])/n - \
                             np.sum(noise[cov == 1, :][:, cov == 1])/n**2
                             for n, cov in zip(ns, self._x_covariates.T)])
            num = rho * np.array([np.trace(self.kinship*block)/n
                                  - np.sum(self.kinship*block)/n**2
                                  for n, block in zip(ns, self._blocks)])
            for k, block in enumerate(self._blocks):
                num += omega[k] * np.array([np.trace(self.kinship*block*block_)/n \
                                            - np.sum(self.kinship*block*block_)/n**2 \
                                            for n, block_ in zip(ns, self._blocks)])

        else:
            # compute PVE using a subset of samples
            ns = np.sum(self._x_covariates[indices], 0)
            den = np.array([np.var(mu[indices][cov == 1])
                            for cov in self._x_covariates[indices].T])
            den += np.array([np.trace(noise[indices, :][:, indices][cov == 1, :][:, cov == 1])/n
                             - np.sum(noise[indices, :][:, indices][cov == 1, :][:, cov == 1])/n**2
                             for n, cov in zip(ns, self._x_covariates[indices].T)])
            num = rho * np.array([np.trace((self.kinship*block)[indices, :][:, indices])/n
                                  - np.sum((self.kinship*block)[indices, :][:, indices])/n**2
                                  for n, block in zip(ns, self._blocks)])
            for k, block in enumerate(self._blocks):
                num += omega[k] * np.array([np.trace((self.kinship*block*block_)[indices, :][:, indices])/n
                                            - np.sum((self.kinship*block*block_)[indices, :][:, indices])/n**2
                                            for n, block_ in zip(ns, self._blocks)])

        den += num
        pve = num / den

        return pve, den

    def compute_total_pve(self, params, indices=None):

        """Compute the total proportion of
        phenotypic variance explained by genetics, given
        estimates of model parameters

        Parameters
        ----------

        params : tuple of ndarrays, len=3
            (sigma, rho, omega)
            See manuscript for definitions of these parameters.

        indices : ndarray, shape (m, ), default=None
            Subset of samples to limit, when estimating parameters

        """

        sigma, rho, omega = params
        beta = self.fit_effect_size(self._X, self._Y, params, indices=indices)
        mu = self._X @ beta
        noise_mat = [np.eye(self.N)] + [np.diag(x_cov) for x_cov in self._x_covariates.T]
        noise = functools.reduce(lambda u, v: u+v, [sig*sig_mat for sig, sig_mat in zip(sigma, noise_mat)])

        if indices is None:
            # compute total PVE using all samples
            den = np.var(mu) + np.trace(noise)/self.N - np.sum(noise)/self.N**2
            num = rho * (np.trace(self.kinship)/self.N - \
                         np.sum(self.kinship)/self.N**2)
            for k, block in enumerate(self._blocks):
                num += omega[k] * (np.trace(self.kinship*block)/self.N - \
                                   np.sum(self.kinship*block)/self.N**2)
        else:
            # compute total PVE using a subset of samples
            N = indices.size
            den = np.var(mu[indices]) + \
                  np.trace(noise[indices, :][:, indices])/N - \
                  np.sum(noise[indices, :][:, indices])/N**2
            num = rho * (np.trace(self.kinship[indices, :][:, indices])/N - \
                         np.sum(self.kinship[indices, :][:, indices])/N**2)
            for k, block in enumerate(self._blocks):
                num += omega[k] * (np.trace((self.kinship*block)[indices, :][:, indices])/N - \
                                   np.sum((self.kinship*block)[indices, :][:, indices])/N**2)

        den += num
        pve = num / den
        return pve, den

    def fit_pve(self, get_serr=False):

        """Fit model parameters and compute the proportion of
        phenotypic variance explained by genetics

        Parameters
        ----------

        get_serr : bool, default=False
            Flag to specify whether to compute standard error
            of estimates of PVE

        """

        # to avoid local optima issues, do 10 random restarts
        rmax = 10
        print("estimating PVE with %d random runs ..."%rmax)
        r = 0
        params = []
        log_likelihoods = np.zeros((rmax, ))
        while r < rmax:
            try:
                param, log_likelihood = self.fit_variance_components(self._X, self._Y, tol=TOL)
                params.append(param)
                log_likelihoods[r] = log_likelihood
                r += 1
                print("completed run %d; log likelihood = %.4f"%(r, log_likelihood))
            except ValueError:
                pass

        # select estimate with highest log likelihood
        optimal_param = params[np.argmax(log_likelihoods)]
        self.sigma, self.rho, self.omega = optimal_param
        self.log_likelihood = log_likelihoods[np.argmax(log_likelihoods)]

        # compute environment-dependent PVE, when there are multiple environments
        try:
            self.pve, self.var = self.compute_pve(optimal_param)
        except NameError:
            self.pve = None
            self.var = None
        # compute total PVE
        self.total_pve, self.total_var = self.compute_total_pve(optimal_param)
        print("estimating PVE, using run with highest log likelihood")

        if get_serr:
            print("estimating std. error of PVE using jackknifing ...")
            hs = []
            var = []
            ths = []
            tvar = []
            jackknifed_samples = utils.jackknife(self.N, self.N//10, 10, X=self._X, blocks=self._blocks)
            for j_samples in jackknifed_samples:
                indices = np.delete(np.arange(self.N), j_samples)
                param_, _ = self.fit_variance_components(self._X, self._Y,
                                                         init_param=optimal_param, 
                                                         indices=indices, tol=TOL)

                # compute environment-dependent PVE, when there are multiple environments
                try:
                    a, b = self.compute_pve(param_, indices=indices)
                    hs.append(a)
                    var.append(b)
                except NameError:
                    pass

                # compute total PVE
                a, b = self.compute_total_pve(param_, indices=indices)
                ths.append(a)
                tvar.append(b)

            self.pve_serr = np.nanstd(hs, 0)
            self.var_serr = np.nanstd(var, 0)
            self.total_pve_serr = np.nanstd(ths)
            self.total_var_serr = np.nanstd(tvar)
            print("finished estimating PVE std. error.")

        else:

            self.pve_serr = None
            self.var_serr = None
            self.total_pve_serr = None
            self.total_var_serr = None

    def _compute_p_value(self, Y, cores, likelihood_ratios, null_param_perm, perm):

        """Compute p-value using a sequential permutation scheme.

        Parameters
        ----------

        Y : ndarray, shape (n,1)
            Phenotype vector

        cores : list
                list of Core instances to evaluate likelihood
                under the null, additive, and interaction models

        likelihood_ratios : list
                            list of true likelihood ratio statistics, one for
                            the additive test and one for interaction test

        null_param_perm : float
                Ratio of residual variance component to genetic variance
                component, for a permuted phenotype vector, under the null model            

        perm : int
               Truncation criterion for sequential permutation scheme.
               Number of permutations where the permuted statistic is
               at least as large as the true statistic, before permutation
               scheme can be stopped.
        """ 

        try:
            null_core, add_core, int_core = cores
            likelihood_ratio, x_likelihood_ratio = likelihood_ratios
        except ValueError:
            null_core, add_core = cores
            likelihood_ratio = likelihood_ratios[0]

        if perm is None:
            p_value = np.nan
            x_p_value = np.nan
        else:
            p = 0
            xp = 0
            P = 0
            xP = 0

            # a sequential permutation scheme
            while min([p, xp]) < perm and min([P, xP]) < MAX_PERM:
                Y_perm = np.random.permutation(Y.ravel()).reshape(Y.shape)

                # compute likelihood under null, additive, and interaction
                # model for permuted phenotype vector
                log_likelihood_perm = add_core.likelihood(np.hstack(null_param_perm), Y_perm)
                if p < perm and P < MAX_PERM:
                    null_log_likelihood_perm = null_core.likelihood(np.hstack(null_param_perm), Y_perm)
                    if (log_likelihood_perm-null_log_likelihood_perm) >= likelihood_ratio:
                        p += 1
                    P += 1
                try:
                    if xp < perm and xP < MAX_PERM:
                        x_log_likelihood_perm = int_core.likelihood(np.hstack(null_param_perm), Y_perm)
                        if (x_log_likelihood_perm-log_likelihood_perm) >= x_likelihood_ratio:
                            xp += 1
                        xP += 1
                except NameError:
                    xp = p
                    xP = P
            p_value = p/P + np.random.random() * (P-p)/P/(P+1)
            x_p_value = xp/xP + np.random.random() * (xP-xp)/xP/(xP+1)

        return p_value, x_p_value

    def run_gwas(self, genotypes, approx=False, perm=None):

        """Test for association between genotype and phenotype,
        at all typed variants.

        Parameters
        ----------

        genotypes : iterator
            Iterator of genotypes over imputed variants in a specific locus.

        approx : bool, default=False
            Use the approximation that variance component parameters are
            the same in the null and alternative models (see EMMAX paper).
            Relevant for computing the observed test statistic alone.

        perm : int, default=None
            Terminate permutations when ``perm`` number of permutations
            have at least as extreme test statistics as the observed data.
            Set to None to turn off permutation to compute p-values.
            Usually, set perm to 10.
        """

        Y = self.phenotype.copy()
        X_null = np.hstack([np.ones((self.N, 1), dtype='float')] + \
                           [covariates.data for covariates in self.covariates
                            if covariates.effect == 'fixed'])
        if np.any([covariates.test_gxe for covariates in self.covariates]):
            Xi_test = np.hstack([covariates.data for covariates in self.covariates
                                 if covariates.test_gxe])

        print("estimating variance components under the null model ...")
        null_param, null_likelihood = self.fit_variance_components(X_null, Y)

        null_param_perm = null_param
        if perm is not None:
            # from a collection of permuted datasets, compute null model
            # parameters and estimate their average.
            # these are kept fixed, in both the null and alternate models,
            # when computing test statistics for each permutation.
            print("estimating expected variance components under the null model for permuted phenotypes ...")
            null_param_perms = []
            for _ in np.arange(NUM_NULL):
                Y_perm = np.random.permutation(Y.ravel()).reshape(Y.shape)
                null_param_perm, _ = self.fit_variance_components(X_null, Y_perm, init_param=null_param)
                null_param_perms.append(null_param_perm)
            null_param_perm = (np.median([param[0] for param in null_param_perms], 0),
                               np.median([param[1] for param in null_param_perms]),
                               np.median([param[2] for param in null_param_perms], 0))

        # loop over all typed variants
        print("association testing at genotyped variants ...")
        for variant, genotype in genotypes:

            # for additive model, append fixed effect covariates
            # with genotype of the focal variant
            X = np.hstack((X_null, genotype.T))

            if approx:
                # keep variance components fixed from null model
                core = Core(X, self._kinships, null_param)
                likelihood = core.likelihood(np.hstack(null_param), Y)
            else:
                # estimate variance components
                _, likelihood = self.fit_variance_components(X, Y, init_param=null_param)

            likelihood_ratio = likelihood - null_likelihood

            # for interaction model, append fixed effect covariates
            # with genotype of the focal variant and product
            # of environment and genotype of focal variant.
            try:
                Xi = np.hstack([X_null,
                                genotype.T,
                                utils.outer_product(genotype, Xi_test.T).T])

                if approx:
                    # keep variance components fixed from null model
                    core = Core(Xi, self._kinships, null_param)
                    x_likelihood = core.likelihood(np.hstack(null_param), Y)
                else:
                    # estimate variance components
                    _, x_likelihood = self.fit_variance_components(Xi, Y, init_param=null_param)

                x_likelihood_ratio = x_likelihood - likelihood
            except NameError:
                pass

            # a core instance for the null, additive,
            # and interaction models
            null_core = Core(X_null, self._kinships, null_param_perm)
            add_core = Core(X, self._kinships, null_param_perm)
            try:
                int_core = Core(Xi, self._kinships, null_param_perm)
                cores = [null_core, add_core, int_core]
                lratios = [likelihood_ratio, x_likelihood_ratio]
            except NameError:
                cores = [null_core, add_core]
                lratios = [likelihood_ratio]

            p_value, x_p_value = self._compute_p_value(self._Y, cores, lratios, null_param_perm, perm)

            # return the test statistics and p-value, for
            # each typed variant
            try:
                result = [variant] + \
                         [likelihood_ratio, p_value] + \
                         [x_likelihood_ratio, x_p_value]
            except NameError:
                result = [variant] + \
                         [likelihood_ratio, p_value]
            yield result

    def run_finemap(self, genotypes, approx=True, perm=None):

        """Test for association between genotype and phenotype
        and estimate effect sizes, at all variants within a locus.

        Parameters
        ----------

        genotypes : iterator
            Iterator of genotypes over imputed variants in a specific locus.

        approx : bool, default=True
            Use the approximation that variance component parameters are
            the same in the null and alternative models (see EMMAX paper).
            Relevant for computing the observed test statistic alone.

        perm : int, default=None
            Terminate permutations when ``perm`` number of permutations
            have at least as extreme test statistics as the observed data.
            Set to None to turn off permutation to compute p-values.
            Usually, set perm to 10.

        """

        Y = self.phenotype
        X_null = np.hstack([np.ones((self.N, 1), dtype='float')] + \
                           [covariates.data for covariates in self.covariates
                            if covariates.effect == 'fixed'])
        if np.any([covariates.test_gxe for covariates in self.covariates]):
            Xi_test = np.hstack([covariates.data for covariates in self.covariates
                                 if covariates.test_gxe])

        print("estimating variance components under the null model ...")
        null_param, null_likelihood = self.fit_variance_components(X_null, Y)

        null_param_perm = null_param
        if perm is not None:
            print("estimating expected variance components under the null model for permuted phenotypes ...")
            null_param_perms = []
            for _ in np.arange(NUM_NULL):
                Y_perm = np.random.permutation(Y.ravel()).reshape(Y.shape)
                null_param_perm, _ = self.fit_variance_components(X_null, Y_perm, init_param=null_param)
                null_param_perms.append(null_param_perm)
            null_param_perm = (np.median([param[0] for param in null_param_perms], 0),
                               np.median([param[1] for param in null_param_perms]),
                               np.median([param[2] for param in null_param_perms], 0))

        # loop over all imputed variants
        print("association testing at all variants in locus ...")
        for variant, genotype in genotypes:

            # for additive model, append fixed effect covariates
            # with genotype of the focal variant
            X = np.hstack((X_null, genotype.T))

            if approx:
                # keep variance components fixed from null model
                core = Core(X, self._kinships, params=null_param)
                likelihood = core.likelihood(np.hstack(null_param), Y)
                # estimate effect sizes and standard errors
                beta, serr = self.fit_effect_size(X, Y, null_param, compute_stderr=True)
            else:
                # estimate variance components
                param_, likelihood = self.fit_variance_components(X, Y, init_param=null_param)
                # estimate effect sizes and standard errors
                beta, serr = self.fit_effect_size(X, Y, param_, compute_stderr=True)

            likelihood_ratio = likelihood - null_likelihood

            try:
                Xi = np.hstack([X_null,
                                genotype.T,
                                utils.outer_product(genotype, Xi_test.T).T])

                if approx:
                    # keep variance components fixed from null model
                    core = Core(Xi, self._kinships, params=null_param)
                    x_likelihood = core.likelihood(np.hstack(null_param), Y)
                    # estimate effect sizes and standard errors
                    x_beta, x_serr = self.fit_effect_size(Xi, Y, null_param, compute_stderr=True)
                else:
                    # estimate variance components
                    param_, x_likelihood = self.fit_variance_components(Xi, Y, init_param=null_param)
                    # estimate effect sizes and standard errors
                    x_beta, x_serr = self.fit_effect_size(Xi, Y, param_, compute_stderr=True)

                x_likelihood_ratio = x_likelihood - likelihood
            except NameError:
                pass

            # a core instance for the null, additive,
            # and interaction models
            null_core = Core(X_null, self._kinships, null_param_perm)
            add_core = Core(X, self._kinships, null_param_perm)
            try:
                int_core = Core(Xi, self._kinships, null_param_perm)
                cores = [null_core, add_core, int_core]
                lratios = [likelihood_ratio, x_likelihood_ratio]
            except NameError:
                cores = [null_core, add_core]
                lratios = [likelihood_ratio]

            p_value, x_p_value = self._compute_p_value(self._Y, cores, lratios, null_param_perm, perm)

            # return the test statistics, p-values, effect sizes, and standard
            # errors for each variant in the locus
            try:
                result = [variant] + \
                         [likelihood_ratio, p_value] + \
                         beta.ravel().tolist() + \
                         serr.ravel().tolist() + \
                         [x_likelihood_ratio, x_p_value] + \
                         x_beta.ravel().tolist() + \
                         x_serr.ravel().tolist()

            except NameError:
                result = [variant] + \
                         [likelihood_ratio, p_value] + \
                         beta.ravel().tolist() + \
                         serr.ravel().tolist()

            yield result
