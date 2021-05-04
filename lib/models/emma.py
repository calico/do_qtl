import functools

import numpy as np
import scipy.optimize as opt

from ..models import utils

TOL = 1e-8
EPS = 1e-12
MAX_PERM = 1e8
NUM_NULL = 10

class Core:

    """
    Model core enables efficient computation of likelihood
    and gradients by maintaining the current parameter state

    Parameters
    ----------
    X : ndarray, shape (n, c)
        Matrix of covariates, where 'n' is the number of samples
        and 'c' is the number of covariates.

    kinship: ndarray, shape (n, n)
             genetic kinship matrix (:math:`\mathcal{K}`)
    """

    def __init__(self, X, kinship):

        self.X = X
        self.N = self.X.shape[0]
        self.kinship = kinship
        self.kinship_eigvals = np.linalg.eigvals(self.kinship)

        # computing the linear projection matrix
        S = np.identity(self.N) - X @ np.linalg.pinv(X.T @ X) @ X.T
        SKS = S @ self.kinship @ S

        # ensure SKS is symmetric; if not, symmetrize
        try:
            assert np.all(SKS==SKS.T)
        except AssertionError:
            SKS = 0.5*(SKS + SKS.T)

        self.SKS_eigvals, self.SKS_eigvecs = np.linalg.eigh(SKS)
        self.mask = 1*(self.SKS_eigvals>EPS)

    def likelihood(self, h, Y):

        """Compute likelihood

        Parameters
        ----------
        h : float
            ratio of residual variance to genetic variance

        Y : ndarray, shape (n, 1)
            phenotype values for 'n' samples
        """

        SKvy = (self.SKS_eigvecs.T @ Y).T[0]**2

        # likelihood of a multivariate normal distribution
        # at the ML estimate of the mean parameter
        L = self.N * np.log(self.N/(2*np.pi)) - \
            self.N - \
            self.N*np.log(np.sum(SKvy*self.mask/(self.SKS_eigvals+h))) - \
            np.sum(np.log(self.kinship_eigvals+h))

        return 0.5*L

    def function(self, h, Y):

        """Compute function (- log likelihood) to be minimized

        Parameters
        ----------
        h : float
            ratio of residual variance to genetic variance

        Y : ndarray, shape (n, 1)
            phenotype values for 'n' samples
        """

        return -np.sum(self.likelihood(h, Y))

    def gradient(self, h, Y):

        """Compute gradient of -log likelihood

        Parameters
        ----------
        h : float
            ratio of residual variance to genetic variance

        Y : ndarray, shape (n, 1)
            phenotype values for 'n' samples
        """

        SKvy = (self.SKS_eigvecs.T @ Y).T[0]**2

        dL = self.N * np.sum(SKvy*self.mask/(self.SKS_eigvals+h)**2)/np.sum(SKvy*self.mask/(self.SKS_eigvals+h)) - \
             np.sum(1./(self.kinship_eigvals+h))

        return -0.5*np.array([dL])

class Emma:

    """Model underlying Efficient Mixed Model Association (EMMA)

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

    sigma_e : float
              Residual variance component

    sigma_g : float
              Genetic variance component

    log_likelihood : float
                     Log likelihood at optimal model parameters

    pve : float
          Proportion of variance explained by genetics

    var : float
          Total genetic variance

    pve_serr : float
               Standard error of PVE

    var_serr : float
               Standard error of total genetic variance
    """

    def __init__(self, kinship, phenotype, covariates):

        self.kinship = kinship
        # eigen decomposition of kinship matrix
        self.kinship_eigvals, self.kinship_eigvecs = np.linalg.eigs(self.kinship)
        self.N = self.kinship.shape[0]

        self.phenotype = phenotype
        self._Y = phenotype

        self.covariates = covariates
        self._X = np.hstack([np.ones((self.N, 1))] + \
                            [covariates.data for covariates in self.covariates
                             if covariates.effect == 'fixed'])
        self.C = self._X.shape[1]

        if np.any([covariates.gxe for covariates in self.covariates]):
            self._x_covariates = np.hstack([covariates.data
                                            for covariates in self.covariates
                                            if covariates.gxe])
        else:
            self._x_covariates = np.zeros((self.N, 0))
        self.E = self._x_covariates.shape[1]

        # attributes to be computed, given data
        # model parameters
        self.sigma_e = None
        self.sigma_g = None
        # likelihood
        self.log_likelihood = None
        # estimates of PVE and std err
        self.pve = None
        self.var = None
        self.pve_serr = None
        self.var_serr = None

    def fit_relative_variance_components(self, X, Y, init_param=None, indices=None, tol=TOL):

        """Estimate variance component parameters

        Parameters
        ----------
        X : ndarray, shape (n, c)
            Matrix of covariates

        Y : ndarray, shape (n, 1)
            Matrix of phenotype values

        init_param : float
                     Ratio of residual variance component to genetic variance component

        indices : ndarray, shape (m, ), m<n
                  Subset of samples to limit, when estimating parameters

        tol : float
              Termination criterion for scipy.optimize

        """

        if indices is None:
            kinship = self.kinship.copy()
            X_ = X.copy()
            Y_ = Y.copy()
            Ke = self.kinship_eigvals
        else:
            N = indices.size
            kinship = self.kinship[indices,:][:,indices]
            X_ = X[indices,:]
            Y_ = Y[indices,:]
            Ke, _ = np.linalg.eigh(kinship)

        retry = 0
        success = False
        while not success and retry<3:
            if init_param is None or retry>0:
                init_param = 10*np.random.rand(1)
            core = Core(X_, kinship)
            args = (Y_,)
            bounds = [(0, np.inf)]

            result = opt.minimize(core.function, init_param, jac=core.gradient, 
                                  args=args, method='L-BFGS-B', bounds=bounds,
                                  tol=tol)

            if result['success']:
                optimal_param = result['x']
                log_likelihood = -result['fun']
                success = True
            else:
                retry += 1

        if success:
            return optimal_param, log_likelihood
        else:
            raise ValueError("Failed to find optimal variance component parameters")
         
    def fit_effect_size(self, X, Y, h, indices=None, compute_stderr=False):

        """Estimate fixed effects, given variance component
        parameters

        Parameters
        ----------
        X : ndarray, shape (n, c)
            Matrix of covariates

        Y : ndarray, shape (n, 1)
            Matrix of phenotype values

        h: float
           Ratio of residual variance component to genetic variance component

        indices : ndarray, shape (m, ), m<n, default=None
            Subset of samples to limit, when estimating parameters

        compute_stderr : bool, default=False
            Flag specifying whether to compute the standard error of the
            parameter estimates, using the Fisher information matrix.
        """

        if indices is None:
            E = np.diag(1./(self.kinship_eigvals+h))
            L = X.T @ E @ X
            R = X.T @ E @ Y
        else:
            Ke = np.linalg.eigvals(self.kinships[indices,:][:,indices])
            E = np.diag(1./(Ke+h))
            L = X[indices].T @ E @ X[indices]
            R = X[indices].T @ E @ Y[indices]
        
        beta = np.linalg.pinv(L) @ R

        if compute_stderr:
            s = X.T @ E @ X
            fisher_information = 0.5*(s+s.T)
            S = np.linalg.pinv(fisher_information)
            serr = np.diag(S)**0.5

            return beta, serr
        else:
            return beta

    def compute_pve(self, h, indices=None):

        """Compute the proportion of phenotypic variance 
        explained by genetics, given estimates of model parameters

        Parameters
        ----------

        h : float
            Ratio of residual variance component to genetic variance
            component

        indices : ndarray, shape (m, ), default=None
            Subset of samples to limit, when estimating parameters
        """

        projX = self.kinship_eigvecs.T @ self._X
        projY = self.kinship_eigvecs.T @ self._Y
        beta = self.fit_effect_size(projX, projY, h, indices=indices)
        
        if indices is None:
            mu = self._X @ beta
            R = (self._Y-mu).T \
                @ np.linalg.pinv(self.kinship \
                                 + h*np.eye(self.N)) \
                @ (self._Y-mu)
            sigma_g = R/self.N
            sigma_e = h*sigma_g

            num = sigma_g * (np.trace(self.kinship)/self.N - np.sum(self.kinship)/self.N**2)
            den = np.var(mu) + num + (1-1./self.N) * sigma_e
        else:
            mu = self._X[indices] @ beta
            R = (self._Y[indices]-mu).T \
                @ np.linalg.pinv(self.kinship[indices,:][:,indices] \
                                 + h*np.eye(indices.size)) \
                @ (self._Y[indices]-mu)
            sigma_g = R/indices.size
            sigma_e = h*sigma_g

            num = sigma_g * (np.trace(self.kinship[indices,:][:,indices])/indices.size \
                             - np.sum(self.kinship[indices,:][:,indices])/indices.size**2)
            den = np.var(mu) + num + (1-1./indices.size) * sigma_e

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

        # to avoid local optima issues, estimate with 10 random restarts 
        params = [] 
        log_likelihoods = np.zeros((10,))
        rmax = 10
        print("estimating PVE with %d random runs ..."%rmax)
        r = 0
        while r<rmax:
            try:
                param, log_likelihood = self.fit_relative_variance_components(self._X, self._Y)
                params.append(param)
                log_likelihoods[r] = log_likelihood
                r += 1
                print("completed run %d; log likelihood = %.4f"%(r, log_likelihood))
            except TypeError:
                pass

        # select estimate with highest log likelihood
        print("estimating PVE, using run with highest log likelihood")
        optimal_param = params[np.argmax(log_likelihoods)]
        self.log_likelihood = log_likelihoods[np.argmax(log_likelihoods)]
        self.pve, self.var = self.compute_pve(optimal_param)

        if get_serr:
            print("estimating std. error of PVE using jackknifing ...")
            hs = []
            var = []
            
            jackknifed_samples = utils.jackknife(self.N, self.N//10, 10, X=self._X)
            for j_samples in jackknifed_samples:
                indices = np.delete(np.arange(self.N), j_samples)
                param_, _ = self.fit_relative_variance_components(self._X, self._Y, 
                                                                  init_param=optimal_param, 
                                                                  indices=indices, tol=TOL)
                a, b = self.compute_pve(param_, indices=indices)
                hs.append(a)
                var.append(b)
            self.pve_serr = np.nanstd(hs)
            self.var_serr = np.nanstd(var)
        else:
            self.pve_serr = None
            self.var_serr = None

    def _compute_p_value(self, Y, cores, likelihood_ratios, null_param_perm, perm):

        """Compute p-value using a sequential permutation scheme.

        Parameters
        ----------

        Y : ndarray, shape (n,1)
            Phenotype vector

        cores : list
                list of Core instances to evaluate likelihood
                under the null, additive, and interaction models

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
                Y_perm = np.random.permutation(self._Y.ravel()).reshape(self._Y.shape)

                # compute likelihood under null, additive, and interaction
                # model for permuted phenotype vector
                log_likelihood_perm = add_core.likelihood(null_param_perm, Y_perm)
                if p < perm and P < MAX_PERM:
                    null_log_likelihood_perm = null_core.likelihood(null_param_perm, Y_perm)
                    if (log_likelihood_perm-null_log_likelihood_perm) >= likelihood_ratio:
                        p += 1
                    P += 1
                try:
                    if xp < perm and xP < MAX_PERM:
                        x_log_likelihood_perm = int_core.likelihood(null_param_perm, Y_perm)
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

        projY = self.kinship_eigvecs.T @ self._Y

        X_null = np.hstack([np.ones((self.N, 1), dtype='float')] + \
                           [covariates.data for covariates in self.covariates
                            if covariates.effect == 'fixed'])
        if np.any([covariates.test_gxe for covariates in self.covariates]):
            Xi_test = np.hstack([covariates.data for covariates in self.covariates
                                 if covariates.test_gxe])

        print("estimating variance components under the null model ...")
        null_param, null_likelihood = self.fit_variance_components(X_null, self._Y)

        if perm is not None:
            # from a collection of permuted datasets, compute null model
            # parameters and estimate their average.
            # these are kept fixed, in both the null and alternate models,
            # when computing test statistics for each permutation.
            print("estimating expected variance components under the null model for permuted phenotypes ...")
            null_param_perms = []
            for _ in np.arange(NUM_NULL):
                Y_perm = np.random.permutation(self._Y.ravel()).reshape(self._Y.shape)
                null_param_perm, _ = self.fit_variance_components(X_null, Y_perm, init_param=null_param)
                null_param_perms.append(null_param_perm)
            null_param_perm = np.median(null_param_perms)

        # loop over all typed variants
        print("association testing at genotyped variants ...")
        for variant, _genotype in genotypes:

            # for additive model, append fixed effect covariates
            # with genotype of the focal variant
            X = np.hstack((X_null, _genotype.T))
            projX = self.kinship_eigvecs.T @ X

            if approx:
                # keep variance components fixed from null model
                core = Core(X, self.kinship)
                likelihood = core.likelihood(null_param, self._Y)
            else:
                # estimate variance components
                _, likelihood = self.fit_variance_components(X, self._Y, init_param=null_param)

            likelihood_ratio = likelihood - null_likelihood

            # for interaction model, append fixed effect covariates
            # with genotype of the focal variant and product
            # of environment and genotype of focal variant.
            try:
                Xi = np.hstack([X_null,
                                genotype.T,
                                utils.outer_product(genotype, Xi_test.T).T])
                projXi = self.kinship_eigvecs.T @ Xi

                if approx:
                    # keep variance components fixed from null model
                    core = Core(Xi, self.kinship)
                    x_likelihood = core.likelihood(null_param, self._Y)
                else:
                    # estimate variance components
                    _, x_likelihood = self.fit_variance_components(Xi, self._Y, init_param=null_param)

                x_likelihood_ratio = x_likelihood - likelihood
            except NameError:
                pass

            # a core instance for the null, additive,
            # and interaction models
            null_core = Core(X_null, self.kinship)
            add_core = Core(X, self.kinship)
            try:
                int_core = Core(Xi, self.kinship)
                cores = [null_core, add_core, int_core]
                lratios = [likelihood_ratio, x_likelihood_ratio]
            except NameError:
                cores = [null_core, add_core]
                lratios = [likelihood_ratio]
                pass
           
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

        projY = self.kinship_eigvecs.T @ self._Y

        X_null = np.hstack([np.ones((self.N, 1), dtype='float')] + \
                           [covariates.data for covariates in self.covariates
                            if covariates.effect == 'fixed'])
        if np.any([covariates.test_gxe for covariates in self.covariates]):
            Xi_test = np.hstack([covariates.data for covariates in self.covariates
                                 if covariates.test_gxe])

        print("estimating variance components under the null model ...")
        null_param, null_likelihood = self.fit_variance_components(X_null, self._Y)

        if perm is not None:
            print("estimating expected variance components under the null model for permuted phenotypes ...")
            null_param_perms = []
            for _ in np.arange(NUM_NULL):
                Y_perm = np.random.permutation(self._Y.ravel()).reshape(self._Y.shape)
                null_param_perm, _ = self.fit_variance_components(X_null, Y_perm, init_param=null_param)
                null_param_perms.append(null_param_perm)
            null_param_perm = np.median(null_param_perms)

        # loop over all imputed variants
        print("association testing at all variants in locus ...")
        for variant,_genotype in genotypes:

            # for additive model, append fixed effect covariates
            # with genotype of the focal variant
            X = np.hstack((X_null, _genotype.T))
            projX = self.kinship_eigvecs.T @ X

            if approx:
                # keep variance components fixed from null model
                core = Core(X, self.kinship)
                likelihood = core.likelihood(null_param, self._Y)
                # estimate effect sizes and standard errors
                beta, serr = self.fit_effect_size(projX, projY, null_param, 
                                                  compute_stderr=True)
            else:
                # estimate variance components
                param, likelihood = self.fit_variance_components(X, self._Y, init_param=null_param)
                # estimate effect sizes and standard errors
                beta, serr = self.fit_effect_size(projX, projY, null_param, 
                                                  compute_stderr=True)

            likelihood_ratio = likelihood - null_likelihood

            # for interaction model, append fixed effect covariates
            # with genotype of the focal variant and product
            # of environment and genotype of focal variant.
            try:
                Xi = np.hstack([X_null,
                                genotype.T,
                                utils.outer_product(genotype, Xi_test.T).T])
                projXi = self.kinship_eigvecs.T @ Xi

                if approx:
                    # keep variance components fixed from null model
                    core = Core(Xi, self.kinship)
                    x_likelihood = core.likelihood(null_param, Y)
                    # estimate effect sizes and standard errors
                    x_beta, x_serr = self.fit_effect_size(Xi, Y, null_param, compute_stderr=True)
                else:
                    # estimate variance components
                    param, x_likelihood = self.fit_variance_components(Xi, Y, init_param=null_param)
                    # estimate effect sizes and standard errors
                    x_beta, x_serr = self.fit_effect_size(Xi, Y, param, compute_stderr=True)

                x_likelihood_ratio = x_likelihood - likelihood
            except NameError:
                pass

            # a core instance for the null, additive,
            # and interaction models
            null_core = Core(X_null, self.kinship)
            add_core = Core(X, self.kinship)
            try:
                int_core = Core(Xi, self.kinship)
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
                         [x_likelihood_ratio, p_value] + \
                         x_beta.ravel().tolist() + \
                         x_serr.ravel().tolist()
            except NameError:
                result = [variant] + \
                         [likelihood_ratio, p_value] + \
                         beta.ravel().tolist() + \
                         serr.ravel().tolist()
            
            yield result

