import csv
import pysam

import numpy as np

class Genotype:

    """
    Loads, processes and returns the kinship matrix,
    parental probabilities at genotyped markers
    and genotypes of all variants.

    Attributes
    ----------

    N_samples : int
                Number of samples used for heritability and association
                analyses.

    N_all_samples : int
                    Total number of samples

    _kinship : ndarray; shape (N_all_samples, N_all_samples)
               Kinship matrix among all samples

    kinship : ndarray; shape (N_samples, N_samples)
              Kinship matrix among subset of samples used for heritability
              and association analyses

    all_samples : list
                  All samples in the dataset

    samples : list
              Samples used for heritability and association analyses

    sample_indices : ndarray
                     Indices of samples in `all_samples` used for heritability
                     and association analyses

    """

    def __init__(self):

        self.kinship_file = None

        self.all_samples = None
        self.N_all_samples = None

        self.samples = None
        self.N_samples = None
        self.sample_indices = None

        self._kinship = None
        self.kinship = None
        self.founders = 'ABCDEFGH'

    def load_genoprobs(self, genoprob_file):

        """Loads and returns the genotypes at typed variants.
        The genotype at each variant is an ndarray of shape (8, `.all_samples`),
        where genotype at each sample is a probability vector over 8 DO
        founder strains.

        Parameters
        ----------

        genoprob_file : str
                        Name of file containing genotypes for each typed
                        variant, represented as a probability vector over
                        founder strains.
        """

        # load genoprob probabilities
        
        with open(genoprob_file, newline='') as csvfile:
            handle = csv.reader(csvfile)
            header = next(handle)
            if self.N_all_samples is None:
                self.all_samples = header[1:]
                self.N_all_samples = len(self.all_samples)
            for row in handle:
                if row[0].startswith(self.founders[0]):
                    variant = row[0][2:]
                    genoprobs = []
                genoprobs.append(np.array(row[1:]).astype('float'))
                if row[0].startswith(self.founders[-1]):
                    genoprobs = np.array(genoprobs)
                    # restrict to subset of samples
                    if self.sample_indices:  
                        genoprobs = genoprobs[:, self.sample_indices]
                    yield variant, genoprobs

    def load_genotypes(self, genotype_file, chromosome=None, start=None, end=None):

        """Loads and returns the genotypes at all bi-allelic variants
        within a specific locus. The genotype at each variant is an
        ndarray of shape (1, `.all_samples`), where genotype at each
        sample is the dosage of the minor allele.

        Parameters
        ----------

        genotype_file : str
                        Name of file containing genotypes for all variants
        """

        # load genotypes
        handle = pysam.TabixFile(genotype_file)
        rows = handle.fetch(chromosome, start, end, parser=pysam.asTuple())
        for row in rows:
            variant = ':'.join(row[:2])
            genotypes = np.array(row[3:3+self.N_all_samples]).astype('float')
            # restrict to subset of samples
            if self.sample_indices:
                genotypes = genotypes[self.sample_indices]
            yield variant, np.expand_dims(genotypes, axis=0)

    def load_kinship(self, kinship_file):

        """Loads the kinship matrix.

        Parameters
        ----------

        kinship_file : str
                       Name of file containing kinship between all samples.
        """

        kinship = dict()
        with open(kinship_file, newline='') as csvfile:
            handle = csv.reader(csvfile)
            header = next(handle)
            self.all_samples = header[1:]
            self.N_all_samples = len(self.all_samples)
            for row, sample in zip(handle, self.all_samples):
                kinship[sample] = np.array(row[1:]).astype('float')
        self._kinship = np.array([kinship[sample] for sample in self.all_samples])

        self.reduce_to(self.all_samples)

    def reduce_to(self, samples):

        """Reduces the set of samples to be analyzed to a specified subset.

        Parameters
        ----------

        samples : list
                  List of samples to retain in the subset.
        """

        self.sample_indices = [self.all_samples.index(sample) for sample in samples]
        self.samples = samples
        self.N_samples = len(samples)
        self.kinship = self._kinship[:, self.sample_indices][self.sample_indices, :]

class Phenotype:

    """
    Loads, processes and returns the matrix of phenotype values.

    Attributes
    ----------

    all_samples : int
                Number of samples used for heritability and association
                analyses.

    N_all_samples : int
                    Total number of samples

    all_data : ndarray; shape (N_all_samples, 1)
               Phenotype values among all samples

    data : ndarray; shape (N_all_samples, 1)
              Phenotype values among subset of samples used for heritability
              and association analyses

    all_samples : list
                  All samples in the dataset

    samples : list
              Samples used for heritability and association analyses

    sample_indices : ndarray
                     Indices of samples in `all_samples` used for heritability
                     and association analyses

    """

    def __init__(self):

        self.all_samples = []
        self.all_data = []
        self.samples = None
        self.data = None

        self.N_all_samples = None
        self.N_samples = None

    def load(self, filename, phenotype_name):

        """Loads the matrix of phenotypes.

        Parameters
        ----------

        filename : str
                   Name of file containing phenotype values

        phenotype_name : str

        """

        # load phenotypes
        with open(filename, newline='') as csvfile:
            handle = csv.reader(csvfile)
            header = next(handle)
            sample_index = header.index('MouseID')
            index = header[1:].index(phenotype_name)
            for row in handle:
                try:
                    self.all_data.append([float(row[1:][index])])
                    self.all_samples.append(row[sample_index])
                except ValueError:
                    continue

        self.all_data = np.array(self.all_data)
        self.N_all_samples = self.all_data.shape[0]

        # remove nans and infs
        indices = np.unique(np.where(~np.logical_or(np.isnan(self.all_data), np.isinf(self.all_data)))[0])
        self.all_data = self.all_data[indices, :]
        self.all_samples = [self.all_samples[index] for index in indices]
        self.N_all_samples = indices.size

    def reduce_to(self, samples):

        """Reduces the set of samples to be analyzed to a specified subset.

        Parameters
        ----------

        samples : list
                  List of samples to retain in the subset.
        """

        sample_indices = [self.all_samples.index(sample) for sample in samples]
        self.samples = samples
        self.data = self.all_data[sample_indices, :]
        self.N_samples = len(self.samples)

class Covariate:

    """
    Loads, processes and returns a matrix of covariate values.

    Parameters
    ----------

    test_gxe : bool (default False)

    effect : str ('fixed' / 'random', default 'fixed')

    gxe : bool (default False)


    Attributes
    ----------

    all_samples : int
                  Number of samples used for heritability and association
                  analyses.

    N_all_samples : int
                    Total number of samples

    all_data : ndarray; shape (N_all_samples, 1)
               Covariate values among all samples

    data : ndarray; shape (N_samples, 1)
           Covariate values among subset of samples used for heritability
           and association analyses

    all_samples : list
                  All samples in the dataset

    samples : list
              Samples used for heritability and association analyses

    sample_indices : ndarray
                     Indices of samples in `N_all_samples` used for heritability
                     and association analyses

    _names : list
             List of names of all covariates.

    names : list
            List of names of covariates with non-zero values for at least one sample.

    N_covars : int
               Number of covariates

    """

    def __init__(self, test_gxe=False, effect='fixed', gxe=False):

        self.all_samples = []
        self.all_data = []
        self._names = []
        self.names = []

        self.samples = None
        self.data = None
        self.N_all_samples = None
        self.N_samples = None
        self.N_covars = None

        self.effect = effect
        self.gxe = gxe
        self.test_gxe = test_gxe
        self.blocks = None


    def load(self, filename):

        """Loads the matrix of covariates.

        Parameters
        ----------

        filename : str
                   Name of file containing covariate values.

        """

        with open(filename, newline='') as csvfile:
            handle = csv.reader(csvfile)
            header = next(handle)
            self._names = header[1:]
            for row in handle:
                self.all_data.append(row[1:])
                self.all_samples.append(row[0])

        self.all_data = np.array(self.all_data).astype('float')

        self.reduce_to(self.all_samples)

    def reduce_to(self, samples):

        """Reduces the set of samples to be analyzed to a specified subset.

        Parameters
        ----------

        samples : list
                  List of samples to retain in the subset.
        """

        sample_indices = [self.all_samples.index(sample) for sample in samples]
        self.samples = samples
        self.N_samples = len(self.samples)

        # subset data to specific samples
        self.data = self.all_data[sample_indices, :]

        # remove environments that aren't populated, in the subset of samples
        columns_to_keep = np.logical_or(~np.all(self.data == 0, 0), 
                                        np.var(self.data, 0) == 0)
        if np.sum(columns_to_keep) > 1:
            self.data = self.data[:, columns_to_keep]
            self.names = [self._names[i] for i in np.where(columns_to_keep)[0]]
        else:
            self.data = np.zeros((self.N_samples, 0))
            self.names = []
        self.N_covars = len(self.names)

        # compute blocking structure,
        # if covariates are to be included in gxe random effects
        if self.gxe:
            self.blocks = [self.data[:, c:c+1] * self.data[:, c:c+1].T
                           for c in np.arange(self.N_covars)]

def intersect_datasets(genotype, phenotype, covariates, at_samples=None):

    """Find intersection of samples in the genotype, phenotype, and covariate
    datasets, and subset each to the intersection.

    Parameters
    ----------

    genotype : object of Genotype

    phenotype : object of Phenotype

    covariates : list of objects of Covariates

    at_samples : list (default None)
                 List of samples to subset the intersection to

    """

    sample_sets = genotype.all_samples + phenotype.all_samples
    for covariate in covariates:
        sample_sets = sample_sets + covariate.all_samples
    all_sets = [sample for sample in sample_sets
                if np.all([sample in samples
                           for samples in [genotype.all_samples, phenotype.all_samples] + \
                                          [covariate.all_samples for covariate in covariates]])]

    if at_samples is None:
        common_samples = list(set(all_sets))
    else:
        common_samples = list(set(all_sets).intersection(at_samples))
    common_samples.sort()

    genotype.reduce_to(common_samples)
    phenotype.reduce_to(common_samples)
    _ = [covariate.reduce_to(common_samples) for covariate in covariates]
