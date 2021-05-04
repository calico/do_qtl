import itertools
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from lib.data_io import Covariate
from lib.data_io import Genotype
from lib.data_io import Phenotype

from lib.data_io import intersect_datasets

PARENTAL_LINES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
TREATMENTS = ['AL', '1D', '2D', '40', '20']
PHENOTYPE = 'phenotype'
COVARIATE = 'covariate'
VARIATE_INDEX_NAME = {PHENOTYPE: 'MouseID', COVARIATE: 'mouse_id'}


def make_variates_filename(variates_path, variate_type):
    return os.path.join(variates_path, f'{variate_type}.csv')


# make names


def make_markers(lines=PARENTAL_LINES, n_variants=100):
    markers = []
    for index in range(n_variants):
        coupled_markers = []
        for line in PARENTAL_LINES:
            coupled_markers.append(f'{line}.Calico{index}')
        markers.append(coupled_markers)
    return markers


def make_mouse_ids(treatments=TREATMENTS, mice_per_treatment=10):
    mouse = 1
    mouse_ids = []
    for treatment in treatments:
        for _ in range(mice_per_treatment):
            mouse_ids.append(f'Calico-{treatment}-{mouse:04}')
            mouse += 1
    return mouse_ids


def make_variates_names(variate_type, n_variates=10):
    return [f'{variate_type}_{i}' for i in range(n_variates)]


# make data frames


def make_genoprobs_data(markers, mouse_ids, seed=1):
    all_markers = list(itertools.chain(*markers))
    mats = []
    np.random.seed(seed)
    for coupled_markers in markers:
        mat = np.random.uniform(size=(len(coupled_markers), len(mouse_ids)))
        mat = mat**10.0
        mat = mat / np.sum(mat, axis=0, keepdims=True)
        mats.append(mat)
    data = np.concatenate(mats, axis=0)
    df = pd.DataFrame(columns=mouse_ids, index=all_markers, data=data)
    return df, mats


def make_variates_df(variates_names, mouse_ids, seed=1, variate_type=None):
    np.random.seed(seed)
    data = np.random.normal(size=(len(mouse_ids), len(variates_names)))
    df = pd.DataFrame(index=mouse_ids, columns=variates_names, data=data)
    df.index.name = VARIATE_INDEX_NAME[variate_type]
    return df


# save data to csv


def save_genoprobs(genoprobs_file, genoprobs_df):
    genoprobs_df.to_csv(genoprobs_file)


def save_variates(variates_filename, variates_df):
    variates_df.to_csv(variates_filename)


# construct genoprob    test files


def make_genoprobs_file(genoprobs_file):
    markers = make_markers()
    mouse_ids = make_mouse_ids()
    genoprobs_df, genoprobs_mats = make_genoprobs_data(markers=markers,
                                                       mouse_ids=mouse_ids)
    save_genoprobs(genoprobs_file=genoprobs_file,
                   genoprobs_df=genoprobs_df)


def make_genoprobs_and_kinship_file(chromosome,
                                    genoprobs_file,
                                    kinship_file,
                                    left_out_chromosome=None):
    make_genoprobs_file(genoprobs_file)
    genotype = Genotype()
    data = np.vstack([genoprob[1] for genoprob in genotype.load_genoprobs(genoprobs_file)])
    kinship = (data.T @ data) / (data.shape[0] / 8)
    index = genotype.all_samples
    columns = genotype.all_samples
    kinship_df = pd.DataFrame(index=index, columns=columns, data=kinship)
    kinship_df.to_csv(kinship_file, index_label='')


def make_genoprobs_and_kinship_subsets_files(chromosome, genoprobs_file,
                                             genoprobs_subset_file, kinship_file,
                                             kinship_subset_file):
    markers = make_markers()
    mouse_ids = make_mouse_ids()
    genoprobs_df, genoprobs_mats = make_genoprobs_data(markers=markers,
                                                       mouse_ids=mouse_ids)
    save_genoprobs(genoprobs_file, genoprobs_df)
    data = genoprobs_df.values
    kinship = (data.T @ data) / (data.shape[0] / 8)
    kinship_df = pd.DataFrame(index=mouse_ids, 
                              columns=mouse_ids, 
                              data=kinship)
    kinship_df.to_csv(kinship_file, index_label='')

    genoprobs_subset_df = genoprobs_df.drop(columns=mouse_ids[25:])
    save_genoprobs(genoprobs_subset_file, genoprobs_subset_df)
    data = genoprobs_subset_df.values
    kinship_subset = (data.T @ data) / (data.shape[0] / 8)
    kinship_subset_df = pd.DataFrame(index=mouse_ids[25:], 
                                     columns=mouse_ids[25:], 
                                     data=kinship_subset)
    kinship_subset_df.to_csv(kinship_subset_file, index_label='')


# construct phenotype test files


def make_variates_file(variates_filename, variate_type):
    variates_names = make_variates_names(variate_type)
    mouse_ids = make_mouse_ids()
    variates_df = make_variates_df(variates_names=variates_names,
                                   mouse_ids=mouse_ids,
                                   variate_type=variate_type)
    save_variates(variates_filename, variates_df)


def make_variates_subsets_files(variates_filename, variates_subset_filename,
                                                                variate_type):
    variates_names = make_variates_names(variate_type)
    mouse_ids = make_mouse_ids()
    variates_df = make_variates_df(variates_names=variates_names,
                                   mouse_ids=mouse_ids,
                                   variate_type=variate_type)
    save_variates(variates_filename, variates_df)
    variates_subset_df = variates_df.drop(index=mouse_ids[25:])
    save_variates(variates_subset_filename, variates_subset_df)


class TestGenoprob(unittest.TestCase):

    def test_load_genoprobs(self):
        chromosome = '1'
        genoprobs_path = tempfile.mkdtemp()
        genoprobs_file = os.path.join(genoprobs_path, chromosome)
        markers = make_markers()
        mouse_ids = make_mouse_ids()
        genoprobs_df, genoprobs_mats = make_genoprobs_data(markers=markers,
                                                           mouse_ids=mouse_ids)
        save_genoprobs(genoprobs_file=genoprobs_file,
                       genoprobs_df=genoprobs_df)
        genotype = Genotype()
        genoprobs = genotype.load_genoprobs(genoprobs_file)
        for mat1, mat2 in zip(genoprobs, genoprobs_mats):
            np.testing.assert_almost_equal(mat1[1], mat2)
        shutil.rmtree(genoprobs_path)

    def test_load_kinship_left_out_chromosome(self):
        chromosome = '1'
        genoprobs_path = tempfile.mkdtemp()
        genoprobs_file = os.path.join(genoprobs_path, chromosome)
        kinship_file = os.path.join(genoprobs_path, 'kinship.csv')

        make_genoprobs_and_kinship_file(chromosome=chromosome,
                                        genoprobs_file=genoprobs_file,
                                        kinship_file=kinship_file,
                                        left_out_chromosome=chromosome)
        genotype = Genotype()
        genotype.load_kinship(kinship_file)
        genoprobs = np.vstack([genoprob[1] for genoprob in genotype.load_genoprobs(genoprobs_file)])
        kinship = (genoprobs.T @ genoprobs) / (genoprobs.shape[0] / 8)
        np.testing.assert_almost_equal(kinship, genotype._kinship)
        shutil.rmtree(genoprobs_path)

    def test_load_kinship(self):
        chromosome = '1'
        genoprobs_path = tempfile.mkdtemp()
        genoprobs_file = os.path.join(genoprobs_path, chromosome)
        kinship_file = os.path.join(genoprobs_path, 'kinship.csv')
        make_genoprobs_and_kinship_file(
                chromosome=chromosome,
                genoprobs_file=genoprobs_file,
                kinship_file=kinship_file,
                left_out_chromosome=None)
        genotype = Genotype()
        genotype.load_kinship(kinship_file)
        genoprobs = np.vstack([genoprob[1] for genoprob in genotype.load_genoprobs(genoprobs_file)])
        kinship = (genoprobs.T @ genoprobs) / (genoprobs.shape[0] / 8)
        np.testing.assert_almost_equal(kinship, genotype._kinship)
        shutil.rmtree(genoprobs_path)

    def test_subset(self):
        chromosome = '1'
        genoprobs_path = tempfile.mkdtemp()
        genoprobs_file = os.path.join(genoprobs_path, chromosome)
        kinship_file = os.path.join(genoprobs_path, 'kinship.csv')
        genoprobs_subset_path = tempfile.mkdtemp()
        genoprobs_subset_file = os.path.join(genoprobs_subset_path, chromosome)
        kinship_subset_file = os.path.join(genoprobs_subset_path, 'kinship.csv')
        make_genoprobs_and_kinship_subsets_files(chromosome=chromosome,
                                                 genoprobs_file=genoprobs_file,
                                                 genoprobs_subset_file=genoprobs_subset_file,
                                                 kinship_file=kinship_file,
                                                 kinship_subset_file=kinship_subset_file)
        genotype = Genotype()
        genotype.load_kinship(kinship_file)
        genotype.reduce_to(genotype.all_samples[:25])
        genoprobs = np.vstack([genoprob[1] for genoprob in genotype.load_genoprobs(genoprobs_file)])

        genotype_subset = Genotype()
        genoprobs_subset = np.vstack([genoprob[1] for genoprob in genotype_subset.load_genoprobs(genoprobs_subset_file)])
        genotype_subset.load_kinship(kinship_subset_file)
        genotype_subset.reduce_to(genotype_subset.all_samples)

        np.testing.assert_almost_equal(genotype.kinship, genotype_subset.kinship)
        np.testing.assert_almost_equal(genoprobs, genoprobs_subset)
        shutil.rmtree(genoprobs_path)
        shutil.rmtree(genoprobs_subset_path)


class TestPhenotype(unittest.TestCase):

    def test_load(self):
        variates_path = tempfile.mkdtemp()
        variate_type = PHENOTYPE
        filename = make_variates_filename(variates_path=variates_path,
                                          variate_type=variate_type)
        variates_names = make_variates_names(variate_type=variate_type)
        mouse_ids = make_mouse_ids()
        variates_df = make_variates_df(variates_names=variates_names,
                                       mouse_ids=mouse_ids,
                                       variate_type=variate_type)
        save_variates(filename, variates_df)
        phenotype = Phenotype()
        phenotype.load(filename, 'phenotype_9')
        np.testing.assert_almost_equal(variates_df.values[:, 9:10], phenotype.all_data)
        shutil.rmtree(variates_path)

    def test_subset(self):
        variates_path = tempfile.mkdtemp()
        variates_subset_path = tempfile.mkdtemp()
        variate_type = PHENOTYPE
        variates_filename = make_variates_filename(variates_path=variates_path,
                                                   variate_type=variate_type)
        variates_subset_filename = make_variates_filename(
                variates_path=variates_subset_path, variate_type=variate_type)
        make_variates_subsets_files(
                variates_filename=variates_filename,
                variates_subset_filename=variates_subset_filename,
                variate_type=variate_type)
        phenotypes = Phenotype()
        phenotypes.load(variates_filename, 'phenotype_9')
        phenotypes.reduce_to(phenotypes.all_samples[:25])
        phenotypes_subset = Phenotype()
        phenotypes_subset.load(variates_subset_filename, 'phenotype_9')
        np.testing.assert_almost_equal(phenotypes.data, phenotypes_subset.all_data)
        shutil.rmtree(variates_path)


class TestCovariate(unittest.TestCase):

    def test_load(self):
        variates_path = tempfile.mkdtemp()
        variate_type = PHENOTYPE
        filename = make_variates_filename(variates_path=variates_path,
                                          variate_type=variate_type)
        variates_names = make_variates_names(variate_type=variate_type)
        mouse_ids = make_mouse_ids()
        variates_df = make_variates_df(variates_names=variates_names,
                                       mouse_ids=mouse_ids,
                                       variate_type=variate_type)
        save_variates(filename, variates_df)
        variates = Covariate()
        variates.load(filename)
        np.testing.assert_almost_equal(variates_df.values, variates.all_data)
        shutil.rmtree(variates_path)

    def test_subset(self):
        variates_path = tempfile.mkdtemp()
        variates_subset_path = tempfile.mkdtemp()

        variate_type = COVARIATE
        variates_filename = make_variates_filename(variates_path=variates_path,
                                                   variate_type=variate_type)
        variates_subset_filename = make_variates_filename(
                variates_path=variates_subset_path, variate_type=variate_type)
        make_variates_subsets_files(
                variates_filename=variates_filename,
                variates_subset_filename=variates_subset_filename,
                variate_type=variate_type)
        covariates = Covariate()
        covariates.load(variates_filename)
        covariates.reduce_to(covariates.all_samples[:25])
        covariates_subset = Covariate()
        covariates_subset.load(variates_subset_filename)
        np.testing.assert_almost_equal(covariates.data, covariates_subset.all_data)
        shutil.rmtree(variates_path)
        shutil.rmtree(variates_subset_path)


class TestIntersection(unittest.TestCase):

    def test_intersect_dataset(self):

        chromosome = '1'
        data_path = tempfile.mkdtemp()
        genoprobs_file = os.path.join(data_path, chromosome)
        kinship_file = os.path.join(data_path, 'kinship.csv')
        data_subset_path = tempfile.mkdtemp()
        genoprobs_subset_file = os.path.join(data_subset_path, chromosome)
        kinship_subset_file = os.path.join(data_subset_path, 'kinship.csv')

        make_genoprobs_and_kinship_subsets_files(chromosome,
                                                 genoprobs_file,
                                                 genoprobs_subset_file,
                                                 kinship_file,
                                                 kinship_subset_file)
        covariates_file = make_variates_filename(variates_path=data_path,
                                                 variate_type=COVARIATE)
        make_variates_file(covariates_file, COVARIATE)
        phenotypes_file = make_variates_filename(
                variates_path=data_path, variate_type=PHENOTYPE)
        make_variates_file(phenotypes_file, PHENOTYPE)

        genotype = Genotype()
        genotype.load_kinship(kinship_file)
        covariate = Covariate()
        covariate.load(covariates_file)
        phenotype = Phenotype()
        phenotype.load(phenotypes_file, 'phenotype_8')

        intersect_datasets(genotype, phenotype, [covariate])

        self.assertTrue(genotype.N_samples == phenotype.N_samples)
        self.assertTrue(genotype.N_samples == covariate.N_samples)
        shutil.rmtree(data_path)
        shutil.rmtree(data_subset_path)
