# do_qtl

![](https://github.com/calico/do-qtl/workflows/test-do_qtl/badge.svg)

## Overview

This repository contains Python implementations of linear mixed models for quantifying
the genetic contribution to phenotypes measured in Diversity Outbred mouse
populations. Specifically, this package can be used to

+ quantify heritability of a phenotype in an age- and environment-dependent manner
+ test and finemap additive genetic and gene-environment interaction effects on a phenotype
+ estimate the additive genetic and gene-environment effects on a phenotype

and includes implementations of the following models

+ [EMMA](https://www.genetics.org/content/178/3/1709)
+ [GxEMM](https://www.sciencedirect.com/science/article/pii/S0002929719304628)

Please cite our work if you find this tool useful.

> **Age and diet shape the genetic architecture of body weight in Diversity Outbred mice**
>
> Kevin Wright, Andrew Deighan, Andrea Di Francesco, Adam Freund, Vladimir Jojic, Gary Churchill, Anil Raj
>

## Installation

Install Python 3.6+, using [Miniconda](https://conda.io/miniconda.html).

Install command line developer tools (only for OSX)
```
xcode-select --install
```

In your conda environment, install `git-lfs`, clone this repository, and install `do_qtl`.
```
$ conda activate <your_env> # skip this, if you want to install in your base environment
(your_env) $ conda install git-lfs
(your_env) $ git lfs install
(your_env) $ git clone https://github.com/calico/do_qtl.git
(your_env) $ cd do_qtl
(your_env) $ python setup.py install
```

If you prefer using your default python installation, you can also use pip to install `do_qtl`.
```
$ pip install git-lfs
$ git lfs install
$ git clone https://github.com/calico/do_qtl.git
$ cd do_qtl
$ pip install .
```

## Data formats 

### Phenotype data
The phenotype data should be provided as a csv file with rows corresponding to samples
and columns corresponding to different phenotypes. The csv file should have a
header containing the phenotype names and the first column should contain
unique sample identifiers. Missing phenotype measurements should be specified
as `nan`. The test file, `test_data/phenotypes.csv`, illustrates how the phenotype
file should be formatted.

```
$ head -n 3 test_data/body_weight.csv
MouseID,bodyweight.100.days,bodyweight.400.days
DO-1D-3001,30.59297157074587,38.86958098329384
DO-1D-3002,27.976137838488697,39.68402895834862
```

### Covariate data
The covariate data should be provided as a csv file with rows corresponding to
samples and columns corresponding to different covariates. The csv file should
have a header containing the names of covariates and the first column should
contain unique sample identifiers. Missing covariate measurements should be
specified as `nan`. The test file, `test_data/diet_covariates.csv`, illustrates how
to format the covariate file.

```
$ head -n 3 test_data/diet_covariates.csv
mouse_id,AL,20,40,1D,2D
DO-1D-3001,0,0,0,1,0
DO-1D-3002,0,0,0,1,0
```

### Genetic data
The genetic data, depending on the analysis, consists of a kinship values
between pairs of samples, the probability of founder of origin for all
genotyped variant (genoprob file; specific to Diversity Outbred populations), 
and the genotypes for all bi-allelic variant (both typed and imputed variants).
Given raw genotype measurements for a sample of Diversity Outbred mice at a set
of variants, the [qtl2](https://kbroman.org/qtl2/) R package provides a 
collection of tools to generate the processed data listed above, which can then
be provided as formatted files as described below.

The kinship matrix should be provided as a csv file with rows and columns
corresponding to samples. The header and the first column of the csv file 
should contain unique sample identifiers. The test file, `test_data/kinship.csv`,
illustrates how to format the kinship file.

```
$ head -n 3 test_data/kinship.csv | cut -f1-3 -d\,
106953,DO-1D-3001,DO-1D-3002
DO-1D-3001,0.540894843312726,0.148986508519352
DO-1D-3002,0.148986508519352,0.540787000125951
```

The probability of founder of origin for genotyped variants (genoprob) should be
provided as a csv file, one file per chromosome, with columns corresponding to 
samples and rows (in sets of 8) corresponding to variants. The header of the csv file should
contain unique sample identifiers and the first column should contain unique
identifiers for the founder and variant. Specifically, for the Diversity
Outbred mice that are generated from 8 founder strains (labeled A - H), the
unique identifiers for the 8 rows corresponding to genetic variant `var001`
are `A.var001, B.var001, ..., H.var001`. The test file,
`test_data/genoprobs.chr12.csv`, illustrates how to format the genoprob file.

```
$ head -n 16 test_data/genoprobs.chr12.csv | cut -f1-3 -d\,
marker_name,DO-1D-3001,DO-1D-3002
A.UNCHS032579,0.0130412374870164,0.0226741939392891
B.UNCHS032579,0.0130412374870036,0.430078098988234
C.UNCHS032579,0.0130412377785297,0.01484907183902
D.UNCHS032579,0.0130412374870038,0.0454419472078346
E.UNCHS032579,0.473920781294331,0.472313774106396
F.UNCHS032579,9.46809002313584e-08,9.46809002313576e-08
G.UNCHS032579,9.4680904639065e-08,9.46809046387549e-08
H.UNCHS032579,0.473914079104309,0.0146427245573256
A.UNCHS032580,0.0129564839166751,0.0225947566686249
B.UNCHS032580,0.0129564839166622,0.430223502477532
C.UNCHS032580,0.0129564842083494,0.0147653159882496
D.UNCHS032580,0.0129564839166624,0.0453750751553371
E.UNCHS032580,0.474090380972926,0.472482486899249
F.UNCHS032580,3.99232718929701e-09,3.99232718929082e-09
G.UNCHS032580,3.99233398083263e-09,3.99233398035766e-09
```

The genotypes at both typed and imputed variants should be provided as a
[Tabix file](https://academic.oup.com/bioinformatics/article/27/5/718/262743), 
a tab-delimited compressed and index file format that is part of the 
[samtools](http://www.htslib.org/doc/tabix.html) package. The test file,
`test_data/genotypes.chr12.tab.gz`, illustrates how to format the genotype file.

```
$ gunzip -c test_data/genotypes.chr12.tab.gz | head -n 3 | cut -f1-6
12	3172832	3172833	0.000	0.000	0.000
12	3173714	3173715	0.474	0.015	0.013
12	3173718	3173719	0.000	0.000	0.000
```

## Usage 

This [colab notebook](https://colab.research.google.com/drive/110yTH0GZZcdNxEbs491Ba563f6srast5?usp=sharing) 
provides detailed examples describing how to use this package 
to quantify environment-dependent heritability, compute association 
test statistics and estimate effect sizes. 

## Testing

All tests can be run as follows

```
python -m unittest tests.test_io tests.test_gxemm
```
