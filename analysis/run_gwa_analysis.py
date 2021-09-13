# import packages
import numpy as np
import argparse
import pickle
import pdb
import sys
import csv
import os
import pandas as pd
sys.path.append('/home/wright/')

##import tools from do_qtl package
from do_qtl.lib.models import gxemm
from do_qtl.lib import data_io as io

def main():

    args = parse_args()
    source_dir = "/home/wright/do_mice/qtl_lifespan_191022_deh_ell_jac/source_data_dir/"
    # load diet covariates 
    #covar_diet_file = source_dir+"covar_diet_200326.csv"
    diet_covariate = io.Covariate(test_gxe=True, effect='fixed', gxe=True) 
    diet_covariate.load(args.covar_diet_file)
    print(diet_covariate.names)

    # load generation covariates
    #covar_gen_file = source_dir+"covar_gen.sty.sex_200326.csv"  
    gen_covariate = io.Covariate(test_gxe=False, effect='fixed', gxe=True) 
    gen_covariate.load(args.covar_gen_file)
    print(gen_covariate.names)

    # store all covariate objects as a list
    covariates = [diet_covariate, gen_covariate]

    # load kinship
    kinship_file = source_dir+"kinship_all_200329_full.csv"
    genotype = io.Genotype()
    genotype.load_kinship(kinship_file)
    
    # load genoprobs - probability of founder of origin genotypes
    genoprobs = genotype.load_genoprobs(args.genotype_file)

    # load phenotype
    phenotype = io.Phenotype()
    phenotype.load(args.phenotype_file, args.phenotype)
    
    ##subsample for testing
    df = pd.read_csv(args.phenotype_file)
    test_samples = df['MouseID'].tolist()
    test_samples = test_samples[::6] ##take every x sample -- ie every 4th

    # subset to common samples
    io.intersect_datasets(genotype, phenotype, covariates, at_samples=test_samples)

    # subset to common samples
    #io.intersect_datasets(genoprob, phenotype, covariates)

    ##model - gxemm
    model = gxemm.Gxemm(genotype.kinship,
                        phenotype.data,
                        covariates)

    # this creates a generator
    results = model.run_gwas(genoprobs, approx=False, perm=10)

    # run gwas and write results to file
    output_file = '/'.join(args.phenotype_file.split('/')[:-1]+["qtl_output.%s.%s.csv"%(args.phenotype,args.chromosome)])
    print(output_file)
    

    # output association statistics
    header = ['variant.id'] + \
             ['additive.LOD', 'additive.p.value'] + \
             ['interaction.LOD', 'interaction.p.value']
    print(header)
    
    with open(output_file, 'w', buffering=1, newline='') as csvfile:
        handle = csv.writer(csvfile)
        handle.writerow(header)
        for result in results:
            handle.writerow(result)
    #        pdb.set_trace()
    

            
def parse_args():

    parser = argparse.ArgumentParser(description="runs QTL analysis for a specified chromosome")

    parser.add_argument("--chromosome",
                        type=str,
                        default=None)

    parser.add_argument("--phenotype_file",
                        type=str)

    parser.add_argument("--phenotype",
                        type=str)
    
    parser.add_argument("--genotype_file",
                        type=str,
                        default=None)
    
    parser.add_argument("--covar_diet_file",
                        type=str)

    parser.add_argument("--covar_gen_file",
                        type=str)    
    args = parser.parse_args()

    print(args.phenotype_file, args.phenotype, args.chromosome, args.genotype_file, 
          args.covar_diet_file, args.covar_gen_file)

    return args

if __name__=="__main__":

    main()

