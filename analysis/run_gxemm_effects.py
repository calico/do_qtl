# import packages
import numpy as np
import argparse
import pickle
import pdb
import sys
import csv
import os
import re
import pandas as pd
sys.path.append('/home/wright/')

##import tools from do_qtl package
from do_qtl.lib.models import gxemm
from do_qtl.lib import data_io as io

def main():

    args = parse_args()
    # load diet covariates 
    diet_covariate = io.Covariate(test_gxe=True, effect='fixed', gxe=True) 
    diet_covariate.load(args.covar_diet_file)
    print(diet_covariate.names)

    # load generation covariates
    gen_covariate = io.Covariate(test_gxe=False, effect='fixed', gxe=True) 
    gen_covariate.load(args.covar_gen_file)
    print(gen_covariate.names)

    # store all covariate objects as a list
    covariates = [diet_covariate, gen_covariate]

    # load kinship
    genotype = io.Genotype()
    genotype.load_kinship(args.kinship_file)
    
    # load genoprobs - probability of founder of origin genotypes
    genoprobs = genotype.load_genoprobs(args.genotype_file)

    # load phenotype
    phenotype = io.Phenotype()
    phenotype.load(args.phenotype_file, args.phenotype)
    
    ##subsample for testing
    #df = pd.read_csv(args.phenotype_file)
    #test_samples = df['MouseID'].tolist()
    #test_samples = test_samples[::4] ##take every x sample -- ie every 4th
    #print('test_samples', test_samples[0:5])
    
    # subset to common samples
    #io.intersect_datasets(genotype, phenotype, covariates, at_samples=test_samples)
    io.intersect_datasets(genotype, phenotype, covariates)
    print('n samples', genotype.N_samples)
    
    ##model - gxemm
    model = gxemm.Gxemm(genotype.kinship,
                        phenotype.data,
                        covariates)


    # this creates a generator
    results = model.run_finemap(genoprobs, approx=False)

    # run gwas and write results to file
    output_file = args.phenotype_file
    output_file = re.sub('.csv$','_'+args.phenotype+'_'+args.chromosome+'.eff.csv',output_file)
    print(output_file)
    
    # output association statistics
    header = ['variant.id'] + \
         ['additive.LOD', 'additive.p.value'] + \
         ['additive.intercept'] + \
         ['additive.effect.size.%s'%diet for diet in diet_covariate.names] + \
         ['additive.effect.size.%s'%gen for gen in gen_covariate.names] + \
         ['additive.effect.size.%s'%founder for founder in genotype.founders] + \
         ['additive.intercept.serr'] + \
         ['additive.effect.size.serr.%s'%diet for diet in diet_covariate.names] + \
         ['additive.effect.size.serr.%s'%gen for gen in gen_covariate.names] + \
         ['additive.effect.size.serr.%s'%founder for founder in genotype.founders] + \
         ['interaction.LOD', 'interaction.p.value'] + \
         ['interaction.intercept'] + \
         ['interaction.effect.size.%s'%diet for diet in diet_covariate.names] + \
         ['interaction.effect.size.%s'%gen for gen in gen_covariate.names] + \
         ['interaction.effect.size.%s'%founder for founder in genotype.founders] + \
         ['interaction.effect.size.%s_x_%s'%(founder,diet) for founder in genotype.founders for diet in diet_covariate.names] + \
         ['interaction.intercept.serr'] + \
         ['interaction.effect.size.serr.%s'%diet for diet in diet_covariate.names] + \
         ['interaction.effect.size.serr.%s'%gen for gen in gen_covariate.names] + \
         ['interaction.effect.size.serr.%s'%founder for founder in genotype.founders] + \
         ['interaction.effect.size.serr.%s_x_%s'%(founder,diet) for founder in genotype.founders for diet in diet_covariate.names]

    print(header)
    
    with open(output_file, 'w', buffering=1, newline='') as csvfile:
        handle = csv.writer(csvfile)
        handle.writerow(header)
        for result in results:
            handle.writerow(result)

            
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
    
    parser.add_argument("--kinship_file",
                        type=str)   
    
    args = parser.parse_args()

    print(args.phenotype_file, args.phenotype, args.chromosome, args.genotype_file, 
          args.covar_diet_file, args.covar_gen_file, args.kinship_file)

    return args

if __name__=="__main__":

    main()

