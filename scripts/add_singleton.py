from space.tools.taxonomy import Lineage
from space.tools.data import H5pyData
import os
import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
import gzip
import csv
import sys
from loguru import logger
import argparse

np.random.seed(42)
csv.field_size_limit(sys.maxsize)

class LLineage(Lineage):

    def __init__(self,node_dmp_zip,group_dir) -> None:

        self.df = pd.read_csv(node_dmp_zip,sep='|',compression='zip',header=None)
        self.eggnog_ancestors = {f.split('.')[0] for f in os.listdir(group_dir) }
        self.group_dir = group_dir

    def common_ancestor(self,taxid_1,
                        taxid_2,l_1,l_2
                        ):


        taxid_1 = int(taxid_1)
        taxid_2 = int(taxid_2)

        for idx,taxid in enumerate(l_1):
            if taxid in l_2:
                common_ancestor = taxid
            else:
                break
        
        ## make sure eggNOG has the common ancestor, and the orthologs are not empty 
        while True:
            if self.check_ortholog_group(taxid_1,taxid_2,common_ancestor):
                break
            idx -= 1
            common_ancestor = l_1[idx-1]
        
        return str(taxid_1),str(taxid_2),int(common_ancestor)
    

def infer_common_ancestors(euk_groups,ncbi_lineage,eggnog_dir,
                           orthologs_dir,):

    euk_groups = pd.read_csv(euk_groups,sep='\t')

    ancestor_finder = LLineage(ncbi_lineage,eggnog_dir)

    lineages = dict(zip(euk_groups['taxid'],euk_groups['lineage']))
    ## change the lineages to list of integers
    lineages = {k:list(map(int,v.split(','))) for k,v in lineages.items()}

    pairs = list()

    for f in os.listdir(f'{orthologs_dir}/seeds'):
        src,tgt = f.split('.')[0].split('_')
        src,tgt = int(src),int(tgt)
        pairs.append((src,tgt))
    
    # data structure is different for non_seeds
    for d in os.listdir(f'{orthologs_dir}/non_seeds'):
        for f in os.listdir(f'{orthologs_dir}/non_seeds/{d}'):
            src,tgt = f.split('.')[0].split('_')
            src,tgt = int(src),int(tgt)
            pairs.append((src,tgt))

    ancestors = list()
    
    for src,tgt in pairs:
        l1 = lineages[src]
        l2 = lineages[tgt]
        ancestors.append(ancestor_finder.common_ancestor(src,tgt,l1,l2))
    return ancestors


def min_max(filename):
    p,e = H5pyData.read(filename, 16)

    min_e = float(np.min(e))

    max_e = float(np.max(e))

    return min_e, max_e

def find_min_max(species_file,directory,num_jobs):

    species_list = open(species_file).read().strip().split('\n')

    with Pool(num_jobs) as p:
        results = p.map(min_max, [f'{directory}/{species}.h5' 
                                  for species in species_list])

    results = list(itertools.chain(*results))

    return min(results), max(results)


#2. scale the embeddings
def scale_fn(filename, scaler, save_dir):

    taxid = filename.split('/')[-1].split('.')[0]

    p,e = H5pyData.read(filename, 16)

    e = e*scaler

    H5pyData.write(proteins=p, embedding=e, 
                   save_path=f'{save_dir}/{taxid}.h5',
                   precision=16)

    return None

def scale_embeddings(species_file,aligned_dir,num_jobs,save_dir,scaler=None):

    min_e, max_e = find_min_max(species_file,aligned_dir,num_jobs)
    
    if not scaler:
        scaler = min(0.99/max_e, abs(0.99/min_e))

    species_list = open(species_file).read().strip().split('\n')

    with Pool(num_jobs) as p:
        results = p.starmap(scale_fn, [(f'{aligned_dir}/{species}.h5', 
                                   scaler, save_dir) 
                                  for species in species_list])
    return None

def filter_singleton(species,embed_dir,sequence_dir,save_dir):

    emb_path = os.path.join(embed_dir,f'{species}.h5')

    interaction_proteins, _ = H5pyData.read(emb_path,16)
    interaction_proteins = set(interaction_proteins)

    sequence_file = f'{sequence_dir}/{species}.protein.sequences.v12.0.fa.gz'
    ## read the sequence file
    sequence_proteins = set()
    with gzip.open(sequence_file,'rt') as f:
        for line in f:
            if line.startswith('>'):
                sequence_proteins.add(line.strip().split()[0][1:])
    
    singleton_proteins = sequence_proteins - interaction_proteins

    with open(f'{save_dir}/{species}.singleton.proteins.v12.0.txt','w') as f:
        for protein in singleton_proteins:
            f.write(f'{protein}\n')
    return None



def filter_singleton_parallel(species,embed_dir,sequence_dir,save_dir,number_jobs):

    with Pool(number_jobs) as p:
        results = p.starmap(filter_singleton, [(species,embed_dir,sequence_dir,save_dir) 
                                               for species in species])
        
    return None

def filter_singleton_in_og(ancestor,singleton_dir,eggnog_dir,save_dir):

    group_file = f'{eggnog_dir}/{ancestor}.tsv.gz'

    singletons = dict()

    for f in os.listdir(singleton_dir):
        f = os.path.join(singleton_dir,f)
        s = open(f).read().strip().split('\n')
        species = f.split('/')[-1].split('.')[0]
        singletons[species] = set(s)

    records = list()

    with gzip.open(group_file,'rt') as f:

        for line in f:
            line = line.strip().split('\t')
            orthologs = set(line[-1].split(','))
            og_name = line[1]
            species = line[4].split(',')

            og_singletons = list()
            
            ## singleton in this og
            for o in orthologs:
                s = o.split('.')[0]
                ## check if this is a singleton
                if o in singletons[s]:
                    og_singletons.append(o)
            
            if len(og_singletons) == 0:
                continue    

            ## check if all the orthologs are singletons
            if len(og_singletons) == len(orthologs):
                records.append((og_name,','.join(species),','.join(og_singletons),'','non-interaction'))
            else:
                non_singleton_og = orthologs - set(og_singletons)
                records.append((og_name,','.join(species),','.join(og_singletons),','.join(non_singleton_og),'partial-interaction'))


    ## save as a gzipped file
    with gzip.open(f'{save_dir}/{ancestor}.tsv.gz','wt') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['og_name','species','singletons','non_singleton_orths','interaction'])
        writer.writerows(records)
    return None

def filter_singleton_in_og_parallel(ancestors,singleton_dir,eggnog_dir,save_dir,num_jobs):

    # ancestors = set([int(l.split('\t')[-1]) for l in ancestors])
    ancestors = set([int(l[-1]) for l in ancestors])

    with Pool(num_jobs) as p:
        results = p.starmap(filter_singleton_in_og, [(ancestor,singleton_dir,eggnog_dir,save_dir) 
                                               for ancestor in ancestors])
    
    return None


def generate_random_embeddings(size=512):
    # Generate random values between 0 and 1
    random_values = np.random.random(size)
    
    # Randomly choose whether each value should be in the negative or positive range
    negative_mask = np.random.random(size) < 0.5
    
    # Transform values to desired ranges
    result = np.where(negative_mask,
                     # For negative range: [-1, -0.99]
                     -1 + (random_values * 0.01),
                     # For positive range: [0.99, 1]
                     0.99 + (random_values * 0.01))
    
    return result


def allocate_orthologous_singletons(singleton_og_dir,save_name):

    noise = 1e-5

    singleton_embs = dict()

    for f in os.listdir(singleton_og_dir):
        f = f'{singleton_og_dir}/{f}'

        df = pd.read_csv(f,compression='gzip',sep='\t',header=None)

        df = df[df.iloc[:,-1] == 'non-interaction']

        for _,line in df.iterrows():
            
            proteins = line[2].split(',')

            random_emb = generate_random_embeddings()

            noise_ = noise * np.random.random(size=(len(proteins),512))

            protein_embs = random_emb + noise_

            for p_idx, p in enumerate(proteins):
                if p in singleton_embs:
                    singleton_embs[p].append(protein_embs[p_idx])
                else:
                    singleton_embs[p] = [protein_embs[p_idx]]

    ## average the embeddings
    for p,e in singleton_embs.items():
        e = np.array(e)
        e = np.mean(e,axis=0).reshape(1,-1)
        singleton_embs[p] = e

    ## save the embeddings
    H5pyData.write(list(singleton_embs.keys()),
                         np.array(list(singleton_embs.values())).reshape(-1,512),
                         save_name,16)
    
    return None


def extract_network_proteins(species,og_proteins,scaled_dir):
    # get the embeddings for og_proteins
    p,e = H5pyData.read(f'{scaled_dir}/{species}.h5',16)

    # make a dictionary of proteins 
    p2indx = {p:i for i,p in enumerate(p)}

    # get the indices of the proteins in the og_proteins
    indices = [p2indx[p] for p in og_proteins]
    used_e = e[indices]
    return og_proteins,used_e

def extract_network_proteins_parallel(singleton_ogs,scaled_dir,save_name,number_jobs):
    og_interaction_proteins = set()

    for f in os.listdir(singleton_ogs):
        f = f'{singleton_ogs}/{f}'

        df = pd.read_csv(f,compression='gzip',sep='\t',header=None)

        df_interaction = df[df.iloc[:,-1] == 'partial-interaction']

        for line in df_interaction.iloc[:,-2]:
            og_interaction_proteins.update(line.split(','))
    
    proteins_per_species = dict()
    for p in og_interaction_proteins:
        species = p.split('.')[0]
        if species in proteins_per_species:
            proteins_per_species[species].append(p)
        else:
            proteins_per_species[species] = [p]
    
    ## read the embeddings
    with Pool(number_jobs) as p:
        results = p.starmap(extract_network_proteins,[(s,p,scaled_dir) 
                                                   for s,p in proteins_per_species.items()])

    # save the embeddings in a single file
    proteins = list()
    embeddings = list()

    ## prepare the data for saving
    for og_proteins,used_e in results:
        proteins.extend(og_proteins)
        embeddings.extend(used_e)
    
    H5pyData.write(proteins,embeddings,save_name,16)

    return None


def collect_embeddings(species:str,interation_og:str,
         singleton_og_dir:str,
         pure_singleton_og:str,
         singleton_dir:str,
         scaled_dir:str,
         save_dir:str):
    
    singleton_embeddings = dict()
    ## load the singletons
    noise = 1e-5
    interaction_og,interaction_og_emb = H5pyData.read(interation_og,16)
    interaction_og2idx = {str(p):i for i,p in enumerate(interaction_og)}

    ## load the singleton orthologs 
    for f in os.listdir(singleton_og_dir):
        f = f'{singleton_og_dir}/{f}'

        df = pd.read_csv(f,compression='gzip',sep='\t',header=None)

        df_interaction = df[df.iloc[:,-1] == 'partial-interaction']

        for _, line in df_interaction.iterrows():
            og_species = line[1].split(',')
            if species not in og_species:
                continue

            ## proteins with interactions, in orthologous groups
            og_proteins = line[3].split(',')

            indices = list(map(lambda x: interaction_og2idx[x],og_proteins))
            og_emb = np.mean(interaction_og_emb[indices],axis=0)
            
            ## species proteins, singletons
            species_proteins = line[2].split(',')
            species_proteins = [p for p in species_proteins if p.split('.')[0] == species]
            noise_ = noise * np.random.random(size=(len(species_proteins),512))

            protein_embs = og_emb + noise_

            for p_idx, p in enumerate(species_proteins):
                if p in singleton_embeddings:
                    singleton_embeddings[p].append(protein_embs[p_idx])
                else:
                    singleton_embeddings[p] = [protein_embs[p_idx]]
    
    ## average the embeddings in orthologous groups
    for p,e in singleton_embeddings.items():
        e = np.array(e)
        e = np.mean(e,axis=0).reshape(-1)
        singleton_embeddings[p] = e
    
    ## put the pure singletons into the dictionary
    ## just to be incorporated in the end
    pure_singleton_og,pure_singleton_ogs_emb = H5pyData.read(pure_singleton_og,16) 
    pure_singleton_og2idx = {p:i for i,p in enumerate(pure_singleton_og)}
    ## filter out other species
    # pure_singleton_og = [p for p in pure_singleton_og if p.split('.')[0] == species]
    pure_singleton_og = list(filter(lambda x: x.split('.')[0] == species,pure_singleton_og))
    indices = list(map(lambda x: pure_singleton_og2idx[x],pure_singleton_og))
    pure_singleton_ogs_emb = pure_singleton_ogs_emb[indices]
    singleton_embeddings.update({p:e for p,e in zip(pure_singleton_og,pure_singleton_ogs_emb)})

    ## load the singletons list
    all_singletons = f'{singleton_dir}/{species}.singleton.proteins.v12.0.txt'
    all_singletons = open(all_singletons).read().strip().split('\n')

    ## 
    unsolved_singletons = set(all_singletons) - set(singleton_embeddings.keys())

    ## give random embeddings for the unsolved singletons
    for p in unsolved_singletons:
        random_emb = generate_random_embeddings()
        singleton_embeddings[p] = random_emb

    ## check if every proteins are solved
    assert len(set(all_singletons) - set(singleton_embeddings.keys())) == 0 

    ## save the embeddings
    proteins,embeddings = H5pyData.read(f'{scaled_dir}/{species}.h5',16)

    proteins = list(singleton_embeddings.keys()) + list(map(str,proteins))
    embeddings = np.concatenate([np.array(list(singleton_embeddings.values())),embeddings],axis=0)

    ## save the embeddings
    H5pyData.write(proteins,embeddings,f'{save_dir}/{species}.h5',16)

    return None

def collect_embeddings_parallel(species:list,interation_og:str,
         singleton_og_dir:str,
         pure_singleton_og:str,
         singleton_dir:str,
         scaled_dir:str,
         save_dir:str,
         number_jobs):
    
    with Pool(number_jobs) as p:
        p.starmap(collect_embeddings,[(s,interation_og,singleton_og_dir,
                                       pure_singleton_og,singleton_dir,scaled_dir,save_dir) 
                                       for s in species])
        
    return None


def main(aligned_dir,species_file,euk_groups,ncbi_lineage,eggnog_dir,
        sequence_dir,
        orthologs_dir,working_dir,full_embedding_save_dir,
        number_jobs,scaler=1.497):
    
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    
    if not os.path.exists(full_embedding_save_dir):
        os.makedirs(full_embedding_save_dir)

    logger.info('Infer common ancestors...')
    common_ancestors = infer_common_ancestors(euk_groups,ncbi_lineage,eggnog_dir,
                           orthologs_dir,)
    
    #save all the scaled embeddings
    scaled_save_dir = f'{working_dir}/scaled'
    if not os.path.exists(scaled_save_dir):
        os.makedirs(scaled_save_dir)
    logger.info('Scaling the embeddings...')
    scale_embeddings(species_file,aligned_dir,number_jobs,
                        scaled_save_dir,scaler=scaler)

    #filter singletons
    logger.info('Filtering singletons...')
    species = open(species_file).read().strip().split('\n')
    if not os.path.exists(f'{working_dir}/singleton_ids'):
        os.makedirs(f'{working_dir}/singleton_ids')
    filter_singleton_parallel(species,f'{working_dir}/scaled',sequence_dir=sequence_dir,
                                save_dir=f'{working_dir}/singleton_ids',
                                number_jobs=number_jobs)
    
    if not os.path.exists(f'{working_dir}/singleton_in_og'):
        os.makedirs(f'{working_dir}/singleton_in_og')
    logger.info('Filtering singletons in orthologous groups...')
    filter_singleton_in_og_parallel(common_ancestors,f'{working_dir}/singleton_ids',
                                    eggnog_dir,f'{working_dir}/singleton_in_og',
                                    number_jobs)
    logger.info('Allocating orthologous singletons...')
    allocate_orthologous_singletons(f'{working_dir}/singleton_in_og',
                                    f'{working_dir}/singleton_in_og_embeddings.h5')
    
    logger.info('Extracting network orthologs...')
    extract_network_proteins_parallel(f'{working_dir}/singleton_in_og',
                                      scaled_dir=scaled_save_dir,
                                      save_name=f'{working_dir}/og_proteins.h5',
                                        number_jobs=number_jobs)
    
    logger.info('Collecting embeddings...')
    collect_embeddings_parallel(species,interation_og=f'{working_dir}/og_proteins.h5',
                                singleton_og_dir=f'{working_dir}/singleton_in_og',
                                pure_singleton_og=f'{working_dir}/singleton_in_og_embeddings.h5',
                                singleton_dir=f'{working_dir}/singleton_ids',
                                scaled_dir=scaled_save_dir,
                                save_dir=full_embedding_save_dir,
                                number_jobs=number_jobs)
    # remove the working directory
    os.system(f'rm -r {working_dir}')

    logger.info('DONE.')
    return None


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Allocate embeddings for singletons')

    argparser.add_argument('--aligned_dir',type=str,required=False,
                           default='data/aligned',
                            help='Directory with aligned embeddings')
    argparser.add_argument('--species_file',type=str,required=False,
                            default='data/euks.txt',
                             help='File with species names')
    argparser.add_argument('--euk_groups',type=str,required=False,
                            default='data/euks_groups.tsv',
                            help='File with eukaryotic groups')
    argparser.add_argument('--ncbi_lineage',type=str,required=False,
                            default='data/ncbi_lineage.zip',
                            help='NCBI taxonomy lineage file')
    argparser.add_argument('--eggnog_dir',type=str,required=False,
                            default='data/eggnog',
                            help='Directory with eggNOG orthologous groups')
    argparser.add_argument('--sequence_dir',type=str,required=False,
                            default='data/sequences',
                            help='Directory with protein sequences')
    argparser.add_argument('--orthologs_dir',type=str,required=False,
                            default='data/orthologs',
                            help='Directory with orthologous groups')
    argparser.add_argument('--working_dir',type=str,required=False,
                            default='temp/singletons',
                            help='Working directory')
    argparser.add_argument('--full_embedding_save_dir',type=str,required=False,
                            default='results/functional_embeddings',
                            help='Directory to save the embeddings')
    argparser.add_argument('--number_jobs',type=int,required=False,
                            default=7,
                            help='Number of parallel jobs')
    argparser.add_argument('--scaler',type=float,required=False,
                            default=1.497,
                            help='Scaling factor for embeddings, default is 1.497 if you use the embeddings in data/aligned. \
                            None if you want to calculate the scaling factor')
    
    args = argparser.parse_args()

    main(args.aligned_dir,args.species_file,args.euk_groups,args.ncbi_lineage,args.eggnog_dir,
        args.sequence_dir,
        args.orthologs_dir,args.working_dir,args.full_embedding_save_dir,
        args.number_jobs,scaler=args.scaler)
    
    

