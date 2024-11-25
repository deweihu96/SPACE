import os
import pandas as pd
import gzip
from itertools import combinations, product
from multiprocessing import Pool
from typing import Set
import re
## for some species, the taxid used in STRING is different from the one used in ncbi
__TAXID_CONVERTER__ = {339724:2059318,     
    1745343:2082293,
    1325735:2604345,
    1658172:1955775,
    56484:2754530,
    1266660:1072105,
    1229665:2502994,
    944018:1913371,
    743788:2126942
}


class Lineage:
    
    def __init__(self,node_dmp_zip,group_dir) -> None:

        self.df = pd.read_csv(node_dmp_zip,sep='|',compression='zip',header=None)
        self.eggnog_ancestors = {f.split('.')[0] for f in os.listdir(group_dir) }
        self.group_dir = group_dir

    def get_lineage(self,taxid):
        taxid = int(taxid)
        line = [taxid]
        while taxid != 1:
            taxid = self.df[self.df.iloc[:,0]==taxid].iloc[0,1]
            line = [taxid] + line
        return line

    def common_ancestor(self,taxid_1,taxid_2):
        taxid_1 = int(taxid_1)
        taxid_2 = int(taxid_2)

        use_taxid_1 = __TAXID_CONVERTER__.get(taxid_1,taxid_1)
        use_taxid_2 = __TAXID_CONVERTER__.get(taxid_2,taxid_2)

        l_1 = self.get_lineage(use_taxid_1)
        l_2 = self.get_lineage(use_taxid_2)
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
    

    def check_ortholog_group(self,taxid_1,taxid_2,ancestor):

        group_file = f'{self.group_dir}/{ancestor}.tsv.gz'

        if not os.path.exists(group_file):
            return False

        with gzip.open(group_file,'rt') as f:

            for line in f:
                line = line.strip().split('\t')
                species_list = line[-2].split(',')

                if str(taxid_1) in species_list and str(taxid_2) in species_list:
                    return True
        
        return False



def infer_common_ancestor(self,seed_species,ncbi_taxonomy_file,eggnog_group_main_folder):
    lineage = Lineage(ncbi_taxonomy_file,group_dir=eggnog_group_main_folder)

    src_tgt = list(combinations(seed_species,2))
    # sort the pairs 
    src_tgt = [sorted(pair) for pair in src_tgt]
    if self.jobs > 1:
        use_jobs = int(self.jobs/2) ## for memory safaty
    with Pool(use_jobs) as p:
        common_ancestors = p.starmap(lineage.common_ancestor,src_tgt)
    
    return common_ancestors

def get_proteins_from_members(tax, proteins):
    pattern = re.compile(rf'{tax}\.[^,]*')
    return pattern.findall(proteins)

def parse_single_og_file(src,tgt,
                         ancestor,
                         eggnog_group_file_main_folder,
                         alpha=1,og_threshold=0.1,
                         save_to:str=None,
                         src_set:Set=None,
                         tgt_set:Set=None):

    group_file = f'{eggnog_group_file_main_folder}/{ancestor}.tsv.gz'

    src_set_index = {p:i for i,p in enumerate(src_set)}
    tgt_set_index = {p:i for i,p in enumerate(tgt_set)}

    results = set() ## store the results for a single species pair

    with gzip.open(group_file, 'rt') as f:
        for line in f: ## each line is an ortholog group

            line = line.strip().split('\t')
            _, species_list, orthologs = line[1], line[-2].split(','), line[-1]

            if str(src) in species_list and str(tgt) in species_list:
                source_proteins = get_proteins_from_members(src, orthologs)
                target_proteins = get_proteins_from_members(tgt, orthologs)

                ## use a filter to remove proteins that are not in the embeddings, if both src_set and tgt_set are provided
                if src_set is not None and tgt_set is not None:
                    source_proteins = list(filter(lambda x: x in src_set, source_proteins))
                    target_proteins = list(filter(lambda x: x in tgt_set, target_proteins))

                ## get the frequency inverse of each protein in the ortholog group
                source_protein_freq_inv = {p: 1/len(source_proteins) for p in source_proteins}
                target_protein_freq_inv = {p: 1/len(target_proteins) for p in target_proteins}

                ## get the product of the proteins with their frequency inverse product
                  
                results.update({'\t'.join([str(src_set_index[p1]), str(tgt_set_index[p2]),  
                                            str((source_protein_freq_inv[p1]*target_protein_freq_inv[p2])**alpha)])
                            for p1, p2 in product(source_proteins, target_proteins)
                            if (source_protein_freq_inv[p1]*target_protein_freq_inv[p2])**alpha >= og_threshold})
    
    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        ## save the results to main_save_folder
        with open(f'{save_to}/{src}_{tgt}.tsv','w') as f:
            f.write('\n'.join(results))
    
    return results ## return the results for a single species pair