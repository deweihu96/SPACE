import faiss
import pandas as pd
from space.tools.data import H5pyData
import numpy as np
import csv
from typing import Tuple
import sys
from sklearn.metrics import auc
import itertools

class BaseRoc:
    """Base class for ROC plot."""

    def __init__(self):
        pass

    
    def load_protein_cluster(self) -> dict:
        """
        Load protein cluster from benchmark dataset.
        """
        raise NotImplementedError("load_protein_cluster() is not implemented.")

    def get_tp_fp(self,sorted_obj,
                  protein_cluster,
                  max_fp=-1,
                  weight_threshold=-1,
                  debug=False) -> Tuple[list, list]:
        """
            Return true positive and false positive of a sorted object.
        """
        try:
            iter(sorted_obj)
            link = sorted_obj
        except TypeError:
            raise TypeError("sorted_obj must be either a list, numpy array or an iterable object.")
            
        tp_cumu = 0
        fp_cumu = 0
        tp_cumu_list = []
        fp_cumu_list = []

        if max_fp == -1:
            max_fp = len(link)

        protein_cluster_keys = set(protein_cluster.keys())

        for row in link:
            if row[2] < weight_threshold*1000:
                if debug:
                    import pdb; pdb.set_trace()
                break
            if row[0] in protein_cluster_keys and row[1] in protein_cluster_keys:
                intersection = protein_cluster[row[0]].intersection(protein_cluster[row[1]])

                if intersection:
                    tp_cumu += 1
                else:
                    fp_cumu += 1
                    if fp_cumu > max_fp:
                        break

                tp_cumu_list.append(tp_cumu)
                fp_cumu_list.append(fp_cumu)
                if fp_cumu > max_fp:
                    break
        return tp_cumu_list, fp_cumu_list


class KEGGRoc(BaseRoc):
    """Class for plotting ROC curve for KEGG."""

    def __init__(self):
        pass

    def load_protein_cluster(self, benchmark_dataset,taxid) -> dict:

        prot_family = {}

        with open(benchmark_dataset, 'r') as f:

            reader = csv.reader(f, delimiter='\t')

            for row in reader:
                if int(row[0]) == int(taxid):

                    if row[1] == 'pfa05144' and int(taxid) == 36329:
                        continue
                    elif row[1] in ['tbr05143','tbr00230'] and int(taxid) == 185431:
                        continue

                    for p in row[-1].split(' '):
                        p = str(taxid)+'.'+p

                        if p not in prot_family:
                            prot_family[p] = set([row[1]])
                        else:
                            prot_family[p].add(row[1])
                else:
                    pass
        self.prot_family = prot_family

        return prot_family
    
def normalize_vectors(vectors):
    """Normalize the vectors to unit length."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-9)

def get_embedding_rank_cos(embed:str|pd.DataFrame,proteins):
    """
        Generate tp fp for node2vec embeddings, with cosine distance
    """

    if isinstance(embed,str):
        _ , data = H5pyData.read(embed,16)
        data = data[proteins]

    elif isinstance(embed,pd.DataFrame):
        data = embed.values[proteins]
    else:
        raise ValueError('embed must be either str or pd.DataFrame')

    data = normalize_vectors(data)

    N, D = data.shape  # Number of samples and dimension

    # Step 1: Load data into FAISS
    index = faiss.IndexFlatIP(D)  # Create a flat (brute force) index
    index.add(data)  # Add vectors to the index

    # Step 2: Compute pairwise distances
    D, I = index.search(data, N + 1)  # Search all vectors (including itself)

    num_pairs = (N * (N - 1)) // 2
    results = np.empty((num_pairs, 3), dtype=object)

    count = 0

    for i in range(N):

        # Get indices where the condition is met
        valid_indices = I[i] > i

        # Number of valid pairs for this row
        num_valid = np.sum(valid_indices)

        # Store the results
        results[count:count + num_valid, 0] = proteins[i]
        results[count:count + num_valid, 1] = proteins[I[i, valid_indices]]
        results[count:count + num_valid, 2] = D[i, valid_indices]

        count += num_valid
    
    results = list(results)

    results.sort(key=lambda x: x[2],reverse=True)  # Sort by cosine distance, descending

    # cum_tp, cum_fp = keggroc.get_tp_fp(results,kegg_clusters,max_fp)

    return results

def get_auc(cumu_tp,cumu_fp):
    
    # Normalize the cumulative counts to get rates
    max_fp = cumu_fp[-1]  # Maximum FP count
    max_tp = cumu_tp[-1]  # Maximum TP count
    
    fp_rates = np.array(cumu_fp) / max_fp
    tp_rates = np.array(cumu_tp) / max_tp
    
    # Ensure the data is sorted by the normalized FP rates
    sorted_indices = np.argsort(fp_rates)
    fp_rates_sorted = fp_rates[sorted_indices]
    tp_rates_sorted = tp_rates[sorted_indices]
    
    # Calculate AUC using the trapezoidal rule
    auc = np.trapz(tp_rates_sorted, fp_rates_sorted)
    
    # print(f"Area Under the Curve (AUC): {auc}")

    return auc

def calculate_auc_with_artificial_cap(cumulative_tp, cumulative_fp, max_tp,max_fp):
    """
    Calculate AUC by setting an artificial maximum number of false positives.
    
    Parameters:
    - cumulative_tp: list of cumulative true positives at each threshold
    - cumulative_fp: list of cumulative false positives at each threshold  
    - max_fp_cap: artificial maximum false positives (default: 100000)
    
    Returns:
    - auc_score: Area under the ROC curve
    - fpr: False positive rates
    - tpr: True positive rates
    """
    
    # Get the maximum TP from your data (total positives found)
    # max_tp = max(cumulative_tp)
    
    # Use the artificial cap as total negatives
    # total_negatives = max_fp_cap
    # total_positives = max_tp  # This assumes you found all positives in your ranking
    
    # Convert to rates
    tpr = [tp / max_tp for tp in cumulative_tp]
    fpr = [fp / max_fp for fp in cumulative_fp]
    
    # Add (0,0) point at the beginning
    tpr = [0] + tpr
    fpr = [0] + fpr
    
    # Calculate AUC using trapezoidal rule
    auc_score = auc(fpr, tpr)
    
    return auc_score, fpr, tpr


def get_max_tp_fp(kegg_proteins,kegg_clusters):
    """
    Get the maximum number of false positives for the given proteins and clusters.
    """
    max_tp, max_fp = 0,0
    for p1,p2 in itertools.combinations(kegg_proteins, 2):
        if p1 in kegg_clusters and p2 in kegg_clusters:
            intersection = kegg_clusters[p1].intersection(kegg_clusters[p2])
            if intersection:
                max_tp += 1
            else:
                max_fp += 1
    return max_tp, max_fp

def run_single_species(taxid,node2vec_dir,t5_dir,aligned_dir,kegg_benchmarking_file,percent_threshold = 0.001):
    # print(f'Running {taxid}')
    node2vec_path = f'{node2vec_dir}/{taxid}.h5'

    t5_path = f'{t5_dir}/{taxid}.h5'

    aligned_path = f'{aligned_dir}/{taxid}.h5'

    node2ind = {k:v for v,k in enumerate(H5pyData.read(node2vec_path,16)[0])}
    
    keggroc = KEGGRoc()
    
    kegg_clusters = keggroc.load_protein_cluster(kegg_benchmarking_file,taxid)
                                                
    ## replace the protein names with the indices
    kegg_clusters = {node2ind[k]:v for k,v in kegg_clusters.items() if k in node2ind.keys() }
    
    kegg_proteins = np.array(list(kegg_clusters.keys()))
    
    # print('Running node2vec rank')
    node2vec_rank = get_embedding_rank_cos(node2vec_path,kegg_proteins)


    # max_fp = 100000
    # max_fp = get_max_fp(kegg_proteins,kegg_clusters)
    max_tp, max_fp = get_max_tp_fp(kegg_proteins,kegg_clusters)
    # use only the top 0.1% of the maximum false positives
    use_max_fp = int(max_fp * percent_threshold) # to reduce the calculation
    
    node2vec_tp, node2vec_fp = keggroc.get_tp_fp(node2vec_rank,kegg_clusters,use_max_fp)

    aligned2ind = {k:v for v,k in enumerate(H5pyData.read(aligned_path,16)[0])}
    kegg_clusters = keggroc.load_protein_cluster(kegg_benchmarking_file,taxid)
    kegg_clusters = {aligned2ind[k]:v for k,v in kegg_clusters.items() if k in aligned2ind.keys() }
    kegg_proteins = np.array(list(kegg_clusters.keys()))
    # print('Running aligned rank')
    aligned_rank = get_embedding_rank_cos(aligned_path,kegg_proteins,)
    aligned_tp, aligned_fp = keggroc.get_tp_fp(aligned_rank,kegg_clusters,use_max_fp,debug=False)

    t52ind = {k:v for v,k in enumerate(H5pyData.read(t5_path,16)[0])}
    kegg_clusters = keggroc.load_protein_cluster(kegg_benchmarking_file,taxid)
    kegg_clusters = {t52ind[k]:v for k,v in kegg_clusters.items() if k in t52ind.keys() }
    kegg_proteins = np.array(list(kegg_clusters.keys()))
    # print('Running t5 rank')
    seq_rank = get_embedding_rank_cos(t5_path,kegg_proteins)
    t5_tp,t5_fp = keggroc.get_tp_fp(seq_rank,kegg_clusters,use_max_fp)


    return node2vec_tp, node2vec_fp, aligned_tp, aligned_fp, t5_tp, t5_fp, max_tp, max_fp


if __name__ == "__main__":

    species = int(sys.argv[1])

    node2vec_embed = './data/node2vec'
    aligned_embed = './data/functional_emb'

    t5_embed = './data/t5_emb'

    benchmark = './kegg_benchmarking.CONN_maps_in.v12.tsv'
    
    
    node2vec_tp, node2vec_fp, aligned_tp, aligned_fp, \
    t5_tp, t5_fp, max_tp, max_fp = run_single_species(species, 
                                                      node2vec_embed,
                                                      t5_embed,
                                                      aligned_embed,
                                                      benchmark)
    
    node2vec_auc,fpr, tpr = calculate_auc_with_artificial_cap(node2vec_tp,node2vec_fp,max_fp=max_fp,max_tp=max_tp)
    aligned_auc,fpr, tpr = calculate_auc_with_artificial_cap(aligned_tp,aligned_fp,max_fp=max_fp,max_tp=max_tp)
    t5_auc,fpr, tpr = calculate_auc_with_artificial_cap(t5_tp,t5_fp,max_fp=max_fp,max_tp=max_tp)


    print(f"{species}\t{node2vec_auc}\t{aligned_auc}\t{t5_auc}\t{max_tp}\t{max_fp}")

