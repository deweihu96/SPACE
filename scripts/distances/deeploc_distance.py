import os
import sys
sys.path.append('./scripts')
from subloc import load_cv_set,filter_and_load_proteins_embeddings
import random
import pandas as pd
import numpy as np
from scipy import stats
import itertools
from tqdm import tqdm

def format_diff(difference):
    """Format the difference for output."""
    if difference > 0:
        return f'+{difference:.2f}'
    else:
        return f'{difference:.2f}'

def normalize_vectors(vectors):
    """Normalize the vectors to unit length."""
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norm, 1e-9)


def mutual_distances(embed_loc, embed_other_loc, number_of_samples):
    
    used_proteins = set()
    sampled_pairs = []
    
    possible_pairs = list(itertools.product(range(len(embed_loc)), range(len(embed_other_loc))))
    
    random.shuffle(possible_pairs)
    for i, j in possible_pairs:
        if i not in used_proteins or j not in used_proteins:
            used_proteins.add(i)
            used_proteins.add(j)
            sampled_pairs.append((i, j))
        if len(sampled_pairs) > number_of_samples:
            break
    
    sampled_indices = np.array(sampled_pairs).T
    src_emb = embed_loc[sampled_indices[0]]
    tgt_emb = embed_other_loc[sampled_indices[1]]
    cosine_similarities = np.sum(src_emb * tgt_emb, axis=1) / (np.linalg.norm(src_emb, axis=1) * np.linalg.norm(tgt_emb, axis=1))

    return cosine_similarities

def inner_distances(embeddings, ):
    """Calculate distances for a given set of embeddings."""
    possible_pairs = list(itertools.combinations(range(len(embeddings)), 2))
    # shuffle the pairs to ensure randomness
    random.shuffle(possible_pairs)
    
    used_proteins = set()
    sampled_pairs = []
    
    for i, j in possible_pairs:
        if i not in used_proteins or j not in used_proteins:
            used_proteins.add(i)
            used_proteins.add(j)
            sampled_pairs.append((i, j))
    
    sampled_indices = np.array(sampled_pairs).T
    
    src_emb = embeddings[sampled_indices[0]]
    tgt_emb = embeddings[sampled_indices[1]]
    
    cosine_similarities = np.sum(src_emb * tgt_emb, axis=1) / (np.linalg.norm(src_emb, axis=1) * np.linalg.norm(tgt_emb, axis=1))  
    
    return cosine_similarities
  
def single_location_p_value(location,embed,label_df):
    embed_loc = embed[label_df[label_df[location] == 1].index]
    
    inner_cosine = inner_distances(embed_loc)
    
    embed_other_loc = embed[label_df[label_df[location] == 0].index]
    
    mutual_cosine = mutual_distances(embed_loc, embed_other_loc, number_of_samples=len(inner_cosine))
    mutual_q1 = round(np.quantile(mutual_cosine, 0.25),2)
    mutual_q3 = round(np.quantile(mutual_cosine, 0.75),2)
    
    
    mutual_median = round(float(np.median(mutual_cosine)),2)
    
    inner_q1 = round(np.quantile(inner_cosine, 0.25),2)
    inner_q3 = round(np.quantile(inner_cosine, 0.75),2)
    inner_median = round(float(np.median(inner_cosine)),2)
        
    stat_cosine, p_value_cosine = stats.mannwhitneyu(mutual_cosine, inner_cosine, alternative='two-sided')
    
    effect_cosine = round((1 - (2*stat_cosine) / (len(mutual_cosine) * len(inner_cosine))),2)
    
    difference = format_diff(float(inner_median - mutual_median))
    
    # use scientific notation for p-value
    p_value_cosine = f'{p_value_cosine:.2e}'
    
    # return p_value_euclidean, p_value_cosine, stat_mutual, stat_inner
    results = {
        'Location': location,
        'N': str(len(mutual_cosine)),
        'Intra-Loc (IQR)': f'{inner_median} ({inner_q1} - {inner_q3})',
        'Inter-Loc (IQR)': f'{mutual_median} ({mutual_q1} - {mutual_q3})',
        'Difference': difference,
        'p-value': p_value_cosine,
        'Effect Size': f'{round(effect_cosine,2)}',
    }    
    return results

def main(cv_set, cv_id_mapping,aligned_dir,
         jobs=1):
    
    cv_ids, cv_labels, cv_label_headers, cv_partitions, cv_species = load_cv_set(cv_set, cv_id_mapping)
    
    cv_ids_aligned, cv_labels_aligned, cv_partitions_aligned, \
    aligned_proteins, aligned_embeddings = filter_and_load_proteins_embeddings(
        cv_ids,
        cv_labels,
        cv_partitions,
        cv_species,
        aligned_dir,
        n_jobs=jobs
    )
    
    df_labels = pd.DataFrame(cv_labels_aligned, columns=cv_label_headers)
    df_labels['id'] = cv_ids_aligned

    # use for loop
    results = []
    # loc = 'Peroxisome'
    # results = [single_location_p_value(loc, aligned_embeddings, df_labels)]
    for loc in tqdm(cv_label_headers):
        if loc != 'id':
            result = single_location_p_value(loc, aligned_embeddings, df_labels)
            results.append(result)
    #         break
    
    df = pd.DataFrame(results)
    # put the location as the first column
    df = df[['Location',] + [col for col in df.columns if col != 'Location']]
    
    df.to_csv('./results/deeploc_location_distances.csv', index=False)



if __name__ == '__main__':
    
    np.random.seed(42)
    random.seed(42)
    
    cv_set = './data/benchmarks/deeploc/Swissprot_Train_Validation_dataset.csv'
    cv_id_mapping = './data/benchmarks/deeploc/cv_idmapping.tsv'
    aligned_dir = './data/functional_emb'
    
    main(cv_set, cv_id_mapping, aligned_dir, jobs=8)
    
    