import gzip
import pandas as pd
from multiprocessing import Pool
from space.tools.data import H5pyData
from tqdm import tqdm
import itertools
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
import seaborn as sns

def read_single_embeddings(taxid):
    proteins, emb = H5pyData.read(f'./data/aligned/{taxid}.h5', 16)
    return taxid, (proteins, emb)

def load_embeddings():
    # load all the embeddings
    aligned_embeddings = dict()

    with open('./data/euks.txt') as f:
        taxids = f.read().strip().split('\n')
        
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(read_single_embeddings, taxids), total=len(taxids), desc="Loading embeddings"))
    
    for taxid, (proteins, emb) in results:
        aligned_embeddings[taxid] = (proteins, emb)
    
    protein_2_index = dict()
    for taxid, (proteins, emb) in aligned_embeddings.items():
        protein_2_index[taxid] = dict(zip(proteins, range(len(proteins))))
        
    return aligned_embeddings, protein_2_index

def sample_proteins(protein_list, probability=0.01):
    """Fastest method using NumPy vectorized operations"""
    protein_array = np.array(protein_list)
    mask = np.random.random(len(protein_array)) < probability
    return protein_array[mask].tolist()

def kick_out_pairs(protein_pairs):
    # randomize the paris
    random.shuffle(protein_pairs)
    
    used_proteins = set()
    filtered_pairs = []
    
    for src, tgt in protein_pairs:
        if src not in used_proteins or tgt not in used_proteins:
            filtered_pairs.append((src, tgt))
            used_proteins.add(src)
            used_proteins.add(tgt)
    return filtered_pairs

def process_single_level(df_ancestor, level, all_proteins, aligned_embeddings):
 
    level = int(level)
    species_taxids = df_ancestor[df_ancestor['ancestor'] == level][['taxid_1', 'taxid_2']]
    
    # Convert to efficient set of integer tuples
    species_pairs = set()
    for taxid_1, taxid_2 in species_taxids.values:
        t1, t2 = int(taxid_1), int(taxid_2)
        species_pairs.add((min(t1, t2), max(t1, t2)))
    
    protein_pairs = list()
    
    with gzip.open(f'./data/eggnog/{level}.tsv.gz', 'rt') as f:
   
        for line in f:
            line = line.strip().split('\t')
            _, species_list, orthologs = line[1], line[-2].split(','), line[-1]
            
            species_list = [int(s) for s in species_list]
            orthologs = orthologs.split(',')
            
            ortholog_per_species = dict()
            
            for og_ in orthologs:
                taxid_int = int(og_.split('.')[0])
                # make sure the taxid is in the aligned embeddings
                if str(taxid_int) not in aligned_embeddings:
                    continue
                if taxid_int not in ortholog_per_species:
                    ortholog_per_species[taxid_int] = 0
                ortholog_per_species[taxid_int] += 1
            
            # make sure we have this protein in the aligned embeddings
            valid_proteins = set(orthologs) & all_proteins
            if len(valid_proteins) < 2:
                continue
            # proteins from the same species can only show up once
            valid_proteins = [og_ for og_ in valid_proteins if ortholog_per_species.get(int(og_.split('.')[0]), 0) == 1]
            if len(valid_proteins) < 2:
                continue
            
            # protein_pairs.extend(itertools.combinations(valid_proteins, 2))
            
            # for each protein we have 0.01% chance to sample it
            # sampled_proteins = np.random.choice(valid_proteins, size=int(len(valid_proteins) * 0.0001), replace=False)
            sampled_proteins = sample_proteins(valid_proteins)
            
            protein_pairs.extend(itertools.combinations(sampled_proteins, 2))
    
        return protein_pairs
    
def get_protein_pairs_distances(protein_pairs, aligned_embeddings, protein_2_index):
    
    src_emb = list()
    tgt_emb = list()
    for src,tgt in protein_pairs:
        src_taxid = src.split('.')[0]
        tgt_taxid = tgt.split('.')[0]
        
        src_index = protein_2_index[src_taxid][src]
        tgt_index = protein_2_index[tgt_taxid][tgt]
        src_emb.append(aligned_embeddings[src_taxid][1][src_index])
        tgt_emb.append(aligned_embeddings[tgt_taxid][1][tgt_index])
    src_emb = np.array(src_emb)
    tgt_emb = np.array(tgt_emb)
    euc_distances = np.linalg.norm(src_emb - tgt_emb, axis=1)
    cos_similarities = np.sum(src_emb * tgt_emb, axis=1) / (np.linalg.norm(src_emb, axis=1) * np.linalg.norm(tgt_emb, axis=1))    
    return euc_distances, cos_similarities

def sample_negative_pairs(protein_pairs, protein_2_index):
    
    negative_pairs = []
    
    for og_1, og_2 in protein_pairs:
        taxid_1 = int(og_1.split('.')[0])
        taxid_2 = int(og_2.split('.')[0])
        
        # sample the same taxid but random index
        while True:
            sample_prot1 = random.choice(list(protein_2_index[str(taxid_1)].keys()))
            sample_prot2 = random.choice(list(protein_2_index[str(taxid_2)].keys()))
            if sample_prot1 != og_1 and sample_prot2 != og_2:
                negative_pairs.append((sample_prot1, sample_prot2))
                break
    return negative_pairs 


def get_stat_report(pos_distance, neg_distance, pos_similarity, neg_similarity):
    _, p_value = scipy.stats.wilcoxon(pos_distance, neg_distance)
    differences = pos_distance - neg_distance
    n_pos = np.sum(differences > 0)
    n_neg = np.sum(differences < 0)
    effect_size = abs(n_pos - n_neg) / len(differences)
    
    if effect_size < 0.1:
        effect_interpretation = "negligible"
    elif effect_size < 0.3:
        effect_interpretation = "small"
    elif effect_size < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    
    _, p_value = scipy.stats.wilcoxon(pos_similarity, neg_similarity)
    differences = pos_similarity - neg_similarity
    n_pos = np.sum(differences > 0)
    n_neg = np.sum(differences < 0)
    effect_size = abs(n_pos - n_neg) / len(differences)
    if effect_size < 0.1:
        effect_interpretation = "negligible"
    elif effect_size < 0.3:
        effect_interpretation = "small"
    elif effect_size < 0.5:
        effect_interpretation = "medium"
    else:
        effect_interpretation = "large"
    stat_dict_cos = {
        'pos_cos_similarity_mean': np.mean(pos_similarity),
        'pos_cos_similarity_std': np.std(pos_similarity),
        'pos_cos_min': np.min(pos_similarity),
        'pos_cos_max': np.max(pos_similarity),
        'pos_cos_median': np.median(pos_similarity),
        'neg_cos_similarity_mean': np.mean(neg_similarity),
        'neg_cos_similarity_std': np.std(neg_similarity),
        'neg_cos_min': np.min(neg_similarity),
        'neg_cos_max': np.max(neg_similarity),
        'neg_cos_median': np.median(neg_similarity),
        'cos_similarity_wilcoxon_effect_size': effect_size,
        'cos_similarity_wilcoxon_effect_interpretation': effect_interpretation,
        'cos_p_value': p_value,
    }
    return stat_dict_cos


def distances_by_group(df,all_proteins,aligned_embeddings,protein_2_index):
    
    unique_levels = df['ancestor'].unique()
    unique_levels = [int(level) for level in unique_levels if level != 'nan']

    unique_levels = sorted(unique_levels)
    
    total_pairs = list()
    for level in tqdm(unique_levels, desc="Processing levels"):
        protein_pairs = process_single_level(df, level, all_proteins, aligned_embeddings)
        total_pairs.extend(protein_pairs)
    
    total_pairs = kick_out_pairs(total_pairs)
    # negative sampling
    negative_pairs = sample_negative_pairs(total_pairs, protein_2_index)
    pos_euc_distances, pos_cos_similarities = get_protein_pairs_distances(total_pairs, aligned_embeddings, protein_2_index)
    neg_euc_distances, neg_cos_similarities = get_protein_pairs_distances(negative_pairs, aligned_embeddings, protein_2_index)
    
    report = get_stat_report(pos_euc_distances, neg_euc_distances, pos_cos_similarities, neg_cos_similarities)
    return report, (pos_euc_distances, pos_cos_similarities), (neg_euc_distances, neg_cos_similarities)


def main():
    random.seed(42)
    np.random.seed(42)
    
    report_results = list()
    distances = list()
    df_ancestor = pd.read_csv('./data/euks_ancestors.tsv',
                            sep='\t')
    seeds = open('./data/seeds.txt').read().strip().split('\n')
    seeds = [int(s) for s in seeds]
    df_ancestor['taxid_1_seed'] = False
    df_ancestor['taxid_2_seed'] = False
    for seed in seeds:
        df_ancestor.loc[df_ancestor['taxid_1'] == seed, 'taxid_1_seed'] = True
        df_ancestor.loc[df_ancestor['taxid_2'] == seed, 'taxid_2_seed'] = True

    aligned_embeddings, protein_2_index = load_embeddings()

    all_proteins = set()
    for proteins, _ in aligned_embeddings.values():
        all_proteins.update(proteins)
        
    # seed with seed
    df_seed_seed = df_ancestor[(df_ancestor['taxid_1_seed'] == True) & (df_ancestor['taxid_2_seed'] == True)]
    report, (pos_euc_distances, pos_cos_similarities), \
        (neg_euc_distances, neg_cos_similarities) = distances_by_group(df_seed_seed, 
                                                                       all_proteins,
                                                                       aligned_embeddings, 
                                                                       protein_2_index)
    report['Group'] = 'Seed with Seed'
    report_results.append(report)
    
    distances.append((pos_euc_distances, pos_cos_similarities, neg_euc_distances, neg_cos_similarities))
    
    # seed with non-seed
    df1 = df_ancestor[(df_ancestor['taxid_1_seed'] == True) & (df_ancestor['taxid_2_seed'] == False)]
    df2 = df_ancestor[(df_ancestor['taxid_1_seed'] == False) & (df_ancestor['taxid_2_seed'] == True)]
    df_seed_nonseed = pd.concat([df1, df2])
    report, (pos_euc_distances, pos_cos_similarities), \
        (neg_euc_distances, neg_cos_similarities) = distances_by_group(df_seed_nonseed, 
                                                                       all_proteins,
                                                                       aligned_embeddings, 
                                                                       protein_2_index)
    report['Group'] = 'Seed with Non-Seed'
    report_results.append(report)
    distances.append((pos_euc_distances, pos_cos_similarities, neg_euc_distances, neg_cos_similarities))
    
    # non-seed with non-seed
    df_nonseed = df_ancestor[(df_ancestor['taxid_1_seed'] == False) & (df_ancestor['taxid_2_seed'] == False)]
    report, (pos_euc_distances, pos_cos_similarities), \
        (neg_euc_distances, neg_cos_similarities) = distances_by_group(df_nonseed, 
                                                                       all_proteins,
                                                                       aligned_embeddings, 
                                                                       protein_2_index)
    report['Group'] = 'Non-Seed with Non-Seed'
    report_results.append(report)
    distances.append((pos_euc_distances, pos_cos_similarities, neg_euc_distances, neg_cos_similarities))
    
    # save the report
    report_df = pd.DataFrame(report_results)
    report_df.to_csv('./results/og_sampling_report.csv', index=False)
    
    # plot the distances
    
    colors = ['#e70148','#bbbbbb']
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=report_df, x='Group', y='Distance', hue='Type',
                showfliers=False,
                widths=0.2,
                gap=0.01,
                palette=colors)
    plt.title('Cosine Similarity Distribution')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Groups')
    # use the custom colors

    plt.grid(True)
    # plt.tight_layout()
    plt.savefig('./results/og_sampling_cosine_similarity.png', dpi=300)
    plt.show()
    