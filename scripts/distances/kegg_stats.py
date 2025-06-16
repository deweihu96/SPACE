import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


def sci_round(x, digits=2):
    """Round a number to scientific notation with specified digits."""
    if x == 0:
        return '0'
    else:
        return f"{x:.{digits}e}"
    
def format_diff(diff:str):
    if diff.startswith('-'):
        return diff
    else:
        return f'+{diff}'
    


def get_stats_report(df,):

    col_a,col_b = df.columns[1], df.columns[2]
    
    if col_b == 'aligned_auc':
        # swap the columns
        col_a, col_b = col_b, col_a
    
    
    
    sample_size = df.shape[0]
    
    # map to the new name
    name_map = {
        'node2vec_auc': 'node2vec',
        'aligned_auc': 'Aligned',
        't5_auc': 'ProtT5',
    }
    
    
    statistic, p_value = stats.wilcoxon(df[col_a], df[col_b],alternative='two-sided')

    differences = df[col_b] - df[col_a]
    n_pos = np.sum(differences > 0)
    n_neg = np.sum(differences < 0)
    effect_size = abs((n_pos - n_neg) / (n_pos + n_neg))
    
    ratios = df[col_a] / df[col_b]
    ratio_q1, ratio_q3 = np.quantile(ratios, [0.25, 0.75])
    ratio_median = np.median(ratios)
    
    ratio_q1 = round(ratio_q1, 2)
    ratio_q3 = round(ratio_q3, 2)
    ratio_median = round(ratio_median, 2)

    p_value = sci_round(p_value, 2)
    effect_size = round(effect_size, 2)
    
    # use a dataframe to store the results
    results = pd.DataFrame({
        'Method A': [name_map.get(col_a, col_a)],
        'Method B': [name_map.get(col_b, col_b)],
        'Sample Size': [sample_size],
        'Ratio A/B Median (IQR)': [f'{ratio_median} ({ratio_q1} - {ratio_q3})'],
        'p-value': [f'{p_value}'],
        'Effect Size': [f'{effect_size}'],
    })
    return results

if __name__ == "__main__":
    
    df = pd.read_csv('./results/kegg_scores.tsv', sep='\t')
    seed_list = open('./data/seeds.txt').read().splitlines()
    seed_list = [int(x) for x in seed_list]
    df['seed'] = False
    for seed in seed_list:
        df.loc[df['species'] == seed, 'seed'] = True
    df_euk_groups = pd.read_csv('./data/euks_groups.tsv', 
                                sep='\t')
    df_euk_groups = df_euk_groups.iloc[:,:-1]
    df_euk_groups.columns = ['species', 'kingdom',] 
    df = df.merge(df_euk_groups, on='species', how='left')
    ## if the kingdom is `other` set it to `protists`
    df.loc[df['kingdom'] == 'other', 'kingdom'] = 'protists' 

    df_all_results = list()
    for kingdom in df['kingdom'].unique():
        df_subset = df[(df['kingdom'] == kingdom)]
        if len(df_subset) > 0:
            for method1,method2 in itertools.combinations(['node2vec_auc', 'aligned_auc','t5_auc'],2):
                df_subset_method = df_subset[['species',method1, method2]] 
                df_result = get_stats_report(df_subset_method)
                # uppercase the first letter of the kingdom
                df_result['Kingdom'] = kingdom.capitalize()
                df_all_results.append(df_result)
    
    df_all_results = pd.concat(df_all_results, ignore_index=True)
    
    ## change the order of the columns
    df_all_results = df_all_results[['Method A', 'Method B','Kingdom',  'Sample Size',
                                     'Ratio A/B Median (IQR)',
                                     'p-value', 'Effect Size']]
    # order by Method A, then Method B
    df_all_results = df_all_results.sort_values(by=['Method A', 'Method B'])
    
    # print(df_all_results)
    df_all_results.to_csv('./results/kegg_stats.csv', 
                          sep=',', index=False)