from space.tools.data import H5pyData
import pandas as pd
import os
from loguru import logger
from multiprocessing import Pool
import numpy as np
from sklearn.linear_model import LogisticRegression
from cafaeval.evaluation import cafa_eval
from sklearn import metrics

def load_data(idmapping,dataset,train):

    idmapping = pd.read_csv(idmapping, sep='\t', )

    labels = pd.read_csv(dataset, sep='\t', header=None)

    labels = labels[labels.iloc[:, 0].isin(idmapping['From'])]

    labels =labels.merge(idmapping, left_on=0, right_on='From', how='inner')

    labels = labels.drop(columns=[3, 'From'])

    labels.columns = ['uniprot', 'label', 'aspect', 'protein']

    labels = labels.drop_duplicates().dropna()

    if train:
        labels = labels.groupby('label').filter(lambda x: len(x) >=10)

    return labels

def load_embeddings_for_species(args):
    s, directory, ids_set = args
    file = f'{directory}/{s}.h5'
    
    if not os.path.exists(file):
        logger.error(f'{file} does not exist')
        return {}

    species_proteins, species_embeddings = H5pyData.read(file, precision=16)
    species_prot2idx = {p: i for i, p in enumerate(species_proteins)}

    # Filter species_proteins to include only those present in cv_ids
    return {p:species_embeddings[species_prot2idx[p]] for p in ids_set.intersection(species_prot2idx)}

def load_embeddings_for_species_parallel(directory,protein_ids_set,species,n_jobs):
    pool_args = [(s, directory, protein_ids_set) for s in species]
    with Pool(n_jobs) as pool:
        results = pool.map(load_embeddings_for_species, pool_args)
    
    proteins = list()
    embeddings = list()

    for r in results:
        for p, e in r.items():
            proteins.append(p)
            embeddings.append(e)
    
    return np.array(proteins), np.array(embeddings)


def prepare_embeddings(train_idmapping, train_dataset, test_idmapping, 
                       test_dataset, aligned_dir, t5_dir, n_jobs):

    train_labels = load_data(train_idmapping, train_dataset, True)

    test_labels = load_data(test_idmapping, test_dataset, False)

    proteins = set(train_labels['protein']).union(set(test_labels['protein']))

    species = list(set(map(lambda x: x.split('.')[0], proteins)))

    aligned_proteins, aligned_embeddings = load_embeddings_for_species_parallel(aligned_dir, 
                                                                                proteins, 
                                                                                species, 
                                                                                n_jobs)
    
    seq_proteins, seq_embeddings = load_embeddings_for_species_parallel(t5_dir, 
                                                                        proteins, 
                                                                        species, 
                                                                        n_jobs)
    # remove the rows with missing embeddings
    train_labels = train_labels[train_labels['protein'].isin(aligned_proteins)]
    test_labels = test_labels[test_labels['protein'].isin(aligned_proteins)]
    
    return train_labels, test_labels, \
            dict(zip(aligned_proteins, aligned_embeddings)), \
            dict(zip(seq_proteins, seq_embeddings))

def predict_single_label(train_labels,test_labels,label,aspect,seq_embeddings,aligned_embeddings,):

    train = train_labels[(train_labels['aspect']==aspect)]

    train_idx_string2uniprot = train[['protein','uniprot']].drop_duplicates()
    train_idx_string2uniprot = dict(zip(train_idx_string2uniprot['protein'],train_idx_string2uniprot['uniprot']))

    train_pos = train[train['label']==label]['protein'].unique()

    train_neg = train[~train['uniprot'].isin(train_pos)]['protein'].unique()

    Y_train = np.array([1]*len(train_pos) + [0]*len(train_neg))

    ## prepare the embeddings array
    X_train_seq = np.array([seq_embeddings[protein] for protein in train_pos] + [seq_embeddings[protein] for protein in train_neg])
    X_train_aligned = np.array([aligned_embeddings[protein] for protein in train_pos] + [aligned_embeddings[protein] for protein in train_neg])


    ## test set
    test = test_labels[test_labels['aspect']==aspect]
    test_idx_string2uniprot = test[['protein','uniprot']].drop_duplicates()
    test_idx_string2uniprot = dict(zip(test_idx_string2uniprot['protein'],test_idx_string2uniprot['uniprot']))

    test_proteins = test['protein'].unique()
    X_test_seq = np.array([seq_embeddings[protein] for protein in test_proteins])
    X_test_aligned = np.array([aligned_embeddings[protein] for protein in test_proteins])

    ## 1. seq
    clf = LogisticRegression(max_iter=1000).fit(X_train_seq, Y_train)
    y_pred_seq = clf.predict_proba(X_test_seq)[:,1]

    ## 2. aligned
    clf = LogisticRegression(max_iter=1000).fit(X_train_aligned, Y_train)
    y_pred_aligned = clf.predict_proba(X_test_aligned)[:,1]

    ## 3. seq concatenated with aligned
    clf = LogisticRegression(max_iter=1000).fit(np.concatenate([X_train_seq,X_train_aligned],axis=1), Y_train)
    y_pred_seq_concat_aligned = clf.predict_proba(np.concatenate([X_test_seq,X_test_aligned],axis=1))[:,1]  

    df_seq = pd.DataFrame(list(zip([test_idx_string2uniprot[protein] for protein in test_proteins], [label]*len(test_proteins), y_pred_seq)), columns=['uniprot','label','prediction'])
    df_aligned = pd.DataFrame(list(zip([test_idx_string2uniprot[protein] for protein in test_proteins], [label]*len(test_proteins), y_pred_aligned)), columns=['uniprot','label','prediction'])
    df_seq_concat_aligned = pd.DataFrame(list(zip([test_idx_string2uniprot[protein] for protein in test_proteins], [label]*len(test_proteins), y_pred_seq_concat_aligned)), columns=['uniprot','label','prediction'])

    df_seq = df_seq[df_seq['prediction'] > 0.01]
    df_aligned = df_aligned[df_aligned['prediction'] > 0.01]
    df_seq_concat_aligned = df_seq_concat_aligned[df_seq_concat_aligned['prediction'] > 0.01]

    return df_seq, df_aligned, df_seq_concat_aligned

def eval_single_modal(ontology, prediction_dir, ground_truth,save_name,n_jobs):
    res = cafa_eval(ontology, prediction_dir, ground_truth, n_cpu=n_jobs, th_step=0.001)

    res[0].to_csv(f'{save_name}_metrics.csv', index=False)

    fmax_row = res[0].sort_values(by='f',ascending=False).head(1)
    fmax,s = float(fmax_row['f'].values[0]), float(fmax_row['s'].values[0])
    pr,rc = fmax_row['pr'].values[0], fmax_row['rc'].values[0]
    
    auprc = metrics.auc(res[0]['rc'], res[0]['pr'])

    record = [fmax, s, auprc, pr, rc]
    # round
    record = [round(x, 3) if isinstance(x, float) else x for x in record]

    return record





if __name__ == "__main__":

    train_idmapping = 'data/benchmarks/netgo/train_idmapping_euk.tsv'

    train_dataset = 'data/benchmarks/netgo/benchmark/train.txt'

    test_idmapping = 'data/benchmarks/netgo/test_idmapping_euk.tsv'

    test_dataset = 'data/benchmarks/netgo/benchmark/test.txt'

    aligned_dir = 'data/full_functional'

    t5_dir = 'data/t5'

    save_dir = 'results/func_pred'

    ontology = 'data/benchmarks/netgo/benchmark/gene_ontology_edit.obo.2017-11-01'

    ground_truth_dir = 'data/benchmarks/netgo'

    n_jobs = 16

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger.info('Preparing embeddings')
    train_labels, test_labels, \
    seq_embeddings, aligned_embeddings = prepare_embeddings(train_idmapping,train_dataset,
                                                        test_idmapping,test_dataset,
                                                        aligned_dir,t5_dir,n_jobs)
    scores = list()
    for aspect in ['cc', 'bp', 'mf']:
        logger.info(f'Predicting {aspect}')
        aspect_labels = train_labels[train_labels['aspect'] == aspect]['label'].unique()
        logger.info(f'Aspect {aspect} has {len(aspect_labels)} labels')
        seq_predictions = []
        aligned_predictions = []
        seq_concat_aligned_predictions = []

        with Pool(n_jobs) as pool:
            results = pool.starmap(predict_single_label, [(train_labels,test_labels,
                                                           label,aspect,seq_embeddings,
                                                           aligned_embeddings) for label in aspect_labels[:20]])
        
        for df_seq, df_aligned, df_seq_concat_aligned in results:
            seq_predictions.append(df_seq)
            aligned_predictions.append(df_aligned)
            seq_concat_aligned_predictions.append(df_seq_concat_aligned)
        
        os.system(f'mkdir -p {save_dir}/{aspect}_seq_pred')
        os.system(f'mkdir -p {save_dir}/{aspect}_aligned_pred')
        os.system(f'mkdir -p {save_dir}/{aspect}_space_pred')

        pd.concat(seq_predictions).to_csv(f'{save_dir}/{aspect}_seq_pred/{aspect}_seq_pred.tsv', header=False,index=False,sep='\t')
        pd.concat(aligned_predictions).to_csv(f'{save_dir}/{aspect}_aligned_pred/{aspect}_aligned_pred.tsv', index=False,header=False,sep='\t')
        pd.concat(seq_concat_aligned_predictions).to_csv(f'{save_dir}/{aspect}_space_pred/{aspect}_space_pred.tsv', index=False,header=False,sep='\t')

        ## evaluate the predictions with cafa-eval
        group_truth = f'{ground_truth_dir}/test_{aspect}_ground_truth.txt'

        # eval
        logger.info('Evaluating predictions')
        seq_record = eval_single_modal(ontology, f'{save_dir}/{aspect}_seq_pred', group_truth, f'{save_dir}/{aspect}_seq_eval', n_jobs)
        aligned_record = eval_single_modal(ontology, f'{save_dir}/{aspect}_aligned_pred', group_truth, f'{save_dir}/{aspect}_aligned_eval', n_jobs)
        seq_concat_aligned_record = eval_single_modal(ontology, f'{save_dir}/{aspect}_space_pred', group_truth, f'{save_dir}/{aspect}_space_eval', n_jobs)

        seq_record = [aspect,'seq'] + seq_record
        aligned_record = [aspect,'aligned'] + aligned_record
        seq_concat_aligned_record = [aspect,'space'] + seq_concat_aligned_record

        scores.append(seq_record)
        scores.append(aligned_record)
        scores.append(seq_concat_aligned_record)

    scores = pd.DataFrame(scores, columns=['aspect','method', 'fmax', 's', 'auprc', 'pr', 'rc'])
    scores.to_csv(f'{save_dir}/scores.csv', index=False)

    logger.info('DONE.')