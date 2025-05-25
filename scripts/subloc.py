from space.tools.data import H5pyData
import pandas as pd
import random
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from loguru import logger
from typing import Dict, List, Tuple
from multiprocessing import Pool
import umap
import matplotlib.pyplot as plt
import argparse




def precision_recall(y_scores_flat,y_true_flat):
    thresholds = np.linspace(0, 1, 1000)

    # Calculate precision and recall for each threshold
    precisions = []
    recalls = []
    for threshold in thresholds:
        y_pred = (y_scores_flat > threshold).astype(int)
        true_positives = np.sum((y_true_flat == 1) & (y_pred == 1))
        false_positives = np.sum((y_true_flat == 0) & (y_pred == 1))
        false_negatives = np.sum((y_true_flat == 1) & (y_pred == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 1
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)

    # Convert to numpy arrays
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort the points by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    return precisions, recalls

def filter_non_existing_proteins(cv_ids,cv_labels,cv_partitions,cv_species,directory:List[str]):

    ## load the proteins from the embeddings
    proteins = set()
    for s in cv_species:
        file = f'{directory}/{s}.h5'
        if not os.path.exists(file):
            logger.error(f'{file} does not exist')
            continue
        
        s_proteins, _ = H5pyData.read(file,precision=16)

        proteins.update(s_proteins)

    proteins = list(proteins)

    ## filter the proteins that are not in the embeddings
    idx = [i for i,p in enumerate(cv_ids) if p in proteins]

    cv_ids = cv_ids[idx]
    cv_labels = cv_labels[idx]
    cv_partitions = cv_partitions[idx]

    return cv_ids, cv_labels, cv_partitions

def init_logistic_regression(random_seed):
    return MultiOutputClassifier(LogisticRegression(max_iter=1000,random_state=random_seed))

def load_cv_set(cv_set, cv_id_mapping, ):
    
    cv_set = pd.read_csv(cv_set,index_col=0)

    ## drop the sequence column
    cv_set = cv_set.drop(columns=['Sequence'])


    ## open the id mapping file
    cv_id_mapping = pd.read_csv(cv_id_mapping,sep='\t')
    cv_id_mapping.columns = ['ACC','STRING_id']


    ## merge the id mapping file with the cross validation set
    cv_set = cv_set.merge(cv_id_mapping,on='ACC').dropna()
    ## if one ACC has multiple STRING_id, keep the first one
    cv_set = cv_set.drop_duplicates(subset='ACC')

    ## extract the species from the string id
    cv_species = cv_set['STRING_id'].apply(lambda x: x.split('.')[0]).unique()

    cv_ids = cv_set['STRING_id']

    cv_label_headers = ['Cytoplasm','Nucleus','Extracellular','Cell membrane',
                        'Mitochondrion','Plastid','Endoplasmic reticulum',
                        'Lysosome/Vacuole','Golgi apparatus','Peroxisome']

    cv_labels = cv_set[cv_label_headers].values.astype(int)

    cv_partitions = cv_set['Partition'].values

    return cv_ids, cv_labels, cv_label_headers, cv_partitions, cv_species

def load_embeddings_for_species(args) -> Dict[str, np.ndarray]:
    s, directory, cv_ids_set = args
    file = f'{directory}/{s}.h5'
    species_proteins, species_embeddings = H5pyData.read(file, precision=16)
    species_prot2idx = {p: i for i, p in enumerate(species_proteins)}
    return {p: species_embeddings[species_prot2idx[p]] for p in cv_ids_set.intersection(species_prot2idx)}

def filter_and_load_proteins_embeddings(
    cv_ids: np.ndarray,
    cv_labels: np.ndarray,
    cv_partitions: np.ndarray,
    cv_species: List[str],
    directory: str,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Convert cv_ids to a set for faster lookup
    cv_ids_set = set(cv_ids)

    pool_args = [(s, directory, cv_ids_set) for s in cv_species]
    with Pool(n_jobs) as pool:
        results = pool.map(load_embeddings_for_species, pool_args)
    protein_to_embedding = dict()
    for result in results:
        protein_to_embedding.update(result)
    
    filtered_idx = [i for i, p in enumerate(cv_ids) if p in protein_to_embedding]
    cv_ids = cv_ids[filtered_idx]
    cv_labels = cv_labels[filtered_idx]
    cv_partitions = cv_partitions[filtered_idx]

    # Extract the ordered embeddings
    output_proteins = cv_ids
    output_embeddings = np.array([protein_to_embedding[p] for p in output_proteins])

    return cv_ids, cv_labels, cv_partitions, np.array(output_proteins), output_embeddings


def evaluate_model(model, X_test:np.ndarray, y_test:np.ndarray,eval_human=False):

    y_test = y_test.astype(int)
    ypred = model.predict(X_test)

    if eval_human:
        ypred = ypred[:,[0,1,3,4,6,8]]
        y_test = y_test[:,[0,1,3,4,6,8]]

    metrics = [f1_score(y_test, ypred, average='micro'),
                f1_score(y_test, ypred, average='macro'),
                accuracy_score(y_test, ypred),
                jaccard_score(y_test, ypred, average='micro')]
    
    mcc = [matthews_corrcoef(y_test[:,i], ypred[:,i]) for i in range(y_test.shape[-1])]

    return metrics, mcc


def benchmark_single_modal_on_cv(cv_partitions,embeddings,cv_labels,cv_label_headers,random_seed):
    # ## 1. run logreg with t5 embeddings
    scores = list()
    mccs = list()

    pred_scores = list()
    y_trues = list()

    for i in range(5):
        training_idx = cv_partitions != i
        val_idx = cv_partitions == i

        X_train = embeddings[training_idx]
        y_train = cv_labels[training_idx]

        X_val = embeddings[val_idx]
        y_val = cv_labels[val_idx]

        clf = init_logistic_regression(random_seed)

        clf.fit(X_train,y_train)

        score, mcc = evaluate_model(clf, X_val, y_val)

        pred_scores.append(np.array([s[:,1] for s in clf.predict_proba(X_val)]).T.flatten())
        y_trues.append(y_val.flatten())

        scores.append(score)
        mccs.append(mcc)

    y_scores_flat = np.concatenate(pred_scores)
    y_trues_flat = np.concatenate(y_trues)

    scores = pd.DataFrame(scores, columns=['f1_micro','f1_macro','accuracy','jaccard'])

    mccs = pd.DataFrame(mccs, columns=cv_label_headers)

    return scores, mccs, np.stack(precision_recall(y_scores_flat,y_trues_flat),axis=1)

def mean_std(scores:pd.DataFrame):
    means = np.round(scores.mean().values,2)
    stddevs = np.round(scores.std().values,2)
    means_std = [str(mn)+' ± '+str(sd) for mn, sd in zip(means, stddevs)]
    return means_std


def plot_umap_projection(projection,cv_labels_aligned,
                         cv_label_headers,save_name,random_seed):
    # import pdb; pdb.set_trace()
    projection = np.stack(projection,axis=1)

    df_cv_labels = pd.DataFrame(cv_labels_aligned,columns=cv_label_headers)
    df_cv_labels['num_labels'] = df_cv_labels.sum(axis=1)

    indices = df_cv_labels[df_cv_labels['num_labels'] == 1].index
    indices_labels = df_cv_labels.iloc[indices, :-1].idxmax(axis=1)

    reducer = umap.UMAP(n_components=2,n_neighbors=100,min_dist=0.1,random_state=random_seed)

    umap_embeddings = reducer.fit_transform(projection[indices])

    colors = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0", "grey"]
    locations = ['Cytoplasm','Cell membrane', 'Nucleus',  'Lysosome/Vacuole',
        'Mitochondrion',  'Plastid','Endoplasmic reticulum',
        'Golgi apparatus', 'Peroxisome', 'Extracellular',]

    loc2color = dict(zip(locations, colors))

    df_umap_emb = pd.DataFrame(umap_embeddings)
    df_umap_emb['label'] = indices_labels.values
    df_umap_emb['color'] = df_umap_emb['label'].apply(lambda x: loc2color[x])

    scatter = plt.scatter(df_umap_emb.iloc[:, 0], df_umap_emb.iloc[:, 1],
                     c=df_umap_emb['color'],
                     s=1)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=colors[i], label=label, markersize=5)
                for i, label in enumerate(locations)]

    plt.legend(handles=legend_elements,
        bbox_to_anchor=(0.95, 0.5),  # Position relative to the plot
        loc='center left',            # Anchor point of the legend
        borderaxespad=0,              # Padding between legend and axes
        bbox_transform=plt.gca().transAxes,
        frameon=False,
        fontsize=13,)

    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    return None



def benchmark_cv(cv_set, cv_id_mapping, t5_dir, aligned_dir:str,save_dir,jobs:int,random_seed:int):
    
    cv_ids, cv_labels, cv_label_headers, cv_partitions, cv_species = load_cv_set(cv_set, cv_id_mapping)

    ## filter the proteins that are not in the embeddings
        
    cv_ids_t5, cv_labels_t5, cv_partitions_t5, \
    t5_proteins, t5_embeddings = filter_and_load_proteins_embeddings(
        cv_ids,
        cv_labels,
        cv_partitions,
        cv_species,
        t5_dir,
        n_jobs=jobs,)
    
    cv_ids_aligned, cv_labels_aligned, cv_partitions_aligned, \
    aligned_proteins, aligned_embeddings = filter_and_load_proteins_embeddings(
        cv_ids,
        cv_labels,
        cv_partitions,
        cv_species,
        aligned_dir,
        n_jobs=jobs
    )
    
    scores = list()
    mccs = list()

    t5_scores, t5_mccs, t5_pr_curves = benchmark_single_modal_on_cv(cv_partitions_t5,t5_embeddings,
                                                                    cv_labels_t5,cv_label_headers,
                                                                    random_seed)
    scores.append(mean_std(t5_scores))
    mccs.append(mean_std(t5_mccs))

    aligned_scores, aligned_mccs, aligned_pr_curves = benchmark_single_modal_on_cv(cv_partitions_t5,aligned_embeddings,
                                                                                   cv_labels_t5,cv_label_headers,
                                                                                   random_seed)
    scores.append(mean_std(aligned_scores))
    mccs.append(mean_std(aligned_mccs))

    space_scores, space_mccs, space_pr_curves = benchmark_single_modal_on_cv(cv_partitions_t5,
                                                np.concatenate([t5_embeddings,aligned_embeddings],axis=-1),
                                                cv_labels_t5,cv_label_headers,
                                                random_seed)


    scores.append(mean_std(space_scores))
    mccs.append(mean_std(space_mccs))

    scores = pd.DataFrame(scores,columns=['f1_micro','f1_macro','accuracy','jaccard'],index=['t5','aligned','t5_concat_aligned'])
    mccs = pd.DataFrame(mccs,columns=cv_label_headers,index=['t5','aligned','t5_concat_aligned'])

    pr_curves = np.concatenate([t5_pr_curves,aligned_pr_curves,space_pr_curves],axis=1)
    pr_curves = pd.DataFrame(pr_curves,
                                columns=['t5_prec','t5_recall',
                                        'aligned_prec','aligned_recall',
                                        'space_prec','space_recall'])
    
    scores.to_csv(f'{save_dir}/cv_scores.csv')
    mccs.to_csv(f'{save_dir}/cv_mccs.csv')
    pr_curves.to_csv(f'{save_dir}/cv_pr_curves.csv')

    t5_clf = init_logistic_regression(random_seed)
    t5_clf.fit(t5_embeddings,cv_labels_t5)

    ## get the projection of the t5 embeddings on clf
    t5_porjection = [est_.decision_function(t5_embeddings) for est_ in t5_clf.estimators_]
    np.save(f'{save_dir}/t5_projection_cv.npy',np.stack(t5_porjection,axis=1),)

    aligned_clf = init_logistic_regression(random_seed)
    aligned_clf.fit(aligned_embeddings,cv_labels_t5)
    aligned_projection = [est_.decision_function(aligned_embeddings) for est_ in aligned_clf.estimators_]
    np.save(f'{save_dir}/aligned_projection_cv.npy',np.stack(aligned_projection,axis=1))

    space_clf = init_logistic_regression(random_seed)
    space_clf.fit(np.concatenate([t5_embeddings,aligned_embeddings],axis=-1),cv_labels_t5)
    space_projection = [est_.decision_function(np.concatenate([t5_embeddings,aligned_embeddings],axis=-1)) 
                        for est_ in space_clf.estimators_]
    np.save(f'{save_dir}/space_projection_cv.npy',np.stack(space_projection,axis=1))

    # plot the umap projection of aligned embeddings
    plot_umap_projection(aligned_projection,cv_labels_aligned,
                         cv_label_headers,f'{save_dir}/aligned_cv_clf_project_umap.png',random_seed)
    
    return {'t5':t5_clf,'aligned':aligned_clf,'space':space_clf}, cv_label_headers


def process_hpa_set(human_alias,hpa_set,save_dir):

    aliases = pd.read_csv(human_alias,compression='gzip',sep='\t')
    hpa = pd.read_csv(hpa_set,sep=',')
    idmapping = aliases[aliases['alias'].isin(hpa['sid'])][['#string_protein_id','alias']].drop_duplicates()
    hpa = pd.merge(hpa,idmapping,left_on='sid',right_on='alias',how='left').drop_duplicates().dropna()

    hpa[['#string_protein_id',
         'Cytoplasm','Nucleus','Cell membrane',
         'Mitochondrion','Endoplasmic reticulum',
         'Golgi apparatus','sid']].to_csv(f'{save_dir}/hpa_testset_mapped.csv',index=False)

    hpa_proteins = hpa['#string_protein_id'].values

    hpa_headers = hpa.columns[1:-4].values
    hpa_labels = hpa.iloc[:,1:-4].values

    use_headers = 'Cytoplasm, Nucleus, Cell membrane, Mitochondrion, Endoplasmic reticulum, Golgi apparatus'
    use_headers = use_headers.split(', ')

    ## filter the headers and labels
    hpa_labels = hpa_labels[:,[i for i,h in enumerate(hpa_headers) if h in use_headers]]
    hpa_headers = [h for h in hpa_headers if h in use_headers]

    return hpa_proteins, hpa_labels, hpa_headers

def load_human_embeddings(file,hpa_proteins):

    proteins, embeddings = H5pyData.read(file,precision=16)

    prot2idx = {p:i for i,p in enumerate(proteins)}

    idx = [prot2idx[p] for p in hpa_proteins]

    return np.array(embeddings)[idx]


def benchmark_hpa(human_alias,hpa_set,t5_path,aligned_path,clfs,cv_headers,save_dir):
    
    hpa_proteins, hpa_labels, hpa_headers = process_hpa_set(human_alias,hpa_set,save_dir)

    hpa_labels_list = list()
    hpa_headers_list = list()

    headers2int = {h:idx for idx,h in enumerate(hpa_headers)}

    for h in cv_headers:
        if h in headers2int:
            hpa_labels_list.append(hpa_labels[:,headers2int[h]])
            hpa_headers_list.append(h)
        else:
            hpa_labels_list.append(np.zeros(hpa_labels.shape[0]))
            hpa_headers_list.append(h)

    hpa_labels = np.array(hpa_labels_list).T
    hpa_headers = np.array(hpa_labels_list)

    t5_embeddings = load_human_embeddings(t5_path,hpa_proteins)
    aligned_embeddings = load_human_embeddings(aligned_path,hpa_proteins)

    metrics_output = list()
    mcc_output = list()
    y_test = hpa_labels[:,[0,1,3,4,6,8]]
    pr_curves = list()

    for model_name, clf in clfs.items():
        if model_name == 't5':

            X_test = t5_embeddings

        elif model_name == 'aligned':
            X_test = aligned_embeddings

        elif model_name == 'space':
            X_test = np.concatenate([t5_embeddings,aligned_embeddings],axis=1)
            
        metrics, mcc = evaluate_model(clf,X_test,hpa_labels,True)

        metrics_output.append([model_name]+metrics)

        mcc_output.append([model_name]+mcc)

        pred_scores = clf.predict_proba(X_test)
        pred_scores = np.array([s[:,1] for s in pred_scores]).T[:,[0,1,3,4,6,8]]
        p,r = precision_recall(pred_scores.flatten(),y_test.flatten())
        pr_curves.append(np.stack([p,r],axis=1))
    
    pr_curves = pd.DataFrame(np.concatenate(pr_curves,axis=1),columns=['t5_prec','t5_recall',
                                                                        'aligned_prec','aligned_recall',
                                                                        'space_prec','space_recall'])
    
    pr_curves.to_csv(f'{save_dir}/hpa_pr_curves.csv')

    hpa_mcc_headers = ['model_name'] + [cv_headers[i] for i in [0,1,3,4,6,8]]

    pd.DataFrame(mcc_output,columns=hpa_mcc_headers).to_csv(f'{save_dir}/hpa_mccs.csv')

    pd.DataFrame(metrics_output,columns=['model_name','f1_micro','f1_macro','accuracy','jaccard']).to_csv(f'{save_dir}/hpa_scores.csv')

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmarking subcellular localization with logistic regression')

    parser.add_argument('--aligned_dir',type=str,help='Directory containing aligned embeddings',required=False,default='data/functional_emb')

    parser.add_argument('--t5_dir',type=str,help='Directory containing t5 embeddings',required=False,default='data/t5_emb')

    parser.add_argument('--cv_set',type=str,help='Cross validation set',required=False,default='data/benchmarks/deeploc/Swissprot_Train_Validation_dataset.csv')

    parser.add_argument('--cv_mapping',type=str,help='Cross validation id mapping',required=False,default='data/benchmarks/deeploc/cv_idmapping.tsv')

    parser.add_argument('--hpa_set',type=str,help='HPA test set',required=False,default='data/benchmarks/deeploc/hpa_testset.csv')

    parser.add_argument('--human_alias',type=str,help='Human alias mapping',required=False,default='data/benchmarks/deeploc/9606.protein.aliases.v12.0.txt.gz')

    parser.add_argument('--save_dir',type=str,help='Directory to save results',required=False,default='results/subloc')

    parser.add_argument('--jobs',type=int,help='Number of jobs to run in parallel',required=False,default=7)

    parser.add_argument('--random_seed',type=int,help='Random seed',required=False,default=5678)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    logger.info('Starting benchmarking')
    logger.info('Benchmarking cross validation set')
    clfs, cv_headers = benchmark_cv(args.cv_set, args.cv_mapping, args.t5_dir, args.aligned_dir, args.save_dir, args.jobs, args.random_seed)
    logger.info('Benchmarking HPA test set')
    benchmark_hpa(args.human_alias, args.hpa_set, f'{args.t5_dir}/9606.h5', f'{args.aligned_dir}/9606.h5', 
                  clfs, cv_headers, args.save_dir)
    logger.info('Results saved in {}'.format(args.save_dir))
    logger.info('Done!')
                                     