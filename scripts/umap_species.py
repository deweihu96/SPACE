import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from space.tools.data import  H5pyData
from tqdm import tqdm
import os
import argparse

argparser = argparse.ArgumentParser(description='UMAP visualization of species embeddings')
argparser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save the results')
argparser.add_argument('--embedding_dir', type=str, default='data/aligned',
                       help='Directory containing the species embeddings')
args = argparser.parse_args()



if os.path.exists(args.save_dir) == False:
    os.makedirs(args.save_dir)

species = [9606,4932,3702,44689,10116]
                 

species_names={4932:'Saccharomyces cerevisiae',3702:'Arabidopsis thaliana',
               9606:'Homo sapiens',44689:'Dictyostelium discoideum',10116:'Rattus norvegicus'}
colors = ["#e60049","#0bb4ff","#50e991","#9b19f5","#ffa300"]
embedding_dir = args.embedding_dir

if __name__ == '__main__':

    emb = list()
    labels = list()
    num_points = list()

    for s in tqdm(species):
        _, e = H5pyData.read(f'{embedding_dir}/{s}.h5',16)
        emb.append(e)
        labels.append([species_names[s]] * e.shape[0])
        num_points.append(e.shape[0])

    umap_emb = umap.UMAP(n_neighbors=100, min_dist=1, metric='cosine').fit_transform(np.concatenate(emb))
    umap_df = pd.DataFrame(umap_emb, columns=['UMAP1', 'UMAP2'])
    umap_df['species'] = np.concatenate(labels)

    plt.figure(figsize=(20, 4))

    for i, s in enumerate(species):
        other_species = [species_names[k] for k in species if k != s]
        plt.subplot(1, 5, i+1)
        
        others = umap_df[umap_df['species'].isin(other_species)]
        plt.scatter(others['UMAP1'], others['UMAP2'], color='#cccccc', s=3)
        
        this_species = umap_df[umap_df['species'] == species_names[s]]
        plt.scatter(this_species['UMAP1'], this_species['UMAP2'], s=3, color=colors[i])
        
        plt.title(species_names[s], fontsize=20, style='italic')

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        if s == 9606:
            plt.xlabel('UMAP1',fontsize=20)
            plt.ylabel('UMAP2',fontsize=20)
            ## change the tick number sizes
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            # remove the top and right spines
        else:
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.yticks([])
            plt.xticks([])

    plt.tight_layout()

    plt.savefig(f'{args.save_dir}/figure_1_umap_species.png', dpi=300)