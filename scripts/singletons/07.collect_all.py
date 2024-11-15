## handle all the singletons here by species
import sys
from space.tools.data import H5pyData
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

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

def main(species:str,interation_og:str,
         singleton_og_dir:str,
         pure_singleton_og:str,
         singleton_dir:str,
         scaled_dir:str,
         save_dir:str):
    
    singleton_embeddings = dict()
    ## load the singletons
    noise = 1e-5
    print(f'handling species: {species}')
    ## load the per interaction-og protein embeddings, which will be added some noise
    print("reading the interaction orthologs")
    interaction_og,interaction_og_emb = H5pyData.read_large(interation_og,16)
    interaction_og2idx = {str(p):i for i,p in enumerate(interaction_og)}

    ## load the singleton orthologs 
    print("reading the singleton orthologs")
    for f in tqdm(os.listdir(singleton_og_dir)):
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
    for p,e in tqdm(singleton_embeddings.items()):
        e = np.array(e)
        e = np.mean(e,axis=0).reshape(-1)
        singleton_embeddings[p] = e

    print(f'handled {len(singleton_embeddings)} singletons by avg orthologs')
    
    ## put the pure singletons into the dictionary
    ## just to be incorporated in the end
    pure_singleton_og,pure_singleton_ogs_emb = H5pyData.read_large(pure_singleton_og,16) 
    pure_singleton_og2idx = {p:i for i,p in enumerate(pure_singleton_og)}
    ## filter out other species
    # pure_singleton_og = [p for p in pure_singleton_og if p.split('.')[0] == species]
    pure_singleton_og = list(filter(lambda x: x.split('.')[0] == species,pure_singleton_og))
    indices = list(map(lambda x: pure_singleton_og2idx[x],pure_singleton_og))
    pure_singleton_ogs_emb = pure_singleton_ogs_emb[indices]
    print(f'handled {len(pure_singleton_og)} singletons by pure singletons orthologs')
    singleton_embeddings.update({p:e for p,e in zip(pure_singleton_og,pure_singleton_ogs_emb)})

    ## load the singletons list
    all_singletons = f'{singleton_dir}/{species}.singleton.proteins.v12.0.txt'
    all_singletons = open(all_singletons).read().strip().split('\n')

    ## 
    unsolved_singletons = set(all_singletons) - set(singleton_embeddings.keys())
    print(f'unsolved singletons: {len(unsolved_singletons)}')
    print('generating random embeddings for unsolved singletons...')
    ## give random embeddings for the unsolved singletons
    for p in unsolved_singletons:
        random_emb = generate_random_embeddings()
        singleton_embeddings[p] = random_emb

    ## check if every proteins are solved
    assert len(set(all_singletons) - set(singleton_embeddings.keys())) == 0 

    ## save the embeddings
    # load the previous scaled embeddings
    print('merging the embeddings')
    proteins,embeddings = H5pyData.read(f'{scaled_dir}/{species}.h5',16)

    proteins = list(singleton_embeddings.keys()) + list(map(str,proteins))
    embeddings = np.concatenate([np.array(list(singleton_embeddings.values())),embeddings],axis=0)

    ## save the embeddings
    H5pyData.write_large(proteins,embeddings,f'{save_dir}/{species}.h5',16)
    print(f'{species} done')



if __name__ == "__main__":

    species = sys.argv[1]

    interation_og = sys.argv[2]

    singleton_og_dir = sys.argv[3]

    pure_singleton_og = sys.argv[4]

    singleton_dir = sys.argv[5]

    scaled_dir = sys.argv[6]

    save_dir = sys.argv[7]


    main(species,interation_og,singleton_og_dir,
         pure_singleton_og,singleton_dir,scaled_dir,save_dir)
