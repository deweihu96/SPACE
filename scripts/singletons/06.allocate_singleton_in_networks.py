import pandas as pd
import os
from tqdm import tqdm
from space.tools.data import H5pyData
from multiprocessing import Pool
import sys

def handle_single_species(species,og_proteins,scaled_dir):
    # get the embeddings for og_proteins
    p,e = H5pyData.read(f'{scaled_dir}/{species}.h5',16)

    # make a dictionary of proteins 
    p2indx = {p:i for i,p in enumerate(p)}

    # get the indices of the proteins in the og_proteins
    indices = [p2indx[p] for p in og_proteins]
    used_e = e[indices]
    print(f'{species} done')
    return og_proteins,used_e


if __name__ == '__main__':


    singleton_ogs = sys.argv[1]

    species_list = sys.argv[2]

    scaled_dir = sys.argv[3]

    save_name = sys.argv[4]


    og_interaction_proteins = set()


    print('Reading the OGs')
    for f in tqdm(os.listdir(singleton_ogs)):
        f = f'{singleton_ogs}/{f}'

        df = pd.read_csv(f,compression='gzip',sep='\t',header=None)

        df_interaction = df[df.iloc[:,-1] == 'partial-interaction']

        for line in df_interaction.iloc[:,-2]:
            og_interaction_proteins.update(line.split(','))
    
    print('Sorting the proteins by species')
    proteins_per_species = dict()
    for p in tqdm(og_interaction_proteins):
        species = p.split('.')[0]
        if species in proteins_per_species:
            proteins_per_species[species].append(p)
        else:
            proteins_per_species[species] = [p]

    print(f'Number of species: {len(proteins_per_species)}')
    print(f'Number of proteins: {len(og_interaction_proteins)}')

    # save the species list
    with open(species_list,'w') as f:
        for s in proteins_per_species.keys():
            f.write(f'{s}\n')



    ## read the embeddings
    with Pool(8) as p:
        results = p.starmap(handle_single_species,[(s,p,scaled_dir) 
                                                   for s,p in proteins_per_species.items()])


    # save the embeddings in a single file
    proteins = list()
    embeddings = list()
    print('Saving the embeddings')

    ## prepare the data for saving
    for og_proteins,used_e in results:
        proteins.extend(og_proteins)
        embeddings.extend(used_e)
    
    print('Saving the data')
    H5pyData.write_large(proteins,embeddings,save_name,16)

    print('Done')
