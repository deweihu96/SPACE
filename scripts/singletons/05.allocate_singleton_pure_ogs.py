from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from space.tools.data import H5pyData
import sys

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

if __name__ == "__main__":
    
    singleton_og_dir = sys.argv[1]

    save_name = sys.argv[2]


    print('Reading the OGs')
    og_singletons = dict()

    noise = 1e-5

    singleton_embs = dict()

    for f in tqdm(os.listdir(singleton_og_dir)):
        f = f'{singleton_og_dir}/{f}'

        df = pd.read_csv(f,compression='gzip',sep='\t',header=None)

        df = df[df.iloc[:,-1] == 'non-interaction']

        for idx,line in df.iterrows():
            og_name = line[0]
            
            proteins = line[2].split(',')

            random_emb = generate_random_embeddings()

            noise_ = noise * np.random.random(size=(len(proteins),512))

            protein_embs = random_emb + noise_

            for p_idx, p in enumerate(proteins):
                if p in singleton_embs:
                    singleton_embs[p].append(protein_embs[p_idx])
                else:
                    singleton_embs[p] = [protein_embs[p_idx]]

    ## average the embeddings
    for p,e in tqdm(singleton_embs.items()):
        e = np.array(e)
        e = np.mean(e,axis=0).reshape(1,-1)
        singleton_embs[p] = e

    ## save the embeddings
    H5pyData.write_large(list(singleton_embs.keys()),
                         np.array(list(singleton_embs.values())).reshape(-1,512),
                         save_name,16)

