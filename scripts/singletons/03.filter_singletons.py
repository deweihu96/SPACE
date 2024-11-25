from space.tools.data import H5pyData
import os
import gzip
import sys

def filter_singleton(species,embed_dir,sequence_dir):

    emb_path = os.path.join(embed_dir,f'{species}.h5')

    interaction_proteins, _ = H5pyData.read(emb_path,16)
    interaction_proteins = set(interaction_proteins)

    sequence_file = f'{sequence_dir}/{species}.protein.sequences.v12.0.fa.gz'
    ## read the sequence file
    sequence_proteins = set()
    with gzip.open(sequence_file,'rt') as f:
        for line in f:
            if line.startswith('>'):
                sequence_proteins.add(line.strip().split()[0][1:])
    
    singleton_proteins = sequence_proteins - interaction_proteins

    return singleton_proteins

if __name__ == "__main__":

    species = sys.argv[1]
    embed_dir = sys.argv[2]
    sequence_dir = sys.argv[3]
    save_dir = sys.argv[4]

    singleton_proteins = filter_singleton(species,embed_dir,sequence_dir)
    with open(f'{save_dir}/{species}.singleton.proteins.v12.0.txt','w') as f:
        for protein in singleton_proteins:
            f.write(f'{protein}\n')
    