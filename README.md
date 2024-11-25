# SPACE: STRING proteins as complementary embeddings
Code for the manuscript: "SPACE: STRING proteins as complementary embeddings" (will be on a preprint website soon). 

The precalculated embeddings can be downloaded directly from [the STRING website](https://string-db.org/cgi/download).

![SPACE](./figures/space_overview.png)

## How to Cite
If you use this work in your research, please cite our upcoming manuscript and the STRING database.
```
@article{szklarczyk2023string,
  title={The STRING database in 2023: protein--protein association networks and functional enrichment analyses for any sequenced genome of interest},
  author={Szklarczyk, Damian and Kirsch, Rebecca and Koutrouli, Mikaela and Nastou, Katerina and Mehryary, Farrokh and Hachilif, Radja and Gable, Annika L and Fang, Tao and Doncheva, Nadezhda T and Pyysalo, Sampo and others},
  journal={Nucleic acids research},
  volume={51},
  number={D1},
  pages={D638--D646},
  year={2023},
  publisher={Oxford University Press}
}
```
## Usage of SPACE embeddings
To have the best prediction results, based on our test, it's better to concatenate the cross-species network embeddings and the ProtT5 sequence embeddings. (That is our **SPACE** embeddings mentioned in the manuscript.)


### How to read the embedding files

Install the `h5py` package to read the single species embedding files. We provide examples of reading the cross-species (aligned) network embeddings. The ProtT5 sequence embedding files have the same format.

The following code reads the cross-species network embedding file `9606.protein.network.embeddings.v12.0.h5`. 

In Python: 
```bash
pip install h5py
```


```Python
import h5py

filename = '9606.protein.network.embeddings.v12.0.h5'

with h5py.File(filename, 'r') as f:
    meta_keys = f['metadata'].attrs.keys()
    for key in meta_keys:
        print(key, f['metadata'].attrs[key])

  embedding = f['embeddings'][:]
  proteins = f['proteins'][:]

  # protein names are stored as bytes, convert them to strings
  proteins = [p.decode('utf-8') for p in proteins]
```

In R:  
Install the `rhdf5` package to read the embedding files. The following code reads the embedding file `9606.protein.network.embeddings.v12.0.h5`.

```R
# Install required packages if not already installed
# install.packages("rhdf5")

# Load the library
library(rhdf5)

filename <- '9606.protein.network.embeddings.v12.0.h5'

metadata <- h5readAttributes(filename, "metadata")
for (key in names(meta_keys)) {
    print(paste(key, meta_keys[[key]]))
}

embeddings <- h5read(filename, "embeddings")
proteins <- h5read(filename, "proteins")
```

Read the combined network embedding file of all eukaryotes  with Python
```Python
import h5py

filename = 'protein.network.embeddings.v12.0.h5'

with h5py.File(filename, 'r') as f:
    meta_keys = f['metadata'].attrs.keys()
    for key in meta_keys:
        print(key, f['metadata'].attrs[key])
  
  species = '4932'  # if we check the brewer's yeast
  embeddings = f['species'][species]['embeddings'][:]
  proteins = f['proteins'][species]['embeddings'][:]

  # protein names are stored as bytes, convert them to strings
  proteins = [p.decode('utf-8') for p in proteins]

```
Read the combined file with R
```R
library(rhdf5)

filename <- 'protein.network.embeddings.v12.0.h5'

meta_keys <- h5attributes(h5file$metadata)
for (key in names(meta_keys)) {
    print(paste(key, meta_keys[[key]]))
}

species <- '4932'  # for brewer's yeast
embeddings <- h5read(filename, paste0('species/', species, '/embeddings'))
proteins <- h5read(filename, paste0('species/', species, '/proteins'))
```



## Key Features
- **Complementary embeddings**: SPACE provides embeddings for each eukaryotic protein in the STRING network, which is complementary to the sequence-based embeddings.
- **Maintenance of original node2vec information**: Aligned embeddings are comparable to the original node2vec embeddings, benchmarked against the KEGG pathways.
- **Improved performance**: SPACE embeddings can improve cross-species predictions such as subcellular localization and protein function prediction.


## To reproduce the results

### Installation
```bash
git clone https://github.com/deweihu96/SPACE.git
conda create -n space python=3.11
conda activate space
cd SPACE
pip install .
```


### Download Data

A processed version of all the files to reproduce can be downloaded from [here](https://erda.ku.dk/archives/c2a0ba424cf75184c39a3cd37e4fe1a6/published-archive.html). You also need to download the STRING network files from the STRING website to run the node2vec embeddings.

```bash
### Run node2vec embeddings
```bash
## change the species_id to the species you want to run
mkdir -p results/node2vec

# with default hyperparameters, and change the input file path
python scripts/node2vec.py \
-i data/networks/{species_id}.protein.links.v12.0.txt.gz \
-o results/node2vec

```

### Seed species alignment
```bash
python scripts/align_seeds.py
```

## Non-seed species alignment
```bash
# align rat to the 48 seed species
python scripts/align_non_seeds.py \
--non_seed_species 10116
```
## License
MIT.