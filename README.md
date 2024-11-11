# SPACE: STRING proteins as complementary embeddings
Code for the manuscript: "SPACE: STRING proteins as complementary embeddings"
add link here. 

The precalculated embeddings can be downloaded directly from [the STRING website](https://string-db.org/)

![SPACE](./figures/space_overview.png)

## How to read the embedding files
Please refer to [the README file](./embed_wiki.md) for the details of the embedding files.

## Key Features
- **Complementary embeddings**: SPACE provides embeddings for each eukaryotic protein in the STRING network, which is complementary to the sequence-based embeddings.
- **Maintenance of original node2vec information**: Aligned embeddings are comparable to the original node2vec embeddings, benchmarked against the KEGG pathways.
- **Improved performance**: SPACE embeddings can improve cross-species predictions such as subcellular localization and protein function prediction.


## Installation
```bash
git clone https://github.com/deweihu96/SPACE.git
conda create -n space python=3.11
conda activate space
cd SPACE
pip install .
```


## Download Data

A processed version of all the files to reproduce can be downloaded from here: xxxxxx.

It contains orthologs pairs, species list, species groups, and pre-calculated embeddings.

The preprocessing steps can be referred to the folder: `scripts/preprocess/`.

All the STRING networks should be downloaded from the STRING website, and put them in `data/networks/`.
The network files used in the paper are: `{taxid}.protein.links.v12.0.txt.gz` .

## Run node2vec embeddings
```bash
## change the species_id to the species you want to run
mkdir -p results/node2vec

# with default hyperparameters
python scripts/node2vec.py \
-i data/networks/{species_id}.protein.links.v12.0.txt.gz \
-o results/node2vec

```
You can skip this step and use the embeddings we provided here: xxxxxxxx

## Seed species alignment
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