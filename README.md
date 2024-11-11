# SPACE: STRING proteins as complementary embeddings
Code for the manuscript: "SPACE: STRING proteins as complementary embeddings"
add link here. 

The precalculated embeddings can be downloaded directly from [the STRING website](https://string-db.org/)

![SPACE](./figures/space_overview.png)

## How to read the embedding files
### Open Embedding Files with Python
Install the `h5py` package to read the embedding files. The following code reads the embedding file `9606.h5` and prints the metadata, the first 5 embeddings, and the first 5 proteins.

```bash
pip install h5py
```


```Python
import h5py

filename = '9606.h5'

with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    print('\n')

    # list the metadata
    print("Metadata: %s" % f['metadata'].attrs.keys())
    print('\n')
    # list the metadata
    for key in f['metadata'].attrs.keys():
        print("%s: %s" % (key, f['metadata'].attrs[key]))

    print('\n')
    # show the first embedding
    print("Embedding: %s" % f['embeddings'][:5])
    print('\n')
    # list the proteins
    print("Proteins before decode: %s" % f['proteins'][:3])
    print('\n')
    ## here the proteins names are stored as bytes, convert them to strings
    proteins = f['proteins'][:3]
    proteins = [p.decode('utf-8') for p in proteins]
    print("Proteins after decode: %s" % proteins)
```

### Open Embedding Files with R
Install the `rhdf5` package to read the embedding files. The following code reads the embedding file `9606.h5` and prints the metadata, the first 5 embeddings, and the first 5 proteins.

```R
# Install required packages if not already installed
# install.packages("rhdf5")

# Load the library
library(rhdf5)

filename <- '9606.h5'

# Open the H5 file
h5file <- H5Fopen(filename)

# List all groups
cat("Keys:", h5ls(filename)$name, "\n\n")

# List the metadata
metadata_attrs <- h5readAttributes(h5file, "metadata")
cat("Metadata:", names(metadata_attrs), "\n\n")

# List metadata values
for (key in names(metadata_attrs)) {
  cat(sprintf("%s: %s\n", key, metadata_attrs[[key]]))
}
cat("\n")

# Show the first 5 embeddings
embeddings <- h5read(h5file, "embeddings")[1:5,]
cat("Embedding:\n")
print(embeddings)
cat("\n")

# List the proteins (before decode)
proteins_raw <- h5read(h5file, "proteins")[1:3]
cat("Proteins before decode:", proteins_raw, "\n\n")

# Convert proteins from raw to character (equivalent to decode)
proteins <- rawToChar(proteins_raw)
# Alternative if stored differently:
# proteins <- sapply(proteins_raw, rawToChar)
cat("Proteins after decode:", proteins, "\n")

# Close the file
H5Fclose(h5file)
```



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