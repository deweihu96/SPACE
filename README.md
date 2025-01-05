# SPACE: STRING proteins as complementary embeddings
Code for the manuscript: ["SPACE: STRING proteins as complementary embeddings"](https://www.biorxiv.org/content/10.1101/2024.11.25.625140v1),
in which we precalculated:
- cross-species network embeddings 
- ProtT5 sequence embeddings  

for all eukaryotic proteins in STRING v12.0.

You can [download all the embeddings from the STRING website](https://string-db.org/cgi/download):
- protein.network.embeddings.v12.0.h5
- protein.sequence.embeddings.v12.0.h5

![SPACE](./figures/space_overview.png)

## How to Cite
If you use this work in your research, please cite our manuscript 
```
@article {Hu2024.11.25.625140,
	author = {Hu, Dewei and Szklarczyk, Damian and Mering, Christian von and Jensen, Lars Juhl},
	title = {SPACE: STRING proteins as complementary embeddings},
	elocation-id = {2024.11.25.625140},
	year = {2024},
	doi = {10.1101/2024.11.25.625140},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/11/26/2024.11.25.625140},
	eprint = {https://www.biorxiv.org/content/early/2024/11/26/2024.11.25.625140.full.pdf},
	journal = {bioRxiv}
}

```

and the STRING database.
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


## Reproduce the results in the manuscript
Please follow this [document](./reproduce.md).
## License
MIT.