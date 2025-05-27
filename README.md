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

## If you find our work interesting, please give us a star!
[![Star History Chart](https://api.star-history.com/svg?repos=deweihu96/SPACE&type=Date)](https://www.star-history.com/#deweihu96/SPACE&Date)

## Reproduce the results in the manuscript
Please follow this [document](./reproduce.md).

## How to Cite
If you use this work in your research, please cite **the SPACE manuscript**:  

Hu, D., Szklarczyk, D., von Mering, C., & Jensen, L. J. (2024). SPACE: STRING proteins as complementary embeddings. bioRxiv, 2024-11. https://doi.org/10.1101/2024.11.25.625140  

and **the STRING database**: 

Szklarczyk, D., Nastou, K., Koutrouli, M., Kirsch, R., Mehryary, F., Hachilif, R., ... & von Mering, C. (2025). The STRING database in 2025: protein networks with directionality of regulation. Nucleic Acids Research, 53(D1), D730-D737. https://doi.org/10.1093/nar/gkae1113

## Usage of SPACE embeddings
To have the best prediction results, based on our test, it's better to concatenate the cross-species network embeddings and the ProtT5 sequence embeddings. (That is our **SPACE** embeddings mentioned [in the manuscript](https://www.biorxiv.org/content/10.1101/2024.11.25.625140v1).)


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
  proteins = f['species'][species]['proteins'][:]

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



## License
MIT.
