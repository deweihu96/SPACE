## Reproduce the results

### 1. Install the dependencies

```bash
git clone https://github.com/deweihu96/SPACE.git
conda create -n space python=3.11
conda activate space
cd SPACE
pip install .
mkdir results
```

### 2. Download the data from
Download the data and decompress it to the `data` folder.
The `data/` is around `212GB` in total, including the networks, sequences, all the embeddings, and benchmark datasets.

If you only need to use the embeddings, you should download from the STRING website: https://string-db.org/cgi/download

This data set only serves as a backup and reference for reimplementation.

```bash
wget https://sid.erda.dk/share_redirect/cZ4tLqQZhv -O data.tar
tar -xvf data.tar 
```
eggNOG dataset is sourced from the eggNOG database: http://eggnog6.embl.de/download/eggnog_6.0/, make sure you cite the eggNOG database if you use the data in your work.
```
Hernández-Plaza, Ana, et al. "eggNOG 6.0: enabling comparative genomics across 12 535 organisms." Nucleic Acids Research 51.D1 (2023): D389-D394.
```


The DeeoLoc dataset is originally from https://services.healthtech.dtu.dk/services/DeepLoc-2.0/
the protein function prediction benchmark has to be downloaded manually from the following link, according the NetGO paper (https://doi.org/10.1093/nar/gkz388): https://drive.google.com/drive/folders/1HLH1aCDxlrVpu1zKvgfdQFEFnbT8gChm

Please cite the original data sources and respect the rules if you use the benchmark data in your work:
```
Thumuluri, Vineet, et al. "DeepLoc 2.0: multi-label subcellular localization prediction using protein language models." Nucleic acids research 50.W1 (2022): W228-W234.

Yao, Shuwei, et al. "NetGO 2.0: improving large-scale protein function prediction with massive sequence, text, domain, family and network information." Nucleic acids research 49.W1 (2021): W469-W475.
```


### 3. Generate the functional embeddings 
You can get the help with each script by running `python scripts/xxx.py -h`.

#### 3.1 Run the node2vec algorithm to generate the node embeddings.
```bash
## for instance, run the node2vec on human network
mkdir results/node2vec

python scripts/node2vec.py \
--input_network data/networks/9606.protein.links.v12.0.txt.gz \
--node2vec_output results/node2vec
```
You could also use other node embedding algorithms as the input to alignment, but make sure that the output embeddings are in the same format as the node2vec embeddings. 
Also the index of the nodes in the embeddings should be the same as the index of the nodes in the network file, or the nodes in the node2vec embeddings.

#### 3.2 Run the FedCoder to align the seed species

```bash
# with the best hyperparameters in the paper
python scripts/align_seeds.py \
--seed_species data/seeds.txt \
--node2vec_dir data/node2vec \
--aligned_embedding_save_dir results/aligned_embeddings
```

#### 3.3 Run the FedCoder to align the non-seed species

```bash
# for instance, align Rattus norvegicus (Norway rat)  
python scripts/align_non_seeds.py \
--node2vec_dir data/node2vec \
--aligned_dir data/aligned \
--non_seed_species 10116 \
--aligned_embedding_save_dir results/aligned_embeddings
```

#### 3.4 Add the singletons to the aligned embedding space

```bash
python scripts/add_singleton.py \
--aligned_dir data/aligned \
--full_embedding_save_dir results/functional_embeddings
```

#### 3.5 Generate the ProtT5 embeddings

```bash
# make sure you have the sentencepiece library installed
pip install sentencepiece

# make sure you have enough GPU memory, otherwise adjust the min_length and max_length to run the script only for sequence within the range
# for example, with min_length=1 and max_length=1000, the sequences: 1<=length<=1000 will be processed. 

# if you have enough GPU memory (~60GB), those sequences with length [1,8000] can be processed.
# we ran the ProtT5 embedding on those super-long sequences with length [8000, 100000] on a CPU server, up to 400GB memory.

# for example, human sequences
python scripts/prott5_emb.py \
--seq_file data/sequences/9606.protein.sequences.v12.0.fa.gz \
--save_path results/prott5/9606.h5 \
--max_length 1000 \
--min_length 1 \
--device cuda
```

Reference: 
```
https://github.com/agemagician/ProtTrans 
https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc
```


#### 4. Evaluate the functional embeddings

To run the following scripts with default parameters, it uses the data inside the `data` folder.

Change the input to your own data if you want to evaluate the functional embeddings on your own networks.

Use the `-h` option to get the help message for each script.
```bash
# subcellular localization prediction, it also generates the umap plot of the subcellular localization
python scripts/subloc.py

# function prediction
python scripts/func_pred.py

# to have the umap plot of the species (fig1.b)
python scripts/umap_species.py

# precision-recall curve
python scripts/pr_curves.py

# due to the copyright issue, we cannot provide the KEGG data and its evaluation
```

#### 5. Alignment quality evaluation
This part is not included in the preprint now.
You can use the following scripts to evaluate the alignment quality of the functional embeddings.

To assess if the orthologous proteins are similar to each other than non-orthologous proteins in the aligned embedding space. We sample the orthologs and non-orthologs to reduce the memory usage.
Please check the code for the details of the sampling strategy.
```bash
python scripts/distances/og_sampling.py
```


To assess if the proteins are similar to each other in the same subcellular localization in the aligned embedding space:
```bash
python scripts/distances/deeploc_distance.py
```