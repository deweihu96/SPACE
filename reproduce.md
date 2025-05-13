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

### 2. Download the data from [here](https://erda.ku.dk/archives/8a9fb8f6d9811fe66574a9a989d4a29d/published-archive.html) and decompress it to the `data` folder.
It contains all the data for the project (`~169GB`). Especially, the benchmark data:
- `data/benchmarks/deeploc/benchmark` is from [DeepLoc 2.0](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/)
- `data/benchmarks/netgo/benchmark` is from [NetGO](https://drive.google.com/drive/folders/1HLH1aCDxlrVpu1zKvgfdQFEFnbT8gChm)

```bash 
# decompress
tar -xvf data.tgz
```
```md5sum data.tgz 471bf89794e84de3670c6b703410d1ac```

and the benchmark data and the STRING files:
```bash
# download the STRING networks and sequences
mkdir -p data/networks
SPECIES="data/euks.txt"
for species in $(cat $SPECIES); do
    # networks
    wget https://stringdb-downloads.org/download/protein.links.v12.0/$species.protein.links.v12.0.txt.gz -O data/networks/$species.protein.links.v12.0.txt.gz -q
    # sequences
    wget https://stringdb-downloads.org/download/protein.sequences.v12.0/$species.protein.sequences.v12.0.fa.gz -O data/networks/$species.protein.sequences.v12.0.fa.gz -q
done

# download the subcellular localization data from DeepLoc2.0
mkdir -p data/benchmarks/deeploc20
wget https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv -O data/benchmarks/deeploc20/Swissprot_Train_Validation_dataset.csv -q
wget https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/hpa_testset.csv -O data/benchmarks/deeploc20/hpa_testset.csv -q

```
the protein function prediction benchmark has to be downloaded manually from the following link, according the NetGO paper (https://doi.org/10.1093/nar/gkz388): https://drive.google.com/drive/folders/1HLH1aCDxlrVpu1zKvgfdQFEFnbT8gChm

### 3. Generate the functional embeddings 
You can get the help with each script by running `python scripts/xxx.py -h`.

#### 3.1 Run the node2vec algorithm to generate the node embeddings.

```bash
## for instance, run the node2vec on human network
mkdir results/node2vec

python scripts/node2vec.py \
-i data/networks/9606.protein.links.v12.0.txt.gz \
-o results/node2vec
```

#### 3.2 Run the FedCoder to align the seed species

```bash
# with the best hyperparameters in the paper
python scripts/align_seeds.py \
--embedding_save_folder results/aligned 
```

#### 3.3 Run the FedCoder to align the non-seed species

```bash
# for instance, align Rattus norvegicus (Norway rat)  
python scripts/align_non_seeds.py \
--non_seed_species 10116 \
--embedding_save_folder results/aligned
```

#### 3.4 Add the singletons to the aligned embedding space

```bash
python scripts/add_singletons.py \
--full_embedding_save_dir results/functional_embeddings
```

#### 3.5 Generate the ProtT5 embeddings
```bash
# ref: https://github.com/agemagician/ProtTrans
python scripts/prott5_emb.py 
```


#### 4. Evaluate the functional embeddings

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
