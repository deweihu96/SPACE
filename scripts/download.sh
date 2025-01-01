# download the preprocessed data
curl https://erda.ku.dk/archives/c2a0ba424cf75184c39a3cd37e4fe1a6/space-2024-10-28/extra_data/data/data.tgz
mkdir -p data
tar -xzvf data.tgz -C data

# download the STRING networks and sequences
mkdir -p data/networks
SPECIES="data/euks.txt"
for species in $(cat $SPECIES); do
    wget https://stringdb-downloads.org/download/protein.links.v12.0/$species.protein.links.v12.0.txt.gz -O data/networks/$species.protein.links.v12.0.txt.gz -q
    wget https://stringdb-downloads.org/download/protein.sequences.v12.0/$species.protein.sequences.v12.0.fa.gz -O data/networks/$species.protein.sequences.v12.0.fa.gz -q
done

# download the subcellular localization data from DeepLoc2.0
mkdir -p data/benchmarks/deeploc20
wget https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv -O data/benchmarks/deeploc20/Swissprot_Train_Validation_dataset.csv -q
wget https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/hpa_testset.csv -O data/benchmarks/deeploc20/hpa_testset.csv -q

# the protein function prediction benchmark has to be downloaded manually from the following link, 
# according the NetGO paper (https://doi.org/10.1093/nar/gkz388): 
# https://drive.google.com/drive/folders/1HLH1aCDxlrVpu1zKvgfdQFEFnbT8gChm