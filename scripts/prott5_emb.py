from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
from Bio import SeqIO
import gzip
import numpy as np
from torch.utils.data import DataLoader, Dataset
from space.tools.data import H5pyData
import sys
from loguru import logger

def load_sequences(filename,max_length=1024):

    file_type = filename.split('.')[-1]
    if file_type in ['fasta', 'fa', 'txt']:
        with open(filename, 'r') as f:
            sequences = [[str(record.id),str(record.seq)[:max_length]] for record in SeqIO.parse(f, 'fasta')]
    elif filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            sequences = [[str(record.id),str(record.seq)[:max_length]] for record in SeqIO.parse(f, 'fasta')]
    else:
        raise ValueError('Unsupported file format')

    return sequences

## randomly cut the sequences into length from 5 to 10
## development only
def cut_sequence(seq):
    length = np.random.randint(5,11)
    start = np.random.randint(0,len(seq)-length)
    return seq[start:start+length]


class SeqDataset(Dataset):

    def __init__(self,sequences) -> None:
        super().__init__()

        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):

        return len(self.sequences[index][1]), self.sequences[index][0], self.sequences[index][1]
    

def main(seq_file,max_length=12000,batch_size=4):

    sequences = load_sequences(seq_file,max_length=max_length)

    # sequences = [[name,cut_sequence(seq)] for name,seq in sequences]  ## development only

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False,cache_dir="temp/seq_models")

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc",cache_dir="temp/seq_models").to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model = model.to(torch.float16) if device!=torch.device("cpu") else model


    seq_loader = DataLoader(SeqDataset(sequences),batch_size=batch_size,shuffle=True)

    output_seq_names = []
    output_seq_embeddings = []

    for seq_len,seq_name,seq in seq_loader:
        ## sort the sequences by length
        seq_len,sort_index = seq_len.sort(descending=False)
        seq_name = [seq_name[i] for i in sort_index]
        seq = [seq[i] for i in sort_index]

        seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq_))) for seq_ in seq]

        ids = tokenizer(seq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        logger.info(f"Processing {seq_name}")   
        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        for i in range(len(seq_len)):
            emb = embedding_repr.last_hidden_state[i][:seq_len[i]].mean(dim=0)
            # output.append([seq_name[i],emb])
            output_seq_names.append(seq_name[i])
            output_seq_embeddings.append(emb.to('cpu').numpy())
        logger.info(f"Processing finished {seq_name}")

    H5pyData.write(output_seq_names,output_seq_embeddings,'temp/seq_embeddings.h5')

    return None

if __name__ == '__main__':

    seq_file = sys.argv[1]

    # change the max_length based on the device
    # gpu: 8000
    # cpu: 100000 # need RAM > 64GB, up to 400GB
    max_length = int(sys.argv[2]) if len(sys.argv)>2 else 8000

    main(seq_file,max_length=max_length,batch_size=1)