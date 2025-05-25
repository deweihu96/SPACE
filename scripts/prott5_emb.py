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
import argparse
import os
from tqdm import tqdm

def load_sequences(filename,max_length=8000,
                   min_length=1):
    
    logger.info(f"Loading sequences from {filename} with max_length={max_length} and min_length={min_length}")

    file_type = filename.split('.')[-1]
    if file_type in ['fasta', 'fa', 'txt']:
        with open(filename, 'r') as f:
            # if the sequence is too long, we drop it
            sequences = []
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)
                if len(seq) <= max_length and len(seq) >= min_length:
                    sequences.append([str(record.id), seq])
    elif filename.endswith('.gz'):
        with gzip.open(filename, 'rt') as f:
            sequences = []
            for record in SeqIO.parse(f, 'fasta'):
                seq = str(record.seq)
                if len(seq) <= max_length and len(seq) >= min_length:
                    sequences.append([str(record.id), seq])
    else:
        raise ValueError('Unsupported file format')

    return sequences



class SeqDataset(Dataset):

    def __init__(self,sequences) -> None:
        super().__init__()

        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):

        return len(self.sequences[index][1]), self.sequences[index][0], self.sequences[index][1]
    

def main(seq_file,
         save_path,
         device,
         max_length=8000,
         min_length=1,
         batch_size=1,):
    
    
    
    # check if the device is valid
    if device not in ['cuda', 'cpu']:
        logger.error("Invalid device specified. Use 'cuda' or 'cpu'.")
        sys.exit(1)
    device = torch.device(device)
    logger.info(f"Using device: {device}")

    sequences = load_sequences(seq_file,max_length=max_length,
                               min_length=min_length)
    # check if the sequences are empty
    if len(sequences) == 0:
        logger.error("No sequences found in the input file.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(sequences)} sequences from {seq_file}")

   
    

    # Load the tokenizer
    if not os.path.exists("temp/seq_models"):
        os.makedirs("temp/seq_models", exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', 
                                            do_lower_case=False,cache_dir="temp/seq_models",
                                            )

    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc",cache_dir="temp/seq_models").to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model = model.to(torch.float16) if device!=torch.device("cpu") else model


    seq_loader = DataLoader(SeqDataset(sequences),batch_size=batch_size,shuffle=True)

    output_seq_names = []
    output_seq_embeddings = []

    for seq_len,seq_name,seq in tqdm(seq_loader, desc="Calculating embeddings"):
        ## sort the sequences by length
        seq_len,sort_index = seq_len.sort(descending=False)
        seq_name = [seq_name[i] for i in sort_index]
        seq = [seq[i] for i in sort_index]

        seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq_))) for seq_ in seq]

        ids = tokenizer(seq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # logger.info(f"Processing {seq_name}")   
        # generate embeddings
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

        for i in range(len(seq_len)):
            emb = embedding_repr.last_hidden_state[i][:seq_len[i]].mean(dim=0)
            # output.append([seq_name[i],emb])
            output_seq_names.append(seq_name[i])
            output_seq_embeddings.append(emb.to('cpu').numpy())
        # logger.info(f"Processing finished {seq_name}")

    # H5pyData.write(output_seq_names,output_seq_embeddings,'temp/seq_embeddings.h5')
    if len(output_seq_names) > 0:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        H5pyData.write(output_seq_names, output_seq_embeddings, 
                        save_path,16)
        logger.info(f"Embeddings saved to {save_path}")
    else:
        logger.warning("No sequences processed, no embeddings saved.")
        logger.warning("Check if the input file is empty or all sequences are too long.")

    return None

if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description="Generate protein embeddings using ProtT5")
    args.add_argument('--seq_file', type=str, required=True, help='Path to the input sequence file (FASTA format)')
    args.add_argument('--save_path', type=str, required=True, help='Path to save the output embeddings (HDF5 format)')
    args.add_argument('--max_length', type=int, default=8000, help='Maximum sequence length to process (default: 8000). ')
    args.add_argument('--min_length', type=int, default=1, help='Minimum sequence length to process (default: 1). Sequences shorter than this will be ignored.')
    args.add_argument('--device', type=str, default='cuda', help='Device to run the model on (default: cuda)')
    args.add_argument('--batch_size', type=int, default=1, help='Batch size for processing sequences (default: 1)')
    args = args.parse_args()
    
    main(seq_file=args.seq_file,
         save_path=args.save_path,
         device=args.device,
         max_length=args.max_length,
         batch_size=1)  # Set batch_size to 1 for simplicity