'''
This module contains the data structures for the space module.
'''
import numpy as np
import h5py
from typing import List, Tuple, Iterable, Union
import gzip
import csv
import os
from multiprocessing import Pool
import itertools
from loguru import logger

class H5pyData:

    @staticmethod
    def write(proteins: Union[np.ndarray, List, Tuple],
            embedding: np.ndarray,
            save_path: str,
            precision: int,
            chunk_size: int = 10000) -> None:
        '''
        Write proteins and embeddings to HDF5 file efficiently.
        
        Args:
            proteins: Array-like of protein identifiers
            embedding: numpy array of embeddings (n_proteins x embedding_dim)
            save_path: Path to save the HDF5 file
            precision: Precision of embeddings (16 or 32)
            chunk_size: Size of chunks for HDF5 dataset
        '''
        # Convert proteins to numpy array if needed
        proteins = np.array(proteins).astype('U').reshape(-1)
        embedding = np.array(embedding)
        
        # Validate inputs
        if len(proteins) != len(embedding):
            raise ValueError(f"Number of proteins ({len(proteins)}) doesn't match number of embeddings ({len(embedding)})")
        
        if precision not in [16, 32]:
            raise ValueError(f"Precision must be 16 or 32, got {precision}")
        
        # Determine dtype for embeddings
        dtype = np.float16 if precision == 16 else np.float32
        embedding = embedding.astype(dtype)
        
        n_proteins = len(proteins)
        embedding_dim = embedding.shape[1]
        
        with h5py.File(save_path, 'w') as f:
            # Create groups to organize data
            f.create_group('metadata')
            
            # Store metadata
            f['metadata'].attrs['n_proteins'] = n_proteins
            f['metadata'].attrs['embedding_dim'] = embedding_dim
            f['metadata'].attrs['precision'] = precision
            
            # Create datasets with chunking and compression
            protein_ds = f.create_dataset(
                'proteins',
                shape=(n_proteins,),
                dtype=h5py.string_dtype(),
                chunks=(min(chunk_size, n_proteins),),
                compression='gzip',
                compression_opts=4
            )
            
            embedding_ds = f.create_dataset(
                'embeddings',
                shape=(n_proteins, embedding_dim),
                dtype=dtype,
                chunks=(min(chunk_size, n_proteins), embedding_dim),
                compression='gzip',
                compression_opts=4
            )
            
            # Write data in chunks for better memory management
            for i in range(0, n_proteins, chunk_size):
                end_idx = min(i + chunk_size, n_proteins)
                
                protein_chunk = proteins[i:end_idx]
                embedding_chunk = embedding[i:end_idx]
                
                protein_ds[i:end_idx] = protein_chunk
                embedding_ds[i:end_idx] = embedding_chunk

    @staticmethod
    def read(file_path: str,precision: int) -> tuple[np.ndarray, np.ndarray]:
        '''
        Read proteins and embeddings from HDF5 file.
        
        Args:
            file_path: Path to HDF5 file
            precision: Precision of embeddings (16 or 32)
            
        Returns:
            tuple: (proteins array, embeddings array)
        '''
        with h5py.File(file_path, 'r') as f:
            proteins = f['proteins'][:]
            proteins = np.vectorize(lambda x: str(x)[2:-1])(proteins)
            embeddings = f['embeddings'][:]
        if precision == 16:
            embeddings = embeddings.astype(np.float16)
        elif precision == 32:
            embeddings = embeddings.astype(np.float32)

        return proteins, embeddings
    

    
    

def query_single_species(querys, precision, aligned_path):

    # logger.info(f'Loading {aligned_path}...')
    proteins,embeddings = H5pyData.read(aligned_path,precision)

    protein2index = {protein: index for index, protein in enumerate(proteins)}

    indices = []
    for query in querys:
        try:
            index = protein2index[query]
            indices.append( index)
        except NameError as e:
            logger.info(f'Query {query} not found. Error: {e}')
    
    ## get the embeddings
    query_proteins = [proteins[index] for index in indices]
    query_embeddings = [embeddings[index] for index in indices]

    return query_proteins, query_embeddings

def query_embedding(querys:Iterable, query_dir:str,precision:int, n_jobs:int) -> Tuple[np.ndarray,np.ndarray]:
    '''
        Query the embeddings for the given proteins in multiple species.
        Args:
            querys: Iterable, the proteins to query.
            aligned_dir: str, the directory of the aligned embeddings.
            precision: int, the precision of the embeddings, either 32 or 16.
            n_jobs: int, the number of jobs to run in parallel.
        Returns:
            output_proteins: np.array, the proteins that are found.
            output_embeddings: np.array, the embeddings of the proteins.

    '''

    ## check if the directory of query_dir exists
    if not os.path.exists(query_dir):
        raise FileNotFoundError(f'{query_dir} does not exist')
    
    if precision not in [32,16]:
        raise ValueError('Precision should be either 32 or 16')

    ## put the quries in a dict where the key is taxid and the value is the list of querys
    query_dict = {}

    for query in querys:
        taxid = query.split('.')[0]
        if taxid not in query_dict:
            query_dict[taxid] = [query]
        else:
            query_dict[taxid].append(query)

    ## for each taxid get the data and the querys
    with Pool(n_jobs) as p:
        results = p.starmap(query_single_species, [(query_dict[taxid],precision, f'{query_dir}/{taxid}.h5') for taxid in query_dict])

    output_proteins = list()
    output_embeddings = list()

    for result in results:
        output_proteins.append(result[0])
        output_embeddings.append(result[1])

    output_proteins = np.array(list(itertools.chain.from_iterable(output_proteins)))
    output_embeddings = np.array(list(itertools.chain.from_iterable(output_embeddings)))

    return output_proteins, output_embeddings


## a class to handle the gz file

class GzipData:
    @staticmethod
    def string2idx(file_path:str,temp_path)->dict:

        nodes = dict()

        ## check if the directory of temp_path exists
        if not os.path.exists(os.path.dirname(temp_path)):
            os.makedirs(os.path.dirname(temp_path))

        edges_writer = csv.writer(open(temp_path, 'w'), delimiter='\t')

        with gzip.open(file_path, 'rt') as f:
            
            reader = csv.reader(f, delimiter=' ')

            next(reader) # skip the header

            for row in reader:
                
                if row[0] not in nodes:
                    nodes[row[0]] = len(nodes)
                if row[1] not in nodes:
                    nodes[row[1]] = len(nodes)
                
                ## get the index of the protein
                src_idx = nodes[row[0]]
                dst_idx = nodes[row[1]]
                weight = int(row[-1])/1000

                edges_writer.writerow([src_idx,dst_idx,weight])

        return nodes
    
    @staticmethod
    def read_nodes(file_path:str)->dict:
        nodes = dict()
        with gzip.open(file_path, 'rt') as f:
            reader = csv.reader(f, delimiter=' ')
            
            next(reader) # skip the header

            for row in reader:
                if row[0] not in nodes:
                    nodes[row[0]] = len(nodes)
                if row[1] not in nodes:
                    nodes[row[1]] = len(nodes)
        return nodes