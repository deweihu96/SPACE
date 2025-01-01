import h5py
import numpy as np
from space.tools.data import H5pyData
import os
from multiprocessing import Pool

def convert_h5_format(paths, precision=16):
    input_file_path, output_file_path = paths
    with h5py.File(input_file_path, 'r') as f:
        proteins = list(f.keys())

        proteins = np.array(proteins).astype('U').reshape(-1)

        embedding = [f[protein][:] for protein in proteins]

        if precision == 32:
            embedding = np.array(embedding).astype(np.float32)
        elif precision == 16:
            embedding = np.array(embedding).astype(np.float16)
    
    H5pyData.write(proteins=proteins, embedding=embedding, 
                   save_path=output_file_path, precision=precision)
    taxid = input_file_path.split('/')[-1].split('.')[0]
    with open(f'logs/fmt/{taxid}.txt', 'w') as f:
        f.write(f'Converted {taxid}')

    return None


# if __name__ == '__main__':

input_dir = 'data/aligned_non_seeds'
save_dir = 'data/new_aligned_non_seeds'

# for f in os.listdir('data/aligned_seeds'):
    
#     if f.endswith('.h5'):
#         input_file_path = os.path.join(input_dir, f)
#         output_file_path = os.path.join(save_dir, f)
#         convert_h5_format(input_file_path, output_file_path, precision=32)
#         print(f'Converted {f}.')

with Pool(7) as p:
    p.map(convert_h5_format, [(f'data/aligned_non_seeds/{f}', f'data/new_aligned_non_seeds/{f}') 
                              for f in os.listdir('data/aligned_non_seeds') if f.endswith('.h5')])
    