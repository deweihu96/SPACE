#1. find out the min max of the embeddings
from space.tools.data import H5pyData
import numpy as np
from multiprocessing import Pool
import itertools
import sys

def min_max(filename):
    p,e = H5pyData.read(filename, 16)

    min_e = float(np.min(e))

    max_e = float(np.max(e))

    return min_e, max_e

def find_min_max(species_file,directory,num_jobs):

    species_list = open(species_file).read().strip().split('\n')

    with Pool(num_jobs) as p:
        results = p.map(min_max, [f'{directory}/{species}.h5' 
                                  for species in species_list])

    results = itertools.chain(*results)

    return min(results), max(results)


#2. scale the embeddings
def scale(filename, scale, save_dir):

    taxid = filename.split('/')[-1].split('.')[0]

    p,e = H5pyData.read(filename, 16)

    e = e*scale

    H5pyData.write(f'{save_dir}/{taxid}.h5', p, e)

    return None

def scale_all(species_file,directory,num_jobs,save_dir):

    print('Finding min and max values...')
    min_e, max_e = find_min_max(species_file,directory,num_jobs)
    print(f'Min: {min_e}, Max: {max_e}')
    
    scale = max(0.99/max_e, abs(0.99/min_e))

    print(f'Scaling factor: {scale}')

    species_list = open(species_file).read().strip().split('\n')

    print('Scaling embeddings...')
    with Pool(num_jobs) as p:
        scale = p.starmap(scale, [(f'{directory}/{species}.h5', 
                                   scale, save_dir) 
                                  for species in species_list])
    print('DONE.')

    return None


if __name__ == '__main__':

    species_file = sys.argv[1]

    directory = sys.argv[2]

    num_jobs = int(sys.argv[3])

    save_dir = sys.argv[4]
    
    scale_all(species_file,directory,num_jobs,save_dir)
