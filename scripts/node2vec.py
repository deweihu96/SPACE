'''
    This script is used to run the node2vec algorithm on the STRING functional networks.
'''

import argparse
import os
from space.models.node2vec import run_single_embedding


def main(args):

    species_name = os.path.basename(args.input_network).split(".")[0]

    # create a temporary file
    temp_path = f"{args.node2vec_output}/{species_name}.tsv"

    # run the single embedding
    run_single_embedding(args.input_network, temp_path, args.node2vec_output, args.dimensions, 
                        args.p, args.q, args.num_walks, args.walk_length, args.window_size, args.sg, 
                        args.negative, args.epochs, args.workers, args.random_state)

    # remove the temporary file
    os.remove(temp_path)

    return None


if __name__ == "__main__":


    parser_single_embedding = argparse.ArgumentParser(description="Run node2vec on STRING functional networks with PecanPy.")

    parser_single_embedding.add_argument('-i','--input_network', type=str, help='File to run the\
                                        embedding for, e.g. <species_name>.tsv.gz. During running, the file will be processed into another temporary tsv file,\n\
                                        proteins will be replaced by integers, and the scores will be converted to float (0~1). Once finished, the temporary file will be deleted.')
                                      
    parser_single_embedding.add_argument('-o','--node2vec_output', type=str, help='Path to the output folder to save the embeddings.\n\
                                         The embeddings will be saved in the format: <output_folder>/<species_name>.h5')
    
    
    ### model parameters, optional
    parser_single_embedding.add_argument('-d', '--dimensions', type=int, default=128,help='The number of dimensions for the embedding.')
    parser_single_embedding.add_argument('-p', '--p', type=float, default=0.3, help='The return parameter for the random walk.')
    parser_single_embedding.add_argument('-q', '--q', type=float, default=0.7, help='The in-out parameter for the random walk.')
    parser_single_embedding.add_argument('--num_walks', type=int, default=10, help='The number of walks to perform.')
    parser_single_embedding.add_argument('--walk_length', type=int, default=50,help='The length of the walk.')
    parser_single_embedding.add_argument('--window_size', type=int, default=5, help='The window size for the skip-gram model.')
    parser_single_embedding.add_argument('--sg', type=int, default=1, help='The type of training to use for the skip-gram model. 0 for cbow, 1 for skip-gram.')
    parser_single_embedding.add_argument('--negative', type=int, default=5, help='The number of negative samples to use for training the model.')
    parser_single_embedding.add_argument('-e', '--epochs', default=5, type=int, help='The number of epochs to train the model.')
    parser_single_embedding.add_argument('--workers', type=int, default=-1, help='The number of workers to use for training the model.')
    parser_single_embedding.add_argument('--random_state', type=int, default=1234, help='The random state to use for the random number generator.')

    args = parser_single_embedding.parse_args()

    main(args)
