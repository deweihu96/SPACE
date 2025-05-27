from space.models.fedcoder import FedCoder
import argparse


def main(args):
    args = vars(args)
    fedcoder = FedCoder(**args)
    fedcoder.fit()

    fedcoder.save_embeddings()

    return None

if __name__ == '__main__':


    argparser = argparse.ArgumentParser(description='Align seed species')

    argparser.add_argument('--seed_species', type=str, default='data/seeds.txt',
                            help='Path to seed species file, including the taxon IDs of the species to align')
    
    argparser.add_argument('--node2vec_dir', type=str, default='data/node2vec',
                            help='Path to node2vec files')
    
    argparser.add_argument('--ortholog_dir', type=str,default='data/orthologs/seeds',
                            help='Path to eggnog group files')
    
    argparser.add_argument('--aligned_embedding_save_dir', type=str,default='results/aligned_embeddings',
                            help='Path to save embeddings')
    
    argparser.add_argument('--save_top_k', type=int, default=3,
                            help='Number of top moldels to save')
    
    argparser.add_argument('--log_dir', type=str, default='logs/seeds',
                            help='Path to save logs')
    
    argparser.add_argument('--input_dim', type=int, default=128,
                            help='Input dimension')
    
    argparser.add_argument('--latent_dim', type=int, default=512,
                            help='Latent dimension')
    
    argparser.add_argument('--hidden_dims', type=int, default=None,
                            help='Hidden dimension')
    
    argparser.add_argument('--activation_fn', type=str, default=None,
                            help='Activation function')
    
    argparser.add_argument('--batch_norm', type=bool, default=False,
                            help='Batch normalization')
    
    argparser.add_argument('--number_iters', type=int, default=10,
                            help='Number of iterations per epoch')
    
    argparser.add_argument('--autoencoder_type', type=str, default='naive',
                            help='Type of autoencoder')
    
    argparser.add_argument('--gamma', type=float, default=0.1,
                            help='Margin of the alignment loss')
    
    argparser.add_argument('--alpha', type=float, default=0.5,
                            help='Balance between reconstruction and alignment (1-alpha) loss')
    
    argparser.add_argument('--lr', type=float, default=1e-2,
                            help='Learning rate')
    
    argparser.add_argument('--device', type=str, default='cpu',
                            help='Device to train on')
    
    argparser.add_argument('--patience', type=int, default=5,
                            help='Patience for early stopping')
    
    argparser.add_argument('--delta', type=float, default=1e-4,
                            help='Delta for early stopping')

    argparser.add_argument('--epochs', type=int, default=500,
                            help='Number of maximum epochs')
    
    argparser.add_argument('--from_pretrained', type=str, default=None,
                            help='Path to pretrained model')

    args = argparser.parse_args()

    main(args)