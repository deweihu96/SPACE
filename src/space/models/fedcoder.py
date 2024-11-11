from space.tools.data import H5pyData
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
import os
from itertools import combinations, product
from typing import Iterable
from loguru import logger
import yaml
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd
from uuid import uuid4

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Encoder(nn.Module):
    def __init__(self,input_dim,latent_dim,hidden_dims:list=None,activation_function=None) -> None:
        super(Encoder,self).__init__()

        self.hidden_dims = hidden_dims
        self.activation_function = activation_function

        self.hidden_layers = nn.ModuleList()
        ## create hidden layers
        if self.hidden_dims is not None:
            for h_dim in self.hidden_dims:
                self.hidden_layers.append(nn.Linear(input_dim,h_dim))
                input_dim = h_dim
        
        self.l1 = nn.Linear(input_dim,latent_dim)

    def forward(self,x):

        if self.hidden_dims is not None:
            if self.activation_function is not None:
                for layer in self.hidden_layers:
                    x = self.activation_function(layer(x))
            else:
                for layer in self.hidden_layers:
                    x = layer(x)
        x = self.l1(x)
        return x

class Decoder(nn.Module):
    def __init__(self,latent_dim,input_dim,hidden_dims:list=None,activation_function=None) -> None:
        super(Decoder,self).__init__()

        self.hidden_dims = hidden_dims
        self.activation_function = activation_function

        self.hidden_layers = nn.ModuleList()
        ## create hidden layers, reverse order
        if self.hidden_dims is not None:
            for h_dim in reversed(self.hidden_dims):
                self.hidden_layers.append(nn.Linear(latent_dim,h_dim))
                latent_dim = h_dim
        
        self.l1 = nn.Linear(latent_dim,input_dim)

    def forward(self,z):
        
        if self.hidden_dims is not None:
            if self.activation_function is not None:
                for layer in self.hidden_layers:
                    z = self.activation_function(layer(z))
            else:
                for layer in self.hidden_layers:
                    z = layer(z)
        z = self.l1(z)
        return z


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, activation_function, batch_norm, ):
        super(VAEEncoder, self).__init__()
        self.layers = nn.ModuleList()
        if batch_norm:
            self.batch_norms = nn.ModuleList()
        self.activation_function = activation_function
        self.batch_norm = batch_norm

        # Create layers
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, h_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(h_dim))
            input_dim = h_dim

        self.fc_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    

    def xavier_init(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc_mean.weight)
        torch.nn.init.xavier_uniform_(self.fc_log_var.weight)

    def forward(self, x):
        
        if self.batch_norm:
            for layer, batch_norm in zip(self.layers, self.batch_norms):
                if self.activation_function:
                    x = self.activation_function(batch_norm(layer(x)))
                else:
                    x = batch_norm(layer(x))
        else:
            for layer in self.layers:
                if self.activation_function:
                    x = self.activation_function(layer(x))
                else:
                    x = layer(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return self.reparameterize(mean, log_var)
    


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dims, activation_function, batch_norm):
        super(VAEDecoder, self).__init__()
        self.layers = nn.ModuleList()
        if batch_norm:
            self.batch_norms = nn.ModuleList()
        self.activation_function = activation_function
        self.batch_norm = batch_norm
        # Create layers
        for h_dim in reversed(hidden_dims):
            self.layers.append(nn.Linear(latent_dim, h_dim))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(h_dim))
            latent_dim = h_dim

        self.final_layer = nn.Linear(hidden_dims[0], output_dim)
        self.xavier_init()

    
    def xavier_init(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)
    

    def forward(self, z):
        if self.batch_norm:
            for layer, batch_norm in zip(self.layers, self.batch_norms):
                if self.activation_function:
                    z = self.activation_function(batch_norm(layer(z)))
                else:
                    z = batch_norm(layer(z))
                
        else:
            for layer in self.layers:
                if self.activation_function:
                    z = self.activation_function(layer(z))
                else:
                    z = layer(z)
        reconstruction = torch.sigmoid(self.final_layer(z))
        return reconstruction



class BaseVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BaseVAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2*latent_dim)
        self.fc2_mean = nn.Linear(2*latent_dim, latent_dim)
        self.fc2_log_var = nn.Linear(2*latent_dim, latent_dim)
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        log_var = self.fc2_log_var(x)
        # z = self.reparameterize(mean, log_var)
        return self.reparameterize(mean, log_var)

class BaseVAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(BaseVAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 2*latent_dim)
        self.fc2 = nn.Linear(2*latent_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        reconstruction = torch.sigmoid(self.fc2(z))
        return reconstruction
    
def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)



class EarlyStopping():

    def __init__(self, patience=5, delta=0.0001, save_models=True):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_models = save_models

    def save_checkpoint(self, models:dict, save_name:str):
        '''Saves model when the metric improves.'''
        logger.info(f'Saving the checkpoint to {save_name}')
        torch.save(models.state_dict(), save_name)

        return True
    
    def reset(self):
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss, model:dict, save_folder:str):
        if self.best_score is None:
            self.best_score = loss
            if self.save_models:
                self.save_checkpoint(model, save_folder)
        elif loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = loss
            if self.save_models:
                self.save_checkpoint(model, save_folder)
            self.counter = 0


class NodeEmbedData(Dataset):

    def __init__(self,file_path) -> None:
        super().__init__()

        self.protein_names, self.embedding = self.read_h5_file(file_path)
    
    def read_h5_file(self,file_path):

        # embed_np = np.load(file_path)['data']
        proteins,embed_np = H5pyData.read(file_path,32)
        return proteins,torch.Tensor(embed_np).requires_grad_(False)

    def __len__(self):

        return len(self.embedding)

    def __getitem__(self, index):

        return self.embedding[index]
    
class OrthologPair:

    def __init__(self,
                 orthologs:Iterable,):
        
        self.pairs, self.weights = self.load_ortholog_pairs(orthologs)


    def __len__(self):

        return self.pairs.shape[0]
    
    def __getitem__(self, index):

        src_idx = self.pairs[:,0][index]
        tgt_idx = self.pairs[:,1][index]

        weight = self.weights[index]

        return src_idx, tgt_idx, weight
    

    def load_ortholog_pairs(self,pairs_file:str):
        pairs_weights = [pair.strip().split('\t') for pair in open(pairs_file,'r').readlines()]
        pairs = np.array(pairs_weights)[:, :-1].astype(int)
        pairs = torch.tensor(pairs).requires_grad_(False)

        weights = np.array(pairs_weights)[:, -1].astype(float)
        weights = torch.tensor(weights).requires_grad_(False)

        return pairs, weights
    

class FedCoder:
    # add the docstring here
    '''
    Parameters
    ----------
    seed_species : str
        The file containing the seed species taxonomy ids.
    node2vec_dir : str
        The directory containing the node2vec embeddings.
    ortholog_dir : str
        The directory containing the ortholog pairs.
    embedding_save_folder : str
        The directory to save the embeddings.
    save_top_k : int, optional
        The number of top models to save, by default 3.
    log_dir : str, optional
        The directory to save the logs, by default None.
    input_dim : int, optional
        The input dimension of the autoencoder, by default 128.
    latent_dim : int, optional
        The latent dimension of the autoencoder, by default 512.
    hidden_dims : list, optional
        The hidden dimensions of the autoencoder, by default None.
    activation_fn : str, optional
        The activation function of the autoencoder, by default None. Only useful when hidden_dims is not None.
    batch_norm : bool, optional
        Whether to use batch normalization, by default False. Only useful when hidden_dims is not None.
    number_iters : int, optional
        The number of iterations per epoch to train the model, by default 10.
    autoencoder_type : str, optional
        The type of autoencoder to use, by default 'naive'.
    gamma : float, optional
        The gamma parameter for the alignment loss, by default 0.1.
    alpha : float, optional
        The alpha parameter for balancing the alignment loss and reconstruction loss, by default 0.5.
    lr : float, optional
        The learning rate, by default 0.01.
    device : str, optional
        The device to use, by default 'cpu'.
    patience : int, optional
        The patience for early stopping, by default 5.
    delta : float, optional
        The delta parameter for early stopping, by default 0.001.
    epochs : int, optional
        The number of epochs to train the model, by default 400.
    from_pretrained : str, optional
        The path to the pretrained model, by default None.
    '''

    def __init__(self,seed_species:str,
                 node2vec_dir:str,
                 ortholog_dir:str,
                 embedding_save_folder:str,
                 save_top_k:int=3,
                 log_dir:str=None,
                 input_dim:int=128,
                 latent_dim:int=512,
                 hidden_dims:list=None,
                 activation_fn:str=None,
                 batch_norm:bool=False,
                 number_iters:int=10,
                 autoencoder_type:str='naive',
                 gamma:float=0.1,
                 alpha:float=0.5,
                 lr:float=0.01,
                 device:str='cpu',
                 patience:int=5,
                 delta:float=0.0001,
                 epochs:int=600,
                 from_pretrained:str=None,
                 ) -> None:
        

        seed_species = open(seed_species,'r').read().strip().split('\n')
        self.seed_species = list(map(int,seed_species))

        self.node2vec_dir = node2vec_dir
        self.ortholog_dir = ortholog_dir
        self.embedding_save_folder = embedding_save_folder
        self.save_top_k = save_top_k
        log_dir = log_dir if log_dir is not None else os.path.join(embedding_save_folder,'logs')
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.log_dir = log_dir + '-' + str(uuid4())
        self.model_save_path = os.path.join(self.log_dir,'model.pth')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.embedding_save_folder):
            os.makedirs(self.embedding_save_folder)
        
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.number_iters = number_iters
        self.autoencoder_type = autoencoder_type
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.device = device
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.from_pretrained = from_pretrained

    def save_hyperparameters(self,save_dict,save_dir):
        ## save as a yaml file
        with open(f'{save_dir}/hyperparameters.yaml','w') as f:
            yaml.dump(save_dict,f)
        return None
    

    def init_everything(self):
        
        def init_autoencoder(input_dim,latent_dim,hidden_dims,activation_fn,batch_norm,device):
            # to simply, use a function to initialize the autoencoder
            if self.autoencoder_type == 'naive':
                return Encoder(input_dim,latent_dim,hidden_dims,activation_fn).to(device), \
                    Decoder(latent_dim,input_dim,hidden_dims,activation_fn).to(device)
            elif self.autoencoder_type == 'vae':
                return VAEEncoder(input_dim,latent_dim,hidden_dims,activation_fn,batch_norm).to(device), \
                    VAEDecoder(latent_dim,input_dim,hidden_dims,activation_fn,batch_norm).to(device)
            else:
                raise ValueError('Unknown autoencoder type')
            
        ## load the node2vec embeddings
        logger.info('Loading the node2vec embeddings')
        self.node2vec_embeddings = dict()
        for species in self.seed_species:
            self.node2vec_embeddings[str(species)] = NodeEmbedData(f'{self.node2vec_dir}/{species}.h5')
        ## dataloader
        self.node2vec_dataloader = {species:DataLoader(embed, 
                                                       batch_size=math.ceil(len(embed)/self.number_iters), 
                                                       shuffle=True) 
                                    for species,embed in self.node2vec_embeddings.items()}


        ## load the ortholog pairs
        logger.info('Loading the ortholog pairs')
        species_pairs = list(combinations(self.seed_species,2))
        # sort the species pairs
        species_pairs = [sorted(pair) for pair in species_pairs]
        self.ortholog_pairs = dict()
        for src,tgt in species_pairs:
            self.ortholog_pairs[f'{src}_{tgt}'] = OrthologPair(f'{self.ortholog_dir}/{src}_{tgt}.tsv')
        ## dataloader
        self.ortholog_dataloader = {pair:DataLoader(ortholog, 
                                                    batch_size=math.ceil(len(ortholog)/self.number_iters), 
                                                    shuffle=True) 
                                    for pair,ortholog in self.ortholog_pairs.items()}

        ## init the models: {'encoder_species':encoder,'decoder_species':decoder}
        logger.info('Initializing the models')
        self.models = dict()
        for species in self.seed_species:
            encoder, decoder = init_autoencoder(self.input_dim,self.latent_dim,self.hidden_dims,self.activation_fn,self.batch_norm,self.device)
            self.models[f'encoder_{species}'] = encoder
            self.models[f'decoder_{species}'] = decoder
        self.models = torch.nn.ModuleDict(self.models)
        
        if self.from_pretrained is not None:
            logger.info(f'Loading the pretrained model from {self.from_pretrained}')
            self.models.load_state_dict(torch.load(self.from_pretrained))
        else:
            for model in self.models.values():
                model.apply(init_xavier)


        ## init the optimizers
        self.parameters = []
        for model in self.models.values():
            self.parameters += list(model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters,lr=self.lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min',patience=3,verbose=True,factor=0.1)

        self.early_stopping = EarlyStopping(patience=self.patience,delta=self.delta,save_models=True)

        # init the tensorboard writer
        logger.info('Initializing the tensorboard writer')
        
        self.writer = SummaryWriter(self.log_dir)

        logger.info('Everything initialized')

        hyperparameters = { 'seed_species':self.seed_species,
                            'node2vec_dir':self.node2vec_dir,
                            'ortholog_dir':self.ortholog_dir,
                            'embedding_save_folder':self.embedding_save_folder,
                            'model_save_path':self.model_save_path,
                            'save_top_k':self.save_top_k,
                            'log_dir':self.log_dir,
                            'input_dim':self.input_dim,
                            'latent_dim':self.latent_dim,
                            'hidden_dims':self.hidden_dims,
                            'activation_fn':self.activation_fn,
                            'batch_norm':self.batch_norm,
                            'number_iters':self.number_iters,
                            'autoencoder_type':self.autoencoder_type,
                            'gamma':self.gamma,
                            'alpha':self.alpha,
                            'lr':self.lr,
                            'device':self.device,
                            'patience':self.patience,
                            'delta':self.delta,
                            'epochs':self.epochs,
                            'from_pretrained':self.from_pretrained,}
        self.save_hyperparameters(hyperparameters,self.log_dir)

        return None


    def reconstruction_loss(self,node_batches):
        loss = list()
        for taxid, batch in node_batches.items():
            
            batch = batch.to(self.device)

            latent = self.models[f'encoder_{taxid}'](batch)

            reconstruction = self.models[f'decoder_{taxid}'](latent)

            loss.append(F.pairwise_distance(batch,reconstruction,p=2).mean().unsqueeze(0))
        
        return torch.cat(loss).mean()


    def alignment_loss(self,pair_batches):
        
        loss = list()

        for src_tgt, (src_index,tgt_index,weight) in pair_batches.items():

            src,tgt = src_tgt.split('_')

            src_batch = self.node2vec_embeddings[src][src_index]
            tgt_batch = self.node2vec_embeddings[tgt][tgt_index]

            src_batch = src_batch.to(self.device)
            tgt_batch = tgt_batch.to(self.device)

            src_latent = self.models[f'encoder_{src}'](src_batch)
            tgt_latent = self.models[f'encoder_{tgt}'](tgt_batch)

            src_tgt_loss = F.pairwise_distance(src_latent,tgt_latent,p=2)

            weight = weight.to(self.device)

            src_tgt_loss = (-F.logsigmoid(self.gamma - src_tgt_loss)*weight).mean().unsqueeze(0)

            loss.append(src_tgt_loss)
        
        return torch.cat(loss).mean()

    def one_epoch(self,crt_epoch):

        loss_dict = {'epoch_loss':0,'reconstruction_loss':0,'alignment_loss':0} 

        node_iterators = {taxid: iter(loader) for taxid, loader in self.node2vec_dataloader.items()}
        pair_iterators = {src_tgt: iter(loader) for src_tgt, loader in self.ortholog_dataloader.items()}

        for iter_ in tqdm(range(self.number_iters)):
            
            node_batches = {taxid: next(node_iterators[str(taxid)],None) for taxid in self.seed_species}

            pair_batches = {src_tgt: next(pair_iterators[src_tgt],None) for src_tgt in self.ortholog_pairs.keys()}
            
            self.optimizer.zero_grad()

            reconstruction_loss = self.reconstruction_loss(node_batches) * self.alpha

            alignment_loss = self.alignment_loss(pair_batches) * (1-self.alpha)

            loss = reconstruction_loss + alignment_loss

            loss_dict['reconstruction_loss'] += reconstruction_loss.item()
            loss_dict['alignment_loss'] += alignment_loss.item()
            loss_dict['epoch_loss'] += loss.item()

            loss.backward()

            self.optimizer.step()

        # log the losses
        for key,value in loss_dict.items():
            self.writer.add_scalar(key,value,crt_epoch+1)
        
        logger.info(f'Epoch {crt_epoch+1} loss: {loss_dict["epoch_loss"]}\n \
                    reconstruction loss: {loss_dict["reconstruction_loss"]}\n \
                    alignment loss: {loss_dict["alignment_loss"]}')

        return tuple(loss_dict.values())

            
    def fit(self):
        self.init_everything()

        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch+1}')
            epoch_loss, reconstruction_loss, alignment_loss = self.one_epoch(epoch)
            self.scheduler.step(alignment_loss,epoch)
        
            logger.info('Evaluating the model')
            self.eval_hitsk_epoch_end(epoch+1)

            ## save the best model
            self.early_stopping(alignment_loss,self.models,self.model_save_path)

            if self.early_stopping.early_stop:
                logger.info('Early stopping')
                break
        
        logger.info(f'Training completed after {epoch+1} epochs')

        return None


    @torch.no_grad()
    def save_embeddings(self,species:int=None):
        
        if species is not None:
            species = [str(species)]
        else:
            species = self.seed_species
        
        for taxid in species:
            taxid = str(taxid)
            encoder = self.models[f'encoder_{taxid}']
            data = self.node2vec_embeddings[str(taxid)].embedding
            encoder.eval()

            data = data.to(self.device)

            latent = encoder(data)

            latent = latent.cpu().detach().numpy()

            H5pyData.write(self.node2vec_embeddings[taxid].protein_names,
                           latent,
                           f'{self.embedding_save_folder}/{taxid}.h5',
                           16)
        return None

    


class FedCoderNonSeed(FedCoder):
    # docstring here
    '''
    Parameters
    ----------
    seed_groups : str
        The file containing the seed species taxonomy ids.
    tax_group : str
        The file containing the taxonomy groups.
    non_seed_species : str|int
        The non seed species taxonomy id.
    node2vec_dir : str
        The directory containing the node2vec embeddings.
    aligned_dir : str   
        The directory containing the aligned embeddings.
    ortholog_dir : str  
        The directory containing the ortholog pairs.
    embedding_save_folder : str
        The directory to save the embeddings.
    save_top_k : int, optional
        The number of top models to save, by default 3.
    log_dir : str, optional
        The directory to save the logs, by default None.
    input_dim : int, optional
        The input dimension of the autoencoder, by default 128.
    latent_dim : int, optional
        The latent dimension of the autoencoder, by default 512.
    hidden_dims : list, optional    
        The hidden dimensions of the autoencoder, by default None.
    activation_fn : str, optional
        The activation function of the autoencoder, by default None. Only useful when hidden_dims is not None.
    batch_norm : bool, optional
        Whether to use batch normalization, by default False. Only useful when hidden_dims is not None.
    number_iters : int, optional
        The number of iterations per epoch to train the model, by default 10.
    autoencoder_type : str, optional
        The type of autoencoder to use, by default 'naive'.
    gamma : float, optional
        The gamma parameter for the alignment loss, by default 0.1.
    alpha : float, optional
        The alpha parameter for balancing the alignment loss and reconstruction loss, by default 0.5.
    lr : float, optional
        The learning rate, by default 0.01.
    device : str, optional
        The device to use, by default 'cpu'.
    patience : int, optional
        The patience for early stopping, by default 5.
    delta : float, optional
        The delta parameter for early stopping, by default 0.0001.
    epochs : int, optional
        The number of epochs to train the model, by default 600.
    from_pretrained : str, optional
        The path to the pretrained model, by default None.
    '''

    def __init__(self,seed_groups:str,
                 tax_group:str,
                 non_seed_species:str|int,
                 node2vec_dir:str,
                 aligned_dir:str,
                 ortholog_dir:str,
                 embedding_save_folder:str,
                 save_top_k:int=3,
                 log_dir:str=None,
                 input_dim:int=128,
                 latent_dim:int=512,
                 hidden_dims:list=None,
                 activation_fn:str=None,
                 batch_norm:bool=False,
                 number_iters:int=10,
                 autoencoder_type:str='naive',
                 gamma:float=0.1,
                 alpha:float=0.5,
                 lr:float=0.01,
                 device:str='cpu',
                 patience:int=5,
                 delta:float=0.0001,
                 epochs:int=600,
                 from_pretrained:str=None,
                 ) -> None:

        self.non_seed_species = int(non_seed_species)

        self.seed_groups = json.load(open(seed_groups,'r'))
        tax_group = pd.read_csv(tax_group,sep='\t')

        seed_species = tax_group[tax_group['taxid']==self.non_seed_species]['group'].values[0]
        self.seed_species = list(map(int,self.seed_groups[seed_species]))

        self.node2vec_dir = node2vec_dir
        self.aligned_dir = aligned_dir
        self.ortholog_dir = f'{ortholog_dir}/{self.non_seed_species}'
        self.embedding_save_folder = embedding_save_folder
        self.save_top_k = save_top_k
        log_dir = log_dir if log_dir is not None else os.path.join(embedding_save_folder,'logs')
        log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        log_dir = log_dir + '-' + str(uuid4())
        self.log_dir = log_dir
        self.model_save_path = os.path.join(self.log_dir,'model.pth')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.embedding_save_folder):
            os.makedirs(self.embedding_save_folder)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.number_iters = number_iters
        self.autoencoder_type = autoencoder_type
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.device = device
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.from_pretrained = from_pretrained
    

    def init_everything(self):
        def init_autoencoder(input_dim,latent_dim,hidden_dims,activation_fn,batch_norm,device):
            # to simply, use a function to initialize the autoencoder
            if self.autoencoder_type == 'naive':
                return Encoder(input_dim,latent_dim,hidden_dims,activation_fn).to(device), \
                    Decoder(latent_dim,input_dim,hidden_dims,activation_fn).to(device)
            elif self.autoencoder_type == 'vae':
                return VAEEncoder(input_dim,latent_dim,hidden_dims,activation_fn,batch_norm).to(device), \
                    VAEDecoder(latent_dim,input_dim,hidden_dims,activation_fn,batch_norm).to(device)
            else:
                raise ValueError('Unknown autoencoder type')


        ## load the node2vec embeddings
        logger.info('Loading the node2vec embeddings')
        self.node2vec_embeddings = dict()
        for species in tqdm(self.seed_species):
            self.node2vec_embeddings[str(species)] = NodeEmbedData(f'{self.aligned_dir}/{species}.h5')
        self.node2vec_embeddings[str(self.non_seed_species)] = NodeEmbedData(f'{self.node2vec_dir}/{self.non_seed_species}.h5')
        ## dataloader
        # only need the dataloader for non seed species
        self.node2vec_dataloader = {str(self.non_seed_species):
                                    DataLoader(self.node2vec_embeddings[str(self.non_seed_species)],
                                            batch_size=math.ceil(len(self.node2vec_embeddings[str(self.non_seed_species)])/self.number_iters),
                                            shuffle=True)}
        
        ## load the ortholog pairs
        logger.info('Loading the ortholog pairs')
        species_pairs = list(product(self.seed_species,[self.non_seed_species]))
        # sort the species pairs
        species_pairs = [sorted(pair) for pair in species_pairs]
        self.ortholog_pairs = dict()
        for src,tgt in species_pairs:
            self.ortholog_pairs[f'{src}_{tgt}'] = OrthologPair(f'{self.ortholog_dir}/{src}_{tgt}.tsv')

        ## dataloader
        self.ortholog_dataloader = {pair:DataLoader(ortholog, 
                                                    batch_size=math.ceil(len(ortholog)/self.number_iters), 
                                                    shuffle=True) 
                                    for pair,ortholog in self.ortholog_pairs.items()}
        ## init the models: {'encoder_species':encoder,'decoder_species':decoder}
        logger.info('Initializing the models')
        self.models = dict()
        encoder, decoder = init_autoencoder(self.input_dim,self.latent_dim,self.hidden_dims,self.activation_fn,self.batch_norm,self.device)
        self.models[f'encoder_{self.non_seed_species}'] = encoder
        self.models[f'decoder_{self.non_seed_species}'] = decoder
        self.models = torch.nn.ModuleDict(self.models)
        
        if self.from_pretrained is not None:
            logger.info(f'Loading the pretrained model from {self.from_pretrained}')
            self.models.load_state_dict(torch.load(self.from_pretrained))
        else:
            for model in self.models.values():
                model.apply(init_xavier)

        ## init the optimizers
        self.parameters = []
        for model in self.models.values():
            self.parameters += list(model.parameters())

        self.optimizer = torch.optim.Adam(self.parameters,lr=self.lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer,'min',patience=3,verbose=True,factor=0.1)

        self.early_stopping = EarlyStopping(patience=self.patience,delta=self.delta,save_models=True)

        # init the tensorboard writer
        logger.info('Initializing the tensorboard writer')
        
        self.writer = SummaryWriter(self.log_dir)

        logger.info('Everything initialized')

        hyperparameters = { 'seed_species':self.seed_species,
                            'node2vec_dir':self.node2vec_dir,
                            'ortholog_dir':self.ortholog_dir,
                            'embedding_save_folder':self.embedding_save_folder,
                            'model_save_path':self.model_save_path,
                            'save_top_k':self.save_top_k,
                            'log_dir':self.log_dir,
                            'input_dim':self.input_dim,
                            'latent_dim':self.latent_dim,
                            'hidden_dims':self.hidden_dims,
                            'activation_fn':self.activation_fn,
                            'batch_norm':self.batch_norm,
                            'number_iters':self.number_iters,
                            'autoencoder_type':self.autoencoder_type,
                            'gamma':self.gamma,
                            'alpha':self.alpha,
                            'lr':self.lr,
                            'device':self.device,
                            'patience':self.patience,
                            'delta':self.delta,
                            'epochs':self.epochs,
                            'from_pretrained':self.from_pretrained,}
        self.save_hyperparameters(hyperparameters,self.log_dir)

        return None
    

    def alignment_loss(self, pair_batches):
        
        loss = list()

        for src_tgt, (src_index,tgt_index,weight) in pair_batches.items():

            src,tgt = src_tgt.split('_')

            src_batch = self.node2vec_embeddings[src][src_index].to(self.device)
            tgt_batch = self.node2vec_embeddings[tgt][tgt_index].to(self.device)

            ## check if src or tgt is the non seed species
            if int(src) == self.non_seed_species:
                src_latent = self.models[f'encoder_{self.non_seed_species}'](src_batch)
                tgt_latent = tgt_batch
            else:
                src_latent = src_batch
                tgt_latent = self.models[f'encoder_{self.non_seed_species}'](tgt_batch)

            src_tgt_loss = F.pairwise_distance(src_latent,tgt_latent,p=2)

            weight = weight.to(self.device)

            src_tgt_loss = (-F.logsigmoid(self.gamma - src_tgt_loss)*weight)

            loss.append(src_tgt_loss)
        
        return torch.cat(loss).mean().unsqueeze(0)
    

    def one_epoch(self,crt_epoch):

        loss_dict = {'epoch_loss':0,'reconstruction_loss':0,'alignment_loss':0} 

        node_iterators = {taxid: iter(loader) for taxid, loader in self.node2vec_dataloader.items()}
        pair_iterators = {src_tgt: iter(loader) for src_tgt, loader in self.ortholog_dataloader.items()}

        for iter_ in tqdm(range(self.number_iters)):
            
            node_batches = {self.non_seed_species: next(node_iterators[str(self.non_seed_species)],None)}

            pair_batches = {src_tgt: next(pair_iterators[src_tgt],None) for src_tgt in self.ortholog_pairs.keys()}
            
            self.optimizer.zero_grad()

            reconstruction_loss = self.reconstruction_loss(node_batches) * self.alpha

            alignment_loss = self.alignment_loss(pair_batches) * (1-self.alpha)

            loss = reconstruction_loss + alignment_loss

            loss_dict['reconstruction_loss'] += reconstruction_loss.item()
            loss_dict['alignment_loss'] += alignment_loss.item()
            loss_dict['epoch_loss'] += loss.item()

            loss.backward()

            self.optimizer.step()

        # log the losses
        for key,value in loss_dict.items():
            self.writer.add_scalar(key,value,crt_epoch+1)
        
        logger.info(f'Epoch {crt_epoch+1} loss: {loss_dict["epoch_loss"]}\n \
                    reconstruction loss: {loss_dict["reconstruction_loss"]}\n \
                    alignment loss: {loss_dict["alignment_loss"]}')

        return tuple(loss_dict.values())
    
    def fit(self):
        self.init_everything()

        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch+1}')
            epoch_loss, reconstruction_loss, alignment_loss = self.one_epoch(epoch)
            self.scheduler.step(alignment_loss,epoch)

            ## save the best model
            self.early_stopping(alignment_loss,self.models,self.model_save_path)

            if self.early_stopping.early_stop:
                logger.info('Early stopping')
                break
        
        logger.info(f'Training completed after {epoch+1} epochs')
        
        return None
    

    ## change the default to save only the non seed species
    @torch.no_grad()
    def save_embeddings(self, species: int=None):

        if species is not None:
            species = [str(species)]
        else:
            species = [str(self.non_seed_species)]
        
        encoder = self.models[f'encoder_{self.non_seed_species}']

        data = self.node2vec_embeddings[str(self.non_seed_species)].embedding

        encoder.eval()

        data = data.to(self.device)

        latent = encoder(data)

        latent = latent.cpu().detach().numpy()

        H5pyData.write(self.node2vec_embeddings[str(self.non_seed_species)].protein_names,
                          latent,
                          f'{self.embedding_save_folder}/{self.non_seed_species}.h5',
                          16)
        return None
