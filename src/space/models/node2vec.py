import warnings
import numba
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from pecanpy.cli import pecanpy
from loguru import logger
from space.tools.data import H5pyData, GzipData



class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self,total_epoch):
        self.epoch = 1
        self.loss_to_be_subed = 0
        self.total_epoch = total_epoch
        self.saved_loss = list()

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.saved_loss.append([self.epoch,loss_now])
        self.epoch += 1



def learn_embeddings(walks,epochs=1,dimensions=64,window_size=10,workers=1,random_state=1234,taxid="",writer=None,hs=0):
    """_summary_

    Args:
        epochs (int, optional): _description_. Defaults to 1.
        walks (int, optional): _description_. Defaults to 20.
        dimensions (int, optional): _description_. Defaults to 64.
        window_size (int, optional): _description_. Defaults to 10.
        workers (int, optional): _description_. Defaults to 1.
        random_state (int, optional): _description_. Defaults to 1234.
        output_path (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """    
    # cb = callback(epochs,taxid,writer) ## de-comment this line if you want to save the loss

    model = Word2Vec(
        walks,
        vector_size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,
        hs=hs,
        workers=workers,
        epochs=epochs,
        seed=random_state,
        compute_loss=True,
        # callbacks=[cb]  ## de-comment this line if you want to save the loss
    )

    return model





def check_mode(g, mode,weighted,p,q):
    """Check mode selection.

    Give recommendation to user for pecanpy mode based on graph size and density.

    """
    # mode = args.mode
    # weighted = args.weighted
    # p = args.p
    # q = args.q

    # Check unweighted first order random walk usage
    if mode == "FirstOrderUnweighted":
        if not p == q == 1 or weighted:
            raise ValueError(
                f"FirstOrderUnweighted only works when weighted = False and "
                f"p = q = 1, got {weighted=}, {p=}, {q=}",
            )
        return

    if mode != "FirstOrderUnweighted" and p == q == 1 and not weighted:
        warnings.warn(
            "When p = 1 and q = 1 with unweighted graph, it is highly "
            f"recommended to use FirstOrderUnweighted over {mode} (current "
            "selection). The runtime could be improved greatly with improved  "
            "memory usage.",
        )
        return

    # Check first order random walk usage
    if mode == "PreCompFirstOrder":
        if not p == q == 1:
            raise ValueError(
                f"PreCompFirstOrder only works when p = q = 1, got {p=}, {q=}",
            )
        return

    if mode != "PreCompFirstOrder" and p == 1 == q:
        warnings.warn(
            "When p = 1 and q = 1, it is highly recommended to use "
            f"PreCompFirstOrder over {mode} (current selection). The runtime "
            "could be improved greatly with low memory usage.",
        )
        return

    # Check network density and recommend appropriate mode
    g_size = g.num_nodes
    g_dens = g.density
    if (g_dens >= 0.2) & (mode != "DenseOTF"):
        warnings.warn(
            f"Network density = {g_dens:.3f} (> 0.2), it is recommended to use "
            f"DenseOTF over {mode} (current selection)",
        )
    if (g_dens < 0.001) & (g_size < 10000) & (mode != "PreComp"):
        warnings.warn(
            f"Network density = {g_dens:.2e} (< 0.001) with {g_size} nodes "
            f"(< 10000), it is recommended to use PreComp over {mode} (current "
            "selection)",
        )
    if (g_dens >= 0.001) & (g_dens < 0.2) & (mode != "SparseOTF"):
        warnings.warn(
            f"Network density = {g_dens:.3f}, it is recommended to use "
            f"SparseOTF over {mode} (current selection)",
        )
    if (g_dens < 0.001) & (g_size >= 10000) & (mode != "SparseOTF"):
        warnings.warn(
            f"Network density = {g_dens:.3f} (< 0.001) with {g_size} nodes "
            f"(>= 10000), it is recommended to use SparseOTF over {mode} "
            "(current selection)",
        )



def read_graph(path,p,q,workers,verbose,weighted,directed,extend,gamma,random_state,mode,delimiter,implicit_ids):
    """Read input network to memory.

    Depending on the mode selected, reads the network either in CSR
    representation (``PreComp`` and ``SparseOTF``) or 2d numpy array
    (``DenseOTF``).

    """

    if directed and extend:
        raise NotImplementedError("Node2vec+ not implemented for directed graph yet.")

    if extend and not weighted:
        print("NOTE: node2vec+ is equivalent to node2vec for unweighted graphs.")

    # if task in ["tocsr", "todense"]:  # perform conversion then save and exit
    #     g = graph.SparseGraph() if task == "tocsr" else graph.DenseGraph()
    #     g.read_edg(path, weighted, directed, delimiter)
    #     g.save(output)
    #     exit()

    pecanpy_mode = getattr(pecanpy, mode, None)
    g = pecanpy_mode(p, q, workers, verbose, extend, gamma, random_state)

    if path.endswith(".npz"):
        g.read_npz(path, weighted, implicit_ids=implicit_ids)
    else:
        g.read_edg(path, weighted, directed, delimiter)

    check_mode(g, mode,weighted,p,q)

    return g


def preprocess(g):
    """Preprocessing transition probabilities with timer."""
    g.preprocess_transition_probs()


def simulate_walks(num_walks, walk_length, g):
    """Simulate random walks with timer."""
    return g.simulate_walks(num_walks, walk_length)


class PecanpyEmbedder():

    def __init__(self,graph_path,p=1,q=1,workers=-1,
                 weighted=True,directed=False,
                 extend=False,gamma=0,random_state=1234,
                 delimiter:str='\t'):
        super().__init__()

        if workers == -1:
            workers = numba.config.NUMBA_DEFAULT_NUM_THREADS

        ## load the graph
        self.graph = read_graph(graph_path,p,q,workers,verbose=False,
                                weighted=weighted,directed=directed,
                                extend=extend,gamma=gamma,
                                random_state=random_state,
                                mode="SparseOTF",delimiter=delimiter,
                                implicit_ids=False)
        preprocess(self.graph)


    def generate_walks(self,num_walks:int,walk_length:int) -> list:
        return simulate_walks(num_walks,walk_length,self.graph)
        



    def learn_embeddings(self, walks, epochs=1,dimensions=128, 
                         window_size=5,workers=-1,
                         negative=5,
                         hs=0,sg=1,
                         random_state=1234) -> Word2Vec:
        """
            Word2Vec API of gensim

            Parameters
            ----------
            walks : list, list of walks
            epochs : int, number of epochs, default 1.
            dimensions : int, number of dimensions, default 128.
            window_size : int, window size, default 5.
            workers : int, number of workers, default -1 (all workers).
            negative : int, if >0, negative sampling will be used, number of negative samples if use negative sampling, default 5.
            hs : int, if 1, use hierarchical softmax; if 0, use negative sampling, default 0.
            sg : int, if 1, use skip-gram; if 0, use CBOW, default 1.
            random_state : int, random state, default 1234.
        """
       
        cb = callback(epochs)

        if workers == -1:
            workers = numba.config.NUMBA_DEFAULT_NUM_THREADS

        model = Word2Vec(
            walks,
            vector_size=dimensions,
            window=window_size,
            min_count=0,
            sg=sg,
            hs=hs,
            workers=workers,
            epochs=epochs,
            seed=random_state,
            compute_loss=True,
            callbacks=[cb],
            negative=negative
        )

        return model



def run_single_embedding(species_file, temp_path, output_folder, dimensions, 
                         p, q, num_walks, walk_length, window_size, sg, 
                         negative, epochs, workers, random_state):
    """Run single species-specific embedding."""
    ## process the gz file
    logger.info(f"Processing {species_file}...")
    nodes = GzipData.string2idx(species_file,temp_path)


    if len(nodes) > 50000:
        logger.warning(f"Number of nodes in {species_file} is {len(nodes)}, if it fails, try larger memory")

    logger.info(f"Running embedding for {species_file}...")
    # Read the graph
    embedder = PecanpyEmbedder(temp_path,p=p,q=q,workers=workers,weighted=True,directed=False,
                               extend=False,gamma=0,random_state=random_state,delimiter='\t')
    # Generate the walks

    embeddings = embedder.learn_embeddings(embedder.generate_walks(num_walks=num_walks,walk_length=walk_length),
                                      epochs=epochs,dimensions=dimensions,window_size=window_size,
                                      workers=workers,negative=negative,hs=0,sg=sg,random_state=random_state)
    
    emb = embeddings.wv.vectors
    index = embeddings.wv.index_to_key

    proteins = list(nodes.keys())

    ## map the index to the protein
    map_proteins = [proteins[int(i)] for i in index]

    ## save the embeddings
    species = species_file.split('/')[-1].split('.')[0]
    save_path = f"{output_folder}/{species}.h5"
    H5pyData.write(map_proteins,emb,save_path,32)
    logger.info(f"Embedding for {species_file} is saved at {save_path}")

    return None