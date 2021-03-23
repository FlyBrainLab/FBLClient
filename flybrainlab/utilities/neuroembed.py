import os
from time import time
import pickle
import shutil

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from graspy.embed import AdjacencySpectralEmbed
from .neurometry import *
try:
    from gem.utils import graph_util, plot_util
    from gem.evaluation import visualize_embedding as viz
    from gem.evaluation import evaluate_graph_reconstruction as gr
    from gem.embedding.gf       import GraphFactorization
    from gem.embedding.hope     import HOPE
    from gem.embedding.lap      import LaplacianEigenmaps
    from gem.embedding.lle      import LocallyLinearEmbedding
    from gem.embedding.node2vec import node2vec
    if shutil.which('node2vec') is None:
        N2VC_available = False
        print('node2vec executable not found. Embedding using node2vec method are not available. Please compile and install from https://github.com/snap-stanford/snap/tree/master/examples/node2vec')
    else:
        N2VC_available = True
    from gem.embedding.sdne     import SDNE
except:
    print('GEM not installed. Please install GEM to enable embedding generation.')
try:
    import umap
except:
    print('UMAP not installed. Please install UMAP to enable embedding visualization.')
    
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler
except:
    print('Tensorflow not installed. Please install Tensorflow to enable GPU-based embedding algorithms.')

from ..graph import NeuronGraph, NeuroNLPResult, NAqueryResult


def construct_connectome_morphology_matrix(G,S):
    """Combines adjacency matrix and morphological features.
    
    # Arguments
        G (networkx Graph): Graph of connectome to use.
        S (numpy array): Input matrix of morphological features.
    # Returns
        numpy array: Combined adjacency matrix-morphological feature matrix.
    """
    W = nx.adjacency_matrix(G).todense()
    X = np.hstack((W,S))
    return X

def spherical_embeddings(Z, n_epochs = 10, batch_size = 128, n_components = 32):
    inputs = layers.Input(shape=(n_components))
    x = layers.Dense(3)(inputs)
    x = layers.LayerNormalization(center = False, scale = False)(x)
    x = layers.Dense(n_components)(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    history = model.fit(Z, Z, epochs = n_epochs, batch_size = batch_size)
    return history, model

def generate_spherical_embeddings_for_adjacency(X, n_components = 32, n_epochs = 10, batch_size = 128):
    """Given an adjacency matrix, first (i) embed the adjacency to a low-dimensional space using PCA, 
    (ii) min-max scale features, (iii) generate spherical embeddings using Tensorflow.
    
    # Arguments
        X (numpy array): Input adjacency matrix.
        n_components (int): Number of PCA components. Defaults to 32.
        n_epochs (int): Number of epochs to run the embeddings for. Defaults to 10.
        batch_size (int): Batch size to use for the embedding generator. Defaults to 128.
    # Returns
        History object: History object of the spherical embedding training.
        Model: Model to use.
    """
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    A = MinMaxScaler().fit_transform(Z)
    history, model = spherical_embeddings(A, n_epochs = n_epochs, batch_size = batch_size, n_components = n_components)
    return history, model

def plot_embedding(X_embedded, names, model_name, y = None):
    """Plots an embedding an saves it.
    
    # Arguments:
        X_embedded (matrix): An embedding matrix.
        names (list): List of neuron names.
        model_name (str): Model name to use.
        y (list): Real labels for the nodes, as a list of integers corresponding to class IDs.
    """
    X_embedded = umap.UMAP().fit_transform(X_embedded)
    fig, ax = plt.subplots(figsize=(40,32))
    for g in np.unique(y):
        i = np.where(y == g)
        ax.scatter(X_embedded[i,0], X_embedded[i,1], label=g, s=40)
    for i, txt in enumerate(names):
        ax.annotate(txt, (X_embedded[i,0], X_embedded[i,1]), fontsize=6)
    
    dg = nx.DiGraph()
    plt.axis('tight')
    plt.title(model_name+'+UMAP')
    plt.savefig(model_name+'.eps')
    plt.savefig(model_name+'.png', dpi=300)

    
def generate_metrics(G, model, embedding, labels, predicted_labels, S=None, cv=5):
    """Generates a series of benchmarks for unsupervised learning (MAP), semi-supervised learning, and supervised learning (cross validation accuracy with random forest classifiers) for the provided input dataset.
    
    # Arguments:
        x (NEGraph): A NeuroEmbed graph.
        cv (int): Optional. Number of cross-validation folds to use.
        
    # Returns:
        dict: A result dictionary with all models and results.
    """
    out_metrics = {}
    clf = RandomForestClassifier(n_estimators = 2000)
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, 
                                                                         model, 
                                                                         embedding, 
                                                                         is_undirected=False, 
                                                                         is_weighted=True)
    out_metrics['MAP'] = MAP
    if labels is not None:
        scores = cross_val_score(clf, embedding, labels, cv=cv)
        print(scores)
        out_metrics['CV'] = scores.mean()
        if S is not None:
            scores = cross_val_score(clf, np.hstack((embedding,S)), labels, cv=cv)
            print(scores)
            out_metrics['CVAnatomy+Graph'] = scores.mean()
            scores = cross_val_score(clf, S, labels, cv=cv)
            print(scores)
            out_metrics['CVAnatomyOnly'] = scores.mean()
        out_metrics['ARC Clustering'] = metrics.adjusted_rand_score(labels, predicted_labels)
        out_metrics['AMI Clustering'] = metrics.adjusted_mutual_info_score(labels, predicted_labels)
    return out_metrics

def benchmark(x, cv=5):
    """This function automatically runs through a series of benchmarks for unsupervised learning (MAP), semi-supervised learning, and supervised learning (cross validation accuracy with random forest classifiers) for the provided input dataset.
    
    # Arguments:
        x (NEGraph): A NeuroEmbed graph.
        cv (int): Optional. Number of cross-validation folds to use.
        
    # Returns:
        dict: A result dictionary with all models and results.
    """
    all_results = {}
    G, X, y, S, names = x.G, x.X, x.y, x.S, x.names
    out_metrics = {}
    model = ASEEmbedding()
    model.fit(X)
    MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, 
                                                                             model, 
                                                                             model.H, 
                                                                             is_undirected=False, 
                                                                             is_weighted=True)
    out_metrics['MAP'] = MAP
    d = model.H.shape[1]//2
    out_metrics = generate_metrics(G, model, model.H, y, model.y, S, cv=cv)
    all_results['ASE'] = out_metrics
    raw_model = RawEmbedding()
    raw_model.fit(X, n_components = d)
    out_metrics = generate_metrics(G, raw_model, raw_model.H, y, raw_model.y, S, cv=cv)
    all_results['Raw'] = out_metrics
    G=nx.from_numpy_matrix(X, create_using=nx.DiGraph)
    Gd=nx.from_numpy_matrix(X+1e-9, create_using=nx.DiGraph)
    models = {}
    if N2VC_available:
        models['node2vec'] = node2vec(d=d, max_iter=10, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1)
    models['HOPE'] = HOPE(d=d, beta=0.01)
    models['Laplacian Eigenmaps'] = LaplacianEigenmaps(d=d)
    for model_name, embedding in models.items():
        if model_name == 'node2vec':
            Xh, t = embedding.learn_embedding(graph=Gd, edge_f=None, is_weighted=True, no_python=True)
            MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(Gd, 
                                                                                     embedding, 
                                                                                     Xh, 
                                                                                     is_undirected=False, 
                                                                                     is_weighted=False)
        else:
            Xh, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
            MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, 
                                                                                     embedding, 
                                                                                     Xh, 
                                                                                     is_undirected=False, 
                                                                                     is_weighted=False)
        Xh = np.real(Xh)
        if y is not None:
            clf = RandomForestClassifier(n_estimators = 200)
            clf = MLPClassifier(alpha=1, max_iter=100000)
            clusterer = GaussianMixture(n_components=Xh.shape[1])
            clusterer.fit(Xh)
            predict_labels = clusterer.predict(Xh)
            scores = cross_val_score(clf, Xh, y, cv=cv)
            out_metrics['CV'] = scores.mean()
            if S is not None:
                scores = cross_val_score(clf, np.hstack((Xh,S)), y, cv=cv)
                out_metrics['CVAnatomy+Graph'] = scores.mean()
                scores = cross_val_score(clf, S, y, cv=cv)
                out_metrics['CVAnatomyOnly'] = scores.mean()
            out_metrics['ARC Clustering'] = metrics.adjusted_rand_score(y, predict_labels)
            out_metrics['AMI Clustering'] = metrics.adjusted_mutual_info_score(y, predict_labels)
        out_metrics['MAP'] = MAP
        print(model_name, out_metrics)
        all_results[model_name] = out_metrics
    return all_results

def reprank(models, set_indices):
    """This function implements the approach from https://arxiv.org/pdf/2005.10700.pdf.
    
    # Arguments:
        models (list): A list of embedding matrices as numpy arrays.
        set_indices (list): A list of integers describing the set of items in the supervision set.
        
    # Returns:
        scipy result instance: A result instance with optimization results.
    """
    dmatrices = []
    variables = []
    for model in models:
        dmatrices.append(distance_matrix(model, model)[set_indices,:]) # Create set of distance matrices
        variables.append(1)
    def reprank_optimizer(x):
        x = np.array(x)
        x = x / np.sum(x)
        A = dmatrices[0] * 0.
        for i in range(len(dmatrices)): # for each distance matrix
            A += x[i] * dmatrices[i] # add weighted
        B = np.argsort(A)
        B = B[:,set_indices]
        return np.max(A)
    # Adapted from https://stackoverflow.com/questions/35631192/element-wise-constraints-in-scipy-optimize-minimize
    def geq_constraint(x):
        """Constrain to x>=0."""
        return x
    def rowsum_constraint(x):
        """Constrain to sum(x)==1."""
        return np.sum(x) - 1
    constraints = [{'type': 'ineq', 'fun': geq_constraint},
                   {'type': 'eq', 'fun': rowsum_constraint}]
    res = minimize(reprank_optimizer, variables, constraints=constraints)
    return res

def get_closest(Xh, i):
    """A function to retrieve the closest neuron to an input using a given embedding.

    # Arguments:
        Xh (numpy array): Embedding matrix, with rows as nodes.
        i (int): The index to  query.

    # Returns:
        int: Index of the closest neuron in the embedding.
    """
    A = Xh.copy()
    A = A - Xh[i,:]
    A = np.sum(np.square(A), axis=1)
    a = np.argsort(A)
    return a

def get_closest_by_name(Xh, names, i):
    """A function to retrieve the closest neuron to an input by name using a given embedding.

    # Arguments:
        Xh (numpy array): Embedding matrix, with rows as nodes.
        names (list): List of names for the neurons.
        i (str): The name to query.
    """
    iname = names.index(i)
    returned = get_closest(Xh, iname)
    return [names[i] for i in returned]

class Embedding:
    """Embedding superclass that implements some functions for NeuroEmbed embeddings.

    """
    def __init__(self):
        """Initialization function that generates a model variable.
        """
        self.model = None

    def embed(self, dataset, morphology = False):
        """Fit function for NEGraph objects.

        # Arguments:
            dataset: An NEGraph object.
            morphology (bool): Whether to use morphological features if available. Defaults to False.
        """
        if morphology == True:
            self.learn_embedding(dataset.G, S = dataset.S)
        else:
            self.learn_embedding(dataset.G)

    def fit(self, X):
        """Fit function, similar to an sklearn-like interface. Stores embedding in self.H and predicted clustering labels in self.y.

        # Arguments:
            X: The input matrix.
        """

    def learn_embedding(self, G, **kwargs):
        """Generates the embedding given a graph and stores it in self.H and predicted clustering labels in self.y.
        
        # Arguments:
            G: The input graph.
        """

    def get_reconstructed_adj(self, *a, **b):
        """Generates a reconstruction of the adjacency matrix input given the embedding. Useful for debugging and analysis.
        """

    def get_closest_by_id(self, x):
        """A function to retrieve the closest neuron to an input.
        
        # Arguments:
            x (int): The index to  query.

        # Returns:
            int: Index of the closest neuron in the embedding.
        """
        return get_closest(model.H, x)

    def get_closest_by_name(self, x):
        """A function to retrieve the closest neuron to an input by name.
        
        # Arguments:
            x (str): The name to query.

        # Returns:
            int: Index of the closest neuron in the embedding.
        """
        return get_closest_by_name(model.H, model.names, x)
        
class GEMEmbedding(Embedding):
    """Implements an interface for GEM embeddings; inherits from the Embedding class.

    """
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, n_components=None, S = None):
        G = nx.from_numpy_matrix(X, create_using=nx.DiGraph)
        Xh, t = self.model.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        if n_components is None:
            n_components = Xh.shape[1]//2
        if S is not None:
            Xh = np.hstack((Xh, S))
        clusterer = GaussianMixture(n_components=n_components)
        clusterer.fit(Xh)
        predict_labels = clusterer.predict(Xh)
        self.y = predict_labels
        self.H = Xh

    def learn_embedding(self, G, n_components=None, S = None, **kwargs):
        Xh, t = self.model.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        if n_components is None:
            n_components = Xh.shape[1]//2
        if S is not None:
            Xh = np.hstack((Xh, S))
        clusterer = GaussianMixture(n_components=n_components)
        clusterer.fit(Xh)
        predict_labels = clusterer.predict(Xh)
        self.y = predict_labels
        self.H = Xh

    def get_reconstructed_adj(self, *a, **b):
        return None
    
class HOPEEmbedding(GEMEmbedding):
    """Implements an interface for HOPE embedding.

    """
    def __init__(self, d = 2, beta=0.01):
        self.model = HOPE(d=d, beta=beta)
        
class N2VEmbedding(GEMEmbedding):
    """Implements an interface for node2vec embedding.

    """
    def __init__(self, d = 2, max_iter=10, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1):
        if not N2VC_available:
            raise RuntimeError('node2vec binary not found on PATH. Please compile and install from https://github.com/snap-stanford/snap/tree/master/examples/node2vec')
        self.model = node2vec(d=d, max_iter=max_iter, walk_len=walk_len, num_walks=num_walks, con_size=con_size, ret_p=ret_p, inout_p=inout_p)
        
class LEEmbedding(GEMEmbedding):
    """Implements an interface for Laplacian Eigenmaps embedding.

    """
    def __init__(self, d = 2):
        self.model = LaplacianEigenmaps(d=d)

class ASEEmbedding(Embedding):
    """Implements an interface for adjacency spectral embedding; inherits from the Embedding class.

    """
    def __init__(self):
        self.model = AdjacencySpectralEmbed()

    def fit(self, X, S = None):
        Xh = np.hstack(self.model.fit_transform(X))
        if S is not None:
            Xh = np.hstack((Xh, S))
        clusterer = GaussianMixture(n_components=Xh.shape[1]//2)
        clusterer.fit(Xh)
        predict_labels = clusterer.predict(Xh)
        self.y = predict_labels
        self.H = Xh

    def learn_embedding(self, G, S = None, **kwargs):
        X = nx.adjacency_matrix(G)
        X = X.todense()
        Xh = np.hstack(self.model.fit_transform(X))
        if S is not None:
            Xh = np.hstack((Xh, S))
        clusterer = GaussianMixture(n_components=Xh.shape[1]//2)
        clusterer.fit(Xh)
        predict_labels = clusterer.predict(Xh)
        self.y = predict_labels
        self.H = Xh

    def get_reconstructed_adj(self, *a, **b):
        return self.model.latent_left_.dot(np.diag(self.model.singular_values_)).dot(self.model.latent_right_.T)
        
class RawEmbedding(Embedding):
    """A trivial embedding in which the adjacency matrix is treated as the embedding.

    """
    def __init__(self):
        pass

    def fit(self, X, n_components=None, S = None):
        Xh = X
        if S is not None:
            Xh = np.hstack((X, S))
        if n_components == None:
            n_components = Xh.shape[1]//2
        clusterer = GaussianMixture(n_components=n_components)
        clusterer.fit(Xh)
        predict_labels = clusterer.predict(Xh)
        self.y = predict_labels
        self.H = Xh

    def learn_embedding(self, G, S = None, **kwargs):
        X = nx.adjacency_matrix(G)
        X = X.todense()
        self.fit(X)

    def get_reconstructed_adj(self, *a, **b):
        return self.H
    
class PCAEmbedding(Embedding):
    """A trivial embedding in which the PCA of the adjacency matrix is treated as the embedding.

    """
    def __init__(self, n_components = 2):
        self.model = PCA(n_components = n_components)
    def fit(self, X, S = None, n_components=None):
        if S is not None:
            X = np.hstack((X, S))
        Xh = self.model.fit_transform(X)
        if n_components == None:
            n_components = Xh.shape[1]//2
        clusterer = GaussianMixture(n_components=n_components)
        clusterer.fit(Xh)
        predict_labels = clusterer.predict(Xh)
        self.y = predict_labels
        self.H = Xh

    def learn_embedding(self, G, S = None, **kwargs):
        X = nx.adjacency_matrix(G)
        X = X.todense()
        if S is not None:
            X = np.hstack((X, S))
        self.fit(X)

    def get_reconstructed_adj(self, *a, **b):
        return np.dot(self.H, self.model.components_)
    
class NEGraph:
    """A class to simplify working with the dataset loading functions for NeuroEmbed graphs.

    # Arguments:
        graph (flybrainlab.graph.NeuronGraph object or nx.DiGraph object): a graph
        names (list): a list of neuron unames, should only be specified when graph is not a NeuroGraph object. If graph is nx.DiGraph and names is None, node names of the graph will be used.
        labels (list): Cell type labels for each node of the graph, should be in the same order as names. If graph is a NeuroGraph object, name (cell type) field of the nodes will be used.
        morphometric (pandas.DataFrame): A data frame in which each row is a morphometric measure and each column is the uname of nodes in the graph.
    """
    def __init__(self, graph, names = None, labels = None, morphometric = None):
        """
        
        """
        if names is None:
            if isinstance(graph, NeuronGraph):
                self.X, self.names = graph.adjacency_matrix()
            else:
                self.X = nx.adj_matrix(graph).todense()
                self.names = list(graph.nodes())
        else:
            if isinstance(graph, NeuronGraph):
                self.X, self.names = graph.adjacency_matrix(graph, uname_order = names)
                self.names = names
            else:
                self.X = nx.adj_matrix(graph, nodelist = names).todense()
                self.names = names
        if labels is None:
            if isinstance(graph, NeuronGraph):
                order_dict = {graph.nodes[n]['uname']: n for n in graph.nodes()}
                rid_order = [order_dict[uname] for uname in self.names]
                cell_types = [graph.nodes[n]['name'] for n in rid_order]
                y = []
                unique_ys = list(np.unique(cell_types))
                for cell_type in cell_types:
                    y.append(unique_ys.index(cell_type))
                self.y = np.array(y)
            else:
                self.y = np.zeros((len(self.names),))
        else:
            self.y = labels

        if morphometric is None:
            self.S = None
        else:
            self.S = morphometric[self.names].to_numpy().T.copy()
        # To make the graph consistent with the order imposed by X, names and labels
        # Other attributes (such as node lables) of the graph are ignored in the embeding anyway
        self.G = nx.from_numpy_matrix(self.X, create_using=nx.DiGraph)

    @classmethod
    def from_NAqueryResult(cls, res, client,
                           synapse_threshold = 5, complete_synapses = True):
        """
        Construct an NEGraph using NAqueryResult

        # Arguments:
            res (flybrainlab.graph.NAqueryResult or flybrianlab.graph.NeuroNLPResult): a NAqueryResult object
            client (flybrainlab.Client.Client): fbl client
            synapse_threshold (int): when retrieving the full connectiivty of neurons in `res`, the threshold of number of synapses to use for filtering out connections with small number of synapses.
            complete_synapses (bool): Whether to complete the synapses between neurons in `res` from NeuroArch database or to use only the synapses in `res`.
        
        # Returns:
            NEGraph: constructed NEGraph object
        """
        graph = client.get_neuron_graph(query_result = res,
                                        synapse_threshold = synapse_threshold,
                                        complete_synapses = complete_synapses)
        if (isinstance(res, NAqueryResult) and res.format == 'morphology') or \
                isinstance(res, NeuroNLPResult):
            morpho = morphometrics(res)
        else:
            morpho = None
        return cls(graph, morphometric = morpho)


