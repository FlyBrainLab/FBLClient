import os
import ast
import pandas as pd
import numpy as np
import networkx as nx
import json

path = os.path.dirname(os.path.abspath(__file__))



def get_presynaptic(X, G, threshold=0.05):
    """ Returns a list of predecessors for a group of neurons.

    # Arguments
        X (list): List of nodes to get the predecessors of.
        G (networkx Graph): Connectivity graph to use.
        threshold (float): Threshold to use for filtering.
    # Returns
        list: List of predecessors.
    """
    predecessors = []
    for x in X:
        if x in G.nodes():
            for i in G.predecessors(x):
                if G.edges[i, x]['weight']>=threshold:
                    predecessors.append(i)
    predecessors = list(set(predecessors))
    return predecessors

def get_search(names, verb = 'show'):
    """ Returns an NLP search query to retrieve neurons given a list of neurons.

    # Arguments
        names (list): List of neurons to retrieve.
        verb (str): Verb to use. Defaults to 'show'. Can be 'show', 'add', 'keep' or 'remove'.
    # Returns
        str: NLP query string.
    """
    _str = verb + ' /:referenceId:['+', '.join([str(i) for i in names])+']'
    print(_str)
    return _str

def get_set_color(names, color):
    """ Returns an NLP search query to color neurons given a list of neurons.

    # Arguments
        names (list): List of neurons to color.
        color (str): Color to use. Refer to the color name list in https://en.wikipedia.org/wiki/Web_colors.
    # Returns
        str: NLP query string.
    """
    _str = 'color /:referenceId:['+', '.join([str(i) for i in names])+'] '+color
    print(_str)
    return _str

def load_normalized_hemibrain():
    """ Returns an NLP search query to color neurons given a list of neurons.

    # Returns
        NetworkX Graph: Connectivity graph for Hemibrain in which edges are weighted based on percentage of inputs.
        dict: Dict of neuron labels that can be searched with get_field or get_field_with_keys functions.
    """
    G = nx.read_gexf('hemi12_G_normalized.gexf')
    neuron_labels = np.load('hemi_neurons.npy', allow_pickle=True).item()
    return G, neuron_labels

def search(inp, neuron_labels):
    """ Utility function to retrieve neuron labels with a filter.

    # Arguments
        inp (str): Keyword to use for filtering.
        neuron_labels (dict): Dictionary of neuron labels, for example generated with load_normalized_hemibrain.
    # Returns
        list: Returns names 
    """
    return [x for x,v in neuron_labels.items() if inp in v]

def threshold_graph(G, threshold=0.05):
    """ Filter edges in a graph by some threshold.

    # Arguments
        G (NetworkX Graph): List of neurons to color.
        threshold (float): Threshold value to use.
    # Returns
        NetworkX Graph: Threshold-filtered NetworkX graph.
    """
    edge_iterator = list(G.edges(data=True))
    edges_to_remove = []
    for n1,n2,data in edge_iterator:
        if data['weight']<threshold:
            edges_to_remove.append((n1,n2))
    for i in edges_to_remove:
        G.remove_edge(i[0],i[1])
    return G

def get_field(X, neuron_labels, i=0):
    """ Return labels for a subset of neuron names.

    # Arguments
        X (list): List of neurons names to retrieve the field of.
        neuron_labels (dict): Dictionary of neuron labels, for example generated with load_normalized_hemibrain.
        i (int): Label field index to retrieve. Label indices in order: status, statusLabel, cropped, instance, notes, type.
    # Returns
        NetworkX Graph: Threshold-filtered NetworkX graph.
    """
    return [neuron_labels[x].split('/')[i] for x in X]

def generate_synapse_data(file_path):
    """ Generates precomputed synaptomics labels from Hemibrain data.

    # Arguments
        file_path (str): First list.
    """
    skeleton_ids = [f.split('.')[0] for f in listdir(file_path) if isfile(join(file_path, f))]

    meshes_to_use = ['bL(L)',
                    'g4(R)',
                    'SIP(L)',
                    'NO1(R)',
                    'EPA(L)',
                    'SLP(R)',
                    'GA(R)',
                    'aL(R)',
                    'PB(L5)',
                    'EBr5',
                    'EB',
                    "b'2(R)",
                    'FBl9',
                    'AB(R)',
                    'PLP(R)',
                    'GF(R)',
                    'PB(R5)',
                    'AOTU(R)',
                    'FBl6',
                    'CRE(R)',
                    'FB-column3',
                    "a'L(R)",
                    'SPS(L)',
                    'PB(L9)',
                    'AL(L)',
                    'b1(R)',
                    'VES(L)',
                    'ME(R)',
                    'FBl7',
                    'EPA(R)',
                    'EBr2r4',
                    "a'3(R)",
                    'SAD',
                    'NO',
                    'FBl3',
                    'NO1(L)',
                    'EBr1',
                    'WED(R)',
                    'BU(R)',
                    'SCL(R)',
                    'VES(R)',
                    'PB(R8)',
                    'PB(R9)',
                    "b'1(R)",
                    'IPS(R)',
                    'b2(R)',
                    'LOP(R)',
                    'g5(R)',
                    'BU(L)',
                    'FBl1',
                    'PB(R3)',
                    "b'L(R)",
                    'MB(R)',
                    'NO2(L)',
                    'RUB(R)',
                    "a'2(R)",
                    'PB(L1)',
                    'PVLP(R)',
                    'PB(L3)',
                    'a2(R)',
                    "a'L(L)",
                    'IB',
                    'mALT(L)',
                    'PB(L2)',
                    'aL(L)',
                    'NO2(R)',
                    "b'L(L)",
                    'SMP(R)',
                    'ICL(R)',
                    'AL-DC3(R)',
                    'PB(R1)',
                    'gL(L)',
                    'g1(R)',
                    'NO3(L)',
                    'AB(L)',
                    'gL(R)',
                    'PB(L8)',
                    'PRW',
                    'AVLP(R)',
                    'PB(R6)',
                    'FB',
                    'PB(L7)',
                    'FBl5',
                    'dACA(R)',
                    'RUB(L)',
                    'g3(R)',
                    'ROB(R)',
                    'NO3(R)',
                    'AMMC',
                    "a'1(R)",
                    'bL(R)',
                    'CA(L)',
                    'PB(L4)',
                    'PB',
                    'PB(R4)',
                    'SAD(-AMMC)',
                    'AME(R)',
                    'SCL(L)',
                    'LAL(R)',
                    'PB(L6)',
                    'vACA(R)',
                    'EBr3am',
                    'NO(R)',
                    'LO(R)',
                    'LAL(-GA)(R)',
                    'FBl4',
                    'a1(R)',
                    'FBl2',
                    'ICL(L)',
                    'EBr6',
                    'AOT(R)',
                    'g2(R)',
                    'EBr3d',
                    'CRE(L)',
                    'FLA(R)',
                    'POC',
                    'GOR(L)',
                    'MB(L)',
                    'ATL(R)',
                    'CAN(R)',
                    'LAL(L)',
                    'LH(R)',
                    'SIP(R)',
                    'GNG',
                    'SMP(L)',
                    'EBr3pw',
                    'a3(R)',
                    'SPS(R)',
                    'PED(R)',
                    'PB(R7)',
                    'AL(R)',
                    'PB(R2)',
                    'NO(L)',
                    'GC',
                    'CA(R)',
                    'GOR(R)',
                    'ATL(L)']

    loc_scale = 0.001
    node_to_id = {}
    # Create the node-to-skeleton-id dict
    # for i in range(len(skeleton_ids)):
    #     node_to_id[str(skeleton_ids[i])] = unames[i]

    all_datas = []
    

    for i in range(1,len(skeleton_ids)):
        syns_batch = []
        syn_n_batch = []
        syn_xs_batch = []
        syn_ys_batch = []
        syn_zs_batch = []
        bodyID = str(skeleton_ids[i])
        data = pd.read_csv(file_path+'/{}.csv'.format(bodyID))
        main_data = np.zeros((6,))
        regions = np.zeros((len(meshes_to_use),))
        
        for j in range(len(data)):
            main_data = np.zeros((6,))
            regions = np.zeros((len(meshes_to_use),))
            main_data[0] = int(data['m.bodyId'][j])
            main_data[1] = int(data['neuron.bodyId'][j])
            
            syn = ast.literal_eval(data['syn'][j]) # this line is slow, care
            coordinates = syn['location']['coordinates']
            # name = str(presyn)+'--'+str(postsyn)
            
                
            main_data[2] = syn['location']['coordinates'][0]
            main_data[3] = syn['location']['coordinates'][1]
            main_data[4] = syn['location']['coordinates'][2]
            main_data[5] = syn['confidence']

            for i in syn.keys():
                if i in meshes_to_use:
                    regions[meshes_to_use.index(i)] = 1.
            all_data = np.hstack((main_data, regions))
            all_datas.append(all_data)

    all_datas2 = np.array(all_datas)
    print('Synapse Data Shape:', all_datas2.shape)
    np.save('syn_data.npy', all_datas2)
            
def intersection(a, b): 
    """ Compute the intersection of two lists.

    # Arguments
        a (list): First list.
        b (list): Second list.

    # Returns
        list: The intersection of the two lists.
    """
    c = [x for x in a if x in b] 
    return c

class HemibrainAnalysis:
    """A class for analyzing a given database with some helper functions. Default files are provided for the Hemibrain dataset.
    """
    def __init__(self):
        """Initializes a HemibrainAnalysis object.
        """
        self.N = pd.read_csv('traced-neurons.csv')
        self.B = pd.read_csv('synapse_list.csv')
        self.N_list = list(self.N['bodyId'])
        self.T = pd.read_csv('all_neurons_reference_fib.csv')
        self.T = self.T[self.T['side'] == 'right']
        self.T = self.T.reset_index(drop=True)
        self.keys_to_neurons = {}
        self.neurons_to_keys = {}
        for i in range(len(self.T)):
            self.keys_to_neurons[self.T.loc[i]['skeleton_id']] = self.T.loc[i]['uname']
            self.neurons_to_keys[self.T.loc[i]['uname']] = self.T.loc[i]['skeleton_id']
    def get_postsynaptic_partners(self, _id, N=0):
        """Get postsynaptic partners of a given neuron.

        # Arguments:
            _id : ID to use.
            N: Synapse threshold to use.

        # Returns:
            list: List of postsynaptic partners.
        """
        if _id in self.keys_to_neurons:
            results = list(self.B[(self.B['N']>=N) & (self.B['presynaptic'] == self.keys_to_neurons[_id])]['postsynaptic'])
            outs = []
            for i in results:
                if i in self.neurons_to_keys:
                    outs.append(self.neurons_to_keys[i])
        else:
            outs = []
            print('ID {} is missing.'.format(str(_id)))
        return outs
    def get_all_postsynaptic_partners(self, ids, N=0):
        """Gets postsynaptic partners of a given set of neurons.

        # Arguments:
            list : list of IDs to use.
            N: Synapse threshold to use.

        # Returns:
            list: List of postsynaptic partners.
        """
        outs = []
        for i in ids:
            outs += self.get_postsynaptic_partners(i, N=N)
        return outs
    def get_presynaptic_partners(self, _id, N=0):
        """Get presynaptic partners of a given neuron.

        # Arguments:
            _id : ID to use.
            N: Synapse threshold to use.

        # Returns:
            list: List of presynaptic partners.
        """
        if _id in self.keys_to_neurons:
            results = list(self.B[(self.B['N']>=N) & (self.B['postsynaptic'] == self.keys_to_neurons[_id])]['presynaptic'])
            outs = []
            for i in results:
                if i in self.neurons_to_keys:
                    outs.append(self.neurons_to_keys[i])
        else:
            outs = []
            print('ID {} is missing.'.format(str(_id)))
        return outs
    def get_all_presynaptic_partners(self, ids, N=0):
        """Gets presynaptic partners of a given set of neurons.

        # Arguments:
            list : list of IDs to use.
            N: Synapse threshold to use.

        # Returns:
            list: List of presynaptic partners.
        """
        outs = []
        for i in ids:
            outs += self.get_presynaptic_partners(i, N=N)
        return outs
    def get_graph(self, _id):
        """Retrieve a graph consisting of only neurons in a list.

        # Arguments:
            _id (list) : list of IDs to use.

        # Returns:
            pandas DataFrame: Graph only composed of the edges whose targets and sources are in the list of IDs.
        """
        results = self.B[self.B['postsynaptic'].isin(_id)]
        results = results[results['presynaptic'].isin(_id)]
        return results
    def get_type(self, _type, N=None):
        """Retrieve neurons whose type contains a specific substring.

        # Arguments:
            _type (str) : type to search.

        # Returns:
            pandas DataFrame: DataFrame with neurons for the query.
        """
        if N is None:
            results = self.N[self.N['type'].str.contains(_type, na=False)]
        else:
            results = N[N['type'].str.contains(_type, na=False)]
        return results
    def get_instance(self, instance, N=None, converter = None):
        """Retrieve neurons whose instance contains a specific substring.

        # Arguments:
            instance (str) : instance to search.

        # Returns:
            pandas DataFrame: DataFrame with neurons for the query.
        """
        if N is None:
            results = self.N[self.N['instance'].str.contains(instance, na=False)]
        else:
            results = N[N['instance'].str.contains(instance, na=False)]
        if converter is not None:
            results = converter(results)
        return results
    def to_id(self, x):
        """Converts body IDs to integers.
        """
        return [int(i) for i in x['bodyId']]
    def to_str_id(self, x):
        """Converts body IDs to strings.
        """
        return [str(i) for i in x['bodyId']]


def load_hemibrain_synaptome():
    """Loads the hemibrain synaptome; useful for .
    
    # Returns:
        numpy array: A matrix in the NeuroSynapsis matrix format.
    """
    return np.load('syn_data.npy'), np.load('hemibrain_volumes.npy', allow_pickle=True).item()

class Synaptome:
    def __init__(self, X, regions = None, synapse_classes = None, confidence = True):
        """Initialize a synaptome object.
    
        # Arguments:
            X (numpy array): A matrix in the NeuroSynapsis matrix format.
            i (int): Numeric ID of the neuron.
            confidence (bool): Whether confidence values exist for the synapses. Optional.
        """
        self.X = X
        self.regions = regions
        self.confidence = confidence

def find_postsynaptic(X, i):
    """Return postsynaptic partners of a neuron.
    
    # Arguments:
        X (numpy array): A matrix in the NeuroSynapsis matrix format.
        i (int): Numeric ID of the neuron.
        
    # Returns:
        numpy array: A matrix in the NeuroSynapsis matrix format.
    """
    vals = np.where(X[:,1] == i)[0]
    return X[vals,:]

def find_presynaptic(X, i):
    """Return presynaptic partners of a neuron.
    
    # Arguments:
        X (numpy array): A matrix in the NeuroSynapsis matrix format.
        i (int): Numeric ID of the neuron.
        
    # Returns:
        numpy array: A matrix in the NeuroSynapsis matrix format.
    """
    vals = np.where(X[:,0] == i)[0]
    return X[vals,:]

def filter_by_maximum(X, region, confidence = True):
    """Filter synapses by maximum.
    
    # Arguments:
        X (numpy array): A matrix in the NeuroSynapsis matrix format.
        
    # Returns:
        numpy array: A matrix in the NeuroSynapsis matrix format.
    """
    vals = np.where((X[:,2] <= i[0])*(X[:,3] <= i[1])*(X[:,4] <= i[2]))[0]
    return X[vals,:]

def filter_by_minimum(X, region):
    """Filter synapses by minimum.
    
    # Arguments:
        X (numpy array): A matrix in the NeuroSynapsis matrix format.
        
    # Returns:
        numpy array: A matrix in the NeuroSynapsis matrix format.
    """
    vals = np.where((X[:,2] >= i[0])*(X[:,3] >= i[1])*(X[:,4] >= i[2]))[0]
    return X[vals,:]

def filter_by_region(X, region, confidence = True):
    """Filter synapses by region.
    
    # Arguments:
        X (numpy array): A matrix in the NeuroSynapsis matrix format.
        region (str): Name of the region to use.
        
    # Returns:
        numpy array: A matrix in the NeuroSynapsis matrix format.
    """
    regions =  ['bL(L)',
                 'g4(R)',
                 'SIP(L)',
                 'NO1(R)',
                 'EPA(L)',
                 'SLP(R)',
                 'GA(R)',
                 'aL(R)',
                 'PB(L5)',
                 'EBr5',
                 'EB',
                 "b'2(R)",
                 'FBl9',
                 'AB(R)',
                 'PLP(R)',
                 'GF(R)',
                 'PB(R5)',
                 'AOTU(R)',
                 'FBl6',
                 'CRE(R)',
                 'FB-column3',
                 "a'L(R)",
                 'SPS(L)',
                 'PB(L9)',
                 'AL(L)',
                 'b1(R)',
                 'VES(L)',
                 'ME(R)',
                 'FBl7',
                 'EPA(R)',
                 'EBr2r4',
                 "a'3(R)",
                 'SAD',
                 'NO',
                 'FBl3',
                 'NO1(L)',
                 'EBr1',
                 'WED(R)',
                 'BU(R)',
                 'SCL(R)',
                 'VES(R)',
                 'PB(R8)',
                 'PB(R9)',
                 "b'1(R)",
                 'IPS(R)',
                 'b2(R)',
                 'LOP(R)',
                 'g5(R)',
                 'BU(L)',
                 'FBl1',
                 'PB(R3)',
                 "b'L(R)",
                 'MB(R)',
                 'NO2(L)',
                 'RUB(R)',
                 "a'2(R)",
                 'PB(L1)',
                 'PVLP(R)',
                 'PB(L3)',
                 'a2(R)',
                 "a'L(L)",
                 'IB',
                 'mALT(L)',
                 'PB(L2)',
                 'aL(L)',
                 'NO2(R)',
                 "b'L(L)",
                 'SMP(R)',
                 'ICL(R)',
                 'AL-DC3(R)',
                 'PB(R1)',
                 'gL(L)',
                 'g1(R)',
                 'NO3(L)',
                 'AB(L)',
                 'gL(R)',
                 'PB(L8)',
                 'PRW',
                 'AVLP(R)',
                 'PB(R6)',
                 'FB',
                 'PB(L7)',
                 'FBl5',
                 'dACA(R)',
                 'RUB(L)',
                 'g3(R)',
                 'ROB(R)',
                 'NO3(R)',
                 'AMMC',
                 "a'1(R)",
                 'bL(R)',
                 'CA(L)',
                 'PB(L4)',
                 'PB',
                 'PB(R4)',
                 'SAD(-AMMC)',
                 'AME(R)',
                 'SCL(L)',
                 'LAL(R)',
                 'PB(L6)',
                 'vACA(R)',
                 'EBr3am',
                 'NO(R)',
                 'LO(R)',
                 'LAL(-GA)(R)',
                 'FBl4',
                 'a1(R)',
                 'FBl2',
                 'ICL(L)',
                 'EBr6',
                 'AOT(R)',
                 'g2(R)',
                 'EBr3d',
                 'CRE(L)',
                 'FLA(R)',
                 'POC',
                 'GOR(L)',
                 'MB(L)',
                 'ATL(R)',
                 'CAN(R)',
                 'LAL(L)',
                 'LH(R)',
                 'SIP(R)',
                 'GNG',
                 'SMP(L)',
                 'EBr3pw',
                 'a3(R)',
                 'SPS(R)',
                 'PED(R)',
                 'PB(R7)',
                 'AL(R)',
                 'PB(R2)',
                 'NO(L)',
                 'GC',
                 'CA(R)',
                 'GOR(R)',
                 'ATL(L)']
    if region in regions:
        k = regions.index(region)+5+confidence
        vals = np.where(X[:,regions.index(region)+5+confidence] == 1.)[0]
        return X[vals,:]
    else:
        print('Region not recognized.')
        
def elbow_kmeans_optimizer(X, k = None, kmin = 1, kmax = 5, visualize = True):
    """k-means clustering with or without automatically determined cluster numbers. 
    Reference: https://pyclustering.github.io/docs/0.8.2/html/d3/d70/classpyclustering_1_1cluster_1_1elbow_1_1elbow.html
    
    # Arguments:
        X (numpy array-like): Input data matrix.
        kmin: Minimum number of clusters to consider. Defaults to 1.
        kmax: Maximum number of clusters to consider. Defaults to 5.
        visualize: Whether to perform k-means visualization or not.
    
    # Returns:
        numpy arraylike: Clusters.
        numpy arraylike: Cluster centers.
    """
    from pyclustering.utils import read_sample
    from pyclustering.samples.definitions import SIMPLE_SAMPLES
    from pyclustering.cluster.kmeans import kmeans
    from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
    from pyclustering.core.wrapper import ccore_library
    from pyclustering.cluster.elbow import elbow
    from pyclustering.cluster.kmeans import kmeans_visualizer
    import pyclustering.core.elbow_wrapper as wrapper
    if k is not None:
        amount_clusters = k
    else:
        elbow_instance = elbow(X, kmin, kmax)
        elbow_instance.process()
        amount_clusters = elbow_instance.get_amount() 
        wce = elbow_instance.get_wce()
    centers = kmeans_plusplus_initializer(X, amount_clusters).initialize()
    kmeans_instance = kmeans(X, centers)
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    centers = kmeans_instance.get_centers()
    kmeans_visualizer.show_clusters(X, clusters, centers)
    return clusters, centers

def return_synapse_positions(X):
    """Filters a NeuroSynapsis matrix and returns only the matrix of synapse positions.
    
    # Arguments:
        X (numpy array): A matrix in the NeuroSynapsis matrix format.
        
    # Returns:
        numpy array: A matrix of size samples x dimensions.
    """
    return X[:,2:5]

def calculate_synapse_density(A, B):
    """Filters a NeuroSynapsis matrix and returns only the matrix of synapse positions.
    
    # Arguments:
        A (numpy array): A matrix in the NeuroSynapsis matrix format.
        B (dict): A NeuroSynapsis volume object.
        
    # Returns:
        dict: Density of synapses, calculated as (# synapses)/(nm^3).
    """
    densities = {}
    for i, v in enumerate(list(B.keys())):
        X = filter_by_region(A, v)
        densities[v] = X.shape[0] / (B[v]*8*8*8)
    return densities