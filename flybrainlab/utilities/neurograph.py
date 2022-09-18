import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import smallworld
try:
    from graphviz import Digraph
except:
    print('GraphViz was not detected. Please install it. Instructions:')
    print('Run "pip install graphviz".')
    print('Run "conda install graphviz pygraphviz -c alubbock -y".')
    print('Create a new terminal and run "dot -c" in the environment.')
import community
from cdlib import algorithms
try:
    from nxcontrol.algorithms import maximum_matching_driver_nodes
except:
    print('nxcontrol was not detected. Please install it from https://github.com/mkturkcan/nxcontrol. Instructions:')
    print('Run "git clone https://github.com/mkturkcan/nxcontrol.git".')
    print('Run "cd nxcontrol".')
    print('Run "pip install -e .".')
from ..graph import NeuronGraph

def connectivity_field_generator(db, query, membership_dict):
    """ Utility function for generating a binary connectivity mask for figuring out how groups connect to one another.
    
    # Arguments:
        db (HemibrainAnalysis object): A HemibrainAnalysis object (from NeuroSynapsis).
        query (list): List of neurons to get connectivity information of.
        membership_dict (dict): Dictionary from member names to their memberships.
    """
    presyns = {}
    postsyns = {}
    member_names = list(membership_dict.keys())
    for i in query:
        try:
            partners = db.get_postsynaptic_partners(i)
            postsyns[i] = []
            for j in partners:
                if j in member_names:
                    postsyns[i].append(membership_dict[j])
            postsyns[i] = list(set(postsyns[i]))
            postsyns[i].sort()
        except:
            print(i,'Failed')
        try:
            partners = db.get_presynaptic_partners(i)
            presyns[i] = []
            for j in partners:
                if j in member_names:
                    presyns[i].append(membership_dict[j])
            presyns[i] = list(set(presyns[i]))
            presyns[i].sort()
        except:
            print(i,'Failed')

    field = np.zeros((len(list(presyns.keys())), len(list(membership_dict.keys()))))
    for i in list(presyns.keys()):
        for j in list(membership_dict.keys()):
            if j in presyns[i]:
                field[list(presyns.keys()).index(i), list(membership_dict.keys()).index(j)] = 1
                
    postfield = np.zeros((len(list(postsyns.keys())), len(list(membership_dict.keys()))))
    for i in list(postsyns.keys()):
        for j in list(membership_dict.keys()):
            if j in presyns[i]:
                postfield[list(membership_dict.keys()).index(j), list(postsyns.keys()).index(i)] = 1
    return field, presyns

def clustering_consistency_check(G):
    """ Check consistency of a community detection algorithm by running it a number of times.
    
    """
    Hun = G.to_undirected()
    Hun = nx.convert_node_labels_to_integers(Hun,label_attribute='skeletonname')

    WHa = np.zeros((len(Hun.nodes()),len(Hun.nodes())))
    for i in range(100):
        partition = community.best_partition(Hun, randomize=None, resolution=1.0)
        for com in set(partition.values()) :
            list_nodes = [nodes for nodes in partition.keys()
                                        if partition[nodes] == com]
            list_nodes = np.array(list_nodes)
            WHa[np.ix_(list_nodes,list_nodes)] += 1
        print('Iteration:', i)
    return WHa

def connectivity_dendrogram(G, 
                            names = None,
                            vmax = None,
                            scale = 'linear',
                            figsize = (15,15),
                            dpi = 100, 
                            xlabel = 'Postsynaptic',
                            ylabel = 'Presynaptic',
                            title = 'Dendrogram',
                            export_format = None,
                            method='ward',
                            metric = 'euclidean',
                            figname = 'default'):
    """ Plot the dendrogram of the graph.
    
    # Arguments:
        G (nx.DiGraph or flybrainlab.graph.NeuroGraph): graph
        names (list): list of names to use for G, ignored when G is a NeuroGraph
        vmax (float): Maximum value for the diagram.
        scale (str): 'linear' or 'log'. Use linear or log scale to cluster.
        figsize (tuple): size of the figure.
        dpi (float): dpi of the figure.
        xlabel (str): Name of the x label.
        ylabel (str): Name of the y label.
        title (str): Title for the diagram.
        export_format (str): if specified, file format to export the diagram.
        method (str): Method to use. Can be "single", "average", "weighted", "centroid", "median" or "ward". Defaults to "ward".
        metric (str): Metric to use. Can be one of the options in https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html#scipy.spatial.distance.pdist . Defaults to "euclidean".

        figname (str): Name for the diagram.
    """
    if isinstance(G, NeuronGraph):
        W, names = G.adjacency_matrix()
    else:
        W = nx.adj_matrix(G)
        if names is None:
            names = list(G.nodes())
    
    if scale == 'log':
        W = np.log10(W+1)
    Mdf = pd.DataFrame(W, index = names, columns = names)
    sns.clustermap(Mdf, xticklabels=1, yticklabels=1, figsize=figsize, vmax=vmax, method=method, metric=metric)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if export_format is not None:
        plt.savefig('{}.{}'.format(figname, export_format), facecolor='white', edgecolor='none')
        
        
def community_detection(G, method = 'un_louvain'):
    """ A function that runs several community detection algorithms.
    
    # Arguments:
        G (networkx.Graph): A networkx graph or subclass object, including flybrainlab.graph.NeuronGraph.
        method (str): Name for the method to use. 
        One of "un_louvain" (undirected louvain), "louvain",
        "label_propagation", "leiden", "rb_pots", "walktrap" and "infomap".
    # Returns:
        np.ndarray: community-ordered connectivity matrix of the undirected graph in linear scale
        list: a list of node orders
        list: a list of list of node ids for each group member
    """
    Gnodes = list(G.nodes())
    Gun = G.to_undirected()
    G_to_return = Gun
    if method == 'un_louvain':
        partition = community.best_partition(Gun, randomize=None, resolution=1.00)
        all_nodes = []
        all_nodes = []
        all_list_nodes = []

        for com in set(partition.values()) :
            # print('Cluster {} Members:'.format(com))
            list_nodes = [nodes for nodes in partition.keys()
                                        if partition[nodes] == com]
            all_list_nodes += [Gnodes.index(nodes) for nodes in partition.keys()
                                        if partition[nodes] == com]
            all_nodes.append(list_nodes)
    else:
        if method == 'label_propagation':
            func = algorithms.label_propagation
        elif method == 'leiden':
            func = algorithms.leiden
        elif method == 'walktrap':
            func = algorithms.walktrap
        elif method == 'louvain':
            func = algorithms.louvain
        elif method == 'infomap':
            func = algorithms.infomap
        elif method == 'rb_pots':
            func = algorithms.rb_pots
        try:
            if method == 'infomap':
                coms = func(G, flags = '--directed')
                G_to_return = G
            elif method == 'rb_pots':
                coms = func(G, weights='weight')
                G_to_return = G
            else:
                coms = func(G)
                G_to_return = G
        except:
            coms = func(Gun)
        all_nodes = []
        all_list_nodes = []
        for i in coms.communities:
            list_nodes = i
            all_list_nodes += [Gnodes.index(j) for j in i]
            all_nodes.append(list_nodes)
    return G_to_return, all_list_nodes, all_nodes


def automatic_community_detector_downsample(G, 
                                            method = 'un_louvain',
                                            downsample_level = 1,
                                            scale = 'log',
                                            figsize = (15,15),
                                            dpi = 100.,
                                            ylabel='Presynaptic Neuron ID',
                                            xlabel='Postsynaptic Neuron ID',
                                            xticklabels = None,
                                            yticklabels = None,
                                            title='Community-Ordered Connectivity Matrix (Log Scale)',
                                            vmax = None,
                                            export_format = None,
                                            figname='example_automatic_community'):
    """Automatically detects communities for a large networkx graph with downsampling for the adjacency matrix.
    
    # Arguments:
        G (network.Graph): A networkx graph or subclass object, including flybrainlab.graph.NeuronGraph.
        method (str): Method to use. One of un_louvain, label_propagation, leiden, louvain, walktrap or infomap.
        downsample_level (int): A downsampling level to use.
        scale (str): 'linear' or 'log'. Use linear or log scale to cluster.
        figsize (tuple): size of the figure.
        dpi (float): dpi of the figure.
        xlabel (str): Name of the x label.
        ylabel (str): Name of the y label.
        xticklabels (list): x tick labels to have.
        yticklabels (list): y tick labels to have.
        title (str): Title for the diagram.
        vmax (float): Maximum value for the diagram.
        export_format (str): if specified, file format to export the diagram.
        figname (str): Name for the output figure.
    
    # Returns:
        np.ndarray: community-ordered connectivity matrix in linear scale
        list: a list of list of node ids for each group member
    """
    Gun, all_list_nodes, all_nodes  = community_detection(G, method = method)

    all_pre_nodes = [i for i in all_list_nodes]
    all_post_nodes = [i for i in all_list_nodes]
    A = nx.adjacency_matrix(G).todense()[np.ix_(all_pre_nodes,all_post_nodes)]
    B = A[::downsample_level,::downsample_level].copy()

    if xticklabels is None:
        if isinstance(G, NeuronGraph):
            xticklabels = sum(nodes_to_unames(G, all_nodes),[])[::downsample_level]
        else:
            xticklabels = sum(all_nodes, [])[::downsample_level]
    if yticklabels is None:
        if isinstance(G, NeuronGraph):
            yticklabels = sum(nodes_to_unames(G, all_nodes),[])[::downsample_level]
        else:
            yticklabels = sum(all_nodes, [])[::downsample_level]
    if scale == 'log':
        Bd = np.log(1.+B)
    else:
        Bd = B
        if title == 'Community-Ordered Connectivity Matrix (Log Scale)':
            title = 'Community-Ordered Connectivity Matrix'

    sizes = np.round(np.cumsum([len(i) for i in all_nodes]) / downsample_level)
    
    gen_heatmap(Bd, figsize = figsize, dpi = dpi, xlabel = xlabel, ylabel = ylabel,
                xticklabels = xticklabels, yticklabels = yticklabels,
                hlines = sizes, vlines = sizes, title = title, vmax = vmax,
                export_format = export_format, figname = figname)
    return B, all_nodes


def automatic_community_detector(G, 
                                 method = 'un_louvain',
                                 scale = 'log',
                                 figsize = (15,15),
                                 dpi = 100.,
                                 ylabel = 'Presynaptic Neuron ID',
                                 xlabel = 'Postsynaptic Neuron ID',
                                 xticklabels = None,
                                 yticklabels = None,
                                 title = 'Community-Ordered Connectivity Matrix (Log Scale)',
                                 vmax = None,
                                 export_format = None,
                                 figname = 'example_automatic_community'):
    """ Automatically detects communities for a large networkx graph.
    
    # Arguments:
        G (network.Graph): A networkx graph or subclass object, including flybrainlab.graph.NeuronGraph.
        method (str): Method to use. One of un_louvain, label_propagation, leiden, louvain, walktrap or infomap.
        scale (str): 'linear', 'log' or 'scaledlog'. Use linear or log scale to cluster. 'scaledlog' uses 50th percentile of nonzero entries as vmax.
        figsize (tuple): size of the figure.
        dpi (float): dpi of the figure.
        xlabel (str): Name of the x label.
        ylabel (str): Name of the y label.
        xticklabels (list): x tick labels to have.
        yticklabels (list): y tick labels to have.
        title (str): Title for the diagram.
        vmax (float): Maximum value for the diagram.
        export_format (str): if specified, file format to export the diagram.
        figname (str): Name for the diagram.
    
    # Returns:
        np.ndarray: community-ordered connectivity matrix in linear scale
        list: a list of list of node ids for each group member
    """
    Gun, all_list_nodes, all_nodes  = community_detection(G, method = method)
    
    all_pre_nodes = [i for i in all_list_nodes]
    all_post_nodes = [i for i in all_list_nodes]
    B = nx.adjacency_matrix(G).todense()[np.ix_(all_pre_nodes,all_post_nodes)].copy()

    if xticklabels is None:
        if isinstance(G, NeuronGraph):
            xticklabels = sum(nodes_to_unames(G, all_nodes),[])
        else:
            xticklabels = sum(all_nodes, [])
    if yticklabels is None:
        if isinstance(G, NeuronGraph):
            yticklabels = sum(nodes_to_unames(G, all_nodes),[])
        else:
            yticklabels = sum(all_nodes, [])
    if scale == 'log':
        Bd = np.log10(1.+B)
        # Bd[Bd>np.percentile(Bd, 90) = np.percentile(Bd, 90)
        # Bd[Bd<np.percentile(Bd, 10) = np.percentile(Bd, 10)
    elif scale == 'scaledlog':
        print('Min B:', np.min(B))
        Bd = np.log10(1.+B)
        # Bd[Bd>np.percentile(Bd, 90)] = np.percentile(Bd, 90)
    else:
        Bd = B
        if title == 'Community-Ordered Connectivity Matrix (Log Scale)':
            title = 'Community-Ordered Connectivity Matrix'

    if scale == 'scaledlog':
        Bd_s = np.array(Bd)
        Bd_s = Bd_s[Bd_s>0.]
        vmax = np.percentile(Bd_s, 50)
        Bd = np.nan_to_num(Bd)
        print('Scaled vmax:', vmax)
    sizes = np.cumsum([len(i) for i in all_nodes])
    gen_heatmap(Bd, figsize = figsize, dpi = dpi, xlabel = xlabel, ylabel = ylabel,
                xticklabels = xticklabels, yticklabels = yticklabels,
                hlines = sizes, vlines = sizes, title = title, vmax = vmax, vmin=0.,
                export_format = export_format, figname = figname)
    return B, all_nodes

def nodes_to_unames(G, all_nodes):
    """ convert node ids of communities from a NeuroGraph to unames

    # Arguments:
        G (flybrainlab.graph.NeuroGraph): NeuroGraph object
        all_nodes (list): list of nodes returned by community detector
    """
    return [[G.nodes[n]['uname'] for n in k] for k in all_nodes]


def matrix_community_detector(W, name='example_automatic_community', 
                                 ylabel='Presynaptic Neuron ID',
                                 xlabel='Postsynaptic Neuron ID',
                                 xticklabels = None,
                                 yticklabels = None,
                                 title='Community-Ordered Connectivity Matrix (Log Scale)',
                                 vmax = 1.0):
    """ Utility function for merging a number of nodes into one in networkx in place.
    
    # Arguments:
        W (numpy array): Adjacency matrix.
        name (str): Name for the diagram.
        xlabel (str): Name of the x label.
        ylabel (str): Name of the y label.
        xticklabels (list): x tick labels to have.
        yticklabels (list): y tick labels to have.
        title (str): Title for the diagram.
        vmax (float): Maximum value for the diagram.
    """
    W2 = np.zeros((W.shape[0]+W.shape[1], W.shape[0]+W.shape[1]))
    W2[:W.shape[0],W.shape[0]:] = W
    G = nx.from_numpy_matrix(W2, nx.DiGraph())
    Gun = G.to_undirected()
    partition = community.best_partition(Gun, randomize=None, resolution=1.00)
    all_nodes = []
    all_nodes = []
    all_list_nodes = []

    Gnodes = list(G.nodes())
    for com in set(partition.values()) :
        print('Cluster {} Members:'.format(com))
        list_nodes = [nodes for nodes in partition.keys()
                                    if partition[nodes] == com]
        all_list_nodes += [Gnodes.index(nodes) for nodes in partition.keys()
                                    if partition[nodes] == com]
        all_nodes.append(list_nodes)

    list_nodes = np.array(list_nodes)
    all_pre_nodes = [i for i in all_list_nodes]
    all_post_nodes = [i for i in all_list_nodes]
    A = nx.adjacency_matrix(G)[all_pre_nodes,:]
    A = A[:,all_post_nodes]
    B = A

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((1,1,1))

    Bd = np.log10(1.+B)
    sns.heatmap(Bd, vmax = vmax, cbar=False, xticklabels=xticklabels, yticklabels=yticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    fig.savefig(name + '.png', facecolor='white', edgecolor='none')
    return B, all_nodes
    
def merge_nodes(G, nodes, new_node, **attr):
    """ Utility function for merging a number of nodes into one in networkx in place.
    
    # Arguments:
        G (nx graph): A networkx graph.
        nodes (list): List of nodes to combine.
        new_node (str): Name of the new node.
        attr (dict): Optional. Additional attributes for the new node.
    """
    G.add_node(new_node, **attr) # Add node corresponding to the merged nodes
    edge_iterator = list(G.edges(data=True))
    for n1,n2,data in edge_iterator:
        if n1 in nodes:
            if G.has_edge(new_node,n2):
                w = data['weight']
                G[new_node][n2]['weight'] += w
            else:
                G.add_edge(new_node,n2,**data)
        elif n2 in nodes:
            G.add_edge(n1,new_node,**data)
            if G.has_edge(n1,new_node):
                w = data['weight']
                G[n1][new_node]['weight'] += w
            else:
                G.add_edge(n1,new_node,**data)
    
    for n in nodes: # remove the merged nodes
        if n in G.nodes():
            G.remove_node(n)

def find_driver_nodes(G):
    """ Finds driver nodes in a given graph.
    
    # Arguments:
        G (nx graph): A networkx graph.
    
    # Returns:
        list: Names of nodes.
    """
    g = nx.convert_node_labels_to_integers(G)
    a = maximum_matching_driver_nodes(g)
    nodes = [G.nodes()[i] for i in a]
    return nodes

def fast_smallworld_sigma(G, iterations = 1000, subgraph_size = 300):
    """ Calculates small-worldness coefficient sigma for very large networks (>10000 nodes) by running analysis on subgraphs.
    
    # Arguments:
        G (nx graph): A networkx graph.
        iterations (int): Number of iterations to use. Defaults to 1000.
        subgraph_size (int): Number of nodes per subgraph.
    
    # Returns:
        float: Mean sigma
        float: Variance of sigma
        numpy array: All raw data from which these values were calculated.
    """
    if isinstance(G, np.ndarray):
        G=nx.from_numpy_matrix(G, create_using=nx.DiGraph())
    Gun = nx.DiGraph.to_undirected(G)
    its = iterations
    ss = []
    for it in range(its):
        random.shuffle(nodes)
        Gsub = Gun.subgraph(nodes[:subgraph_size]).copy()

        if not nx.is_connected(Gsub):
            sub_graphs = list(nx.connected_component_subgraphs(Gsub))
            main_graph = sub_graphs[0]
            for sg in sub_graphs:
                if len(sg.nodes()) > len(main_graph.nodes()):
                    main_graph = sg
            cur_graph = main_graph
            print(len(cur_graph.nodes()))
        s = smallworld.sigma(cur_graph, niter = 1, nrand = 1)
        ss.append(s)
    ss = np.array(ss)
    mean = np.mean(ss)
    var = np.var(ss)
    return mean, var, ss

def draw_graph_with_groups(G, groups, graphname = 'mygraph', namekey = 'uname',
                 graph_struct = {'splines': 'ortho', 
                 'pad': '0.5',
                 'ranksep': '1.5',
                 'concentrate': 'true',
                 'newrank': 'true',
                 'rankdir': 'LR'},
                 node_struct = {'shape': 'box', 'height': '0.05'},
                 edge_struct = {'arrowsize': '0.5'},
                 group_struct = {'color': 'lightgrey'}):
    """ Basic utility function for generating a graph describing how groups connects to one another that relies on get_neuron_graph graphs.
    
    # Arguments:
        G (NetworkX graph): A NetworkX graph.
        groups (dict): A dictionary in which keys are groups and values are lists of corresponding neuron names (identifiers).
        graphname (str): Name of the graph to use for saving. Defaults to 'mygraph'.
        namekey (str): Key to use in the NetworkX graph for naming. Defaults to 'uname', can also be 'name'.
        graph_struct (dict): A dictionary structure for graph parameters. Optional.
        node_struct (dict): A dictionary structure for node parameters. Optional.
        edge_struct (dict): A dictionary structure for edge parameters. Optional.
        group_struct (dict): A dictionary structure for group parameters. Optional.
    # Example:
        client = fbl.get_client()
        _res = client.executeNLPquery('$VA1v$')
        G = client.get_neuron_graph(query_result = _res)
        nodes = list(G.nodes())
        groups = {'inputs': nodes[:5], 'locals': nodes[5:10], 'outputs': nodes[10:15]} # Create some arbitrary groups
        draw_graph_with_groups(G, groups)
    """

    g = Digraph('G', filename = graphname + '.gv',graph_attr = graph_struct)
    valid_nodes = []
    for group in groups.keys():
        with g.subgraph(name = 'cluster_'+group) as c:
            c.attr(style = 'filled', color = group_struct['color'])
            for _pre in groups[group]:
                valid_nodes.append(_pre)
                c.node(G.nodes[_pre][namekey], shape = node_struct['shape'], height = node_struct['height'])

    for i, j in G.edges():
        if i in valid_nodes and j in valid_nodes:
            g.edge(G.nodes[i][namekey], G.nodes[j][namekey], arrowsize = edge_struct['arrowsize'])

    g.attr(size='25,25')

    g.save(graphname + '.gv')

    g.render(graphname + '.svg', view=False)  
    return g

def graphviz_visualize(G, groups, graph_name = 'graphviz', visualize_in_notebook = True, update_dict = {}):
    """ Uses graphviz to visualize connections in a graph. Superseded by draw_graph_with_groups.
    
    # Arguments:
        G (nx graph): A networkx graph.
        groups (dict): A dictionary whose keys are groups and values are lists of neuron nodes comprising it.
        graph_name (str): Graph name to use for saving.
        visualize_in_notebook: Whether to return the graph for visualization in the notebook.
    
    # Returns:
        g: The graphviz graph object that will be drawn in the notebook.
    """
    graph_struct = {'splines': 'ortho', 
                    'pad': '0.5',
                    'ranksep': '1.5',
                     'concentrate': 'true',
                     'newrank': 'true',
                     'node_shape': 'box', 
                     'rankdir': 'LR',
                     'node_shape': 'box'}
    for i in update_dict:
        graph_struct[i] = update_dict[i]

    g = Digraph('G', filename=graph_name+'.gv', graph_attr = {'splines': 'ortho', 
                                                        'pad': '0.1',
                                                        'nodesep': '0.05',
                                                        'ranksep': '1.5',
                                                        'concentrate': 'true',
                                                        'newrank': 'true',
                                                        'rankdir': 'LR'})

    for i in groups:
        with g.subgraph(name='cluster_'+i) as c:
            c.attr(style='filled', color='lightgrey')
            for _pre in groups[i]:
                c.node(str(_pre), shape = graph_struct['node_shape'], height = '0.05')

    for i, j in G.edges():
        g.edge(str(i), str(j), arrowsize = '0.5')

    g.attr(size = '25,25')

    g.save(graph_name + '.gv')

    g.render(graph_name + '.svg', view=False)  
    if visualize_in_notebook:
        return g
    

def gen_heatmap(W,
                figsize = (10,10), 
                dpi = 100.,
                xlabel = 'Postsynaptic Neuron ID', 
                ylabel = 'Presynaptic Neuron ID',
                xticklabels = None,
                yticklabels = None,
                hlines = None,
                vlines = None,
                title = 'Connectivity Matrix (Log Scale)',
                vmax = None,
                export_format = 'png',
                figname = None):
    """ Utility function for visualizing an adjacency matrix heatmap.
    
    # Arguments:
        W (numpy array): Adjacency matrix.
        figsize (tuple): size of the figure.
        dpi (float): dpi of the figure.
        xlabel (str): Name of the x label.
        ylabel (str): Name of the y label.
        xticklabels (list): x tick labels to have.
        yticklabels (list): y tick labels to have.
        hlines (list): a list of y coordinates to draw horizontal lines on.
        vlines (list): a list of x coordinates to draw verticle lines on.
        title (str): Title for the diagram.
        vmax (float): Maximum value for the diagram.
        export_format (str): if specified, file format to export the diagram.
        figname (str): Name for the diagram.
    """
    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor((1, 1, 1))

    ax = sns.heatmap(W, vmax = vmax, xticklabels = xticklabels, yticklabels = yticklabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.setp(ax.get_xticklabels(),  fontsize=6)
    plt.setp(ax.get_yticklabels(),  fontsize=6)

    ax.hlines(hlines, *ax.get_xlim(), colors = 'w')
    ax.vlines(vlines, *ax.get_xlim(), colors = 'w')
    plt.show()
    if export_format is not None:
        fig.savefig('{}.{}'.format(figname, export_format), facecolor='white', edgecolor='none')

        