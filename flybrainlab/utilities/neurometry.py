import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
import networkx as nx

def generate_neuron_stats(_input, scale = 'mum', scale_coefficient = 1., log=False):
    """Generates statistics for a given neuron.
    
    # Arguments:
        _input (str or np.array): Name of the file to use, or the numpy array to get as input.
        scale (str): Optional. Name of the measurement scale. Default to 'mum'.
        scale_coefficient (float): Optional. A number to multiply the input with if needed. Defaults to 1.
        
    # Returns:
        dict: A result dictionary with all results.
    """
    if isinstance(_input, str):
        a = pd.read_csv(_input, sep=' ', header=None, comment='#')
        X = a.values
    else:
        X = _input
    if X.shape[1]>7:
        X = X[:, X.shape[1]-7:]
    G = nx.DiGraph()
    distance = 0
    surface_area = 0
    volume = 0
    X[:,2:5] = X[:,2:5] * scale_coefficient
    for i in range(X.shape[0]):
        if X[i,6] != -1:
            G.add_node(i)
            parent = np.where(X[:,0] == X[i,6])[0][0]
            x_parent = X[parent,2:5]
            x = X[i,2:5]
            h = np.sqrt(np.sum(np.square(x_parent-x)))
            G.add_edge(parent,i,weight=h)
            distance += h
            r_parent = X[parent,5]
            r = X[i,5]
            surface_area += np.pi * (r + r_parent) * np.sqrt(np.square(r-r_parent)+np.square(h))
            volume += np.pi/3.*(r*r+r*r_parent+r_parent*r_parent)*h

    XX = X[:,2:5]
    w = np.abs(np.max(XX[:,0])-np.min(XX[:,0]))
    h = np.abs(np.max(XX[:,1])-np.min(XX[:,1]))
    d = np.abs(np.max(XX[:,2])-np.min(XX[:,2]))
    bifurcations = len(X[:,6])-len(np.unique(X[:,6]))
    max_euclidean_dist = np.max(pdist(XX))
    max_path_dist = nx.dag_longest_path_length(G)
    if log == True:
        print('Total Length: ', distance, scale)
        print('Total Surface Area: ', surface_area, scale+'^2')
        print('Total Volume: ', volume, scale+'^3')
        print('Maximum Euclidean Distance: ', max_euclidean_dist, scale)
        print('Width (Orientation Variant): ', w, scale)
        print('Height (Orientation Variant): ', h, scale)
        print('Depth (Orientation Variant): ', d, scale)
        print('Average Diameter: ', 2*np.mean(X[:,5]), scale)
        print('Number of Bifurcations:', bifurcations)
        print('Max Path Distance: ', max_path_dist, scale)
    results = {}
    results['Total Length'] = distance
    results['Total Surface Area'] = surface_area
    results['Total Volume'] = volume
    results['Maximum Euclidean Distance'] = max_euclidean_dist
    results['Width (Orientation Variant)'] = w
    results['Height (Orientation Variant)'] = h
    results['Depth (Orientation Variant)'] = d
    results['Average Diameter'] = 2*np.mean(X[:,5])
    results['Number of Bifurcations'] = bifurcations
    results['Max Path Distance'] = max_path_dist
    return results

def generate_naquery_neuron_stats(res, node):
    """Generates statistics for a given NAqueryResult.
    
    # Arguments:
        res (NAqueryResult): Name of the NAqueryResult structure to use.
        node (str): id of the node to use.
        
    # Returns:
        dict: A result dictionary with all results.
    """
    x = res.graph.nodes[node]
    X = np.vstack((np.array(x['sample']),
                   np.array(x['identifier']),
                   np.array(x['x']),
                   np.array(x['y']),
                   np.array(x['z']),
                   np.array(x['r']),
                   np.array(x['parent']))).T
    return generate_neuron_stats(X)

def morphometrics(res):
    """ computes the morphometric measurements of neurons in NAqueryResult.

    # Arguments:
        res (flybrainlab.graph.NAqueryResult): query result from an NeuroArch query.

    # Returns
        pandas.DataFrame: a data frame with morphometric measurements in each row and neuron unames in each column
    """
    metrics = {}
    for rid, attributes in res.neurons.items():
        morphology_data = [res.graph.nodes[n] for n in res.getData(rid) \
                           if res.graph.nodes[n]['class'] == 'MorphologyData' \
                              and res.graph.nodes[n]['morph_type'] == 'swc']
        if len(morphology_data):
            x = morphology_data[0]
            X = np.vstack((np.array(x['sample']),
                   np.array(x['identifier']),
                   np.array(x['x']),
                   np.array(x['y']),
                   np.array(x['z']),
                   np.array(x['r']),
                   np.array(x['parent']))).T
            uname = attributes['uname']
            metrics[uname] = generate_neuron_stats(X)
    return pd.DataFrame.from_dict(metrics)


def generate_neuron_shape(_input, scale = 'mum', scale_coefficient = 1., log=False):
    """Generates shape structures for the specified neuron.
    
    # Arguments:
        _input (str or np.array): Name of the file to use, or the numpy array to get as input.
        scale (str): Optional. Name of the measurement scale. Default to 'mum'.
        scale_coefficient (float): Optional. A number to multiply the input with if needed. Defaults to 1.
        
    # Returns:
        X: A result matrix with the contents of the input.
        G: A directed networkx graph with the contents of the input.
        distances: List of all distances in the .swc file.
    """
    if isinstance(_input, str):
        a = pd.read_csv(_input, sep=' ', header=None, comment='#')
        X = a.values
    else:
        X = _input
    if X.shape[1]>7:
        X = X[:, X.shape[1]-7:]
    G = nx.DiGraph()
    X[:,2:5] = X[:,2:5] * scale_coefficient
    distances = []
    for i in range(X.shape[0]):
        if X[i,6] != -1:
            parent = np.where(X[:,0] == X[i,6])[0][0]
            x_parent = X[parent,2:5]
            G.add_node(i, position_data = X[i,2:5], parent_position_data = X[parent,2:5], r = X[i,5])
            x = X[i,2:5]
            h = np.sqrt(np.sum(np.square(x_parent-x)))
            G.add_edge(parent,i,weight=h)
            distances.append(h)
        else:
            G.add_node(i, position_data = X[i,2:5], parent_position_data = X[i,2:5], r = X[i,5])
    return X, G, distances

def fix_swc(swc_file, new_swc_file,
            percentile_cutoff = 50,
            similarity_cutoff = 0.40,
            distance_multiplier = 5):
    """Tries to fix connectivity errors in a given swc file.
    
    # Arguments:
        swc_file (str or np.array): Name of the file to use, or the numpy array to get as input.
        new_swc_file (str): Name of the new swc file to use as output.
        percentile_cutoff (int): Optional. Percentile to use for inter-node cutoff distance during reconstruction for connecting two nodes. Defaults to 50.
        similarity_cutoff (float): Optional. Cosine similarity cutoff value between two endpoints' branches during reconstruction. Defaults to 0.8.
        distance_multiplier (float): Optional. A multiplier to multiply percentile_cutoff with. Defaults to 8.
        
    # Returns:
        G: A directed networkx graph with the contents of the input.
        G_d: A directed networkx graph with the contents of the input after the fixes.

    """
    X, G, distances = generate_neuron_shape(swc_file)
    endpoints = []
    endpoint_vectors = []
    endpoint_dirs = []
    for i in G.nodes():
        if len(list(G.successors(i)))==0:
            endpoints.append(i)
            endpoint_vectors.append(G.nodes()[i]['position_data'])
            direction = G.nodes()[i]['position_data'] - G.nodes()[i]['parent_position_data']
            if np.sqrt(np.sum(np.square(direction)))>0.:
                direction = direction / np.sqrt(np.sum(np.square(direction)))
            endpoint_dirs.append(direction)
    endpoint_vectors = np.array(endpoint_vectors)
    endpoint_dirs = np.array(endpoint_dirs)


    distance_cutoff = np.percentile(distances,percentile_cutoff)
    X_additions = []
    X_a_idx = int(np.max(X[:,0]))+1
    G_d = G.copy()
    for idx_a_i in range(len(endpoints)):
        for idx_b_j in range(idx_a_i+1,len(endpoints)):
            idx_a = endpoints[idx_a_i]
            idx_b = endpoints[idx_b_j]
            if np.abs(np.sum(np.multiply(endpoint_dirs[idx_a_i],endpoint_dirs[idx_b_j])))>similarity_cutoff:
                x = X[idx_b,2:5]
                x_parent = X[idx_a,2:5]
                if np.sqrt(np.sum(np.square(x_parent-x)))<distance_multiplier * distance_cutoff:
                    X_additions.append([X_a_idx,0,X[idx_b,2],X[idx_b,3],X[idx_b,4],X[idx_b,5],X[idx_a,0]])
                    X_a_idx += 1
                    G_d.add_edge(idx_a, idx_b)
    X_additions = np.array(X_additions)
    X_all = np.vstack((X, X_additions))
    X_pd = pd.DataFrame(X_all)
    X_pd[0] = X_pd[0].astype(int)
    X_pd[1] = X_pd[1].astype(int)
    X_pd[6] = X_pd[6].astype(int)
    X_pd.to_csv(new_swc_file, sep=' ', header=None, index=None)
    return G, G_d



def fix_swc_components(swc_file, new_swc_file,
                       percentile_cutoff = 50,
                       similarity_cutoff = 0.40,
                       distance_multiplier = 5):
    """Tries to fix connectivity errors in a given swc file and connect disconnected components.
    
    # Arguments:
        swc_file (str or np.array): Name of the file to use, or the numpy array to get as input.
        new_swc_file (str): Name of the new swc file to use as output.
        percentile_cutoff (int): Optional. Percentile to use for inter-node cutoff distance during reconstruction for connecting two nodes. Defaults to 50.
        similarity_cutoff (float): Optional. Cosine similarity cutoff value between two endpoints' branches during reconstruction. Defaults to 0.8.
        distance_multiplier (float): Optional. A multiplier to multiply percentile_cutoff with. Defaults to 8.
        
    # Returns:
        G: A directed networkx graph with the contents of the input.
        G_d: A directed networkx graph with the contents of the input after the fixes.
        G_d_uncon: An undirected networkx graph with the contents of the input after the fixes with no disconnected components.

    """
    X, G, distances = generate_neuron_shape(swc_file)
    endpoints = []
    endpoint_vectors = []
    endpoint_dirs = []
    for i in G.nodes():
        if len(list(G.successors(i)))==0:
            endpoints.append(i)
            endpoint_vectors.append(G.nodes()[i]['position_data'])
            direction = G.nodes()[i]['position_data'] - G.nodes()[i]['parent_position_data']
            if np.sqrt(np.sum(np.square(direction)))>0.:
                direction = direction / np.sqrt(np.sum(np.square(direction)))
            endpoint_dirs.append(direction)
    endpoint_vectors = np.array(endpoint_vectors)
    endpoint_dirs = np.array(endpoint_dirs)


    distance_cutoff = np.percentile(distances,percentile_cutoff)
    X_additions = []
    X_a_idx = int(np.max(X[:,0]))+1
    G_d = G.copy()
    for idx_a_i in range(len(endpoints)):
        for idx_b_j in range(idx_a_i+1,len(endpoints)):
            idx_a = endpoints[idx_a_i]
            idx_b = endpoints[idx_b_j]
            if np.abs(np.sum(np.multiply(endpoint_dirs[idx_a_i],endpoint_dirs[idx_b_j])))>similarity_cutoff:
                x = X[idx_b,2:5]
                x_parent = X[idx_a,2:5]
                if np.sqrt(np.sum(np.square(x_parent-x)))<distance_multiplier * distance_cutoff:
                    X_additions.append([X_a_idx,0,X[idx_b,2],X[idx_b,3],X[idx_b,4],X[idx_b,5],X[idx_a,0]])
                    X_a_idx += 1
                    G_d.add_edge(idx_a, idx_b)
                    
    
    G_d_uncon = nx.Graph(G_d)
    processing = True
    X_disconnected_additions = []
    while processing == True:
        components = []
        for component in nx.connected_components(G_d_uncon):
            components.append(list(component))
        if len(components)<2:
            processing = False
        else:
            print(len(components))
            components_endpoints = []
            component_matrices = []
            for component in components:
                component_endpoints = []
                component_matrix = []
                for i in component:
                    if i in endpoints:
                        component_endpoints.append(i)
                        component_matrix.append(G_d_uncon.nodes()[i]['position_data'])
                component_matrix = np.array(component_matrix)
                components_endpoints.append(component_endpoints)
                component_matrices.append(component_matrix)


            max_dist = 10000.
            min_a = 0
            min_b = 0
            min_vals = None
            for component_idx in range(len(components)):
                for component_idx_b in range(component_idx+1, len(components)):
                    DD = pairwise_distances(component_matrices[component_idx], component_matrices[component_idx_b])
                    if np.min(DD)<max_dist:
                        max_dist = np.min(DD)
                        min_a = component_idx
                        min_b = component_idx_b
                        min_vals = np.unravel_index(DD.argmin(), DD.shape)
            G_d_uncon.add_edge(components_endpoints[min_a][min_vals[0]], components_endpoints[min_b][min_vals[1]])
            print(min_a, min_b)
            X_disconnected_additions.append([X_a_idx,0,X[components_endpoints[min_a][min_vals[0]],2],X[components_endpoints[min_a][min_vals[0]],3],X[components_endpoints[min_a][min_vals[0]],4],X[components_endpoints[min_a][min_vals[0]],5],X[components_endpoints[min_b][min_vals[1]],0]])
            X_a_idx += 1
                    
    X_additions = np.array(X_additions)
    X_disconnected_additions = np.array(X_disconnected_additions)
    X_all = np.vstack((X, X_additions, X_disconnected_additions))
    X_pd = pd.DataFrame(X_all)
    X_pd[0] = X_pd[0].astype(int)
    X_pd[1] = X_pd[1].astype(int)
    X_pd[6] = X_pd[6].astype(int)
    X_pd.to_csv(new_swc_file, sep=' ', header=None, index=None)
    return G, G_d, G_d_uncon