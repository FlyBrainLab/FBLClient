
import time

import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from .utils import chunks


class NAqueryResult(object):
    """
    Query result of an NeuroArch query via Client.executeNAquery.

    # Arguments
        task (dict):
            query task in NeuroArch JSON format
        x_scale, y_scale, z_scale, r_scale (float):
            factors to scale the morphology data (default: 1.0).
        x_shift, y_shift, z_shift, r_shift (float):
            factors to shift 3D coordinates of the morphology data and
            default radius (default: 0.0)

    # Attributes:
        task (dict):
            storage for the input task
        format (string):
            format of the NA query return.
        queryID (string):
            ID of the NA query.
        NLPquery (string):
            If the NA query is initiated by a NLP query, the original NLPquery.
        graph (networkx.MultiDiGraph):
            a graph representing the data retrieved from NA database.
    """
    def __init__(self, task, x_scale = 1.0, y_scale = 1.0, z_scale = 1.0,
                 r_scale = 1.0, x_shift = 0.0, y_shift = 0.0, z_shift = 0.0,
                 r_shift = 0.0):
        self.task = task
        self.format = task.get('format', 'morphology')
        self.queryID = task.get('queryID', '')
        self.NLPquery = task.get('NLPquery', None)
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.z_scale = z_scale
        self.r_scale = r_scale
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.z_shift = z_shift
        self.r_shift = r_shift
        self.graph = nx.MultiDiGraph()
        # self.initialize()

    def initialize(self):
        if self.format == 'no_result':
            self.data = {}
            return

        if self.format == 'morphology':
            self.data= {'nodes': {}, 'edges': []}
        elif self.format == 'nx':
            self.data = {'nodes': {}, 'edges': []}
        self.locked = False

    def receive_data(self, data):
        if self.format == 'no_result':
            self.data = {}
            return

        if self.format in ['morphology', 'nx']:
            self._receive_data_from_nx_query(data)
        elif self.format == 'df':
            self._get_result_from_df()
        elif self.format == 'get_data':
            self._get_result_from_get_data()
        elif self.format == 'nk':
            self._get_result_from_nk()
        else:
            raise ValueError('NAres format "{}" unrecognized'.format(format))

    def finalize(self):
        if self.format == 'no_result':
            return

        if self.format == 'morphology':
            self._finalize_morphology()
        elif self.format == 'nx':
            self._finalize_nx()
        self.data = {}

    def send_morphology(self, Comm, threshold = None):
        data = {}
        for rid, v in self.get('MorphologyData').items():
            morphology_data = v.copy()
            node = self.fromData(rid)
            if node is not None:
                morphology_data.update(self.graph.nodes[node])
            data[rid] = morphology_data
            data[rid]['orid'] = node

        if threshold is None:
            threshold = self.task.get('threshold', 5)
        if threshold == 'auto':
            threshold = 20
        for c in chunks(data, threshold):
            a = {}
            a["messageType"] = "Data"
            a["widget"] = "NLP"
            a["data"] = {"data": c, "queryID": self.queryID}
            Comm(a)

    def _receive_data_from_nx_query(self, data):
        while self.locked:
            time.sleep(1)

        self.locked = True
        try:
            self.data['nodes'].update(data['nodes'])
            self.data['edges'].extend(data['edges'])
            self.locked = False
        except:
            self.locked = False
            raise

    def _finalize_morphology(self):
        while self.locked:
            time.sleep(1)

        G = self.graph
        G.add_nodes_from(list(self.data['nodes'].items()))
        G.add_edges_from(self.data['edges'])

        for rid, morphology_data in self.get('MorphologyData').items():
            if "x" in morphology_data:
                morphology_data["x"] = [x*self.x_scale+self.x_shift for x in morphology_data["x"]]
            if "y" in morphology_data:
                morphology_data["y"] = [y*self.y_scale+self.y_shift for y in morphology_data["y"]]
            if "z" in morphology_data:
                morphology_data["z"] = [z*self.z_scale+self.z_shift for z in morphology_data["z"]]
            if "r" in morphology_data:
                morphology_data["r"] = [r*self.r_scale+self.r_shift for r in morphology_data["r"]]
            if "vertices" in morphology_data:
                vertices = morphology_data["vertices"]
                for j in range(len(vertices)//3):
                    vertices[j*3] = vertices[j*3]*self.x_scale + self.x_shift
                    vertices[j*3+1] = vertices[j*3+1]*self.y_scale + self.y_shift
                    vertices[j*3+2] = vertices[j*3+2]*self.z_scale + self.z_shift


    def _finalize_nx(self):
        while self.locked:
            time.sleep(1)

        G = self.graph
        G.add_nodes_from(list(self.data['nodes'].items()))
        G.add_edges_from(self.data['edges'])


    @property
    def neurons(self):
        return self.get('Neuron')

    @property
    def synapses(self):
        return self.get(['Synapse', 'InferredSynapse'])

    def fromData(self, data_rid):
        obj_rids = [rid for rid, _, v in self.graph.in_edges(data_rid, data=True) if v.get('class', None) == 'HasData']
        if len(obj_rids) == 1:
            return obj_rids[0]
        elif len(obj_rids) == 0:
            return None
        else:
            raise ValueError('Data found to be owned by 2 nodes, should not possible.')

    def getData(self, rid):
        data_rids = [data_rid for _, data_rid, v in self.graph.out_edges(rid, data=True) if v.get('class', None) == 'HasData']
        return data_rids

    def get(self, cls):
        """
        Get all data that belongs to the class cls

        # Arguments:
            cls (list or str):
                A list of str of the classes to be retrieved

        # Returns
            dict: A dict with OrientDB record IDs as keys with values the attributes of data.
        """
        if isinstance(cls, str):
            cls = [cls]
        assert isinstance(cls, list)
        return {rid: v for rid, v in self.graph.nodes(data=True) if v['class'] in cls}


class NeuroNLPResult(NAqueryResult):
    """
    A buffer processing commands and mirroring data to be sent to the NeuroNLP window.

    # Arguments:
        enableResets (bool):
            If False, will not reset the NeuroNLP window on 'show' queries.
            default: True.
    """
    def __init__(self, enableResets = True):
        self.commands = []
        self.processed_commands = []
        self.enableResets = enableResets
        self.graph = nx.DiGraph() #using only DiGraph as this class only deals with the morphology and it is one-to-one
        self.uname_to_rid = {}

    def receive_cmd(self, data):
        if 'commands' in data:
            self.commands.append(data['commands'])

    def clear_cmd(self):
        self.commands = []

    def process_commands(self, Comm):
        while len(self.commands):
            command =self.commands.pop(0)
            self.processed_commands.append(command)
            if 'reset' in command and self.enableResets == False:
                continue
            a = {"data": {'commands': command},
                 "messageType": "Command",
                 "widget": "NLP"}
            Comm(a)
            if 'remove' in command:
                to_remove = command['remove'][0]
                objs_to_remove = list(set([self.fromData(m_rid) for m_rid in to_remove])-set([None]))
                self.graph.remove_nodes_from(objs_to_remove+to_remove)
                self._refresh_data_map()
            if 'reset' in command:
                self.reset()
                
    def clear_history(self):
        self.processed_commands = []

    def reset(self):
        self.graph.clear()
        self._refresh_data_map()

    def _refresh_data_map(self):
        self.uname_to_rid = {v['uname']: rid for rid, v in self.graph.nodes(data=True)
                             if 'uname' in v and v.get('class', None) != 'MorphologyData'}

    def process_data(self, na_query_result, Comm):
        if na_query_result.format == 'morphology':
            self.graph.update(na_query_result.graph)
            self._refresh_data_map()
            na_query_result.send_morphology(Comm)

    #TODO
    def getInfo(self, rid):
        pass

    #TODO
    def getStats(self, rid = None, neuron_name = None):
        pass


    @property
    def rids(self):
        return list(self.uname_to_rid.values())

    def __getitem__(self, key):
        if key.startswith('#'):
            node = self.graph.nodes[key]
            if node.get('class', None) == 'MorphologyData':
                obj_node = self.graph.nodes[self.fromData(key)].copy()
                data_node = node.copy()
            else:
                data_nodes = [self.graph.nodes[n].copy() for n in self.getData(key)]
                obj_node = node.copy()
            for i, data_node in enumerate(data_nodes):
                obj_node[data_node.get('class', 'Data{}'.format(i))] = data_node
            return obj_node
        else:
            rid = self.uname_to_rid.get(key, None)
            if rid is None:
                raise KeyError('Node with uname {} is not in the NLP result'.format(key))
            data_node = self.graph.nodes[self.getData(rid)].copy()
            obj_node = self.graph.nodes[rid].copy()
            obj_node[data_node.get('class', 'Data')] = data_node
            return obj_node

    def __setitem__(self, key, value):
        if key.startswith('#'):
            rid = key
        else:
            rid = self.uname_to_rid.get(key, None)
            if rid is None:
                raise KeyError('Node with uname {} is not in the NLP result'.format(key))
        self.graph.add_node(rid, value)


class NeuronGraph(nx.DiGraph):
    """
    Construct a graph of neurons where nodes are neurons and
    synapses are represented as edges. The weight of an edge equals to
    the number of synapses between the two connected neurons.

    # Arguments
        connectivity_query_result (NeuroNLPResult or networkx.(Multi)DiGraph):
            query result from Client.getConnectivity() or equivalent query
    """
    def __init__(self, connectivity_query_result):
        super(NeuronGraph, self).__init__()

        if isinstance(connectivity_query_result, NAqueryResult):
            if connectivity_query_result.graph is None:
                raise AttributeError('query result does not have a graph')
            else:
                graph = connectivity_query_result.graph
        elif isinstance(connectivity_query_result, nx.Graph):
            graph = connectivity_query_result
        neurons = {n: v for n, v in graph.nodes(data = True) \
                    if v['class'] in ['Neuron']}
        synapses = {n: v for n, v in graph.nodes(data = True) \
                    if v['class'] in ['Synapse', 'InferredSynapse']}
        pre_to_synapse_edges = {post:pre for pre, post, prop in graph.edges(data = True) \
                                if prop.get('class', None) == 'SendsTo' and pre in neurons}
        synapse_to_post_edges = {pre:post for pre, post, prop in graph.edges(data = True) \
                                 if prop.get('class', None) == 'SendsTo' and post in neurons}

        connections = [(pre, synapse_to_post_edges[syn], synapses[syn]['N']) \
                       for syn, pre in pre_to_synapse_edges.items()]

        self.add_nodes_from(list(neurons.items()))
        self.add_weighted_edges_from(connections)
    def names(self):
        return sorted([self.nodes[n]['uname'] for n in self.nodes()])

    def adjacency_matrix(self, uname_order = None, rid_order = None):
        """
        Get adjacency matrix between Neurons.

        # Arguments
            uname_order (list):
                A list of the uname of neurons to order the rows and columns of the adjacency matrix.
                If None, use rid_order. If rid_order is None, will sort uname for order.
            rid_order (list):
                A list of the rids of neurons to order the rows and columns of the adjacency matrix.
                If None, use uname_order. if uname_order is None, will sort uname for order.
        # Returns
            M (numpy.ndarray):
                A graph representing the connectivity of the neurons.
            uname_oder (list):
                A list of unames by which the rows and columns of M are ordered.
        """
        if uname_order is None and rid_order is None:
            order = sorted([(self.nodes[n]['uname'], n) for n in self.nodes()])
            uname_order = [uname for uname, _ in order]
            rid_order = [rid for _, rid in order]
        elif uname_order is None:
            # rid_order
            uname_order = [self.nodes[n]['uname'] for n in rid_order]
        else:
            # uname_order
            order_dict = {self.nodes[n]['uname']: n for n in self.nodes()}
            rid_order = [order_dict[uname] for uname in uname_order]
        M = nx.adjacency_matrix(self, nodelist = rid_order).todense()
        return M, uname_order


class CircuitGraph(nx.MultiDiGraph):
    def __init__(self, connectivity_query_result):
        super(CircuitGraph, self).__init__()
        if isinstance(connectivity_query_result, NAqueryResult):
            if connectivity_query_result.graph is None:
                raise AttributeError('query result does not have a graph')
            else:
                graph = connectivity_query_result.graph
        elif isinstance(connectivity_query_result, nx.Graph):
            graph = connectivity_query_result
        self.add_nodes_from(list(graph.nodes(data = True)))
        self.add_edges_from(graph.edges(data = True))

    def copy(self):
        return self.__class__(self)

