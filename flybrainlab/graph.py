
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
        format (string): format of the NA query return.
        queryID (string): ID of the NA query.
        NLPquery (string): If the NA query is initiated by a NLP query,
                           the original NLPquery.
        command
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
        self.graph = None
        # self.initialize()

    def initialize(self):
        if self.format == 'no_result':
            self.neurons = {}
            self.synapses = {}
            return

        self.morphology_to_send = set()
        if self.format == 'morphology':
            self.data= {}
        elif self.format == 'nx':
            self.data = {'nodes': {}, 'edges': []}
        self.locked = False

    def receive_data(self, data):
        if self.format == 'no_result':
            self.data = None
            return

        if self.format == 'morphology':
            self._receive_data_from_morphology_query(data)
        elif self.format == 'nx':
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

    def send_morphology(self, Comm, threshold = None):
        sent = set()
        data = {}
        for rid in self.morphology_to_send:
            node = self.data[rid].copy()
            morphology_data = node.pop('MorphologyData')
            node.update(morphology_data)
            data[morphology_data['rid']] = node
            sent.add(rid)

        if threshold is None:
            threshold = self.task.get('threshold', 5)
        for c in chunks(data, threshold):
            a = {}
            a["messageType"] = "Data"
            a["widget"] = "NLP"
            a["data"] = {"data": c, "queryID": self.queryID}
            Comm(a)
        self.morphology_to_send -= sent

    def _receive_data_from_morphology_query(self, data):
        # morphology graph will only return neurons/synapses and their morphology nodes as attribute.
        # neurons = {rid: v for rid, v in data.items() if v['class'] == 'Neuron'}
        # synapses = {rid: v for rid, v in data.items() if v['class'] == 'Synapse'}
        # assert len(set(data.keys()) - set(neurons.keys()) - set(synapses.keys())) == 0
        while self.locked:
            time.sleep(1)

        self.locked = True
        try:
            for rid, v in data.items():
                if 'MorphologyData' in v:
                    self.morphology_to_send.add(rid)
                    morphology_data = v['MorphologyData']
                    if "x" in morphology_data:
                        morphology_data["x"] = [x*self.x_scale+self.x_shift for x in morphology_data["x"]]
                    if "y" in morphology_data:
                        morphology_data["y"] = [y*self.y_scale+self.y_shift for y in morphology_data["y"]]
                    if "z" in morphology_data:
                        morphology_data["z"] = [z*self.z_scale+self.z_shift for z in morphology_data["z"]]
                    if "r" in morphology_data:
                        morphology_data["r"] = [r*self.r_scale+self.r_shift for r in morphology_data["r"]]

            self.data.update(data)
            self.locked = False
        except:
            self.locked = False
            raise

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

    def _finalize_nx(self):
        while self.locked:
            time.sleep(1)
        if self.format == 'nx':
            G = nx.MultiDiGraph()
            G.add_nodes_from(list(self.data['nodes'].items()))
            G.add_edges_from(self.data['edges'])
            self.graph = G

    def neurons(self):
        if self.graph is not None:
            return self.get('Neuron')
        else:
            return {rid: v for rid, v in self.data.items() if v['class'] == 'Neuron'}

    def synapses(self):
        if self.graph is not None:
            return self.get(['Synapses', 'InferredSynapses'])
        else:
            return {rid: v for rid, v in self.data.items() if v['class'] in ['Synapses', 'InferredSynapse']}

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
            cls = list(cls)
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
        self.enableResets = enableResets
        self.data = {}
        self.morphology_to_send = set()
        self.uname_to_rid = {}
        self.morphology_rid_to_uname_rid = {}

    def receive_cmd(self, data):
        if 'commands' in data:
            self.commands.append(data['commands'])

    def clear_cmd(self):
        self.commands = []

    def process_commands(self, Comm):
        while len(self.commands):
            command =self.commands.pop(0)
            if 'reset' in command and self.enableResets == False:
                continue
            a = {"data": {'commands': command},
                 "messageType": "Command",
                 "widget": "NLP"}
            Comm(a)
            if 'remove' in command:
                to_remove = command['remove'][0]
                for m_rid in to_remove:
                    try:
                        self.data.pop(self.morphology_rid_to_uname_rid[m_rid])
                    except KeyError:
                        pass
            if 'reset' in command:
                self.reset()

    def reset(self):
        self.data = {}
        self._refresh_data_map()

    def _refresh_data_map(self):
        self.uname_to_rid = {v['uname']: rid for rid, v in self.data.items() if 'uname' in v}
        self.morphology_rid_to_uname_rid = {v['MorphologyData']['rid']: rid \
                                            for rid, v in self.data.items()}

    def process_data(self, na_query_result, Comm):
        if na_query_result.format == 'morphology':
            self.data.update(na_query_result.data)
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
        return list(self.data.keys())

    def __getitem__(self, key):
        if key.startswith('#'):
            return self.data[key]
        else:
            rid = self.uname_to_rid.get(key, None)
            if rid is None:
                raise KeyError('Node with uname {} is not in the NLP result'.format(key))
            return self.data[rid]

    def __setitem__(self, key, value):
        if key.startswith('#'):
            rid = key
        else:
            rid = self.uname_to_rid.get(key, None)
            if rid is None:
                raise KeyError('Node with uname {} is not in the NLP result'.format(key))
        self.data[rid] = value


class NeuronGraph(nx.DiGraph):
    """
    Construct a graph of neurons where nodes are neurons and
    synapses are represented as edges. The weight of an edge equals to
    the number of synapses between the two connected neurons.

    # Arguments
        connectivity_query_result (NeuroNLPResult):
            query result from Client.getConnectivity() or equivalent query
    """
    def __init__(self, connectivity_query_result):
        super(NeuronGraph, self).__init__()

        if connectivity_query_result.graph is None:
            raise AttributeError('query result does not have a graph')
        graph = connectivity_query_result.graph
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

    def adjacency_matrix(self, uname_order = None, rid_order = None):
        """
        Get adjacency matrix between Neurons.

        # Arguments
            query_result (graph.NeuroNLPResult):
                If None, currently active Neurons in the NeuroNLP window will be used (default).
                If supplied, the neurons in the query_result will be used.
            uname_order (list):
                A list of the uname of neurons to order the rows and columns of the adjacency matrix.
                If None, use rid_order. If rid_order is None, will sort uname for order.
            rid_order (list):
                A list of the rids of neurons to order the rows and columns of the adjacency matrix.
                If None, use uname_order. if uname_order is None, will sort uname for order.
        # Returns
            M (networkx.MultiDiGraph):
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
        M = nx.adj_matrix(self, nodelist = rid_order).todense()
        return M, uname_order
