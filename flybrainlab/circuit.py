
import time
import json
from copy import deepcopy
import traceback, sys

import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from .graph import NAqueryResult, CircuitGraph
from .exceptions import FlyBrainLabNAserverException, FlyBrainLabNKserverException

def numberGenerator():
    number = -1
    while True:
        yield str(number)
        number -= 1

class ExecutableCircuit(object):
    def __init__(self, client, circuit = None, model_name = None, model_version = None,
                 complete_synapses = True, synapse_threshold = 0, callback = True):
        """
        Initilize the construction of an executable circuit.

        # Arguments
            client (flybrainlab.Client):
                The client to be connected to.
            circuit (graph.NAqueryResult or None):
                If None, will start constructing the model from retrieving model from the DB using model_name and version.
                If supplied, will be used to construct a graph.CircuitGraph.
            synapse_threshold (int or None):
                The connections between neurons that are larger than synapse_threshold will added (default 5).
            complete_synapses (bool or None):
                Whether or not to include all synapses between pairs of neurons in the query_result.
                If True, all synapses will be considered even they are not in the query_result.
                If False, only the synapses in the query_result will be used to construct the graph.
                (default True).
            model_name (str or None), model_version (str or None):
                If specified, the model_name and model_version will be used to retrieve a executable circuit
        """
        self.client = client
        if callback:
            client.experimentWatcher = self
        self.number_generator = numberGenerator()
        self._js = []

        # obtain self.circuit as a CircuitGraph
        # and self.model circuit as an empty graph or CircuitGraph
        if circuit is not None:
            self.circuit = client.get_circuit_graph(
                                circuit,
                                synapse_threshold = synapse_threshold,
                                complete_synapses = complete_synapses)
            existing_models = self.query_for_models()
            if model_name is not None:
                if model_name in existing_models:
                    if model_version is None:
                        self.current_model = self.prompt_to_choose_model(
                                    {rid: v for rid, v in existing_models.items() \
                                     if v['name'] == model_name})
                    else:
                        tmp = [rid for rid, v in existing_models.items() \
                               if v['name'] == model_name and v['version'] == model_version]
                        if len(tmp):
                            self.current_model = tmp[0]
                        else:
                            print('Model with name {} version {} does not exist. \
                                   Initializing a new executable circuit'.format(model_name, model_version))
                else:
                    print('Model with name {} version {} does not exist. \
                           Initializing a new executable circuit'.format(model_name, model_version))
                    self.current_model = None
            else:
                if len(existing_models):
                    self.current_model = self.prompt_to_choose_model(
                                {rid: v for rid, v in existing_models.items()})
                else:
                    self.current_model = None
                    print('Initializing a new executable circuit')
            if self.current_model is None:
                self.graph = self.circuit.copy()
                if model_name is None or model_version is None:
                    raise ValueError("model_name and model_version must both be specified to initialize a new exeuctable circuit.")
                else:
                    self.current_model_name = model_name
                    self.current_model_version = model_version
            else:
                print(self.current_model)
                tmp = existing_models[self.current_model]
                self.current_model_name = tmp['name']
                self.current_model_version = tmp['version']
                self.graph = self._get_graph_from_circuit_and_model(self.circuit, self.current_model)
                self._get_diagram(self.current_model)
        else:
            if model_name is None:
                raise ValueError('Executable Circuit must be speicified by a circuit or model name.')
            else:
                existing_models = self.query_for_models_by_name(model_name)
                if len(existing_models) == 0:
                    raise ValueError('Model with name {} does not exist. \
                           Cannot initialize without a circuit'.format(model_name))
                if model_version is None:
                    self.current_model = self.prompt_to_choose_model(
                                {rid: v for rid, v in existing_models.items() \
                                 if v['name'] == model_name})
                else:
                    tmp = [v['rid'] for name, v in existing_models.items() \
                           if name == model_name and v['version'] == model_version]
                    if len(tmp):
                        self.current_model = tmp[0]
                    else:
                        raise ValueError('Model with name {} version {} does not exist. \
                               Cannot initialize without a circuit'.format(model_name, model_version))

            tmp = existing_models[self.current_model]
            self.current_model_name = tmp['name']
            self.current_model_version = tmp['version']
            self.ciruit = self._get_circuit_from_model(self.current_model)
            self.graph = self._get_graph_from_model(self.current_model)
            self._get_diagram(self.current_model)
        self._updated = False

    def prompt_to_choose_model(self, models):
        list_of_models = list(models.items())
        res = input('Please select from the exisiting models to initialize the executable circuit, or press a to abort\n{}'.format('\n'.join(['{}: {} version {} (rid {})'.format(i, v['name'], v['version'], rid) for i, (rid, v) in enumerate(list_of_models)])))
        while True:
            if res.isnumeric():
                res = int(res)
                if res < len(list_of_models):
                    model_rid = list_of_models[res][0]
                    break
            elif res == 'a':
                raise KeyboardInterrupt
            elif res == 'n':
                model_rid = None
                break
        return model_rid

    def query_for_models(self):
        task = {"format": "nx",
                "query":[
                         {"action":{"method": {'query': {} }},
                          "object":{"rid": list(self.circuit.nodes())}
                         },
                         {"action":{"method": {"gen_traversal_in":{"pass_through": [["Models"]], "min_depth": 1}}},
                          "object":{"memory": 0}
                         },
                         {"action":{"method": {"traverse_owned_by":{"cls": "ExecutableCircuit", "max_levels": 2}}},
                          "object":{"memory": 0}
                         },
                        ]
                }
        res = self.client.executeNAquery(task, temp = True)
        return {rid: {'name': model['name'], 'version': model['version']} for rid, model in res.get('ExecutableCircuit').items()}

    def query_for_models_by_name(self, model_name):
        task = {"format": "nx",
                "query":[
                         {"action":{"method": {'has': {'name': model_name} }},
                          "object":{"class": 'ExecutableCircuit'}
                         },
                        ]
                }
        res = self.client.executeNAquery(task, temp = True)
        return  {rid: {'name': model['name'], 'version': model['version']} for rid, model in res.get('ExecutableCircuit').items()}

    def _get_graph_from_model(self, model_rid):
        task = {"format": "nx",
                "query":[
                        {"action": {"method": {"query": {}}},
                         "object": {"rid": [model_rid]}},
                        {"action": {"method": {"owns": {}}},
                         "object":{"memory": 0}},
                        {"action": {"method": {"gen_traversal_out": {"pass_through": [["Models"]], "min_depth": 0}}},
                         "object":{"memory": 0}},
                ]}
        res = self.client.executeNAquery(task, temp = True)
        return CircuitGraph(res)

    def _get_circuit_from_model(self, model_rid):
        task = {"format": "nx",
                "query":[
                        {"action": {"method": {"query": {}}},
                         "object": {"rid": [model_rid]}},
                        {"action": {"method": {"owns": {}}},
                         "object":{"memory": 0}},
                        {"action": {"method": {"gen_traversal_out": {"pass_through": [["Models"]], "min_depth": 1}}},
                         "object":{"memory": 0}},
                ]}
        res = self.client.executeNAquery(task, temp = True)
        return CircuitGraph(res)

    def _get_graph_from_circuit_and_model(self, circuit, model_rid):
        task = {"format": "nx",
                "query":[
                        {"action": {"method": {"query": {}}},
                         "object": {"rid": list(circuit.nodes())}},
                        {"action": {"method": {"gen_traversal_in": {"pass_through": [["Models"]], "min_depth": 1}}},
                         "object": {"memory": 0}},
                        {"action": {"method": {"query": {}}},
                         "object": {"rid": [model_rid]}},
                        {"action": {"method": {"gen_traversal_out": {"pass_through": [["Owns"], ["Owns"]], 'min_depth': 2}}},
                         "object": {"memory": 0}},
                        {"action": {"op": {"__and__": {"memory": 2}}},
                         "object": {"memory": 0}},
                        {"action": {"op": {"__add__": {"memory": 4}}},
                         "object": {"memory": 0}},
                ]}
        res = self.client.executeNAquery(task, temp = True)
        return CircuitGraph(res)

    @property
    def uname_to_rid(self):
        return {v['uname']: n for n, v in self.graph.nodes(data=True) if 'uname' in v and v['class'] in ['Neuron', 'Synapse']}

    def _disable_single_neuron(self, neuron_name):
        """
        Main part of single neuron removal.

        Steps: 1. Remove entries in subregion_arborization
               1. remove edges assocated with ArborizesIn
               2. remove synapses that are assciated with the neuron
               3. remove all edges associated with the neuron
        """
        rid = self.uname_to_rid[neuron_name]

        in_edges = self.graph.in_edges(rid, data = True)
        out_edges = self.graph.out_edges(rid, data = True)

        for pre, post, k in in_edges:
            if k['class'] == 'SendsTo':
                if self.graph.nodes[pre]['class'] in ['Synapse', 'InferredSynapse']:
                    self._disable_synapse(self.graph.nodes[pre]['uname'])

        for pre, post, k in out_edges:
            if k['class'] == 'SendsTo':
                if self.graph.nodes[post]['class'] == 'Synapse':
                    self._disable_synapse(self.graph.nodes[post]['name'])

    def _disable_neuron(self, neuron_name):
        """
        Remove a neuron from the diagram.
        Should be called by GUI after simExperimentConfig is updated.

        Parameters
        ----------
        neuron_name: str
                     name of the neuron
        """
        if neuron_name in self.config['active']['neuron']:
            self.config['inactive']['neuron'][neuron_name] = \
                self.config['active']['neuron'].pop(neuron_name)
            self._disable_single_neuron(neuron_name)

    def disable_neurons(self, neurons):
        """
        Disable neurons that exist in the diagram.

        Parameters
        ----------
        neurons:  list of neuron names
        """
        for neuron in neurons:
            self._disable_neuron(neuron)
        self.send_to_GFX()

    def _disable_synapse(self, synapse_name):
        """
        Remove a synapse from the diagram.
        Should be called by GUI after simExperimentConfig is updated.

        Parameters
        ----------
        synapse_name: str
                     uname of the synapse
        """
        if synapse_name in self.config['active']['synapse']:
            self.config['inactive']['synapse'][synapse_name] = \
                self.config['active']['synapse'].pop(synapse_name)


    def disable_synapses(self, synapses):
        """
        Disable synapses that exist in the diagram.

        Parameters
        ----------
        synapses:  list of synapse names
        """
        for synapse in synapses:
            self._disable_synapse(synapse)
        self.send_to_GFX()

    def enable_neurons(self, neurons):
        """
        Enable neurons that exist in the diagram.

        Parameters
        ----------
        neurons:  list of neuron names
        """
        for neuron in neurons:
            self._enable_neuron(neuron)
        self.send_to_GFX()

    def _enable_neuron(self, neuron_name):
        """
        Add a neuron that exists in diagram.
        Should be called by GUI after simExperimentConfig is updated.

        Parameters
        ----------
        neuron_name: str
                     uname of the neuron
        """
        if neuron_name in self.config['inactive']['neuron']:
            self.config['active']['neuron'][neuron_name] = \
                self.config['inactive']['neuron'].pop(neuron_name)

            rid = self.uname_to_rid[neuron_name]

            in_edges = self.graph.in_edges(rid, data = True)
            out_edges = self.graph.out_edges(rid, data = True)

            for pre, post, k in list(in_edges):
                if k['class'] == 'SendsTo':
                    if self.graph.nodes[pre]['class'] == 'Synapse':
                        self._enable_synapse(self.graph.nodes[pre]['uname'])

            for pre, post, k in list(out_edges):
                if k['class'] == 'SendsTo':
                    if self.graph.nodes[post]['class'] == 'Synapse':
                        self._enable_synapse(self.graph.nodes[post]['uname'])

    def enable_synapses(self, synapses):
        """
        Enable synapses that exist in the diagram.

        Parameters
        ----------
        synapses:  list of synapse unames
        """
        for synapse in synapses:
            self._enable_synapse(synapse)
        self.send_to_GFX()

    def _enable_synapse(self, synapse_name):
        if synapse_name in self.config['inactive']['synapse']:
            self.config['active']['synapse'][synapse_name] = \
                self.config['inactive']['synapse'].pop(synapse_name)

    def loadExperimentConfig(self, x):
        try:
            lastObject = x['lastObject']
            lastLabel = x['lastLabel']
            action = x['lastAction']
            if lastObject == "neuron":
                if action == "deactivated":
                    self._disable_neuron(lastLabel)
                elif action == "activated":
                    self._enable_neuron(lastLabel)
                elif action == "toggled":
                    if lastLabel in self.config['active']['neuron']:
                        self._disable_neuron(lastLabel)
                    elif lastLabel in self.config['inactive']['neuron']:
                        self._enable_neuron(lastLabel)
            elif lastObject == "synapse":
                if lastLabel in self.config['active']['synapse']:
                    self._disable_synapse(lastLabel)
                elif lastLabel in self.config['inactive']['synapse']:
                    self._enable_synapse(lastLabel)
        except KeyError:
            pass
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self.client.raise_error(e, "An error occured during last action update:\n" + tb)

        try:
            n = x['lastUpdated']
        except KeyError:
            raise

        try:
            circuit_node = self.graph.nodes[self.uname_to_rid[n]]
            v = deepcopy(x[self.current_model_name][n])
            if not v.get('params', {}).get('name', 'Default') == 'Default':
                self.update_model(n, v['params'], v['states'], no_send = True)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
            self.client.raise_error(e, "An error occured during model update:\n" + tb)
        self.send_to_GFX()

    def find_model(self, rid):
        return {pre: self.graph.nodes[pre] for pre, post, v in self.graph.in_edges(rid, data = True) if v['class'] == 'Models'}

    def find_neurotransmitter(self, rid):
        return {post: self.graph.nodes[post] for pre, post, v in self.graph.out_edges(rid, data = True) if v['class'] == 'HasData' and self.graph.nodes[post]['class'] == 'NeurotransmitterData'}

    def all_inactive(self):
        return list(self.config['inactive']['neuron'].keys()) + \
               list(self.config['inactive']['synapse'].keys())

    def send_to_GFX(self):
        print('sending circuit configuration to GFX')
        config = {self.current_model_name: {k: v for items in self.config['active'].values()\
                            for k, v in items.items()}}
        config[self.current_model_name]['disabled'] = self.all_inactive()
        config[self.current_model_name]['updated'] = []

        config_tosend = json.dumps(config)
        self.client.JSCall(messageType = 'setExperimentConfig',
                        data = config_tosend)
        time.sleep(2)

    def _get_diagram(self, model_rid):
        task = {"format": "nx",
                "query":[
                         {"action":{"method": {'query': {} }},
                          "object":{"rid": [model_rid]}
                         },
                         {"action":{"method": {"gen_traversal_out":{"pass_through": [["HasData", "CircuitDiagram"]], "min_depth": 1}}},
                          "object":{"memory": 0}
                         }
                        ]
                }
        res = self.client.executeNAquery(task, temp = True)
        circuit_diagrams = {rid: v for rid, v in res.graph.nodes(data = True) if v['class'] == 'CircuitDiagram'}
        tmp = circuit_diagrams.popitem()[1]
        diagram = tmp['diagram']
        js = tmp['js']
        self._load_diagram_str(diagram)
        self._load_js_str(js)

    def load_diagram(self, filename):
        with open(filename, 'r') as file:
            data = file.read()
        self._load_diagram_str(data)

    def _load_diagram_str(self, data):
        self._diagram = data
        self.client.tryComms({'widget':'GFX',
                           'messageType': 'loadCircuitFromString',
                           'data': {'string': data, 'name': 'diagram'}})
        time.sleep(1)
        self.initialize_diagram_config()

    def initialize_diagram_config(self):
        config = {'inactive': {'neuron': {},
                               'synapse': {}},
                  'active': {'neuron': {},
                             'synapse': {}}}

        for rid, v in self.get('Neuron').items():
            model_node = self.find_model(rid)
            if len(model_node):
                node_data = deepcopy(model_node.popitem()[1])

                new_node_data = {'params': {k: float(v) for k, v in node_data.pop('params', {}).items()},
                                 'states': {k: float(v) for k, v in node_data.pop('states', {}).items()}}
                new_node_data['params']['name'] = node_data.pop('class')
                new_node_data.update(node_data)
                config['active']['neuron'][v['uname']] = new_node_data
            else:
                node_data = {'name': 'Default'}
                new_node_data = {'params': node_data,
                                 'states': {}}
                config['active']['neuron'][v['uname']] = new_node_data
        for rid, v in self.get('Synapse').items():
            model_node = self.find_model(rid)
            if len(model_node):
                node_data = deepcopy(model_node.popitem()[1])
                new_node_data = {'params': {k: float(v) for k, v in node_data.pop('params', {}).items()},
                                 'states': {k: float(v) for k, v in node_data.pop('states', {}).items()}}
                new_node_data['params']['name'] = node_data.pop('class')
                new_node_data.update(node_data)
                config['active']['synapse'][v['uname']] = new_node_data
            else:
                node_data = {'name': 'Default'}
                new_node_data = {'params': node_data,
                                 'states': {}}
                config['active']['synapse'][v['uname']] = new_node_data

        self.config = config
        self.send_to_GFX()

    def load_js(self, filename):
        with open(filename, 'r') as file:
            data = file.read()
        self._load_js_str(data)

    def _load_js_str(self, data):
        self._js.append(data)
        self.client.tryComms({'messageType': 'eval', 'widget':'GFX',
                           'data': {'data': data}})
        time.sleep(1)

    def clear_js(self):
        self._js = []

    def create_executable_graph(self, model_name):
        not_modeled_nodes = [v['uname'] for n, v in self.graph.nodes(data=True) \
                          if v['class'] in ['Neuron', 'Synapse', 'InferredSynapse'] \
                          and not self._exists_model(n)]
        if len(not_modeled_nodes):
            raise ValueError('The following neurons/synapses in the circuit do not have a corresponding model:\n\
                             {}'.format(','.join(not_modeled_nodes)))

        g = nx.MultiDiGraph()
        neurons_modeled = {rid: v for rid, v in self.graph.nodes(data = True) \
                       if v['class'] in ['Neuron']}
        synapses_modeled = {rid: v for rid, v in self.graph.nodes(data = True) \
                       if v['class'] in ['Synapse', 'InferredSynapse']}
        for rid in list(neurons_modeled.keys())+list(synapses_modeled.keys()):
            model_rid, v = self.find_model(rid).popitem()
            params = {p: float(n) for p, n in v['params'].items()}
            params['class'] = v['class']
            params.update({'init{}'.format(k): float(v) for k, v in v['states'].items()})
            g.add_node(model_rid, **params)

        for synapse in synapses_modeled:
            post_neuron = [self.find_model(post).popitem()[0] for pre, post, v in self.graph.out_edges(synapse, data = True) if v['class'] == 'SendsTo'][0]
            pre_neuron = [self.find_model(pre).popitem()[0] for pre, post, v in self.graph.in_edges(synapse, data = True) if v['class'] == 'SendsTo'][0]
            synapse_model = self.find_model(synapse).popitem()
            if synapse_model[1]['class'] == 'SynapseNMDA':
                g.add_edge(pre_neuron, synapse_model[0], variable = 'spike_state')
                g.add_edge(synapse_model[0], post_neuron)
                g.add_edge(post_neuron, synapse_model[0], variable = 'V')
            else:
                g.add_edge(pre_neuron, synapse_model[0])
                g.add_edge(synapse_model[0], post_neuron)
        return g

    def update_model(self, node, params, states = None, no_send = False):
        circuit_node = self.graph.nodes[self.uname_to_rid[node]]
        model_node = self.find_model(self.uname_to_rid[node])
        if len(model_node) == 1:
            rid, p = model_node.popitem()
            v = {'uname': p['uname']}
            v['params'] = deepcopy(params)
            try:
                v['class'] = v['params'].pop('name')
            except KeyError:
                assert 'class' in v
            if v['class'] in ['Neuron', 'Synapse', 'InferredSynapse']: # replace with subclass call to models
                raise ValueError('Model cannot use biological node class')
            v['states'] = states if states is not None else p['states']
            nx.set_node_attributes(self.graph, {rid: v})
        elif len(model_node) == 0:
            v = {'uname': node}
            v['params'] = deepcopy(params)
            try:
                v['class'] = v['params'].pop('name')
            except KeyError:
                assert 'class' in v
            if v['class'] in ['Neuron', 'Synapse', 'InferredSynapse']:
                raise ValueError('Model cannot use biological node class')
            v['states'] = states if states is not None else params['states']
            new_node_id = next(self.number_generator)
            self.graph.add_node(new_node_id, **v)
            self.graph.add_edge(new_node_id, self.uname_to_rid[node], **{'class': 'Models'})
        else:
            raise ValueError('Duplicate nodes {} detected.'.format(node))
        node_type = circuit_node['class'].lower()

        pp = deepcopy(v)
        pp['params']['name'] = pp.pop('class')
        pp.pop('uname')
        if node in self.config['active'][node_type]:
            self.config['active'][node_type][node] = pp
        elif n in self.config['inactive'][node_type]:
            self.config['inactive'][node_type][node] = pp
        else:
            raise ValueError('{} not in either active or inactive'.format(n))
        self._updated = True
        if not no_send:
            self.send_to_GFX()

    def update_model_like(self, nodes, node_to_copy):
        if not isinstance(nodes, list):
            nodes = [nodes]
        # circuit_node = self.graph.nodes[self.uname_to_rid[node_to_copy]]
        model_node = self.find_model(self.uname_to_rid[node_to_copy])
        if len(model_node) == 1:
            rid, v = model_node.popitem()
        elif len(model_node) == 0:
            raise ValueError('No node {} detected.'.format(node_to_copy))
        else:
            raise ValueError('Duplicate nodes {} detected.'.format(node_to_copy))
        vv = deepcopy(v)
        vv['params']['name'] = vv['class']
        for node in nodes:
            self.update_model(node, vv['params'], vv['states'], no_send = True)
        self.send_to_GFX()

    def update_models(self, node_params):
        for node, v in node_params.items():
            self.update_model(node, v['params'], states = v.get('states', None),
                              no_send = True)
        self.send_to_GFX()

    def flush_model(self, model_name = None, model_version = None):
        not_modeled_nodes = [v['uname'] for n, v in self.graph.nodes(data=True) \
                          if v['class'] in ['Neuron', 'Synapse', 'InferredSynapse'] \
                          and not self._exists_model(n)]
        if len(not_modeled_nodes):
            raise ValueError('The following neurons/synapses in the circuit do not have a corresponding model:\n\
                             {}'.format(','.join(not_modeled_nodes)))

        graph = self._reform_graph()
        res = self.client.rpc('ffbo.na.NeuroArch.write.{}'.format(self.client.naServerID),
                        'create_model_from_circuit',
                        model_name if model_name is not None else self.current_model_name,
                        model_version if model_version is not None else self.current_model_version,
                        {'nodes': list(graph.nodes(data=True)),
                         'edges': list(graph.edges(data=True))},
                        circuit_diagram = self._diagram, js = '\n'.join(self._js))
        if 'success' in res:
            rid_map = res['success']['data']
        else:
            raise FlyBrainLabNAserverException(res['error']['message']+'\n'+res['error']['exception'])
        self._update_node_rids(rid_map)
        if model_name is not None:
            self.current_model_name = model_name
        if model_version is not None:
            self.current_model_version = model_version

    def _reform_graph(self):
        graph = self.graph.copy()
        neurons_modeled = {rid: v for rid, v in graph.nodes(data = True) \
                       if v['class'] in ['Neuron']}
        synapses_modeled = {rid: v for rid, v in graph.nodes(data = True) \
                       if v['class'] in ['Synapse', 'InferredSynapse']}
        for rid in list(neurons_modeled.keys())+list(synapses_modeled.keys()):
            model_rid, v = self.find_model(rid).popitem()
            new_v = {}
            new_v['params'] = {p: float(n) for p, n in v['params'].items()}
            new_v['class'] = v['class']
            new_v['states'] = {k: float(v) for k, v in v['states'].items()}
            nx.set_node_attributes(graph, {model_rid: new_v})
        return graph

    def _update_node_rids(self, mapping):
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=False)
        self._updated = False

    def _exists_model(self, rid):
        return len(self.find_model(rid)) > 0

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

    def remove_components(self):
        """
        Query database to remove disabled neurons and synapses.
        This should be called right before self.execution()
        """
        disabled_neurons = list(self.config['inactive']['neuron'].keys())
        disabled_synapses = list(self.config['inactive']['synapse'].keys())
        query_list = [
                {"action": {"method":
                    {"query":{"name": self.current_model_name,
                              "version": self.current_model_version}}},
                 "object":{"class":"ExecutableCircuit"}},
                {"action": {"method":{"traverse_owns":{}}},
                 "object":{"memory":0}}
              ]
        res = self.client.executeNAquery({'query': query_list, 'format': 'nx'})

        if len(disabled_neurons):
            query_list = [{"action":{"method":{"has":{"name": disabled_neurons}}},"object":{"state":0}},
                    {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":0}},
                    {"action":{"op":{"find_matching_ports_from_selector":{"memory":0}}},"object":{"state":0}},
                    {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"method":{"gen_traversal_in":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":0}},
                    {"action":{"method":{"gen_traversal_out":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":1}},
                    {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"op":{"__add__":{"memory":3}}},"object":{"memory":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"state":0}}]
        else:
            query_list = []

        if len(disabled_synapses):
            if len(query_list):
                query_list.extend(
                    [{"action":{"method":{"has":{"name": disabled_synapses}}},"object":{"state":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"memory":1}}])
            else:
                query_list.extend(
                    [{"action":{"method":{"has":{"name": disabled_synapses}}},"object":{"state":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"state":0}}])

        if len(query_list):
            res = self.client.executeNAquery({'query': query_list, 'format': 'nx'})
        return res

    def execute(self, input_processors = {}, output_processors = {},
                steps = None, dt = None, name = None):
        self.remove_components()
        if name is None:
            name = '{}/{}'.format(self.current_model_name, self.current_model_version)
        self.client.execute_multilpu(name, inputProcessors = input_processors,
                                     outputProcessors = output_processors,
                                     steps = steps, dt = dt)

    def run(self, input_processors = {}, output_processors = {},
            steps = None, dt = None, name = None, version = None):
        if self._updated:
            if name is None:
                res = input('Please assign a name to the model: ')
                name = res
            if version is None:
                res = input('Please assign a version to the model: ')
                version = res
            self.flush_model(name, version)

        # may need to update rids of input/output processor component
        self.execute(input_processors = input_processors,
                     output_processors = output_processors,
                     steps = steps, dt = dt, name = None)


    # should be done as a callback
    def get_result(self, name = None):
        if name is None:
            name = '{}/{}'.format(self.current_model_name, self.current_model_version)
        label_dict = {}
        for rid, node in self.get(['Neuron', 'Synpase', 'InferredSynapse']).items():
            label_dict[self.find_model(rid).popitem()[0]] = node['uname']
        self.client.updateSimResultLabel(name, label_dict)
        self.result = self.client.exec_result[name]
        return self.client.exec_result[name]
