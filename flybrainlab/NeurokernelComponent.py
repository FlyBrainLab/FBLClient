import sys
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.logger import Logger

from autobahn.twisted.util import sleep
from autobahn.twisted.wamp import ApplicationSession
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp import auth
import networkx as nx
import numpy as np
import pycuda.driver as cuda
import h5py
from time import gmtime, strftime
from configparser import ConfigParser
import os
from os.path import expanduser
import pickle
import math
import time
import six
import simplejson as json
import ast
from pathlib import Path
try:
    import neuroballad as nb
    from neuroballad import *
    from HDF5toJSON import *
    from diagram_generator import *
    from circuit_execution import *
    import matplotlib.pyplot as plt
    import pygraphviz
except:
    pass

from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core
from neurokernel.pattern import Pattern
from neurokernel.tools.timing import Timer
from neurokernel.LPU.LPU import LPU
from retina.InputProcessors.RetinaInputIndividual import RetinaInputIndividual
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
import neurokernel.LPU.utils.simpleio as si
from neuroarch import nk

from retina.configreader import ConfigReader
from retina.NDComponents.MembraneModels.PhotoreceptorModel import PhotoreceptorModel
from retina.NDComponents.MembraneModels.BufferPhoton import BufferPhoton
from retina.NDComponents.MembraneModels.BufferVoltage import BufferVoltage
from configparser import ConfigParser
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor

import urllib
import requests

import subprocess
import argparse
import txaio
import time
import traceback

home = str(Path.home())
if not os.path.exists(os.path.join(home, '.ffbolab')):
    os.makedirs(os.path.join(home, '.ffbolab'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','data')):
    os.makedirs(os.path.join(home, '.ffbolab', 'data'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','config')):
    os.makedirs(os.path.join(home, '.ffbolab', 'config'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','lib')):
    os.makedirs(os.path.join(home, '.ffbolab', 'lib'), mode=0o777)


## Create the home directory
import os
import urllib
home = str(Path.home())
if not os.path.exists(os.path.join(home, '.ffbolab')):
    os.makedirs(os.path.join(home, '.ffbolab'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','data')):
    os.makedirs(os.path.join(home, '.ffbolab', 'data'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','config')):
    os.makedirs(os.path.join(home, '.ffbolab', 'config'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','lib')):
    os.makedirs(os.path.join(home, '.ffbolab', 'lib'), mode=0o777)

_FFBOLabDataPath = os.path.join(home, '.ffbolab', 'data')
_FFBOLabExperimentPath = os.path.join(home, '.ffbolab', 'experiments')

print(os.path.exists(_FFBOLabDataPath))

import binascii
from os import listdir
from os.path import isfile, join
from time import sleep
import txaio
import random
from autobahn.twisted.util import sleep
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from twisted.internet._sslverify import OpenSSLCertificateAuthorities
from twisted.internet.ssl import CertificateOptions
import OpenSSL.crypto
from autobahn.wamp.types import RegisterOptions
from autobahn.wamp import auth
from autobahn_sync import publish, call, register, subscribe, run, AutobahnSync
from autobahn.wamp import PublishOptions, RegisterOptions
from pathlib import Path
import os

import logging

logging.basicConfig()
logging.getLogger('twisted').setLevel(logging.CRITICAL)


def urlRetriever(url, savePath, verify = False):
    """Retrieves and saves a url in Python 3.

    # Arguments:
        url (str): File url.
        savePath (str): Path to save the file to.
    """
    with open(savePath, 'wb') as f:
        resp = requests.get(url, verify=verify)
        f.write(resp.content)

def create_graph_from_database_returned(x):
    """Builds a NetworkX graph using processed data from NeuroArch.

    # Arguments:
        x (dict): File url.

    # Returns:
        g (NetworkX MultiDiGraph): A MultiDiGraph instance with the circuit graph.
    """
    g = nx.MultiDiGraph()
    g.add_nodes_from(x['nodes'].items())
    for pre,v,attrs in x['edges']:
        g.add_edge(pre, v, **attrs)
    return g

def get_config_obj(conf_name = 'configurations/retina.cfg', conf_specname = 'configurations/retina_template.cfg'):
    """Reads and returns a configuration reader object.

    # Arguments:
        conf_name (str): Optional. The config file to use for execution.
        conf_specname (str): Optional. The specification file to use for execution.

    # Returns:
        ConfigReader: A ConfigReader instance opening the configuration files.
    """
    # Append file extension if not exist
    conf_filename = conf_name if '.' in conf_name else ''.join(
        [conf_name, '.cfg'])
    

    return ConfigReader(conf_filename, conf_specname)


class neurokernel_server(object):
    """ A Neurokernel server launcher instance. """

    def __init__(self):
        cuda.init()
        if cuda.Device.count() < 0:
            raise ValueError("No GPU found on this device")

    def launch(self, user_id, task):
        neuron_uid_list = [str(a) for a in task['neuron_list']]

        conf_obj = get_config_obj()
        config = conf_obj.conf

        if config['Retina']['intype'] == 'Natural':
            coord_file = config['InputType']['Natural']['coord_file']
            tmp = os.path.splitext(coord_file)
            config['InputType']['Natural']['coord_file'] = '{}_{}{}'.format(
                    tmp[0], user_id, tmp[1])

        setup_logger(file_name = 'neurokernel_'+user_id+'.log', screen = True)

        manager = core.Manager()

        lpus = {}
        patterns = {}
        G = task['data']
        with open('G.pickle', 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(G)
        print(G.keys())
        print(G['LPU'])
        print(G['LPU'].keys())

        # get graph and output_uid_list for each LPU
        for k, lpu in G['LPU'].items():
            lpus[k] = {}
            g_lpu_na = create_graph_from_database_returned(lpu)
            lpu_nk_graph = nk.na_lpu_to_nk_new(g_lpu_na)
            lpus[k]['graph'] = lpu_nk_graph
            lpus[k]['output_uid_list'] = list(
                        set(lpu_nk_graph.nodes()).intersection(
                            set(neuron_uid_list)))
            lpus[k]['output_file'] = '{}_output_{}.h5'.format(k, user_id)

        for kkey, lpu in lpus.items():
            graph = lpu['graph']

            for uid, comp in graph.node.items():
                if 'attr_dict' in comp:
                    print('Found attr_dict; fixing...')
                    nx.set_node_attributes(graph, {uid: comp['attr_dict']})
                    # print('changed',uid)
                    graph.nodes[uid].pop('attr_dict')
            for i,j,k,v in graph.edges(keys=True, data=True):
                if 'attr_dict' in v:
                    for key in v['attr_dict']:
                        nx.set_edge_attributes(graph, {(i,j,k): {key: v['attr_dict'][key]}})
                    graph.edges[(i,j,k)].pop('attr_dict')
            lpus[kkey]['graph'] = graph

        # get graph for each Pattern
        for k, pat in G['Pattern'].items():
            l1,l2 = k.split('-')
            if l1 in lpus and l2 in lpus:
                g_pattern_na = create_graph_from_database_returned(pat)
                pattern_nk = nk.na_pat_to_nk(g_pattern_na)
                print(lpus[l1]['graph'].nodes(data=True))
                lpu_ports = [node[1]['selector'] \
                             for node in lpus[l1]['graph'].nodes(data=True) \
                             if node[1]['class']=='Port'] + \
                            [node[1]['selector'] \
                             for node in lpus[l2]['graph'].nodes(data=True) \
                             if node[1]['class']=='Port']
                pattern_ports = pattern_nk.nodes()
                patterns[k] = {}
                patterns[k]['graph'] = pattern_nk.subgraph(
                    list(set(lpu_ports).intersection(set(pattern_ports))))

        dt = config['General']['dt']
        if 'dt' in task:
            dt = task['dt']
            print(dt)


        # add LPUs to manager
        for k, lpu in lpus.items():
            lpu_name = k
            graph = lpu['graph']

            for uid, comp in graph.node.items():
                if 'attr_dict' in comp:
                    nx.set_node_attributes(graph, {uid: comp['attr_dict']})
                    # print('changed',uid)
                    graph.nodes[uid].pop('attr_dict')
            for i,j,ko,v in graph.edges(keys=True, data=True):
                if 'attr_dict' in v:
                    for key in v['attr_dict']:
                        nx.set_edge_attributes(graph, {(i,j,ko): {key: v['attr_dict'][key]}})
                    graph.edges[(i,j,ko)].pop('attr_dict')
            nx.write_gexf(graph,'name.gexf')
            with open(lpu_name + '.pickle', 'wb') as f:
                pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            comps =  graph.node.items()

            #for uid, comp in comps:
            #    if 'attr_dict' in comp:
            #        nx.set_node_attributes(graph, {uid: comp['attr_dict']})
            #        print('changed',uid)
            #    if 'class' in comp:

            if k == 'retina':
                prs = [node for node in graph.nodes(data=True) \
                       if node[1]['class'] == 'PhotoreceptorModel']
                for pr in prs:
                    graph.node[pr[0]]['num_microvilli'] = 3000
                input_processors = [RetinaInputIndividual(config, prs, user_id)]
                extra_comps = [PhotoreceptorModel]
                retina_input_uids = [a[0] for a in prs]
            elif k == 'EB':
                input_processor = StepInputProcessor('I', [node[0] for node in graph.nodes(data=True) \
                       if node[1]['class'] == 'LeakyIAF'], 40.0, 0.0, 1.0)
                input_processors = [input_processor]
                extra_comps = [BufferVoltage]
            else:
                input_processors = []
                extra_comps = [BufferVoltage]
            if 'inputProcessors' in task:
                input_processors = loadExperimentSettings(task['inputProcessors'])
            output_processor = FileOutputProcessor(
                                    [('V', lpu['output_uid_list'])],
                                    lpu['output_file'], sample_interval=10)


            (comp_dict, conns) = LPU.graph_to_dicts(graph)



            # print(comp_dict)
            # print(conns)
            print(k)
            manager.add(LPU, k, dt, comp_dict, conns,
                        device = 0, input_processors = input_processors,
                        output_processors = [output_processor],
                        extra_comps = extra_comps, debug=True)

        # connect LPUs by Patterns
        for k, pattern in patterns.items():
            l1,l2 = k.split('-')
            if l1 in lpus and l2 in lpus:
                print('Connecting {} and {}'.format(l1, l2))
                pat, key_order = Pattern.from_graph(nx.DiGraph(pattern['graph']),
                                                    return_key_order = True)
                print(l1,l2)
                print(key_order)
                with Timer('update of connections in Manager'):
                    manager.connect(l1, l2, pat,
                                    int_0 = key_order.index(l1),
                                    int_1 = key_order.index(l2))

        # start simulation
        steps = config['General']['steps']
        ignored_steps = config['General']['ignored_steps']
        if 'steps' in task:
            steps = task['steps']
        if 'ignored_steps' in task:
            ignored_steps = task['ignored_steps']
        # ignored_steps = 0
        # steps = 100
        manager.spawn()
        manager.start(steps=steps)
        manager.wait()

        time.sleep(5)
        print(task)

        # post-processing inputs (hard coded, can be better organized)
        inputs = {u'ydomain': 1.0,
                      u'xdomain': dt*(steps-ignored_steps),
                      u'dt': dt*10,
                      u'data': {}}
        if 'retina' in lpus:
            input_array = si.read_array(
                    '{}_{}.h5'.format(config['Retina']['input_file'], user_id))
            inputs[u'ydomain'] = input_array.max()
            for i, item in enumerate(retina_input_uids):
                inputs['data'][item] = np.hstack(
                    (np.arange(int((steps-ignored_steps)/10)).reshape((-1,1))*dt*10,
                     input_array[ignored_steps::10,i:i+1])).tolist()

            del input_array

        # post-processing outputs from all LPUs and combine them into one dictionary
        result = {u'ydomain': 1,
                  u'xdomain': dt*(steps-ignored_steps),
                  u'dt': dt*10,
                  u'data': {}}

        for k, lpu in lpus.items():
            with h5py.File(lpu['output_file']) as output_file:
                uids = output_file['V']['uids'][:]
                output_array = output_file['V']['data'][:]
                for i, item in enumerate(uids):
                    output = output_array[int(ignored_steps/10):,i:i+1]
                    # tmp = output.max()-output.min()
                    # if tmp <= 0.01: #mV
                    #     output = (output - output.min()) + 0.5
                    # else:
                    #     output = (output - output.min())/tmp*0.9+0.1
                    result['data'][item] = np.hstack(
                        (np.arange(int((steps-ignored_steps)/10)).reshape((-1,1))*dt*10, output)).tolist()

        return inputs, result

def printHeader(name):
    return '[' + name + ' ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] '



def loadExperimentSettings(X):
    inList = []
    for a in X:
        if a['name'] == 'InIGaussianNoise':
            inList.append(InIGaussianNoise(a['node_id'], a['mean'], a['std'], a['t_start'], a['t_end']))
        elif a['name'] == 'InISinusoidal':
            inList.append(InISinusoidal(a['node_id'], a['amplitude'], a['frequency'], a['t_start'], a['t_end'], a['mean'], a['shift'], a['frequency_sweep'], a['frequency_sweep_frequency'], a['threshold_active'], a['threshold_value']))
        elif a['name'] == 'InIBoxcar':
            inList.append(InIBoxcar(a['node_id'], a['I_val'], a['t_start'], a['t_end']))
        else:
            inList.append(StepInputProcessor(*a['args'], **a['kwargs']))
    return inList

class ffbolabComponent:
    def __init__(self, ssl = True, debug = True, authentication = True, user = u"ffbo", secret = u"", url = u'wss://neuronlp.fruitflybrain.org:7777/ws', realm = u'realm1', ca_cert_file = 'isrgrootx1.pem', intermediate_cert_file = 'letsencryptauthorityx3.pem', FFBOLabcomm = None):
        if os.path.exists(os.path.join(home, '.ffbolab', 'lib')):
            print(printHeader('FFBOLab Client') + "Downloading the latest certificates.")
            # CertificateDownloader = urllib.URLopener()
            if not os.path.exists(os.path.join(home, '.ffbolab', 'lib')):  
                urlRetriever("https://data.flybrainlab.fruitflybrain.org/config/FBLClient.ini",
                                  os.path.join(home, '.ffbolab', 'config','FBLClient.ini'))
            urlRetriever("https://data.flybrainlab.fruitflybrain.org/lib/isrgrootx1.pem",
                              os.path.join(home, '.ffbolab', 'lib','caCertFile.pem'))
            urlRetriever("https://data.flybrainlab.fruitflybrain.org/lib/letsencryptauthorityx3.pem",
                              os.path.join(home, '.ffbolab', 'lib','intermediateCertFile.pem'))
            config_file = os.path.join(home, '.ffbolab', 'config','FBLClient.ini')
            ca_cert_file = os.path.join(home, '.ffbolab', 'lib','caCertFile.pem')
            intermediate_cert_file = os.path.join(home, '.ffbolab', 'lib','intermediateCertFile.pem')
        config = ConfigParser()
        config.read(config_file)
        # user = config["ComponentInfo"]["user"]
        # secret = config["ComponentInfo"]["secret"]
        # url = config["ComponentInfo"]["url"]
        self.FFBOLabcomm = FFBOLabcomm
        self.NKSimState = 0
        self.executionSettings = []
        extra = {'auth': authentication}
        self.lmsg = 0
        st_cert=open(ca_cert_file, 'rt').read()
        c=OpenSSL.crypto
        ca_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)
        st_cert=open(intermediate_cert_file, 'rt').read()
        intermediate_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)
        certs = OpenSSLCertificateAuthorities([ca_cert, intermediate_cert])
        ssl_con = CertificateOptions(trustRoot=certs)

        FFBOLABClient = AutobahnSync()
        self.client = FFBOLABClient

        @FFBOLABClient.on_challenge
        def on_challenge(challenge):
            if challenge.method == u"wampcra":
                print("WAMP-CRA challenge received: {}".format(challenge))
                if u'salt' in challenge.extra:
                    # salted secret
                    salted_key = auth.derive_key(secret,
                                          challenge.extra['salt'],
                                          challenge.extra['iterations'],
                                          challenge.extra['keylen'])
                    salted_key = (salted_key).decode('utf-8')
                    print(salted_key)
                #if user==u'ffbo':
                    # plain, unsalted secret
                #    salted_key = u"kMU73GH4GS1WGUpEaSdDYwN57bdLdB58PK1Brb25UCE="
                #print(salted_key)
                # compute signature for challenge, using the key
                signature = auth.compute_wcs(salted_key, challenge.extra['challenge'])

                # return the signature to the router for verification
                return signature

            else:
                raise Exception("Invalid authmethod {}".format(challenge.method))

        FFBOLABClient.run(url=url, authmethods=[u'wampcra'], authid=user, ssl=ssl_con)
        self.client_data = []
        self.data = []
        self.launch_queue = []

        @FFBOLABClient.register('ffbo.nk.launch.' + str(FFBOLABClient._async_session._session_id))
        def nk_launch_progressive(task, details=None):
            # print(task['user'])
            user_id = str(task['user'])
            self.launch_queue.append((user_id, task))
            def mock_result():
                result = {u'ydomain': 1,
                          u'xdomain': 1,
                          u'dt': 10,
                          u'data': {}}
                """
                res = {u'ydomain': 1,
                          u'xdomain': 1,
                          u'dt': 10,
                          u'data': {}}
                print(task['forward'])
                res_to_processor = yield self.call(six.u(task['forward']), res)
                """
                return result, result
            res = mock_result()
            return res
            # res = server.launch(user_id, task)
            # returnValue(res)
        print("Procedure nk_launch_progressive Registered...")

        res = FFBOLABClient.session.call(u'ffbo.server.register',FFBOLABClient._async_session._session_id,'nk','nk_server')
        print("Registered self...")
        
def mainThreadExecute(Component, server):
    #self.execution_settings = json.loads(settings)
    if len(Component.launch_queue)>0:
        user_id, task = Component.launch_queue[0]
        # try:
        inputs, res = server.launch(user_id, task)
        # print(res)
        for key in res.keys():
            if type(key) is not str:
                try:
                    res[str(key)] = res[key]
                except:
                    try:
                        res[repr(key)] = res[key]
                    except:
                        pass
                del res[key]
        for key in res['data'].keys():
            if type(key) is not str:
                try:
                    res['data'][str(key)] = res['data'][key]
                except:
                    try:
                        res['data'][repr(key)] = res['data'][key]
                    except:
                        pass
                del res['data'][key]
        # print(res['data'].keys())
        res =  six.u(res)
        """
        res = {u'ydomain': 1,
                    u'xdomain': 1,
                    u'dt': 10,
                    u'data': {}}
        """
        print('Printing task...')
        print(task)
        r = json.dumps(res)
        res_to_processor = Component.client.session.call(six.u(task['forward']), r)
        # except:
        #     print('There was an error...')
        Component.launch_queue.pop(0)
    else:
        return False