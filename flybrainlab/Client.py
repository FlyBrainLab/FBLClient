from time import sleep
import txaio
import random
import h5py
from autobahn.twisted.util import sleep
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from twisted.internet._sslverify import OpenSSLCertificateAuthorities
from twisted.internet.ssl import CertificateOptions
import OpenSSL.crypto
from collections import Counter
from autobahn.wamp.types import RegisterOptions, CallOptions
from functools import partial
from autobahn.wamp import auth
from autobahn_sync import publish, call, register, subscribe, run, AutobahnSync
from IPython.display import clear_output
from pathlib import Path
from functools import partial
from configparser import ConfigParser
import numpy as np
import os
import json
import binascii
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
import numpy as np
import neuroballad as nb
import networkx as nx
import importlib
from time import gmtime, strftime



## Create the home directory
import os
import urllib
import requests
home = str(Path.home())
if not os.path.exists(os.path.join(home, '.ffbolab')):
    os.makedirs(os.path.join(home, '.ffbolab'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','data')):
    os.makedirs(os.path.join(home, '.ffbolab', 'data'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','config')):
    os.makedirs(os.path.join(home, '.ffbolab', 'config'), mode=0o777)
if not os.path.exists(os.path.join(home, '.ffbolab','lib')):
    os.makedirs(os.path.join(home, '.ffbolab', 'lib'), mode=0o777)

# Generate the data path to be used for imports
_FFBOLabDataPath = os.path.join(home, '.ffbolab', 'data')
_FFBOLabConfigPath = os.path.join(home, '.ffbolab', 'config', 'ffbo.flybrainlab.ini')

def urlRetriever(url, savePath, verify = False):
    """Retrieves and saves a url in Python 3.

    # Arguments:
        url (str): File url.
        savePath (str): Path to save the file to.
    """
    with open(savePath, 'wb') as f:
        resp = requests.get(url, verify=verify)
        f.write(resp.content)

def guidGenerator():
    """Unique query ID generator for handling the backend queries

    # Returns:
        str: The string with time format and brackets.
    """
    def S4():
        return str(((1+random.random())*0x10000))[1]
    return (S4()+S4()+"-"+S4()+"-"+S4()+"-"+S4()+"-"+S4()+S4()+S4())

def printHeader(name):
    """Header printer for the console messages. Useful for debugging.

    # Arguments:
        name (str): Name of the component.

    # Returns:
        str: The string with time format and brackets.
    """
    return '[' + name + ' ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '] '

class Client:
    """FlyBrainLab Client class. This class communicates with JupyterLab frontend and connects to FFBO components.

    # Attributes:
        FFBOLabcomm (obj): The communication object for sending and receiving data.
        circuit (obj): A Neuroballad circuit that enables local circuit execution and facilitates circuit modification.
        dataPath (str): Data path to be used.
        experimentInputs (list of dicts): Inputs as a list of dicts that can be parsed by the GFX component.
        compiled (bool): Circuits need to be compiled into networkx graphs before being sent for simulation. This is necessary as circuit compilation is a slow process.
        sendDataToGFX (bool): Whether the data received from the backend should be sent to the frontend. Useful for code-only projects.
    """
    def tryComms(self, a):
        """Communication function to communicate with a JupyterLab frontend if one exists.

        # Arguments:
            a (obj): Arbitrarily formatted data to be sent via communication.
        """
        try:
            self.FFBOLabcomm.send(data=a)
        except:
            pass

    def __init__(self, ssl = True, debug = True, authentication = True, user = 'guest', secret = 'guestpass', url = u'wss://neuronlp.fruitflybrain.org:7777/ws', realm = u'realm1', ca_cert_file = 'isrgrootx1.pem', intermediate_cert_file = 'letsencryptauthorityx3.pem', FFBOLabcomm = None, legacy = False):
        """Initialization function for the ffbolabClient class.

        # Arguments:
            ssl (bool): Whether the FFBO server uses SSL.
            debug (bool) : Whether debugging should be enabled.
            authentication (bool): Whether authentication is enabled.
            user (str): Username for establishing communication with FFBO components.
            secret (str): Password for establishing communication with FFBO components.
            url (str): URL of the WAMP server with the FFBO Processor component.
            realm (str): Realm to be connected to.
            ca_cert_file (str): Path to the certificate for establishing connection.
            intermediate_cert_file (str): Path to the intermediate certificate for establishing connection.
            FFBOLabcomm (obj) Communications object for the frontend.
        """
        if os.path.exists(os.path.join(home, '.ffbolab', 'lib')):
            print(printHeader('FFBOLab Client') + "Downloading the latest certificates.")
            # CertificateDownloader = urllib.URLopener()
            if not os.path.exists(os.path.join(home, '.ffbolab', 'config', 'FBLClient.ini')):  
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
        user = config["ClientInfo"]["user"]
        secret = config["ClientInfo"]["secret"]
        if url is None:
            url = config["ClientInfo"]["url"]
        self.FFBOLabcomm = FFBOLabcomm # Current Communications Object
        self.C = nb.Circuit() # The Neuroballd Circuit object describing the loaded neural circuit
        self.dataPath = _FFBOLabDataPath
        extra = {'auth': authentication}
        self.lmsg = 0
        self.experimentInputs = [] # List of current experiment inputs
        self.compiled = False # Whether the current circuit has been compiled into a NetworkX Graph
        self.sendDataToGFX = True # Shall we send the received simulation data to GFX Component?
        self.executionSuccessful = False # Used to wait for data loading
        self.experimentQueue = [] # A queue for experiments
        self.simExperimentConfig = None # Experiment configuration (disabled neurons etc.) for simulations
        self.simExperimentRunners = {} # Experiment runners for simulations
        self.simData = {} # Locally loaded simulation data obtained from server
        self.clientData = [] # Servers list
        self.data = [] # A buffer for data from backend; used in multiple functions so needed
        self.legacy = legacy
        st_cert=open(ca_cert_file, 'rt').read()
        c=OpenSSL.crypto
        ca_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)
        st_cert=open(intermediate_cert_file, 'rt').read()
        intermediate_cert=c.load_certificate(c.FILETYPE_PEM, st_cert)
        certs = OpenSSLCertificateAuthorities([ca_cert, intermediate_cert])
        ssl_con = CertificateOptions(trustRoot=certs)



        FFBOLABClient = AutobahnSync()

        @FFBOLABClient.on_challenge
        def on_challenge(challenge):
            """The On Challenge function that computes the user signature for verification.

            # Arguments:
                challenge (obj): The challenge object received.

            # Returns:
                str: The signature sent to the router for verification.
            """
            print(printHeader('FFBOLab Client') + 'Initiating authentication.')
            if challenge.method == u"wampcra":
                print(printHeader('FFBOLab Client') + "WAMP-CRA challenge received: {}".format(challenge))
                print(challenge.extra['salt'])
                if u'salt' in challenge.extra:
                    # Salted secret
                    print(printHeader('FFBOLab Client') + 'Deriving key...')
                    salted_key = auth.derive_key(secret,
                                          challenge.extra['salt'],
                                          challenge.extra['iterations'],
                                          challenge.extra['keylen'])
                    print(salted_key.decode('utf-8'))

                if user=='guest':
                    # A plain, unsalted secret for the guest account
                    salted_key = u"C5/c598Gme4oALjmdhVC2H25OQPK0M2/tu8yrHpyghA="

                # compute signature for challenge, using the key
                signature = auth.compute_wcs(salted_key, challenge.extra['challenge'])

                # return the signature to the router for verification
                return signature

            else:
                raise Exception("Invalid authmethod {}".format(challenge.method))

        FFBOLABClient.run(url=url, authmethods=[u'wampcra'], authid='guest', ssl=ssl_con) # Initialize the communication right now!

        @FFBOLABClient.subscribe('ffbo.server.update.' + str(FFBOLABClient._async_session._session_id))
        def updateServers(data):
            """Updates available servers.

            # Arguments:
                data (obj): Obtained servers list.

            """
            self.clientData.append(data)
            print("Updated the Servers")

        print("Subscribed to topic 'ffbo.server.update'")
        @FFBOLABClient.register('ffbo.ui.receive_cmd.' + str(FFBOLABClient._async_session._session_id))
        def receiveCommand(data):
            """The Receive Command function that receives commands and sends them to the frontend.

            # Arguments:
                data (dict): Data to be sent to the frontend

            # Returns:
                bool: Whether the data has been received.
            """
            self.clientData.append('Received Command')
            a = {}
            a['data'] = data
            a['messageType'] = 'Command'
            a['widget'] = 'NLP'
            self.data.append(a)
            print(printHeader('FFBOLab Client NLP') + "Received a command.")
            self.tryComms(a)
            return True
        print(printHeader('FFBOLab Client') + "Procedure ffbo.ui.receive_cmd Registered...")

        @FFBOLABClient.register('ffbo.ui.receive_gfx.' + str(FFBOLABClient._async_session._session_id))
        def receiveGFX(data):
            """The Receive GFX function that receives commands and sends them to the GFX frontend.

            # Arguments:
                data (dict): Data to be sent to the frontend.

            # Returns:
                bool: Whether the data has been received.
            """
            self.clientData.append('Received GFX Data')
            self.data.append(data)
            print(printHeader('FFBOLab Client GFX') + "Received a message for GFX.")
            if self.sendDataToGFX == True:
                self.tryComms(data)
            else:
                if 'messageType' in data.keys():
                    if data['messageType'] == 'showServerMessage':
                        print(printHeader('FFBOLab Client GFX') + "Execution successful for GFX.")
                        if len(self.experimentQueue)>0:
                            print(printHeader('FFBOLab Client GFX') + "Next execution now underway. Remaining simulations: " + str(len(self.experimentQueue)))
                            a = self.experimentQueue.pop(0)
                            res = self.client.session.call('ffbo.gfx.sendExperiment', a)
                            res = self.client.session.call('ffbo.gfx.startExecution', {'name': a['name']})
                        else:
                            self.executionSuccessful = True
                            self.parseSimResults()
                            print(printHeader('FFBOLab Client GFX') + "GFX results successfully parsed.")
            return True
        print(printHeader('FFBOLab Client') + "Procedure ffbo.ui.receive_gfx Registered...")

        @FFBOLABClient.register('ffbo.ui.get_circuit.' + str(FFBOLABClient._async_session._session_id))
        def get_circuit(X):
            """Obtain a circuit and save it to the local FFBOLab folder.

            # Arguments:
                X (str): Name of the circuit.

            # Returns:
                bool: Whether the process has been successful.
            """
            name = X['name']
            G = binascii.unhexlify(X['graph'].encode())
            with open(os.path.join(_FFBOLabDataPath, name + '.gexf.gz'), "wb") as file:
                file.write(G)
            return True
        print("Procedure ffbo.ui.get_circuit Registered...")

        @FFBOLABClient.register('ffbo.ui.get_experiment' + str(FFBOLABClient._async_session._session_id))
        def get_experiment(X):
            """Obtain an experiment and save it to the local FFBOLab folder.

            # Arguments:
                X (str): Name of the experiment.

            # Returns:
                bool: Whether the process has been successful.
            """
            print(printHeader('FFBOLab Client GFX') + "get_experiment called.")
            name = X['name']
            data = json.dumps(X['experiment'])
            with open(os.path.join(_FFBOLabDataPath, name + '.json'), "w") as file:
                file.write(data)
            output = {}
            output['success'] = True
            print(printHeader('FFBOLab Client GFX') + "Experiment save successful.")
            return True
        print("Procedure ffbo.ui.get_experiment Registered...")

        @FFBOLABClient.register('ffbo.ui.receive_data.' + str(FFBOLABClient._async_session._session_id))
        def receiveData(data):
            """The Receive Data function that receives commands and sends them to the NLP frontend.

            # Arguments:
                data (dict): Data from the backend.

            # Returns:
                bool: Whether the process has been successful.
            """
            self.clientData.append('Received Data')
            a = {}
            if self.legacy == True:
                a['data'] = {'data': data, 'queryID': guidGenerator()}
            else:
                a['data'] = data
            a['messageType'] = 'Data'
            a['widget'] = 'NLP'
            self.data.append(a)
            print(printHeader('FFBOLab Client NLP') + "Received data.")
            self.tryComms(a)
            return True
        print(printHeader('FFBOLab Client') + "Procedure ffbo.ui.receive_data Registered...")

        @FFBOLABClient.register('ffbo.ui.receive_partial.' + str(FFBOLABClient._async_session._session_id))
        def receivePartial(data):
            """The Receive Partial Data function that receives commands and sends them to the NLP frontend.

            # Arguments:
                data (dict): Data from the backend.

            # Returns:
                bool: Whether the process has been successful.
            """
            self.clientData.append('Received Data')
            a = {}
            a['data'] = {'data': data, 'queryID': guidGenerator()}
            a['messageType'] = 'Data'
            a['widget'] = 'NLP'
            self.data.append(a)
            print(printHeader('FFBOLab Client NLP') + "Received partial data.")
            self.tryComms(a)
            return True
        print(printHeader('FFBOLab Client') + "Procedure ffbo.ui.receive_partial Registered...")

        if legacy == False:
            @FFBOLABClient.register('ffbo.gfx.receive_partial.' + str(FFBOLABClient._async_session._session_id))
            def receivePartialGFX(data):
                """The Receive Partial Data function that receives commands and sends them to the NLP frontend.

                # Arguments:
                    data (dict): Data from the backend.

                # Returns:
                    bool: Whether the process has been successful.
                """
                self.clientData.append('Received Data')
                a = {}
                a['data'] = {'data': data, 'queryID': guidGenerator()}
                a['messageType'] = 'Data'
                a['widget'] = 'NLP'
                self.data.append(a)
                print(printHeader('FFBOLab Client NLP') + "Received partial data.")
                self.tryComms(a)
                return True
            print(printHeader('FFBOLab Client') + "Procedure ffbo.gfx.receive_partial Registered...")

        @FFBOLABClient.register('ffbo.ui.receive_msg.' + str(FFBOLABClient._async_session._session_id))
        def receiveMessage(data):
            """The Receive Message function that receives commands and sends them to the NLP frontend.

            # Arguments:
                data (dict): Data from the backend.

            # Returns:
                bool: Whether the process has been successful.
            """
            self.clientData.append('Received Message')
            a = {}
            a['data'] = data
            a['messageType'] = 'Message'
            a['widget'] = 'NLP'
            self.data.append(a)
            print(printHeader('FFBOLab Client NLP') + "Received a message.")
            self.tryComms(a)
            return True
        print(printHeader('FFBOLab Client') + "Procedure ffbo.ui.receive_msg Registered...")

        self.client = FFBOLABClient # Set current client to the FFBOLAB Client

        self.findServerIDs() # Get current server IDs



    def findServerIDs(self):
        """Find server IDs to be used for the utility functions.
        """
        res = self.client.session.call(u'ffbo.processor.server_information')

        for i in res['na']:
            if 'na' in res['na'][i]['name']:
                print(printHeader('FFBOLab Client') + 'Found working NA Server: ' + res['na'][i]['name'])
                self.naServerID = i
        for i in res['nlp']:
            self.nlpServerID = i

    def executeNLPquery(self, query = None, language = 'en', uri = None, queryID = None, returnNAOutput = False):
        """Execute an NLP query.

        # Arguments:
            query (str): Query string.
            language (str): Language to use.
            uri (str): Currently not used; for future NLP extensions.
            queryID (str): Query ID to be used. Generated automatically.
            returnNAOutput (bool): Whether the corresponding NA query should not be executed.

        # Returns:
            dict: NA output or the NA query itself, depending on the returnNAOutput setting.
        """
        if query is None:
            print(printHeader('FFBOLab Client') + 'No query specified. Executing test query "eb".')
            query = 'eb'
        if query.startswith("load "):
            self.sendSVG(query[5:])
        else:
            # if self.legacy == False:
            uri = 'ffbo.nlp.query.' + self.nlpServerID
            queryID = guidGenerator()
            try:
                resNA = self.client.session.call(uri , query, language)
            except:
                a = {}
                a['data'] = {'info': {'timeout': 'This is a timeout.'}}
                a['messageType'] = 'Data'
                a['widget'] = 'NLP'
                self.tryComms(a)
                return a
            print(printHeader('FFBOLab Client NLP') + 'NLP successfully parsed query.')

            if returnNAOutput == True:
                return resNA
            else:
                try:
                    self.compiled = False
                    res = self.executeNAquery(resNA, queryID = queryID)
                    self.sendNeuropils()
                    """
                    a = {}
                    a['data'] = {'info': {'success': 'Finished fetching results from database'}}
                    a['messageType'] = 'Data'
                    a['widget'] = 'NLP'
                    self.tryComms(a)
                    """
                    return res
                except:
                    a = {}
                    a['data'] = {'info': {'timeout': 'This is a timeout.'}}
                    a['messageType'] = 'Data'
                    a['widget'] = 'NLP'
                    self.tryComms(a)
                    return resNA
            """
            else:
                msg = {}
                msg['username'] = "Guest  "
                msg['servers'] = {}
                msg['data_callback_uri'] = 'ffbo.ui.receive_partial'
                msg['language'] = language
                msg['servers']['nlp'] = self.nlpServerID
                msg['servers']['na'] = self.naServerID
                msg['nlp_query'] = query
                def on_progress(x, res):
                    res.append({'data': x, 'queryID': guidGenerator()})
                res_list = []
                resNA = self.client.session.call('ffbo.processor.nlp_to_visualise', msg, options=CallOptions(
                                                    on_progress=partial(on_progress, res=res_list), timeout = 20))
                if returnNAOutput == True:
                    return resNA
                else:
                    self.compiled = False
                    # res = self.executeNAquery(resNA, queryID = queryID)
                    self.sendNeuropils()
                    return resNA
            """

    def executeNAquery(self, res, language = 'en', uri = None, queryID = None, progressive = True, threshold = 20):
        """Execute an NA query.

        # Arguments:
            res (dict): Neuroarch query.
            language (str): Language to use.
            uri (str): A custom FFBO query URI if desired.
            queryID (str): Query ID to be used. Generated automatically.
            progressive (bool): Whether the loading should be progressive. Needs to be true most of the time for the connection to be stable.
            threshold (int): Data chunk size. Low threshold is required for the connection to be stable.

        # Returns:
            bool: Whether the process has been successful.
        """
        def on_progress(x, res):
            res.append(x)
        if isinstance(res, str):
            res = json.loads(res)
        if uri == None:
            uri = 'ffbo.na.query.' + self.naServerID
            if "uri" in res.keys():
                uri = res["uri"] + "." + self.naServerID
        if queryID == None:
            queryID = guidGenerator()
        # del self.data # Reset the data in the backend
        # self.data = []

        res['queryID'] = queryID
        res['threshold'] = threshold
        res['data_callback_uri'] = 'ffbo.ui.receive_data'
        res_list = []
        if self.legacy == False:
            res = self.client.session.call(uri, res, options=CallOptions(
                    on_progress=partial(on_progress, res=res_list), timeout = 3000
                ))
        else:
            res = self.client.session.call(uri, res)
        a = {}
        a['data'] = res
        a['messageType'] = 'Data'
        a['widget'] = 'NLP'
        if "retrieve_tag" in uri:
            a['messageType'] = 'TagData'
            self.tryComms(a)
            self.executeNAquery({"command": {"retrieve": {"state": 0}}})
        if progressive == True:
            self.tryComms(a)
            self.data.append(a)
            return self.data
        else:
            self.tryComms(a)
            return a

    def createTag(self, tagName):
        """Creates a tag.

        # Returns:
            bool: True.
        """
        metadata = {"color":{},"pinned":{},"visibility":{},"camera":{"position":{},'up':{}},'target':{}};
        self.executeNAquery({
            "tag": tagName,
            "metadata": metadata,
            "uri": 'ffbo.na.create_tag'
        })
        return True

    def loadTag(self, tagName):
        """Loads a tag.

        # Returns:
            bool: True.
        """
        self.executeNAquery({
            "tag": tagName,
            "uri": 'ffbo.na.retrieve_tag'
        })
        return True

    def addByUname(self, uname, verb="add"):
        """Adds some neurons by the uname.

        # Returns:
            bool: True.
        """
        self.executeNAquery({
            "verb": verb,
            "query": [
                        {
                        'action': { 'method': { 'query': { 'uname': uname } } },
                        'object': { 'class': ["Neuron", "Synapse"] }
                        }
                    ]
        })
        return True

    def runLayouting(self, type="auto", model="auto"):
        """Sends a request for the running of the layouting algorithm.

        # Returns:
            bool: True.
        """
        self.prepareCircuit(model = model)
        self.sendCircuit(name = "auto")
        a = {}
        a['data'] = "auto"
        a['messageType'] = 'runLayouting'
        a['widget'] = 'GFX'
        self.tryComms(a)
        return True

    def getNeuropils(self):
        """Get the neuropils the neurons in the workspace reside in.

        # Returns:
            list of strings: Set of neuropils corresponding to neurons.
        """
        res = {}
        res['query'] = []
        res['format'] = 'nx'
        res['user'] = 'test'
        res['temp'] = True
        res['query'].append({'action': {'method': {'traverse_owned_by': {'cls': 'Neuropil'}}},
           'object': {'state': 0}})
        res = self.executeNAquery(res)
        neuropils = []
        for i in res:
            try:
                if 'data' in i.keys():
                    if 'data' in i['data'].keys():
                        if 'nodes' in i['data']['data'].keys():
                            a = i['data']['data']['nodes']
                            for j in a.keys():
                                name = a[j]['name']
                                neuropils.append(name)
            except:
                pass
        neuropils = list(set(neuropils))
        return neuropils

    def sendNeuropils(self):
        """Pack the list of neuropils into a GFX message.

        # Returns:
            bool: Whether the messaging has been successful.
        """
        a = {}
        a['data'] = self.getNeuropils()
        print(a['data'])
        a['messageType'] = 'updateActiveNeuropils'
        a['widget'] = 'GFX'
        self.tryComms(a)
        return True

    def getInfo(self, args):
        """Get information on a neuron.

        # Arguments:
            args (str): Database ID of the neuron or node.

        # Returns:
            dict: NA information regarding the node.
        """
        res = {"uri": 'ffbo.na.get_data.', "id": args}
        queryID = guidGenerator()
        res = self.executeNAquery(res, uri = res['uri'] + self.naServerID, queryID = queryID, progressive = False)
        # res['data']['data']['summary']['rid'] = args
        a = {}
        a['data'] = res
        a['messageType'] = 'Data'
        a['widget'] = 'INFO'
        self.tryComms(a)
        print(res)

        if self.compiled == True:
            a = {}
            name = res['data']['data']['summary']['name']
            if name in self.node_keys.keys():
                data = self.C.G.node['uid' + str(self.node_keys[name])]
                data['uid'] = str(self.node_keys[name])
                a['data'] = data
                a['messageType'] = 'Data'
                a['widget'] = 'JSONEditor'
                self.tryComms(a)

        return res

    def GFXcall(self, args):
        """Arbitrary call to a GFX procedure in the GFX component format.

        # Arguments:
            args (list): A list whose first element is the function name (str) and the following are the arguments.

        # Returns:
            dict OR string: The call result.
        """
        if isinstance(args, str):
            res = self.client.session.call(args)
        else:
            res = self.client.session.call(args[0], args[1:])
        if type(res) == dict:
            a = res
            a['widget'] = 'GFX'
        else:
            a = {}
            a['data'] = res
            a['messageType'] = 'Data'
            a['widget'] = 'GFX'
        self.tryComms(a)
        return res

    def getSimData(self,url):
        url = 'https://data.flybrainlab.fruitflybrain.org/simresults/' + url
        urlRetriever(url, url.split('/')[-1])
        filename = url.split('/')[-1]
        f = h5py.File(filename, 'r')
        data = f['V']['data'][()].tolist()
        uids = f['V']['uids'][()].tolist()
        uids = [i.decode('ascii') for i in uids]
        h5data = {}
        h5data['data'] = data
        h5data['uids'] = uids
        a = {}
        a['data'] = h5data
        a['messageType'] = 'PlotResults'
        a['widget'] = 'Master'
        self.data.append(a)
        print(printHeader('FFBOLab Client Master') + "Sending simulation data.")
        self.tryComms(a)
        json_str = json.dumps(h5data)
        with open(filename.split('.')[0]+'.json', 'w') as f:
            f.write(json_str)
        self.simData = h5data

    def updateBackend(self, type = "Null", data = "Null"):
        """Updates variables in the backend using the data in the Editor.

        # Arguments:
            type (str): A string, either "WholeCircuit" or "SingleNeuron", specifying the type of the update.
            data (str): A stringified JSON

        # Returns:
            bool: Whether the update was successful.
        """

        data = json.loads(data)
        print(type)
        print(data)
        return True

    def getConnectivity(self):
        """Obtain the connectivity matrix of the current circuit in NetworkX format.

        # Returns:
            dict: The connectivity dictionary.
        """
        res = json.loads("""
        {"format":"nx","query":[{"action":{"method":{"add_connecting_synapses":{}}},"object":{"state":0}}],"temp":true}
        """)
        res = self.executeNAquery(res)
        return res


    def sendExecuteReceiveResults(self, circuitName = "temp", dt = 1e-5, tmax = 1.0, compile = False):
        """Compiles and sends a circuit for execution in the GFX backend.

        # Arguments:
            circuitName (str): The name of the circuit for the backend.
            compile (bool): Whether to compile the circuit first.

        # Returns:
            bool: Whether the call was successful.
        """
        print(printHeader('FFBOLab Client GFX') + 'Initiating remote execution for the current circuit.')
        if self.compiled == False:
            compile = True
        if compile == True:
            print(printHeader('FFBOLab Client GFX') + 'Compiling the current circuit.')
            self.prepareCircuit()
        print(printHeader('FFBOLab Client GFX') + 'Circuit prepared. Sending to FFBO servers.')
        self.sendCircuitPrimitive(self.C, args = {'name': circuitName})
        print(printHeader('FFBOLab Client GFX') + 'Circuit sent. Queuing execution.')
        res = self.client.session.call('ffbo.gfx.startExecution', {'name': circuitName, 'dt': dt, 'tmax': tmax})
        return True
        
    def getConnectivityMatrix(self):
        M = np.zeros((len(self.out_nodes),len(self.out_nodes)))
        for i in self.out_edges:
            M[self.out_nodes.index(i[0]),self.out_nodes.index(i[1])] += 1
        return M
        
    def prepareCircuit(self, model = "auto"):

        """Prepares the current circuit for the Neuroballad format.
        """
        res = self.getConnectivity()

        for data in self.data:
            if data['messageType'] == 'Data':
                if 'data' in data:
                    if 'data' in data['data']:
                        connectivity = data['data']['data']

        out_nodes, out_edges, out_edges_unique = self.processConnectivity(connectivity)
        self.out_nodes = out_nodes
        self.out_edges = out_edges
        self.out_edges_unique = out_edges_unique
        C, node_keys = self.GenNB(self.out_nodes, self.out_edges, model=model)
        self.C = C
        self.node_keys = node_keys
        self.compiled = True

    def getSlowConnectivity(self):
        

        hashids = []
        names = []
        synapses = []

        for data in self.data:
                    if data['messageType'] == 'Data':
                        if 'data' in data:
                            if 'data' in data['data']:
                                keys = list(data['data']['data'].keys())
                                for key in keys:
                                    if 'uname' in data['data']['data'][key].keys():
                                        hashids.append(key)
                                        names.append(data['data']['data'][key]['uname'])
        

        for i in range(len(hashids)):
            res = self.getInfo(hashids[i])
            if 'connectivity' in res['data']['data'].keys():
                presyn = res['data']['data']['connectivity']['pre']['details']

                for syn in presyn:
                    synapses.append([syn['uname'], names[i], syn['number']])

                postsyn = res['data']['data']['connectivity']['pre']['details']
                for syn in postsyn:
                    synapses.append([names[i], syn['uname'], syn['number']])
                clear_output()
        connectivity = {'hashids': hashids, 'names': names, 'synapses': synapses}
        return connectivity

    def sendCircuit(self, name = 'temp'):
        """Sends a circuit to the backend.

        # Arguments:
            name (str): The name of the circuit for the backend.
        """
        self.sendCircuitPrimitive(self.C, args = {'name': name})


    def processConnectivity(self, connectivity):
        """Processes a Neuroarch connectivity dictionary.

        # Returns:
            tuple: A tuple of nodes, edges and unique edges.
        """
        edges = connectivity['edges']
        nodes = connectivity['nodes']

        csv = ''
        out_edges = []
        out_nodes = []
        for e_pre in edges:
            if nodes[e_pre]['class'] == 'Neuron':
                if 'uname' in nodes[e_pre].keys():
                    pre = nodes[e_pre]['uname']
                else:
                    pre = nodes[e_pre]['name']
            synapse_nodes = edges[e_pre]

            for synapse in synapse_nodes:
                if nodes[synapse]['class'] == 'Synapse':
                    inferred=0
                else:
                    inferred=1
                if 'N' in nodes[synapse].keys():
                    N = nodes[synapse]['N']
                else:
                    N = 0
                try:
                    post_node = nodes[list(edges[synapse].keys())[0]]
                    if 'uname' in post_node:
                        post = post_node['uname']
                    else:
                        post = post_node['name']
                    csv = csv +  ('\n' + str(pre) + ',' + str(post) + ',' + str(N) + ',' + str(inferred))
                    for i in range(N):
                        out_edges.append((str(pre), str(post)))
                        out_nodes.append(str(pre))
                        out_nodes.append(str(post))
                except:
                    pass
        out_nodes = list(set(out_nodes))
        out_edges_unique = list(set(out_edges))
        return out_nodes, out_edges, out_edges_unique

    def getSynapses(self, presynapticNeuron, postsynapticNeuron):
        """Returns the synapses between a given presynaptic neuron and a postsynaptic neuron.

        # Arguments:
            presynapticNeuron (str): The name of the presynaptic neuron.
            postsynapticNeuron (str): The name of the postsynaptic neuron.

        # Returns:
            float: The number of synapses.
        """
        if self.compiled == False:
            self.prepareCircuit()
        try:
            presynapticIndex = self.out_nodes.index(presynapticNeuron)
        except:
            raise Exception("The presynaptic neuron given as input to 'getSynapses' is not present in the current workspace.")
        try:
            postsynapticIndex = self.out_nodes.index(postsynapticNeuron)
        except:
            raise Exception("The postsynaptic neuron given as input to 'getSynapses' is not present in the current workspace.")
        M = self.getConnectivityMatrix()
        return M[presynapticIndex, postsynapticIndex]

    def getPresynapticNeurons(self, postsynapticNeuron):
        """Returns a dictionary of all presynaptic neurons for a given postsynaptic neuron.

        # Arguments:
            postsynapticNeuron (str): The name of the postsynaptic neuron.

        # Returns:
            dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given postsynaptic neuron.
        """
        if self.compiled == False:
            self.prepareCircuit()
        postsynapticIndex = self.out_nodes.index(postsynapticNeuron)
        if postsynapticIndex<0:
            raise Exception("The postsynaptic neuron given as input to 'getPresynapticNeurons' is not present in the current workspace.")
        M = self.getConnectivityMatrix()
        connDict = {}
        for i in range(M.shape[0]):
            if M[i,postsynapticIndex]>0:
                connDict[self.out_nodes[i]] = M[i,postsynapticIndex]
        return connDict

    def getPostsynapticNeurons(self, presynapticNeuron):
        """Returns a dictionary of all postsynaptic neurons for a given presynaptic neuron.

        # Arguments:
            presynapticNeuron (str): The name of the presynaptic neuron.

        # Returns:
            dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given presynaptic neuron.
        """
        if self.compiled == False:
            self.prepareCircuit()
        presynapticIndex = self.out_nodes.index(presynapticNeuron)
        if presynapticIndex<0:
            raise Exception("The presynaptic neuron given as input to 'getPostsynapticNeurons' is not present in the current workspace.")
        M = self.getConnectivityMatrix()
        connDict = {}
        for i in range(M.shape[0]):
            if M[i,presynapticIndex]>0:
                connDict[self.out_nodes[i]] = M[i,presynapticIndex]
        return connDict

    def GenNB(self, nodes, edges, model = "auto", config = {}, default_neuron = nb.MorrisLecar(),  default_synapse = nb.AlphaSynapse()):
        """Processes the output of processConnectivity to generate a Neuroballad circuit.

        # Returns:
            tuple: A tuple of the Neuroballad circuit, and a dictionary that maps the neuron names to the uids.
        """
        edge_strengths = []
        unique_edges = list(set(edges))

        edge_strengths = Counter(edges)
        neuron_models = []
        neuron_edges = []
        C = nb.Circuit()
        node_keys = {}
        for i in nodes:
            if i not in config:
                idx = C.add_cluster(1, default_neuron)[0]
                node_keys[i] = idx
        for i, v in enumerate(list(C.G.nodes())):
            C.G.nodes()[v]['name'] = list(node_keys.keys())[i]
        if model == "auto":
            for i in edges:
                if i not in config:
                    idx = C.add_cluster(1, default_synapse)[0]
                    C.join([[node_keys[i[0]],idx]])
                    C.join([[idx, node_keys[i[1]]]])
                    C.G.nodes()["uid" + str(idx)]['name'] = "Synapse from " + i[0] + " to " + i[1]
                    # print(C.G.nodes()["uid" + str(idx)]['BioName'])
        if model == "simple":
            for i in edges:
                if i not in config:
                    C.join([[node_keys[i[0]],node_keys[i[1]]]])

        return C, node_keys

    def getConnectivityDendrogram(self):
        self.prepareCircuit()
        M = self.getConnectivityMatrix()
        M = pd.DataFrame(M, index = self.out_nodes, columns = self.out_nodes)
        sns.clustermap(M)

    def sendCircuitPrimitive(self, C, args = {'name': 'temp'}):
        """Sends a NetworkX graph to the backend.
        """
        C.compile(model_output_name = os.path.join(self.dataPath,
                                                   args['name'] + '.gexf.gz'))
        with open(os.path.join(self.dataPath, args['name'] + '.gexf.gz'), 'rb') as file:
            data=file.read()
        a = {}
        a['name'] = args['name']
        a['experiment'] = self.experimentInputs
        a['graph'] = binascii.hexlify(data).decode()
        res = self.client.session.call('ffbo.gfx.sendCircuit', a)
        res = self.client.session.call('ffbo.gfx.sendExperiment', a)
        #print(_FFBOLABClient.client.session.call('ffbo.gfx.sendCircuit', a))

    def alter(self, X):
        """Alters a set of models with specified Neuroballad models.

       # Arguments:
            X (list of lists): A list of lists. Elements are lists whose first element is the neuron ID (str) and the second is the Neuroballad object corresponding to the model.
        """
        if any(isinstance(el, list) for el in X): # Check if input is a list of lists
            pass
        else:
            X = [X]
        for x in X:
            if x[0] in self.node_keys:
                self.C.G.node['uid' + str(self.node_keys[x[0]])].clear()
                params = x[1].params
                params['name'] = params['name'] + str(self.node_keys[x[0]])
                self.C.G.node['uid' + str(self.node_keys[x[0]])].update(params)
            else:
                raise Exception('The rule you passed named', x, 'does match a known node name.')

    def addInput(self, x):
        """Adds an input to the experiment settings. The input is a Neuroballad input object.

        # Arguments:
            x (Neuroballad Input Object): The input object to append to the list of inputs.

        # Returns:
            dict: The input object added to the experiment list.
        """
        self.experimentInputs.append(x.params)
        data = self.experimentInputs
        a = {}
        a['data'] = data
        a['messageType'] = 'Data'
        a['widget'] = 'JSONEditor'
        self.tryComms(a)
        return x.params

    def listInputs(self):
        """Sends the current experiment settings to the frontend for displaying in an editor.
        """
        a = {}
        data = self.experimentInputs
        a['data'] = data
        a['messageType'] = 'Data'
        a['widget'] = 'JSONEditor'
        self.tryComms(a)
        return self.experimentInputs

    def fetchCircuit(self, X, local = True):
        """Deprecated function that locally saves a circuit file via the backend.
           Deprecated because of connectivity issues with large files.
        """
        X = self.client.session.call(u'ffbo.gfx.getCircuit', X)
        X['data'] = binascii.unhexlify(X['data'].encode())
        if local == False:
            with open(os.path.join(_FFBOLabDataPath, X['name'] + '.gexf.gz'), "wb") as file:
                file.write(X['data'])
        else:
            with open(os.path.join(X['name'] + '.gexf.gz'), "wb") as file:
                file.write(X['data'])
        return True

    def fetchExperiment(self, X, local = True):
        """Deprecated function that locally saves an experiment file via the backend.
           Deprecated because of connectivity issues with large files.
        """
        X = self.client.session.call(u'ffbo.gfx.getExperiment', X)
        X['data'] = json.dumps(X['data'])
        if local == False:
            with open(os.path.join(_FFBOLabDataPath, X['name'] + '.json'), "w") as file:
                file.write(X['data'])
        else:
            with open(os.path.join(X['name'] + '.json'), "w") as file:
                file.write(X['data'])
        return True

    def JSCall(self, messageType='getExperimentConfig', data = {}):
        a = {}
        a['data'] = data
        a['messageType'] = messageType
        a['widget'] = 'GFX'
        self.tryComms(a)

    def getExperimentConfig(self):
        self.JSCall()

    def fetchSVG(self, X, local = True):
        """Deprecated function that locally saves an SVG via the backend.
           Deprecated because of connectivity issues with large files.
        """
        X = self.client.session.call(u'ffbo.gfx.getSVG', X)
        X['data'] = binascii.unhexlify(X['data'].encode())
        # X['data'] = json.dumps(X['data'])
        if local == False:
            with open(os.path.join(_FFBOLabDataPath, X['name'] + '.svg'), "wb") as file:
                file.write(X['data'])
        else:
            with open(os.path.join(X['name'] + '.svg'), "wb") as file:
                file.write(X['data'])
        return True

    def _sendSVG(self, X):
        """Deprecated function that loads an SVG via the backend.
           Deprecated because of connectivity issues with large files.
        """
        name = X
        #with open(os.path.join(_FFBOLabDataPath, name + '.svg'), "r") as file:
        #        svg = file.read()
        a = {}
        #a['data'] = svg
        a['data'] = X
        a['messageType'] = 'loadCircuit'
        a['widget'] = 'GFX'
        self.tryComms(a)


    def sendSVG(self, name, file):
        """Sends an SVG to the FBL fileserver. Useful for storing data and using loadSVG.

        # Arguments:
            name (str): Name to use when saving the file; '_visual' gets automatically appended to it.
            file (str): Path to the SVG file.
        """
        with open(file, 'r') as ifile:
            data = ifile.read()
        data = json.dumps({'name': name, 'svg': data})
        self.client.session.call('ffbo.gfx.sendSVG',data)

    def loadSVG(self, name):
        """Loads an SVG in the FBL fileserver.

        # Arguments:
            name (str): Name to use when loading the file.
        """
        self.tryComms({'widget':'GFX','messageType': 'loadCircuit', 'data': name})

    def FICurveGenerator(self, model):
        """Sample library function showing how to do automated experimentation using FFBOLab's Notebook features. Takes a simple abstract neuron model and runs experiments on it.

        # Arguments:
            model (Neuroballad Model Object): The model object to test.

        # Returns:
            numpy array: A tuple of NumPy arrays corresponding to the X and Y of the FI curve.
        """
        del self.data
        self.data = []
        self.sendDataToGFX = False
        del self.C
        self.C = nb.Circuit()

        self.experimentInputs = []
        #self.addInput(
        #    nb.InIStep(0, 5., 0., 1.))
        self.executionSuccessful = True
        circuitName = "FITest"



        for stepAmplitude in range(30):
            idx = self.C.add_cluster(1, model)[0]
            self.addInput(
                nb.InIStep(idx, float(stepAmplitude), 0., 1.))
        self.sendCircuitPrimitive(self.C, args = {'name': circuitName})
        print(printHeader('FFBOLab Client GFX') + 'Circuit sent. Queuing execution.')
        #while self.executionSuccessful == False:
        #    sleep(1)
        #self.experimentInputs = []

        #
        a = {}
        a['name'] = "FITest"
        a['experiment'] = self.experimentInputs
        self.experimentQueue.append(a)
        self.executionSuccessful = False
        a = self.experimentQueue.pop(0)
        # self.parseSimResults()
        res = self.client.session.call('ffbo.gfx.sendExperiment', a)
        res = self.client.session.call('ffbo.gfx.startExecution', {'name': circuitName})


        return True

    def parseSimResults(self):
        """Parses the simulation results.
        """
        numpyData = {}
        for x in self.data:
            if type(x['data']) is dict:
                print(x['data'].keys())
                for i in x['data'].keys():
                    if i not in numpyData.keys():
                        numpyData[i] = x['data'][i]
                    else:
                        numpyData[i] += x['data'][i]
        self.simData = numpyData

    def FICurvePlotSimResults(self):
        """Plots some result curves for the FI curve generator example.
        """
        import matplotlib
        import numpy as np
        import matplotlib.pyplot as plt
        import re
        X = []
        Y = []
        for key in self.simData.keys():
            if 'spike' in key:
                a = np.sum(self.simData[key])
                keynums = [float(s) for s in re.findall(r'-?\d+\.?\d*', key)]
                X.append(keynums[0])
                Y.append(a)

        X = np.array(X)
        Y = np.array(Y)
        Y = Y[np.argsort(X)]
        X = np.sort(X)
        plt.plot(np.array(X), np.array(Y))
        plt.xlabel('Input Amplitude (muA)')
        plt.ylabel('Spike Rate (Spikes/Second)')
        plt.title('F-I Curve for the Queried Model')

    def loadCartridge(self, cartridgeIndex = 100):
        """Sample library function for loading cartridges, showing how one can build libraries that work with flybrainlab.
        """
        self.executeNAquery(
            {"query":[
                {"action":{"method":{"query":{"name":["lamina"]}}},"object":{"class":"LPU"}},
                {"action":{"method":{"traverse_owns":{"cls":"CartridgeModel","name":'cartridge_' + str(cartridgeIndex)}}},"object":{"memory":0}},
                {"action":{"method":{"traverse_owns":{"instanceof":"MembraneModel"}}},"object":{"memory":0}},
                {"action":{"method":{"traverse_owns":{"instanceof":"DendriteModel"}}}, "object":{"memory":1}},
                {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                {"action":{"method":{"traverse_owns":{"cls":"Port"}}},"object":{"memory":3}},
                {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                {"action":{"method":{"gen_traversal_in":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","MembraneModel","instanceof"]]}}},"object":{"memory":0}},
                {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                {"action":{"method":{"gen_traversal_in":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","Aggregator","instanceof"]]}}},"object":{"memory":2}},
                {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                {"action":{"method":{"gen_traversal_out":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","MembraneModel","instanceof"]]}}},"object":{"memory":4}},
                {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                {"action":{"method":{"gen_traversal_out":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","Aggregator","instanceof"]]}}},"object":{"memory":6}},
                {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":2}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":6}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":8}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":11}}},"object":{"memory":0}},
                {"action":{"method":{"get_connecting_synapsemodels":{}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":1}},
                {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                {"action":{"method":{"query":{"name":["retina-lamina"]}}},"object":{"class":"Pattern"}},
                {"action":{"method":{"owns":{"cls":"Interface"}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                {"action":{"op":{"find_matching_ports_from_selector":{"memory":20}}},"object":{"memory":1}},
                {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                {"action":{"method":{"query":{"name":["retina"]}}},"object":{"class":"LPU"}},
                {"action":{"op":{"find_matching_ports_from_selector":{"memory":1}}},"object":{"memory":0}},
                {"action":{"method":{"gen_traversal_in":{"pass_through":["SendsTo", "MembraneModel","instanceof"]}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":10}}},"object":{"memory":0}},
                {"action":{"op":{"__add__":{"memory":4}}},"object":{"memory":0}}
            ],
            "format":"no_result"})

        res = self.executeNAquery({"query":[{"action":{"method":{"has":{}}},"object":{"state":0}}],"format":"nx"})

        data = []
        for i in res:
            if 'data' in i:
                if 'data' in i['data']:
                    if 'nodes' in i['data']['data']:
                        data.append(i['data']['data'])
        G=nx.Graph(data[0])
        self.C.G = G
        return True

    def loadExperimentConfig(self, x):
        """Updates the simExperimentConfig attribute using input from the diagram.

        # Arguments:
            x (string): A JSON dictionary as a string.

        # Returns:
            bool: True.
        """
        print('Obtained Experiment Configuration: ', x)
        self.simExperimentConfig = json.loads(x)
        return True

    def initiateExperiments(self):
        """Initializes and executes experiments for different LPUs.
        """
        print('Initiating experiments...')
        print('Experiment Setup: ', self.simExperimentConfig)
        for key in self.simExperimentConfig.keys():
            if key in self.simExperimentRunners.keys():
                try:
                    module = importlib.import_module(i)
                    print('Loaded LPU {}.'.format(i))
                    self.simExperimentRunners[key] = getattr(module, 'sim')
                except:
                    print('Failed to load LPU {}.'.format(i))
                run_func = self.simExperimentRunners[key]
                run_func(self.simExperimentConfig)
            else:
                print('No runner(s) were found for Diagram {}.'.format(key))
        return True

    def prune_retina_lamina(self, removed_neurons = [], removed_labels=[], retrieval_format="nk"):
        """Prunes the retina and lamina circuits.

        # Arguments:
            cartridgeIndex (int): The cartridge to load. Optional.

        # Returns:
            dict: A result dict to use with the execute_lamina_retina function.

        # Example:
            res = load_retina_lamina(nm[0])
            execute_multilpu(nm[0], res)
        """
        list_of_queries = [
        {"command":{"swap":{"states":[0,1]}},"format":"nx", "user": self.client._async_session._session_id, "server": self.naServerID},
        {"query":[{"action":{"method":{"has":{"name": removed_neurons}}},"object":{"state":0}},{"action":{"method":{"has":{"label": removed_labels}}},"object":{"state":0}},{"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":0}},
                    {"action":{"method":{"has":{"via":['+removed_via+']}}},"object":{"state":0}},{"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"op":{"find_matching_ports_from_selector":{"memory":0}}},"object":{"state":0}},
                    {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                    {"action":{"method":{"gen_traversal_in":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":0}},
                    {"action":{"method":{"gen_traversal_out":{"min_depth":1, "pass_through":["SendsTo", "SynapseModel","instanceof"]}}},"object":{"memory":1}},
                    {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},{"action":{"op":{"__add__":{"memory":3}}},"object":{"memory":0}},
                    {"action":{"op":{"__sub__":{"memory":0}}},"object":{"state":0}}],
        "format":retrieval_format, "user": self.client._async_session._session_id, "server": self.naServerID}
        ]
        res = self.client.session.call('ffbo.processor.neuroarch_query', list_of_queries[0])
        print('Pruning ', removed_neurons)
        print('Pruning ', removed_labels)
        res = self.client.session.call('ffbo.processor.neuroarch_query', list_of_queries[1])
        return res


    def load_retina_lamina(self, cartridgeIndex=11, removed_neurons = [], removed_labels=[], retrieval_format="nk"):
        """Loads retina and lamina.

        # Arguments:
            cartridgeIndex (int): The cartridge to load. Optional.

        # Returns:
            dict: A result dict to use with the execute_lamina_retina function.

        # Example:
            nm[0].getExperimentConfig() # In a different cell
            experiment_configuration = nm[0].load_retina_lamina(cartridgeIndex=126)
            experiment_configuration = experiment_configuration['success']['result']
            nm[0].execute_multilpu(experiment_configuration)
        """

        inp = {"query":[
                        {"action":{"method":{"query":{"name":["lamina"]}}},"object":{"class":"LPU"}},
                        {"action":{"method":{"traverse_owns":{"cls":"CartridgeModel","name":'cartridge_' + str(cartridgeIndex)}}},"object":{"memory":0}},
                        {"action":{"method":{"traverse_owns":{"instanceof":"MembraneModel"}}},"object":{"memory":0}},
                        {"action":{"method":{"traverse_owns":{"instanceof":"DendriteModel"}}}, "object":{"memory":1}},
                        {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                        {"action":{"method":{"traverse_owns":{"cls":"Port"}}},"object":{"memory":3}},
                        {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                        {"action":{"method":{"gen_traversal_in":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","MembraneModel","instanceof"]]}}},"object":{"memory":0}},
                        {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                        {"action":{"method":{"gen_traversal_in":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","Aggregator","instanceof"]]}}},"object":{"memory":2}},
                        {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                        {"action":{"method":{"gen_traversal_out":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","MembraneModel","instanceof"]]}}},"object":{"memory":4}},
                        {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                        {"action":{"method":{"gen_traversal_out":{"min_depth":2,"pass_through":[["SendsTo","SynapseModel","instanceof"],["SendsTo","Aggregator","instanceof"]]}}},"object":{"memory":6}},
                        {"action":{"method":{"has":{"name":"Amacrine"}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":2}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":6}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":8}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":11}}},"object":{"memory":0}},
                        {"action":{"method":{"get_connecting_synapsemodels":{}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                        {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":1}},
                        {"action":{"op":{"__add__":{"memory":1}}},"object":{"memory":0}},
                        {"action":{"method":{"query":{"name":["retina-lamina"]}}},"object":{"class":"Pattern"}},
                        {"action":{"method":{"owns":{"cls":"Interface"}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                        {"action":{"op":{"find_matching_ports_from_selector":{"memory":20}}},"object":{"memory":1}},
                        {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                        {"action":{"method":{"get_connected_ports":{}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":0}}},"object":{"memory":1}},
                        {"action":{"method":{"query":{"name":["retina"]}}},"object":{"class":"LPU"}},
                        {"action":{"op":{"find_matching_ports_from_selector":{"memory":1}}},"object":{"memory":0}},
                        {"action":{"method":{"gen_traversal_in":{"pass_through":["SendsTo", "MembraneModel","instanceof"]}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":10}}},"object":{"memory":0}},
                        {"action":{"op":{"__add__":{"memory":4}}},"object":{"memory":0}}
                    ],
                    "format":"no_result",
                    "user": self.client._async_session._session_id,
                    "server": self.naServerID}

        res = self.client.session.call('ffbo.processor.neuroarch_query', inp)
        

        inp = {"query":[
                        {"action":{"method":{"has":{}}},"object":{"state":0}}
                    ],
                    "format":"nx",
                    "user": self.client._async_session._session_id,
                    "server": self.naServerID}

        

        res = self.client.session.call('ffbo.processor.neuroarch_query', inp)

        res_info = self.client.session.call(u'ffbo.processor.server_information')
        msg = {"user": self.client._async_session._session_id,
            "servers": {'na': self.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.client.session.call(u'ffbo.na.query.' + msg['servers']['na'], {'user': msg['user'],
                                'command': {"retrieve":{"state":0}},
                                'format': "nk"})


        neurons = self.get_current_neurons(res)
        if 'cartridge_' + str(cartridgeIndex) in self.simExperimentConfig:
            if 'disabled' in self.simExperimentConfig['cartridge_' + str(cartridgeIndex)]:
                removed_neurons = removed_neurons + self.simExperimentConfig['cartridge_' + str(cartridgeIndex)]['disabled']
                print('Updated Disabled Neuron List: ', removed_neurons)
        removed_neurons = self.ablate_by_match(res, removed_neurons)

        res = self.prune_retina_lamina(removed_neurons = removed_neurons, removed_labels = removed_labels, retrieval_format=retrieval_format)
        """
        res_info = self.client.session.call(u'ffbo.processor.server_information')
        msg = {"user": self.client._async_session._session_id,
            "servers": {'na': self.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.client.session.call(u'ffbo.na.query.' + msg['servers']['na'], {'user': msg['user'],
                                'command': {"retrieve":{"state":0}},
                                'format': "nk"})
        """
        # print(res['data']['LPU'].keys())
        print('Retina and lamina circuits have been successfully loaded.')
        return res

    def get_current_neurons(self, res):
        labels = []
        for i in res['data']['LPU']:
            for j in res['data']['LPU'][i]['nodes']:
                if 'label' in res['data']['LPU'][i]['nodes'][j]:
                    label = res['data']['LPU'][i]['nodes'][j]['label']
                    if 'port' not in label and 'synapse' not in label:
                        labels.append(label)
        return labels

    def ablate_by_match(self, res, neuron_list):
        neurons = self.get_current_neurons(res)
        removed_neurons = []
        for i in neuron_list:
            removed_neurons = removed_neurons + [j for j in neurons if i in j]
        removed_neurons = list(set(removed_neurons))
        return removed_neurons

    def execute_multilpu(self, res):
        """Executes a multilpu circuit. Requires a result dictionary.

        # Arguments:
            res (dict): The result dictionary to use for execution.

        # Returns:
            bool: A success indicator.
        """
        labels = []
        for i in res['data']['LPU']:
            for j in res['data']['LPU'][i]['nodes']:
                if 'label' in res['data']['LPU'][i]['nodes'][j]:
                    label = res['data']['LPU'][i]['nodes'][j]['label']
                    if 'port' not in label and 'synapse' not in label:
                        labels.append(label)

        res = self.client.session.call(u'ffbo.processor.server_information')
        msg = {'neuron_list': labels,
            "user": self.client._async_session._session_id,
            "servers": {'na': self.naServerID, 'nk': list(res['nk'].keys())[0]}}

        print(res)
        res = []
        def on_progress(x, res):
            res.append(x)
        res_list = []
        res = self.client.session.call('ffbo.processor.nk_execute', msg, options=CallOptions(
                            on_progress=partial(on_progress, res=res_list), timeout = 30000000000
                        ))
        print('Execution request sent. Please wait.')
    


import importlib

LPU_list = ['cx','mb']

for i in LPU_list:
    try:
        module = importlib.import_module(i)
        print('Loaded LPU {}.'.format(i))
        sim_func = getattr(module, 'sim')
        sim_func({'hello': 'world'})
    except:
        print('Failed to load LPU {}.'.format(i))


ffbolabClient = Client


