import os
import sys
import subprocess
import logging
import importlib.util

# Install all necessary packages
## We attempt to resolve package installation errors during the import.

def install(package):
    if package == 'neuroballad':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'git+https://github.com/FlyBrainLab/Neuroballad.git'])
    else:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def check_then_import(package):
    spec = importlib.util.find_spec(package.replace('pypiwin32','win32').replace('-','_'))
    if spec is None:
        install(package)

package_list = ['txaio','twisted','autobahn','crochet','service_identity','autobahn-sync','matplotlib','h5py','seaborn','fastcluster','networkx','msgpack','pandas','scipy','sympy','nose','neuroballad','jupyter','jupyterlab']
if os.name == 'nt':
    package_list.append('pypiwin32')
for i in package_list:
    check_then_import(i)

# Go ahead with imports

from time import sleep, gmtime, strftime
import os, sys, json, binascii, warnings, urllib
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
import random
import pickle
from collections import Counter
from functools import partial
from pathlib import Path
from configparser import ConfigParser
import importlib

import numpy as np
import matplotlib.pyplot as plt
import txaio
import h5py
import pandas as pd
import networkx as nx
import autobahn
from autobahn.twisted.util import sleep
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner
from autobahn.wamp.exception import ApplicationError
from autobahn.wamp import auth
from autobahn.websocket.protocol import WebSocketClientFactory
from autobahn.wamp.types import RegisterOptions, CallOptions
from autobahn_sync import publish, call, register, subscribe, run, AutobahnSync
from twisted.internet._sslverify import OpenSSLCertificateAuthorities
from twisted.internet.ssl import CertificateOptions
import OpenSSL.crypto
from IPython.display import clear_output
import requests
import seaborn as sns
sns.set(color_codes=True)

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import neuroballad as nb
from .utils import setProtocolOptions
from .exceptions import *


# Create the home directory
## The home directory is situated at the home directory of the user, and is named ".ffbo".

home = str(Path.home())
if not os.path.exists(os.path.join(home, ".ffbo")):
    os.makedirs(os.path.join(home, ".ffbo"), mode=0o777)
if not os.path.exists(os.path.join(home, ".ffbo", "data")):
    os.makedirs(os.path.join(home, ".ffbo", "data"), mode=0o777)
if not os.path.exists(os.path.join(home, ".ffbo", "config")):
    os.makedirs(os.path.join(home, ".ffbo", "config"), mode=0o777)
if not os.path.exists(os.path.join(home, ".ffbo", "lib")):
    os.makedirs(os.path.join(home, ".ffbo", "lib"), mode=0o777)

# Generate the data path to be used for imports
_FBLDataPath = os.path.join(home, ".ffbo", "data")
_FBLConfigPath = os.path.join(home, ".ffbo", "config", "ffbo.flybrainlab.ini")

logging.basicConfig(format = '[%(name)s %(asctime)s] %(message)s')

def convert_from_bytes(data):
    """Attempt to decode data from bytes; useful for certain data types retrieved from servers.

    # Arguments
        data (object): Data in bytes, or some other data structure whose elements are in bytes.
    # Returns
        object: The object that was in bytes.
    """
    if isinstance(data, bytes):      return data.decode()
    if isinstance(data, dict):       return dict(map(convert_from_bytes, data.items()))
    if isinstance(data, tuple):      return tuple(map(convert_from_bytes, data))
    if isinstance(data, list):       return list(map(convert_from_bytes, data))
    if isinstance(data, set):        return set(map(convert_from_bytes, data))
    return data

def urlRetriever(url, savePath, verify=False):
    """Retrieves and saves a url in Python 3.

    # Arguments
        url (str): File url.
        savePath (str): Path to save the file to.
    """
    with open(savePath, "wb") as f:
        resp = requests.get(url, verify=verify)
        f.write(resp.content)


def guidGenerator():
    """Unique query ID generator for handling the backend queries

    # Returns
        str: The string with time format and brackets.
    """

    def S4():
        return str(((1 + random.random()) * 0x10000))[1]

    return S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4()


def printHeader(name):
    """Header printer for the console messages. Useful for debugging.

    # Arguments
        name (str): Name of the component.

    # Returns
        str: The string with time format and brackets.
    """
    return "[" + name + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "] "


class MetaClient:
    """FlyBrainLab MetaClient class that tracks available FBL clients and connected frontend widgets.

    # Attributes:
        clients (obj): A list of dictionaries with the following keys: (i) 'name': Contains the common name of the client. (ii) 'client': A reference to the client object. (iii) 'widgets': List of widget names associated with the client.
    """

    def __init__(self, initializer=None):
        """Initialization function for the FBL MetaClient class.

        # Arguments
            initializer (list): A list of dictionaries with initialization data for connections.
        """
        self.clients = {}
        self.latest_client = None # latest active client
        if initializer is not None:
            for i in list(initializer.keys()):
                self.clients[i] = initializer[i]

    def add_client(self, client_name, client, client_widgets=[]):
        """Adds a client with optional existing widgets to the MetaClient.

        # Arguments
            client_name (str): Name of the FlyBrainLab client.
            client (obj): A FlyBrainLab client object.
            client_widgets (list): A list of strings corresponding to the names of the currently connected client widgets. Defaults to empty list.
        """
        new_client = {}
        new_client["client"] = client
        new_client["widgets"] = client_widgets
        self.clients[client_name] = new_client

    def delete_client(self, client_name):
        """Delete a client from the MetaClient.

        # Arguments
            client_name (str): Name of the FlyBrainLab client.
        """
        if client_name in list(self.clients.keys()):
            del self.clients[client_name]

    def add_widget(self, client_name, widget_name):
        """Add a widget to a client in the MetaClient.

        # Arguments
            client_name (str): Name of the FlyBrainLab client.
            widget_name (str): Name of the new NeuroMynerva widget.
        """
        if client_name in self.clients:
            self.clients[client_name]["widgets"].append(widget_name)

    def delete_widget(self, client_name, widget_name):
        """Delete a widget to a client in the MetaClient.

        # Arguments
            client_name (str): Name of the FlyBrainLab client.
            widget_name (str): Name of the new NeuroMynerva widget.
        """
        if client_name in self.clients:
            if widget_name in self.clients[client_name]["widgets"]:
                idx = self.clients[client_name]["widgets"].index(widget_name)
                del self.clients[client_name]["widgets"][idx]

    def get_client(self, client_name):
        """Get a client in the MetaClient by name.

        # Arguments
            client_name (str): Name of the FlyBrainLab client.

        # Returns
            obj: The associated FlyBrainLab client.
        """
        return self.clients[client_name]["client"]

    def update_client_names(self):
        """Update all client names with naming scheme from the frontend. Used for synchronization.
        """
        for i in self.clients:
            self.clients[i]['client'].name = i
            self.clients[i]['client'].widgets = self.clients[i]['widgets']

    def get_client_info(self):
        """Receives client data.

        # Example:
            fbl.client_manager.get_client_info()
            cl = fbl.client_manager.clients['client-Neu3D-1-473e4837-8b1d-440a-a361-172507db1b38']['client']
            cl.get_client_info()

        # Arguments
            client_name (str): Name of the FlyBrainLab client.

        # Returns
            obj: The associated FlyBrainLab client.
        """
        self.update_client_names()
        res = {}
        for i in self.clients:
            client = self.clients[i]
            client_data = {}
            client_data['widgets'] = client['widgets']
            client_data['name'] = i
            client_data['species'] = client['client'].species
            res[client['client'].name] = client_data
        return res

class MetaComm:
    """A meta-communications object to assist in sending messages to the frontend.

    # Attributes:
        name (str): Name of the communications object.
        manager (obj): A WidgetManager  object that manages the widgets connected to this object..
    """
    def __init__(self, name, manager):
        """Initialization function for the MetaComm class.

        # Arguments
            name (str): Name of the communications object.
            manager (obj): A WidgetManager  object that manages the widgets connected to this object..
        """
        self.name = name
        self.manager = manager


    def send(self, data=None):
        """A function to send data to all widgets connected to the manager of this meta-communications object.

        # Arguments
            data (object): The data object to send.
        """
        for widget_name in self.manager.client_manager.clients[self.name]['widgets']:
            self.manager.widget_manager.widgets[widget_name].comm.send(data=data)


class Client:
    """FlyBrainLab Client class. This class communicates with JupyterLab frontend and connects to FFBO components.

    # Attributes:
        FBLcomm (obj): The communication object for sending and receiving data.
        circuit (obj): A Neuroballad circuit that enables local circuit execution and facilitates circuit modification.
        dataPath (str): Data path to be used.
        experimentInputs (list of dicts): Inputs as a list of dicts that can be parsed by the GFX component.
        compiled (bool): Circuits need to be compiled into networkx graphs before being sent for simulation. This is necessary as circuit compilation is a slow process.
        sendDataToGFX (bool): Whether the data received from the backend should be sent to the frontend. Useful for code-only projects.
    """

    def tryComms(self, a):
        """Communication function to communicate with a JupyterLab frontend if one exists.

        # Arguments
            a (obj): Arbitrarily formatted data to be sent via communication.
        """
        try:
            for i in fbl.widget_manager.widgets:
                if fbl.widget_manager.widgets[i].widget_id not in fbl.client_manager.clients[fbl.widget_manager.widgets[i].client_id]['widgets']:
                    fbl.client_manager.clients[fbl.widget_manager.widgets[i].client_id]['widgets'].append(fbl.widget_manager.widgets[i].widget_id)
        except:
            pass
        try:
            self.FBLcomm.send(data=a)
        except:
            pass

    def __init__(
        self,
        ssl=False,
        debug=True,
        authentication=True,
        user="guest",
        secret="guestpass",
        custom_salt=None,
        url=u"wss://flycircuitdev.neuronlp.fruitflybrain.org/ws",
        realm=u"realm1",
        ca_cert_file="isrgrootx1.pem",
        intermediate_cert_file="letsencryptauthorityx3.pem",
        FFBOLabcomm=None,
        FBLcomm=None,
        legacy=False,
        initialize_client=True,
        name = None,
        species = '',
        use_config = False,
        custom_config = None,
        widgets = [],
        dataset = 'default',
        log_level = 'info'
    ):
        """Initialization function for the FBL Client class.


        # Arguments
            ssl (bool): Whether the FFBO server uses SSL.
            debug (bool) : Whether debugging should be enabled.
            authentication (bool): Whether authentication is enabled.
            user (str): Username for establishing communication with FFBO components.
            secret (str): Password for establishing communication with FFBO components.
            url (str): URL of the WAMP server with the FFBO Processor component.
            realm (str): Realm to be connected to.
            ca_cert_file (str): Path to the certificate for establishing connection.
            intermediate_cert_file (str): Path to the intermediate certificate for establishing connection.
            FFBOLabcomm (obj): Communications object for the frontend.
            FBLcomm (obj): Communications object for the frontend.
            legacy (bool): Whether the server uses the old FFBO server standard or not. Should be False for most cases. Defaults to False.
            initialize_client (bool): Whether to connect to the database or not. Defaults to True.
            name (str): Name for the client. String. Defaults to None.
            use_config (bool): Whether to read the url from config instead of as arguments to the initializer. Defaults to False. False recommended for new users.
            species (str): Name of the species to use for client information. Defaults to ''.
            custom_config (str): A .ini file name to use to initiate a custom connection. Defaults to None. Used if provided.
            widgets (list): List of widgets associated with this client. Optional.
            dataset (str): Name of the dataset to use. Not used right now, but included for future compatibility.
            log_level (str): Log level, can be any of the standard Python logging.logger levels: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET. (see also https://docs.python.org/3/library/logging.html#logging-levels)
        """
        self.log_level = log_level.upper()
        self.log = {'Client': logging.getLogger('FBL Client'),
                    'NA': logging.getLogger('FBL NA'),
                    'NLP': logging.getLogger('FBL NLP'),
                    'GFX': logging.getLogger('FBL GFX'),
                    'NK': logging.getLogger('FBL NK'),
                    'Master': logging.getLogger('FBL Master')}
        self.set_log_level(log_level.upper(), logger_names = None)
        self.name = name
        self.species = species
        self.url = url
        self.widgets = widgets
        if FBLcomm is None and FFBOLabcomm is not None:
            FBLcomm = FFBOLabcomm
        if os.path.exists(os.path.join(home, ".ffbo", "lib")):
            self.log['Client'].debug("Downloading the latest certificates.")
            # CertificateDownloader = urllib.URLopener()
            if not os.path.exists(
                os.path.join(home, ".ffbo", "config", "FBLClient.ini")
            ):
                urlRetriever(
                    "https://data.flybrainlab.fruitflybrain.org/config/FBLClient.ini",
                    os.path.join(home, ".ffbo", "config", "FBLClient.ini"),
                )
            if not os.path.exists(
                os.path.join(home, ".ffbo", "config", "flycircuit_config.ini")
            ):
                urlRetriever(
                    "https://data.flybrainlab.fruitflybrain.org/config/flycircuit_config.ini",
                    os.path.join(home, ".ffbo", "config", "flycircuit_config.ini"),
                )
            if not os.path.exists(
                os.path.join(home, ".ffbo", "config", "hemibrain_config.ini")
            ):
                urlRetriever(
                    "https://data.flybrainlab.fruitflybrain.org/config/hemibrain_config.ini",
                    os.path.join(home, ".ffbo", "config", "hemibrain_config.ini"),
                )
            if not os.path.exists(
                os.path.join(home, ".ffbo", "config", "larva_config.ini")
            ):
                urlRetriever(
                    "https://data.flybrainlab.fruitflybrain.org/config/larva_config.ini",
                    os.path.join(home, ".ffbo", "config", "larva_config.ini"),
                )
            urlRetriever(
                "https://data.flybrainlab.fruitflybrain.org/lib/isrgrootx1.pem",
                os.path.join(home, ".ffbo", "lib", "caCertFile.pem"),
            )
            urlRetriever(
                "https://data.flybrainlab.fruitflybrain.org/lib/letsencryptauthorityx3.pem",
                os.path.join(home, ".ffbo", "lib", "intermediateCertFile.pem"),
            )
            config_file = os.path.join(home, ".ffbo", "config", "FBLClient.ini")
            ca_cert_file = os.path.join(home, ".ffbo", "lib", "caCertFile.pem")
            intermediate_cert_file = os.path.join(
                home, ".ffbo", "lib", "intermediateCertFile.pem"
            )
        # config = ConfigParser()
        # print(config_file)
        # config.read(config_file)

        # This is a temporary fix. The configuration should be provided when instantiating a Client instance
        if use_config:
            root = os.path.expanduser("/")
            homedir = os.path.expanduser("~")
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_files = []
            os.path.join(home, ".ffbo", "config", "FFBO.ini"),
            config = ConfigParser()
            configured = False
            file_type = 0
            for config_file in config_files:
                if os.path.exists(config_file):
                    config.read(config_file)
                    configured = True
                    break
                file_type += 1
            if not configured:
                raise Exception("No config file exists for this component")

            user = config["USER"]["user"]
            secret = config["USER"]["secret"]
            ssl = eval(config["AUTH"]["ssl"])
            websockets = "wss" if ssl else "ws"
            if "ip" in config["SERVER"]:
                split = config["SERVER"]["ip"].split(':')
                ip = split[0]
                if len(split) == 2:
                    port = split[1]
                    url =  "{}://{}:{}/ws".format(websockets, ip, port)
                else:
                    url =  "{}://{}/ws".format(websockets, ip)
            else:
                ip = "localhost"
                port = int(config["NLP"]['port'])
                url =  "{}://{}:{}/ws".format(websockets, ip, port)
            realm = config["SERVER"]["realm"]
            authentication = eval(config["AUTH"]["authentication"])
            debug = eval(config["DEBUG"]["debug"])
            ssl = False # override ssl for connections
            if 'dataset' in config["SERVER"]:
                dataset = config["SERVER"]['dataset']
        if custom_config is not None:
            root = os.path.expanduser("/")
            homedir = os.path.expanduser("~")
            filepath = os.path.dirname(os.path.abspath(__file__))
            config_files = []
            config_files.append(os.path.join(home, ".ffbo", "config", custom_config))
            config_files.append(os.path.join(homedir, ".ffbo", "config", custom_config))
            config_files.append(os.path.join(root, ".ffbo", "config", custom_config))
            config = ConfigParser()
            configured = False
            file_type = 0
            for config_file in config_files:
                if os.path.exists(config_file):
                    config.read(config_file)
                    configured = True
                    break
                file_type += 1
            if not configured:
                raise Exception("No config file exists for this component")

            user = config["USER"]["user"]
            secret = config["USER"]["secret"]
            ssl = eval(config["AUTH"]["ssl"])
            websockets = "wss" if ssl else "ws"
            if "ip" in config["SERVER"]:
                ip = config["SERVER"]["ip"]
            else:
                ip = "localhost"
            if "port" in config["SERVER"]:
                port = int(config["SERVER"]["port"])
                url =  "{}://{}:{}/ws".format(websockets, ip, port)
            else:
                url =  u"{}://{}/ws".format(websockets, ip)

            realm = config["SERVER"]["realm"]
            # authentication = eval(config["AUTH"]["authentication"])
            ssl = False # override ssl for connections
            if 'dataset' in config["SERVER"]:
                dataset = config["SERVER"]['dataset']
        # end of temporary fix
        self.url = url
        self.FBLcomm = FBLcomm # Current Communications Object
        self.C = (
            nb.Circuit()
        )  # The Neuroballd Circuit object describing the loaded neural circuit
        self.neuron_data = {}
        self.active_data = []
        self.dataPath = _FBLDataPath
        extra = {"auth": authentication}
        self.lmsg = 0
        self.dataset = dataset
        self.NLPInterpreters = []
        self.enableResets = True  # Enable resets
        self.addToRemove = False  # Switch adds to removals
        self.history = []
        self.experimentInputs = []  # List of current experiment inputs
        self.compiled = (
            False  # Whether the current circuit has been compiled into a NetworkX Graph
        )
        self.sendDataToGFX = (
            True  # Shall we send the received simulation data to GFX Component?
        )
        self.executionSuccessful = False  # Used to wait for data loading
        self.experimentQueue = []  # A queue for experiments
        self.simExperimentConfig = (
            {}
        )  # Experiment configuration (disabled neurons etc.) for simulations
        self.simExperimentRunners = {}  # Experiment runners for simulations
        self.simData = {}  # Locally loaded simulation data obtained from server
        self.clientData = []  # Servers list
        self.data = (
            []
        )  # A buffer for data from backend; used in multiple functions so needed
        self.uname_to_rid = {}  # local map from unames to rid's
        self.legacy = legacy
        self.neuronStats = {}
        self.query_threshold = 20
        self.naServerID = None
        self.experimentWatcher = None
        self.exec_result = {}
        self.current_exec_result = None
        self.connected = False

        if self.legacy:
            self.query_threshold = 2
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.z_scale = 1.0
        self.r_scale = 1.0
        self.x_shift = 0.0
        self.y_shift = 0.0
        self.z_shift = 0.0
        self.r_shift = 0.0
        self.errors = [] # Buffer that stores errors
        st_cert = open(ca_cert_file, "rt").read()
        c = OpenSSL.crypto
        ca_cert = c.load_certificate(c.FILETYPE_PEM, st_cert)
        st_cert = open(intermediate_cert_file, "rt").read()
        intermediate_cert = c.load_certificate(c.FILETYPE_PEM, st_cert)
        """ Some alternative approaches for certificates:
        # import certifi
        # st_cert = open(certifi.where(), "rt").read()
        # certifi_cert = c.load_certificate(c.FILETYPE_PEM, st_cert)
        # import twisted
        # print(twisted.internet.ssl.platformTrust())
        """
        certs = OpenSSLCertificateAuthorities([ca_cert, intermediate_cert])
        ssl_con = CertificateOptions(trustRoot=certs)
        if initialize_client:
            self.ssl = ssl
            self.user = user
            self.secret = secret
            self.custom_salt = custom_salt
            self.url = url
            self.ssl_con = ssl_con
            self.legacy = legacy
            self.dataset = dataset
            self.init_client(ssl, user, secret, custom_salt, url, ssl_con, legacy)
            self.findServerIDs(dataset)  # Get current server IDs
            self.connected = True

    def set_log_level(self, level, logger_names = None):
        """
        Set the log level of the Client instance.

        # Arguments
            level (str):  Log level, can be any of the standard Python logging.logger levels: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET. (see also https://docs.python.org/3/library/logging.html#logging-levels)

            logger_names (list): the names of the logger in a list. Defaults to None and will set level for all logs.
        """
        if logger_names is None:
            logger_names = list(self.log.keys())
        for log in logger_names:
            try:
                self.log[log].setLevel(level.upper())
            except KeyError:
                Warning('log {} does not exist - level not set'.format(log))
                pass

    def reconnect(self):
        try:
            self.init_client( self.ssl,  self.user,  self.secret,  self.custom_salt,  self.url,  self.ssl_con,  self.legacy)
            self.connected = True
            try:
                self.findServerIDs(self.dataset)  # Attempt to retrieve current server IDs
                self.connected = True
            except Exception as e: # Server finding fails
                self.raise_error(e, 'There was an error in trying to find servers. Check your server configuration or contact the backend administrator.')
                print(e)
                self.connected = False
            if len(self.active_data)>0:
                self.active_data = [i for i in self.active_data if '#' in i]
                self.executeNAquery({'query': [{'action': {'method': {'query': {} }},
                                                'object': {'rid': self.active_data }
                                            },
                                                {'action': {'method': {'gen_traversal_in': {'pass_through': ['HasData'], 'min_depth': 1} }},
                                                'object': {'memory': 0 }
                                            },
                                            ],
                                    'user': 'test',
                                    'format': 'no_result'})
        except Exception as e:
            self.raise_error(e, 'Failed to connect to the server. Check your server configuration or contact the backend administrator. Alternatively, use the client.reconnect function.')
            print(e)
            self.connected = False



    def init_client(self, ssl, user, secret, custom_salt, url, ssl_con, legacy):
        FBLClient = AutobahnSync()
        @FBLClient.on_challenge
        def on_challenge(challenge):
            """The On Challenge function that computes the user signature for verification.

            # Arguments
                challenge (obj): The challenge object received.

            # Returns
                str: The signature sent to the router for verification.
            """
            self.log['Client'].debug("Initiating authentication.")
            if challenge.method == u"wampcra":
                self.log['Client'].debug("WAMP-CRA challenge received: {}".format(challenge))
                self.log['Client'].debug(challenge.extra['salt'])
                if custom_salt is not None:
                    salted_key = custom_salt
                else:
                    if u'salt' in challenge.extra:
                        # Salted secret
                        self.log['Client'].debug('Deriving key...')
                        salted_key = auth.derive_key(secret,
                                              challenge.extra['salt'],
                                              challenge.extra['iterations'],
                                              challenge.extra['keylen'])
                        #print(salted_key.decode('utf-8'))

                # compute signature for challenge, using the key
                signature = auth.compute_wcs(salted_key, challenge.extra["challenge"])

                # return the signature to the router for verification
                return signature

            else:
                raise Exception("Invalid authmethod {}".format(challenge.method))

        if ssl:
            FBLClient.run(
                url=url, authmethods=[u"wampcra"], authid=user, ssl=ssl_con
            )  # Initialize the communication right now!
        else:
            FBLClient.run(url=url, authmethods=[u'wampcra'], authid=user)

        setProtocolOptions(FBLClient._async_session._transport,
                           maxFramePayloadSize = 0,
                           maxMessagePayloadSize = 0,
                           autoFragmentSize = 65536)

        @FBLClient.subscribe(
            'ffbo.server.update.' + str(FBLClient._async_session._session_id)
            )

        def updateServers(data):
            """Updates available servers.

            # Arguments
                data (obj): Obtained servers list.

            """
            self.clientData.append(data)
            print("Updated the Servers")

        self.log['Client'].debug("Subscribed to topic 'ffbo.server.update'")

        @FBLClient.register(
            "ffbo.ui.receive_cmd." + str(FBLClient._async_session._session_id)
        )
        def receiveCommand(data):
            """The Receive Command function that receives commands and sends them to the frontend.

            # Arguments
                data (dict): Data to be sent to the frontend

            # Returns
                bool: Whether the data has been received.
            """
            self.clientData.append("Received Command")
            a = {}
            data = convert_from_bytes(data)
            a["data"] = data
            a["messageType"] = "Command"
            a["widget"] = "NLP"

            if "commands" in data:
                if "remove" in data["commands"]:
                    to_remove = data["commands"]['remove'][0]
                    for i in to_remove:
                        if i in self.active_data:
                            self.active_data.remove(i)
            #self.data.append(a)
            self.log['NLP'].debug("Received a command.")
            to_send = True
            if self.enableResets == False:
                if "commands" in data:
                    if "reset" in data["commands"]:
                        to_send = False
            if to_send == True:
                self.tryComms(a)
            return True

        self.log['Client'].debug("Procedure ffbo.ui.receive_cmd Registered...")

        @FBLClient.register(
            "ffbo.ui.receive_gfx." + str(FBLClient._async_session._session_id)
        )
        def receiveGFX(data):
            """The Receive GFX function that receives commands and sends them to the GFX frontend.

            # Arguments
                data (dict): Data to be sent to the frontend.

            # Returns
                bool: Whether the data has been received.
            """
            self.clientData.append("Received GFX Data")
            data = convert_from_bytes(data)
            self.data.append(data)
            self.log['GFX'].debug("Received a message for GFX.")
            if self.sendDataToGFX == True:
                self.tryComms(data)
            else:
                if "messageType" in data.keys():
                    if data["messageType"] == "showServerMessage":
                        self.log['GFX'].info("Execution successful for GFX.")
                        if len(self.experimentQueue) > 0:
                            self.log['GFX'].info(
                                "Next execution now underway. Remaining simulations: "
                                + str(len(self.experimentQueue)))
                            a = self.experimentQueue.pop(0)
                            res = self.rpc("ffbo.gfx.sendExperiment", a)
                            res = self.rpc(
                                "ffbo.gfx.startExecution", {"name": a["name"]}
                            )
                        else:
                            self.executionSuccessful = True
                            self.parseSimResults()
                            self.log['GFX'].info("GFX results successfully parsed.")
            return True

        self.log['Client'].debug("Procedure ffbo.ui.receive_gfx Registered...")

        @FBLClient.register(
            "ffbo.ui.get_circuit." + str(FBLClient._async_session._session_id)
        )
        def get_circuit(X):
            """Obtain a circuit and save it to the local FBL folder.

            # Arguments
                X (str): Name of the circuit.

            # Returns
                bool: Whether the process has been successful.
            """
            name = X["name"]
            G = binascii.unhexlify(X["graph"].encode())
            with open(os.path.join(_FBLDataPath, name + ".gexf.gz"), "wb") as file:
                file.write(G)
            return True

        self.log['Client'].debug("Procedure ffbo.ui.get_circuit Registered...")

        @FBLClient.register(
            "ffbo.ui.get_experiment" + str(FBLClient._async_session._session_id)
        )
        def get_experiment(X):
            """Obtain an experiment and save it to the local FBL folder.

            # Arguments
                X (str): Name of the experiment.

            # Returns
                bool: Whether the process has been successful.
            """
            self.log['GFX'].debug("get_experiment called.")
            name = X["name"]
            data = json.dumps(X["experiment"])
            with open(os.path.join(_FBLDataPath, name + ".json"), "w") as file:
                file.write(data)
            output = {}
            output["success"] = True
            self.log['GFX'].info("Experiment save successful.")
            return True

        self.log['Client'].debug("Procedure ffbo.ui.get_experiment Registered...")

        @FBLClient.register(
            "ffbo.ui.receive_data." + str(FBLClient._async_session._session_id)
        )
        def receiveData(data):
            """The Receive Data function that receives commands and sends them to the NLP frontend.

            # Arguments
                data (dict): Data from the backend.

            # Returns
                bool: Whether the process has been successful.
            """
            if self.log_level in ['debug']:
                self.clientData.append("Received Data")
            a = {}
            data = convert_from_bytes(data)
            if isinstance(data['data'],dict):
                if self.legacy == True:
                    a["data"] = {"data": data, "queryID": guidGenerator()}
                else:
                    a["data"] = data
                try:
                    if 'data' in data:
                        if isinstance(data['data'],dict):
                            for i in data['data'].keys():
                                if 'MorphologyData' in data['data'][i]:
                                    data['data'][i].update(data['data'][i]['MorphologyData'])
                except Exception as e:
                    self.raise_error(e, 'A potential error was detected during data parsing.')
                    print(e)
                if isinstance(data['data'],dict):
                    self.neuron_data.update(data['data'])
                    self.active_data = list(set(self.active_data + list(data['data'].keys())))
                else:
                    print(data)
                a["messageType"] = "Data"
                a["widget"] = "NLP"
                if self.addToRemove == True:
                    if "data" in data:
                        keys = list(data["data"].keys())
                        data["commands"] = {"remove": [keys, []]}
                        del data["data"]
                        a["data"] = data
                        a["messageType"] = "Command"
                # Change scales
                try:
                    if a["messageType"] == "Data":
                        if "data" in a["data"]:
                            for i in a["data"]["data"].keys():
                                if "name" in a["data"]["data"][i]:
                                    a["data"]["data"][i]["x"] = [
                                        i * self.x_scale + self.x_shift
                                        for i in a["data"]["data"][i]["x"]
                                    ]
                                    a["data"]["data"][i]["y"] = [
                                        i * self.y_scale + self.y_shift
                                        for i in a["data"]["data"][i]["y"]
                                    ]
                                    a["data"]["data"][i]["z"] = [
                                        i * self.z_scale + self.z_shift
                                        for i in a["data"]["data"][i]["z"]
                                    ]
                                    a["data"]["data"][i]["r"] = [
                                        i * self.r_scale + self.r_shift
                                        for i in a["data"]["data"][i]["r"]
                                    ]
                except Exception as e:
                    self.raise_error(e, 'There was an error when scaling data.')
                    self.errors.append(e)
                self.data.append(a)
                displayDict = {
                    "totalLength": "Total Length (µm)",
                    "totalSurfaceArea": "Total Surface Area (µm<sup>2</sup>)",
                    "totalVolume": "Total Volume (µm<sup>3</sup>)",
                    "maximumEuclideanDistance": "Maximum Euclidean Distance (µm)",
                    "width": "Width (µm)",
                    "height": "Height (µm)",
                    "depth": "Depth (µm)",
                    "maxPathDistance": "Max Path Distance (µm)",
                    "averageDiameter": "Average Diameter (µm)",
                }
                if a["messageType"] == "Data":
                    if "data" in a["data"]:
                        try:
                            for i in a["data"]["data"].keys():
                                if "name" in a["data"]["data"][i]:
                                    self.uname_to_rid[a["data"]["data"][i]["uname"]] = i
                                    self.neuronStats[a["data"]["data"][i]["uname"]] = {}
                                    for displayKey in displayDict.keys():
                                        try:
                                            self.neuronStats[a["data"]["data"][i]["uname"]][
                                                displayKey
                                            ] = a["data"]["data"][i][displayKey]
                                        except:
                                            pass
                        except:
                            print(a["data"]["data"])
                if self.log_level>1:
                    print(printHeader("FBL Client NLP") + "Received data.")
                self.tryComms(a)
                return True

        self.log['Client'].debug("Procedure ffbo.ui.receive_data Registered...")

        @FBLClient.register(
            "ffbo.ui.receive_partial." + str(FBLClient._async_session._session_id)
        )
        def receivePartial(data):
            """The Receive Partial Data function that receives commands and sends them to the NLP frontend.

            # Arguments
                data (dict): Data from the backend.

            # Returns
                bool: Whether the process has been successful.
            """
            if self.log_level in ['debug']:
                self.clientData.append("Received Data")
            a = {}
            data = convert_from_bytes(data)
            a["data"] = {"data": data, "queryID": guidGenerator()}
            a["messageType"] = "Data"
            a["widget"] = "NLP"
            self.data.append(a)
            self.log['NLP'].debug("Received partial data.")
            self.tryComms(a)
            return True

        self.log['Client'].debug("Procedure ffbo.ui.receive_partial Registered...")

        if legacy == False:
            # @FBLClient.register('ffbo.gfx.receive_partial.' + str(FBLClient._async_session._session_id))
            # def receivePartialGFX(data):
            #     """The Receive Partial Data function that receives commands and sends them to the NLP frontend.
            #
            #     # Arguments
            #         data (dict): Data from the backend.
            #
            #     # Returns
            #         bool: Whether the process has been successful.
            #     """
            #     self.clientData.append('Received Data')
            #     a = {}
            #     a['data'] = {'data': data, 'queryID': guidGenerator()}
            #     a['messageType'] = 'Data'
            #     a['widget'] = 'NLP'
            #     self.data.append(a)
            #     print(printHeader('FBL Client NLP') + "Received partial data.")
            #     self.tryComms(a)
            #     return True
            # print(printHeader('FBL Client') + "Procedure ffbo.gfx.receive_partial Registered...")

            @FBLClient.register('ffbo.gfx.receive_partial.' + str(FBLClient._async_session._session_id))
            def receivePartialGFX(data):
                """The Receive Partial Data function that receives commands and sends them to the NLP frontend.

                # Arguments
                    data (dict): Data from the backend.

                # Returns
                    bool: Whether the process has been successful.
                """
                if self.log_level in ['debug']:
                    self.clientData.append('Received Data')
                if self.current_exec_result is None:
                    try:
                        temp = msgpack.unpackb(data)
                    except (msgpack.ExtraData, msgpack.UnpackValueError):
                        pass
                    else:
                        self.current_exec_result = temp['execution_result_start']
                        self.exec_result[self.current_exec_result] = []
                        self.log['GFX'].info("Receiving Execution Result for {}.  Please wait .....".format(self.current_exec_result))
                else:
                    try:
                        temp = msgpack.unpackb(data)
                    except (msgpack.ExtraData, msgpack.UnpackValueError):
                        self.exec_result[self.current_exec_result].append(data)
                    else:
                        if isinstance(temp, dict) and 'execution_result_end' in temp:
                            assert temp['execution_result_end'] == self.current_exec_result

                            result = msgpack.unpackb(b''.join(self.exec_result[self.current_exec_result]))
                            result_name = self.current_exec_result
                            self.current_exec_result = None

                            if 'error' in result:
                                print(result['error']['message'], file = sys.stderr)
                                raise ValueError(result['error']['exception'])
                            meta = result['success'].pop('meta')
                            temp = result['success']['result']

                            formatted_result = {'sensory': {}, 'input': {}, 'output': {}, 'meta': meta}
                            for data_type, data in temp.items():
                                for key, value in data.items():
                                    k = eval(key).decode('utf-8') if key[0]=='b' else key
                                    if data_type == 'sensory':
                                        formatted_result[data_type][k] = [{'dt': val['dt'],
                                                                 'data': val['data']}#np.array(val['data'])}\
                                                                for val in value]
                                    else:
                                        formatted_result[data_type][k] = {kk: {'data': v['data'],# np.array(v['data']),
                                                                       'dt': v['dt']} \
                                                                   for kk, v in value.items()}
                            self.exec_result[result_name] = formatted_result
                            self.log['GFX'].info( "Received Execution Result for {}. Result stored in Client.exec_result['{}']".format(result_name, result_name))
                            # self.tryComms(a)
                        else:
                            self.exec_result[self.current_exec_result].append(data)
                return True

            self.log['Client'].debug("Procedure ffbo.gfx.receive_partial Registered...")

        @FBLClient.register(
            "ffbo.ui.receive_msg." + str(FBLClient._async_session._session_id)
        )
        def receiveMessage(data):
            """The Receive Message function that receives commands and sends them to the NLP frontend.

            # Arguments
                data (dict): Data from the backend.

            # Returns
                bool: Whether the process has been successful.
            """
            self.clientData.append("Received Message")
            data = convert_from_bytes(data)
            a = {}
            a["data"] = data
            a["messageType"] = "Message"
            a["widget"] = "NLP"
            #self.data.append(a)

            if self.log_level in ['debug']:
                message_type = list(data.keys())[0]
                status = data[message_type]
                if 'success' in status:
                    message = status['success']
                    print(printHeader("FBL Client") + "Received a {message_type} message: {message}".format(message_type = message_type, message = message))
                elif 'error' in status:
                    message = status['error']
                    print(printHeader("FBL Client") + "Received an Error message: {message}".format(message = message))
                else:
                    message = list(status.values())[0]
                    print(printHeader("FBL Client") + "Received a {message_type} message: {message}".format(message_type = message_type, message = message))
            self.tryComms(a)
            return True

        self.log['Client'].debug("Procedure ffbo.ui.receive_msg Registered...")

        self.client = FBLClient  # Set current client to the FBLClient Client

    def findServerIDs(self, dataset = None):
        """Find server IDs to be used for the utility functions.

        # Arguments
            dataset (str): Name of the dataset to connect to. Optional.
        """
        res = self.rpc(u"ffbo.processor.server_information")
        res = convert_from_bytes(res)

        if not res["processor"]["autobahn"].split('.')[0] == autobahn.__version__.split('.')[0]:
            self.raise_error(Exception(), "Autobahn major version mismatch between your environment {} and the backend servers {}.\nPlease update your autobahn version to match with the processor version by running ``pip install --upgrade autobahn`` in your terminal.".format(autobahn.__version__, res["processor"]["autobahn"]))

        default_mode = False

        server_dict = {}
        for server_id, server_config in res["na"].items():
            if 'dataset' not in server_config:
                server_config['dataset'] = 'default'
                default_mode = True
            if server_config['dataset'] not in server_dict:
                server_dict[server_config['dataset']] = {'na': [], 'nlp': []}
            server_dict[server_config['dataset']]['na'].append(server_id)
        for server_id, server_config in res["nlp"].items():
            if 'dataset' not in server_config:
                server_config['dataset'] = 'default'
                default_mode = True
            if server_config['dataset'] not in server_dict:
                server_dict[server_config['dataset']] = {'na': [], 'nlp': []}
            server_dict[server_config['dataset']]['nlp'].append(server_id)
        valid_datasets = []
        for dataset_name, server_lists in server_dict.items():
            if len(server_lists['na']) and len(server_lists['nlp']):
                valid_datasets.append(dataset_name)

        if dataset is None:
            if default_mode:
                if len(valid_datasets):
                    pass
                else:
                    raise RuntimeError("No valid datasets cannot be found.\nIf you are running the NeuroArch and NeuroNLP servers locally, please check if the servers are on and connected. If you are connecting to a public server, please contact server admin.")
            else:
                if len(valid_datasets) == 1:
                    dataset = valid_dataset[0]
                elif len(valid_datasets) > 1:
                    raise RuntimeError("Multiple valid datasets are available on the specified FFBO processor. However, you did not specify which dataset to connect to. Available datasets on the FFBO processor are the following:\n{}\n\n. Please choose one of the above datasets during Client connection by passing the dataset argument.".format('\n- '.join(valid_datasets)))
                # print(
                #     printHeader("FBL Client")
                #     + "Found following datasets: "
                #     + ', '.join(valid_datasets)
                # )
                # print(
                #     printHeader("FBL Client")
                #     + "Please choose a dataset from the above valid datasets by"
                #     + " Client.findServerIDs(dataset = 'any name above')"
                # )
                else: #len(valid_datasets) == 0
                    raise RuntimeError("No valid datasets cannot be found.\nIf you are running the NeuroArch and NeuroNLP servers locally, please check if the servers are on and connected. If you are connecting to a public server, please contact server admin.")
                    # print(
                    #     printHeader("FBL Client")
                    #     + "No valid datasets found."
                    # )

        server_dict = {'na': [], 'nlp': []}
        for server_id, server_config in res["na"].items():
            if 'dataset' in server_config:
                if server_config['dataset'] == dataset:
                    server_dict['na'].append(server_id)
            else:
                server_dict['na'].append(server_id)
                break
        for server_id, server_config in res["nlp"].items():
            if 'dataset' in server_config:
                if server_config['dataset'] == dataset:
                    server_dict['nlp'].append(server_id)
            else:
                server_dict['nlp'].append(server_id)
                break
        if len(server_dict['na']):
            if self.naServerID is None:
                self.log['Client'].debug("Found working NeuroArch Server for dataset {}: ".format(dataset)
                    + res["na"][server_dict['na'][0]]['name'])
                self.naServerID = server_dict['na'][0]
            else:
                if self.naServerID not in server_dict['na']:
                    self.log['Client'].warning(
                        "Previous NeuroArch Server not found, switching NeuroArch Servre to: "
                        + res["na"][server_dict['na'][0]]['name']
                        + " Prior query states may not be accessible."
                    )
                    self.naServerID = server_dict['na'][0]
        else:
            raise RuntimeError("NeuroArch Server with {} dataset cannot be found. Available dataset on the FFBO processor is the following:\n{}\n\nIf you are running the NeuroArch server locally, please check if the server is on and connected. If you are connecting to a public server, please contact server admin.".format(dataset, '\n- '.join(valid_datasets)))
            # print(
            #     printHeader("FBL Client")
            #     + "NA Server with {} dataset not found".format(dataset)
            # )
        if len(server_dict['nlp']):
            self.log['Client'].debug(
                "Found working NeuroNLP Server for dataset {}: ".format(dataset)
                + res["nlp"][server_dict['nlp'][0]]['name']
            )
            self.nlpServerID = server_dict['nlp'][0]
        else:
            raise RuntimeError("NeuroNLP Server with {} dataset cannot be found. Available dataset on the FFBO processor is the following:\n{}\n\nIf you are running the NeuroNLP server locally, please check if the server is on and connected. If you are connecting to a public server, please contact server admin.".format(dataset, '\n- '.join(valid_datasets)))
            # print(
            #     printHeader("FBL Client")
            #     + "NLP Server with {} dataset not found".format(dataset)
            # )

        if len(res["nk"]) == 0:
            self.log['Client'].warning("Neurokernel Server not found on the FFBO processor. Circuit execution on the server side is not supported.")
            Warning("Neurokernel Server not found on the FFBO processor. Circuit execution on the server side is not supported.")

    @property
    def rpc(self):
        return self.client.session.call

    def get_client_info(self, fbl=None):
        """Receive client data for this client only.

        # Arguments
            fbl (Object): MetaClient object. Optional. Gives us.

        # Returns
            dict: dict of dicts with client name as key and widgets, name and species as keys of the value.
        """
        if fbl is None:
            res = {}
            client_data = {}
            client_data['widgets'] = self.widgets
            client_data['name'] = self.name
            client_data['species'] = self.species
            res[self.name] = client_data
            return res
        else:
            fbl.client_manager.update_client_names()
            res = {}
            for i in fbl.client_manager.clients:
                client = fbl.client_manager.clients[i]
                client_data = {}
                client_data['widgets'] = client['widgets']
                client_data['name'] = i
                client_data['species'] = client['client'].species
                res[client['client'].name] = client_data
            return res

    def executeNLPquery(
        self, query=None, language="en", uri=None, queryID=None, returnNAOutput=False
    ):
        """Execute an NLP query.

        # Arguments
            query (str): Query string.
            language (str): Language to use.
            uri (str): Currently not used; for future NLP extensions.
            queryID (str): Query ID to be used. Generated automatically.
            returnNAOutput (bool): Whether the corresponding NA query should not be executed.

        # Returns
            dict: NA output or the NA query itself, depending on the returnNAOutput setting.
        """
        if self.connected == False:
            self.reconnect()
        if self.connected == False:
            return False
        if query is None:
            self.log['Client'].warning('No query specified. Executing test query "eb".')
            query = "eb"
        self.JSCall(messageType="GFXquery", data=query)
        if query.startswith("load "):
            self.sendSVG(query[5:])
        else:
            # if self.legacy == False:
            uri = "ffbo.nlp.query.{}".format(self.nlpServerID)
            queryID = guidGenerator()
            try:
                resNA = self.rpc(uri, query, language)
                # Send the parsed query to the fronedned to be displayed if need be
                if resNA == {}:
                    self.raise_error('Interpretation Error','The query could not be interpreted. Look at the server to check for potential errors.')
                a = {}
                a["data"] = resNA
                a["messageType"] = "ParsedQuery"
                a["widget"] = "NLP"
            except:
                a = {}
                a["data"] = {"info": {"timeout": "This is a timeout."}}
                a["messageType"] = "Data"
                a["widget"] = "NLP"
                self.tryComms(a)
                return a
            self.log['NLP'].info("NLP successfully parsed query.")

            if returnNAOutput == True:
                return resNA
            else:
                try:
                    self.compiled = False
                    res = self.executeNAquery(
                        resNA, queryID=queryID, threshold=self.query_threshold
                    )
                    if 'show ' in query or 'add ' in query or 'remove ' in query:
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
                    a["data"] = {"info": {"timeout": "This is a timeout."}}
                    a["messageType"] = "Data"
                    a["widget"] = "NLP"
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
                resNA = self.rpc('ffbo.processor.nlp_to_visualise', msg, options=CallOptions(
                                                    on_progress=partial(on_progress, res=res_list), timeout = 20))
                if returnNAOutput == True:
                    return resNA
                else:
                    self.compiled = False
                    # res = self.executeNAquery(resNA, queryID = queryID)
                    self.sendNeuropils()
                    return resNA
            """

    def executeNAquery(
        self, res, language="en", uri=None, queryID=None, progressive=True, threshold=5
    ):
        """Execute an NA query.

        # Arguments
            res (dict): Neuroarch query.
            language (str): Language to use.
            uri (str): A custom FFBO query URI if desired.
            queryID (str): Query ID to be used. Generated automatically.
            progressive (bool): Whether the loading should be progressive. Needs to be true most of the time for the connection to be stable.
            threshold (int): Data chunk size. Low threshold is required for the connection to be stable.

        # Returns
            bool: Whether the process has been successful.
        """

        # def on_progress(x, res):
        #     x = convert_from_bytes(x)
        #     res.append(x)

        if isinstance(res, str):
            res = json.loads(res)
        if uri == None:
            uri = "ffbo.na.query.{}".format(self.naServerID)
            if "uri" in res.keys():
                uri = "{}.{}".format(res["uri"], self.naServerID)
        if queryID == None:
            queryID = guidGenerator()
        # del self.data # Reset the data in the backend
        # self.data = []

        """
        if 'verb' in res:
            if res['verb'] == 'remove':
                self.data.append(res)
                if 'query' in res:
                    try:
                        rids = res['query'][0]['action']['method']['query']['rid']
                        for i in rids:
                            if i in self.active_data:
                                self.active_data.remove(i)
                    except Exception as e:
                        pass
                    try:
                        unames = res['query'][0]['action']['method']['query']['uname']
                        for i in unames:
                            if i in self.uname_to_rid:
                                rid = self.uname_to_rid[i]
                                if rid in self.active_data:
                                    self.active_data.remove(rid)
                    except Exception as e:
                        pass
        """

        res["queryID"] = queryID
        res["threshold"] = threshold
        res["data_callback_uri"] = "ffbo.ui.receive_data"
        res_list = []
        if self.legacy == False and progressive == True:
            try:
                res = self.client.session.call(
                    uri,
                    res,
                    options=CallOptions(
                        on_progress=partial(on_progress, res=res_list), timeout=300000
                    ),
                )
            except Exception as e:
                #TODO: this reconnect also affects runtime error in the backend, should only reconnect
                #      when seeing an error from autobahn, like autobahn.wamp.exception.TransportLost
                self.raise_error(e,'A connection error occured during a progressive NLP call. Check client.errors for more details. Attempting to reconnect.')
                print(e)
                try:
                    self.reconnect()
                    self.raise_message('Successfully reconnected to server. Note that previous workspace state will be lost in the backend (for add or remove queries).')
                except Exception as e:
                    self.raise_error(e, 'There was an error during the reconnection attempt. Check the server.')
                    print(e)
        else:
            try:
                res = self.client.session.call(uri, res)
            except Exception as e:
                self.raise_error(e,'A connection error occured during a NeuroArch call. Check client.errors for more details.')
                print(e)
                try:
                    self.reconnect()
                    self.raise_message('Successfully reconnected to server. Note that previous workspace state will be lost in the backend (for add or remove queries).')
                except Exception as e:
                    self.raise_error(e, 'There was an error during the reconnection attempt. Check the server.')
                    print(e)
        res = convert_from_bytes(res)

        try:
            if 'data' in res:
                self.neuron_data.update(res['data'])
                for i in res['data'].keys():
                    if 'MorphologyData' in res['data'][i]:
                        res['data'][i].update(res['data'][i]['MorphologyData'])
        except Exception as e:
            self.raise_error(e, 'A potential error was detected during data parsing.')
            print(e)

        a = {}
        a["data"] = res
        a["messageType"] = "Data"
        a["widget"] = "NLP"

        if progressive == True:
            self.tryComms(a)
            #self.data.append(a)
            return self.data
        else:
            self.tryComms(a)
            return a

    def createTag(self, tagName):
        """Creates a tag.

        # Returns
            bool: True.
        """
        metadata = {
            "color": {},
            "pinned": {},
            "visibility": {},
            "camera": {"position": {}, "up": {}},
            "target": {},
        }
        # res = self.executeNAquery(
        #     {"tag": tagName, "metadata": metadata, "uri": "ffbo.na.create_tag"}
        # )
        task = {"tag": tagName, "metadata": metadata}
        res = self.rpc('ffbo.na.create_tag.{}'.format(self.naServerID), task)
        if 'success' in res['info']:
            self.log['Client'].info(res['info']['success'])
        elif 'error' in res['info']:
            raise FlyBrainLabNAserverException(res['info']['error'])
        return res

    def loadTag(self, tagName):
        """Loads a tag.

        # Returns
            bool: True.
        """
        # self.executeNAquery({"tag": tagName, "uri": "ffbo.na.retrieve_tag"})
        task = {"tag": tagName}
        res = self.rpc('ffbo.na.retrieve_tag.{}'.format(self.naServerID), task)
        if 'success' in res['info']:
            self.log['Client'].info(res['info']['success'])
        elif 'error' in res['info']:
            raise FlyBrainLabNAserverException(res['info']['error'])
        a = {}
        a["data"] = res
        a["messageType"] = "TagData"
        a["widget"] = "NLP"

        self.tryComms(a)
        res = self.executeNAquery({"command": {"retrieve": {"state": 0}},
                                   "loadtag": tagName})
        return res

    def addByUname(self, uname, verb="add"):
        """Adds some neurons by the uname.

        # Returns
            bool: True.
        """
        self.history.append([uname,verb])
        default_run = True
        if len(self.NLPInterpreters) > 0:
            default_run = False
            for i in self.NLPInterpreters:
                default_run = i(self, uname, verb)
        if default_run == True or len(self.NLPInterpreters) == 0:
            self.executeNAquery(
                {
                    "verb": verb,
                    "query": [
                        {
                            "action": {"method": {"query": {"uname": uname}}},
                            "object": {"class": ["Neuron", "Synapse"]},
                        }
                    ],
                }
            )
            return True
        return True

    def removeByUname(self, uname):
        """Removes some neurons by the uname.

        # Returns
            bool: True.
        """
        return self.addByUname(uname, verb="remove")

    def runLayouting(self, type="auto", model="auto"):
        """Sends a request for the running of the layouting algorithm.

        # Returns
            bool: True.
        """
        self.prepareCircuit(model=model)
        self.sendCircuit(name="auto")
        a = {}
        a["data"] = "auto"
        a["messageType"] = "runLayouting"
        a["widget"] = "GFX"
        self.tryComms(a)
        return True

    def raise_message(self, message):
        """Raises an message in the frontend.

        # Arguments
            message (str): String message to raise.
        """
        self.tryComms({'data': {'info': {'success': message}},
            'messageType': 'Message',
            'widget': 'NLP'})

    def raise_error(self, e, error):
        """Raises an error in the frontend.

        # Arguments
            e (str): The error string to add to self.errors.
            error (str): String error to raise.
        """
        self.errors.append(e)
        self.tryComms({'data': {'info': {'error': error}},
            'messageType': 'Message',
            'widget': 'NLP'})

    def getStats(self, neuron_name):
        """Print various statistics for a given neuron.

        # Arguments
            neuron_name (str): Name of the neuron to print the data for. The neuron must have been queried beforehand.
        """
        displayDict = {
            "totalLength": "Total Length (µm)",
            "totalSurfaceArea": "Total Surface Area (µm^2)",
            "totalVolume": "Total Volume (µm^3)",
            "maximumEuclideanDistance": "Maximum Euclidean Distance (µm)",
            "width": "Width (µm)",
            "height": "Height (µm)",
            "depth": "Depth (µm)",
            "maxPathDistance": "Max Path Distance (µm)",
            "averageDiameter": "Average Diameter (µm)",
        }
        if neuron_name in self.neuronStats.keys():
            print("Statistics for " + neuron_name + ":")
            print("-----------")
            for i in displayDict.keys():
                print(displayDict[i] + ":", self.neuronStats[neuron_name][i])
        else:
            print("No statistics found for " + str(neuron_name) + ".")
        return None

    def getNeuropils(self):
        """Get the neuropils the neurons in the workspace reside in.

        # Returns
            list of strings: Set of neuropils corresponding to neurons.
        """
        res = {}
        res["query"] = []
        res["format"] = "nx"
        res["user"] = "test"
        res["temp"] = True
        res["query"].append(
            {
                "action": {"method": {"traverse_owned_by": {"cls": "Neuropil"}}},
                "object": {"state": 0},
            }
        )
        res = self.executeNAquery(res)
        neuropils = []
        for i in res:
            try:
                if "data" in i.keys():
                    if "data" in i["data"].keys():
                        if "nodes" in i["data"]["data"].keys():
                            a = i["data"]["data"]["nodes"]
                            for j in a.keys():
                                name = a[j]["name"]
                                neuropils.append(name)
            except:
                pass
        neuropils = list(set(neuropils))
        return neuropils

    def sendNeuropils(self):
        """Pack the list of neuropils into a GFX message.

        # Returns
            bool: Whether the messaging has been successful.
        """
        a = {}
        a["data"] = self.getNeuropils()
        self.log['Client'].debug('Available Neuropils: {}'.format(a["data"]))
        a["messageType"] = "updateActiveNeuropils"
        a["widget"] = "GFX"
        self.tryComms(a)
        return True

    def loadSWC(self, file_name, scale_factor=1., uname=None):
        """Loads a neuron skeleton stored in the .swc format.

        # Arguments
            file_name (str): Database ID of the neuron or node.
            scale_factor (float): A scale factor to scale the neuron's dimensions with. Defaults to 1.
            uname (str): Unique name to use in the frontend. Defaults to the file_name.

        """
        neuron_pd = pd.read_csv(file_name,
                        names=['sample','identifier','x','y','z','r','parent'],
                        comment='#',
                        delim_whitespace=True)
        if uname == None:
            uname = file_name.split('.')[0]
        rid = '#'+file_name
        neuron_data = {'data': {'data': {rid: {'name': file_name,
                'uname': uname,
                'morph_type': 'swc',
                'x': list(scale_factor * neuron_pd['x']),
                'y': list(scale_factor * neuron_pd['y']),
                'z': list(scale_factor * neuron_pd['z']),
                'r': list(scale_factor * neuron_pd['r']),
                'parent': list(neuron_pd['parent']),
                'identifier': list(neuron_pd['identifier']),
                'sample': list(neuron_pd['sample']),
                'class': 'MorphologyData'}},
              'queryID': '0-0'},
             'messageType': 'Data',
             'widget': 'NLP'}
        self.tryComms(neuron_data)

        return True

    def getInfo(self, dbid):
        """Get information on a neuron.

        # Arguments
            dbid (str): Database ID of the neuron or node.

        # Returns
            dict: NA information regarding the node.
        """
        task = {"id": dbid, 'queryID': guidGenerator()}
        res = self.rpc('ffbo.na.get_data.{}'.format(self.naServerID), task)
        a = {}
        a["data"] = {"data": res, "messageType": "Data", "widget": "NLP"} # the extra message type seems to be needed to update info panel, why?
        a["messageType"] = "Data"
        a["widget"] = "INFO"
        self.tryComms(a)
        # print(res)

        if self.compiled == True:
            try:
                a = {}
                name = res["data"]["summary"]["uname"]
                if name in self.node_keys.keys():
                    data = self.C.G.node["uid" + str(self.node_keys[name])]
                    data["uid"] = str(self.node_keys[name])
                    a["data"] = data
                    a["messageType"] = "Data"
                    a["widget"] = "JSONEditor"
                    self.tryComms(a)
            except:
                pass
        return res

    def GFXcall(self, args):
        """Arbitrary call to a GFX procedure in the GFX component format.

        # Arguments
            args (list): A list whose first element is the function name (str) and the following are the arguments.

        # Returns
            dict OR string: The call result.
        """
        if isinstance(args, str):
            res = self.rpc(args)
        else:
            res = self.rpc(args[0], args[1:])
        if type(res) == dict:
            a = res
            a["widget"] = "GFX"
        else:
            a = {}
            a["data"] = res
            a["messageType"] = "Data"
            a["widget"] = "GFX"
        self.tryComms(a)
        return res

    def getSimData(self, url):
        url = "https://data.flybrainlab.fruitflybrain.org/simresults/" + url
        urlRetriever(url, url.split("/")[-1])
        filename = url.split("/")[-1]
        f = h5py.File(filename, "r")
        data = f["V"]["data"][()].tolist()
        uids = f["V"]["uids"][()].tolist()
        uids = [i.decode("ascii") for i in uids]
        h5data = {}
        h5data["data"] = data
        h5data["uids"] = uids
        a = {}
        a["data"] = h5data
        a["messageType"] = "PlotResults"
        a["widget"] = "Master"
        self.data.append(a)
        self.log['Master'].info("Sending simulation data.")
        self.tryComms(a)
        json_str = json.dumps(h5data)
        with open(filename.split(".")[0] + ".json", "w") as f:
            f.write(json_str)
        self.simData = h5data

    def updateBackend(self, type="Null", data="Null"):
        """Updates variables in the backend using the data in the Editor.

        # Arguments
            type (str): A string, either "WholeCircuit" or "SingleNeuron", specifying the type of the update.
            data (str): A stringified JSON

        # Returns
            bool: Whether the update was successful.
        """

        data = json.loads(data)
        print(type)
        print(data)
        return True

    def getConnectivity(self):
        """Obtain the connectivity matrix of the current circuit in NetworkX format.

        # Returns
            dict: The connectivity dictionary.
        """
        res = json.loads(
            """
        {"format":"nx","query":[{"action":{"method":{"add_connecting_synapses":{}}},"object":{"state":0}}],"temp":true}
        """
        )
        res = self.executeNAquery(res)
        return res

    def sendExecuteReceiveResults(
        self, circuitName="temp", dt=1e-5, tmax=1.0, inputProcessors=[], compile=False
    ):
        """Compiles and sends a circuit for execution in the GFX backend.

        # Arguments
            circuitName (str): The name of the circuit for the backend.
            compile (bool): Whether to compile the circuit first.

        # Returns
            bool: Whether the call was successful.
        """
        self.log['GFX'].info("Initiating remote execution for the current circuit.")
        if self.compiled == False:
            compile = True
        if compile == True:
            self.log['GFX'].info("Compiling the current circuit.")
            self.prepareCircuit()
        self.log['GFX'].info("Circuit prepared. Sending to FFBO servers.")
        self.sendCircuitPrimitive(self.C, args={"name": circuitName})
        self.log['GFX'].info("Circuit sent. Queuing execution.")
        if len(inputProcessors) > 0:
            res = self.rpc(
                "ffbo.gfx.startExecution",
                {
                    "name": circuitName,
                    "dt": dt,
                    "tmax": tmax,
                    "inputProcessors": inputProcessors,
                },
            )
        else:
            res = self.rpc(
                "ffbo.gfx.startExecution", {"name": circuitName, "dt": dt, "tmax": tmax}
            )
        return True

    def getConnectivityMatrix(self):
        M = np.zeros((len(self.out_nodes), len(self.out_nodes)))
        for i in self.out_edges:
            M[self.out_nodes.index(i[0]), self.out_nodes.index(i[1])] += 1
        return M

    def prepareCircuit(self, model="auto"):
        """Prepares the current circuit for the Neuroballad format.
        """
        res = self.getConnectivity()

        for data in self.data:
            if data["messageType"] == "Data":
                if "data" in data:
                    if "data" in data["data"]:
                        connectivity = data["data"]["data"]
                        break

        connectivity = res[-2]["data"]["data"]
        # print(connectivity)
        out_nodes, out_edges, out_edges_unique = self.processConnectivity(connectivity)
        self.out_nodes = out_nodes
        self.out_edges = out_edges
        self.out_edges_unique = out_edges_unique
        C, node_keys = self.genNB(self.out_nodes, self.out_edges, model=model)
        self.C = C
        self.node_keys = node_keys
        self.compiled = True

    def getSlowConnectivity(self):
        """Obtain the connectivity matrix of the current circuit in a custom dictionary format. Necessary for large circuits.

        # Returns
            dict: The connectivity dictionary.
        """
        hashids = []
        names = []
        synapses = []

        for data in self.data:
            if data["messageType"] == "Data":
                if "data" in data:
                    if "data" in data["data"]:
                        keys = list(data["data"]["data"].keys())
                        for key in keys:
                            if isinstance(data["data"]["data"][key], dict):
                                if "uname" in data["data"]["data"][key].keys():
                                    hashids.append(key)
                                    names.append(data["data"]["data"][key]["uname"])

        for i in range(len(hashids)):
            res = self.getInfo(hashids[i])
            if "connectivity" in res["data"].keys():
                presyn = re["data"]["connectivity"]["pre"]["details"]

                for syn in presyn:
                    synapses.append([syn["uname"], names[i], syn["number"]])

                postsyn = res["data"]["connectivity"]["post"]["details"]
                for syn in postsyn:
                    synapses.append([names[i], syn["uname"], syn["number"]])
                clear_output()
        connectivity = {"hashids": hashids, "names": names, "synapses": synapses}
        return connectivity

    def sendCircuit(self, name="temp"):
        """Sends a circuit to the backend.

        # Arguments
            name (str): The name of the circuit for the backend.
        """
        self.sendCircuitPrimitive(self.C, args={"name": name})

    def autoLayout(self):
        """Layout raw data from NeuroArch and save results as G_auto.*.
        """
        import json
        res = json.loads(
                    """
                {"format":"nx","query":[{"action":{"method":{"add_connecting_synapses":{}}},"object":{"state":0}}],"temp":true}
                """
                )
        res = self.executeNAquery(res)
        nodes = res[-2]["data"]["data"]['nodes']
        edges = res[-2]["data"]["data"]['edges']
        import networkx as nx
        G = nx.DiGraph()
        for e_pre in nodes:
            G.add_node(e_pre, **{'uname': nodes[e_pre]['uname']})
        for edge in edges:
            G.add_edge(edge[0],edge[1])
        from graphviz import Digraph
        graph_struct = {'splines': 'ortho',
                        'pad': '0.5',
                        'ranksep': '1.5',
                        'concentrate': 'true',
                        'newrank': 'true',
                        'rankdir': 'LR'}

        g = Digraph('G', filename='G_ex.gv',graph_attr = {'splines': 'line',
                                                        'pad': '0.1',
                                                        'nodesep': '0.03',
                                                        'outputorder': 'edgesfirst',
                                                        'ranksep': '0.2',
                                                        'bgcolor': '#212529',
                                                        'concentrate': 'true',
                                                        'newrank': 'true',
                                                        'rankdir': 'LR'})
        # g.attr(bgcolor='black')
        valid_nodes = []
        for pre, post, data in G.edges(data=True):
            pre_name = G.nodes(data=True)[pre]['uname']
            post_name = G.nodes(data=True)[post]['uname']
            g.edge(str(pre_name), str(post_name), arrowsize='0.2', color='dimgray')
            valid_nodes.append(str(pre_name))
            valid_nodes.append(str(post_name))
            valid_nodes = list(set(valid_nodes))

        for _pre in valid_nodes:
            if '--' in str(_pre):
                g.node(str(_pre),
                    shape='circle',
                    height='0.05',
                    label='',
                    fontsize='4.0',
                    fixedsize='true',
                    color='dimgray',
                    style='filled')
            else:
                g.node(str(_pre),
                    shape='circle',
                    height='0.20',
                    fontsize='3.0',
                    fixedsize='true',
                    fontcolor='white',
                    color='dodgerblue3',
                    style='filled')

        g.attr(size='25,25')
        try:
            g.save('G_auto.gv')
            g.render('G_auto', format = 'svg', view=False)
            g.render('G_auto', format = 'png', view=False)
        except Exception as e:
            self.raise_error(e, 'There was an error during diagram generation. Please execute "conda install -c anaconda graphviz" in your terminal in your conda environment, or try to install GraphViz globally from https://graphviz.org/download/.')
            print(e)
    def processConnectivity(self, connectivity):
        """Processes a Neuroarch connectivity dictionary.

        # Returns
            tuple: A tuple of nodes, edges and unique edges.
        """
        edges = connectivity["edges"]
        nodes = connectivity["nodes"]

        out_edges = []
        out_nodes = []
        csv = ""
        for e_pre in nodes:
            if "class" in nodes[e_pre]:
                if "uname" not in nodes[e_pre].keys():
                    nodes[e_pre]["uname"] = nodes[e_pre]["name"]
        for e_pre in nodes:
            # e_pre = node
            pre = None
            if "class" in nodes[e_pre]:
                if nodes[e_pre]["class"] == "Neuron":
                    if "uname" in nodes[e_pre].keys():
                        pre = nodes[e_pre]["uname"]
                    else:
                        pre = nodes[e_pre]["name"]
                        nodes[e_pre]["uname"] = nodes[e_pre]["name"]
                # print(pre)
                if pre is not None:
                    synapse_nodes = [
                        i[1]
                        for i in edges
                        if (nodes[i[0]]["name"] == pre
                        and (
                            nodes[i[1]]["class"] == "Synapse"
                            or nodes[i[1]]["class"] == "InferredSynapse"
                        )) or (nodes[i[0]]["uname"] == pre
                        and (
                            nodes[i[1]]["class"] == "Synapse"
                            or nodes[i[1]]["class"] == "InferredSynapse"
                        ))
                    ]
                    # print(len(synapse_nodes))
                    for synapse in synapse_nodes:
                        if "class" in nodes[synapse]:
                            if nodes[synapse]["class"] == "Synapse":
                                inferred = 0
                            else:
                                inferred = 1
                            if "N" in nodes[synapse].keys():
                                N = nodes[synapse]["N"]
                            else:
                                N = 0
                            # try:
                            post_nodes = [
                                i[1]
                                for i in edges
                                if i[0] == synapse
                                and (nodes[i[1]]["class"] == "Neuron")
                            ]
                            for post_node in post_nodes:
                                post_node = nodes[post_node]
                                if "uname" in post_node:
                                    post = post_node["uname"]
                                else:
                                    post = post_node["name"]
                                if post is not None:
                                    csv = csv + (
                                        "\n"
                                        + str(pre)
                                        + ","
                                        + str(post)
                                        + ","
                                        + str(N)
                                        + ","
                                        + str(inferred)
                                    )
                                    for i in range(N):
                                        out_edges.append((str(pre), str(post)))
                                        out_nodes.append(str(pre))
                                        out_nodes.append(str(post))
                            # except:
                            #     pass
        out_nodes = list(set(out_nodes))
        out_edges_unique = list(set(out_edges))
        return out_nodes, out_edges, out_edges_unique

    def getSynapses(self, presynapticNeuron, postsynapticNeuron):
        """Returns the synapses between a given presynaptic neuron and a postsynaptic neuron.

        # Arguments
            presynapticNeuron (str): The name of the presynaptic neuron.
            postsynapticNeuron (str): The name of the postsynaptic neuron.

        # Returns
            float: The number of synapses.
        """
        if self.compiled == False:
            self.prepareCircuit()
        try:
            presynapticIndex = self.out_nodes.index(presynapticNeuron)
        except:
            raise Exception(
                "The presynaptic neuron given as input to 'getSynapses' is not present in the current workspace."
            )
        try:
            postsynapticIndex = self.out_nodes.index(postsynapticNeuron)
        except:
            raise Exception(
                "The postsynaptic neuron given as input to 'getSynapses' is not present in the current workspace."
            )
        M = self.getConnectivityMatrix()
        return M[presynapticIndex, postsynapticIndex]

    def getPresynapticNeurons(self, postsynapticNeuron):
        """Returns a dictionary of all presynaptic neurons for a given postsynaptic neuron.

        # Arguments
            postsynapticNeuron (str): The name of the postsynaptic neuron.

        # Returns
            dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given postsynaptic neuron.
        """
        if self.compiled == False:
            self.prepareCircuit()
        postsynapticIndex = self.out_nodes.index(postsynapticNeuron)
        if postsynapticIndex < 0:
            raise Exception(
                "The postsynaptic neuron given as input to 'getPresynapticNeurons' is not present in the current workspace."
            )
        M = self.getConnectivityMatrix()
        connDict = {}
        for i in range(M.shape[0]):
            if M[i, postsynapticIndex] > 0:
                connDict[self.out_nodes[i]] = M[i, postsynapticIndex]
        return connDict

    def getPostsynapticNeurons(self, presynapticNeuron):
        """Returns a dictionary of all postsynaptic neurons for a given presynaptic neuron.

        # Arguments
            presynapticNeuron (str): The name of the presynaptic neuron.

        # Returns
            dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given presynaptic neuron.
        """
        if self.compiled == False:
            self.prepareCircuit()
        presynapticIndex = self.out_nodes.index(presynapticNeuron)
        if presynapticIndex < 0:
            raise Exception(
                "The presynaptic neuron given as input to 'getPostsynapticNeurons' is not present in the current workspace."
            )
        M = self.getConnectivityMatrix()
        connDict = {}
        for i in range(M.shape[0]):
            if M[i, presynapticIndex] > 0:
                connDict[self.out_nodes[i]] = M[i, presynapticIndex]
        return connDict

    def genNB(
        self,
        nodes,
        edges,
        model="auto",
        config={},
        default_neuron=nb.HodgkinHuxley(),
        default_synapse=nb.AlphaSynapse(),
    ):
        """Processes the output of processConnectivity to generate a Neuroballad circuit.

        # Returns
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
            C.G.nodes()[v]["name"] = list(node_keys.keys())[i]
        if model == "auto":
            for i in edges:
                if i not in config:
                    idx = C.add_cluster(1, default_synapse)[0]
                    C.join([[node_keys[i[0]], idx]])
                    C.join([[idx, node_keys[i[1]]]])
                    try:
                        C.G.nodes()["uid" + str(idx)]["name"] = (
                            "Synapse from " + i[0] + " to " + i[1]
                        )
                    except:
                        pass
                    # print(C.G.nodes()["uid" + str(idx)]['BioName'])
        if model == "simple":
            for i in edges:
                if i not in config:
                    C.join([[node_keys[i[0]], node_keys[i[1]]]])

        return C, node_keys

    def getConnectivityDendrogram(self):
        self.prepareCircuit()
        M = self.getConnectivityMatrix()
        M = pd.DataFrame(M, index=self.out_nodes, columns=self.out_nodes)
        sns.clustermap(M)

    def sendCircuitPrimitive(self, C, args={"name": "temp"}):
        """Sends a NetworkX graph to the backend.
        """
        C.compile(
            model_output_name=os.path.join(self.dataPath, args["name"] + ".gexf.gz")
        )
        with open(os.path.join(self.dataPath, args["name"] + ".gexf.gz"), "rb") as file:
            data = file.read()
        a = {}
        a["name"] = args["name"]
        a["experiment"] = self.experimentInputs
        a["graph"] = binascii.hexlify(data).decode()
        res = self.rpc("ffbo.gfx.sendCircuit", a)
        res = self.rpc("ffbo.gfx.sendExperiment", a)
        # print(_FBLClient.rpc('ffbo.gfx.sendCircuit', a))

    def alter(self, X):
        """Alters a set of models with specified Neuroballad models.

        # Arguments
            X (list of lists): A list of lists. Elements are lists whose first element is the neuron ID (str) and the second is the Neuroballad object corresponding to the model.
        """
        if any(isinstance(el, list) for el in X):  # Check if input is a list of lists
            pass
        else:
            X = [X]
        for x in X:
            if x[0] in self.node_keys:
                self.C.G.node["uid" + str(self.node_keys[x[0]])].clear()
                params = x[1].params
                params["name"] = params["name"] + str(self.node_keys[x[0]])
                self.C.G.node["uid" + str(self.node_keys[x[0]])].update(params)
            else:
                raise Exception(
                    "The rule you passed named", x, "does match a known node name."
                )

    def addInput(self, x):
        """Adds an input to the experiment settings. The input is a Neuroballad input object.

        # Arguments
            x (Neuroballad Input Object): The input object to append to the list of inputs.

        # Returns
            dict: The input object added to the experiment list.
        """
        self.experimentInputs.append(x.params)
        data = self.experimentInputs
        a = {}
        a["data"] = data
        a["messageType"] = "Data"
        a["widget"] = "JSONEditor"
        self.tryComms(a)
        return x.params

    def listInputs(self):
        """Sends the current experiment settings to the frontend for displaying in an editor.
        """
        a = {}
        data = self.experimentInputs
        a["data"] = data
        a["messageType"] = "Data"
        a["widget"] = "JSONEditor"
        self.tryComms(a)
        return self.experimentInputs

    def fetchCircuit(self, X, local=True):
        """Deprecated function that locally saves a circuit file via the backend.
           Deprecated because of connectivity issues with large files.
        """
        X = self.rpc(u"ffbo.gfx.getCircuit", X)
        X["data"] = binascii.unhexlify(X["data"].encode())
        if local == False:
            with open(
                os.path.join(_FBLDataPath, X["name"] + ".gexf.gz"), "wb"
            ) as file:
                file.write(X["data"])
        else:
            with open(os.path.join(X["name"] + ".gexf.gz"), "wb") as file:
                file.write(X["data"])
        return True

    def fetchExperiment(self, X, local=True):
        """Deprecated function that locally saves an experiment file via the backend.
           Deprecated because of connectivity issues with large files.
        """
        X = self.rpc(u"ffbo.gfx.getExperiment", X)
        X["data"] = json.dumps(X["data"])
        if local == False:
            with open(os.path.join(_FBLDataPath, X["name"] + ".json"), "w") as file:
                file.write(X["data"])
        else:
            with open(os.path.join(X["name"] + ".json"), "w") as file:
                file.write(X["data"])
        return True

    def JSCall(self, messageType="getExperimentConfig", data={}):
        a = {}
        a["data"] = data
        a["messageType"] = messageType
        a["widget"] = "GFX"
        self.tryComms(a)

    def getExperimentConfig(self):
        self.JSCall()

    def fetchSVG(self, X, local=True):
        """Deprecated function that locally saves an SVG via the backend.
           Deprecated because of connectivity issues with large files.
        """
        X = self.rpc(u"ffbo.gfx.getSVG", X)
        X["data"] = binascii.unhexlify(X["data"].encode())
        # X['data'] = json.dumps(X['data'])
        if local == False:
            with open(os.path.join(_FBLDataPath, X["name"] + ".svg"), "wb") as file:
                file.write(X["data"])
        else:
            with open(os.path.join(X["name"] + ".svg"), "wb") as file:
                file.write(X["data"])
        return True

    def _sendSVG(self, X):
        """Deprecated function that loads an SVG via the backend.
           Deprecated because of connectivity issues with large files.
        """
        name = X
        # with open(os.path.join(_FBLDataPath, name + '.svg'), "r") as file:
        #        svg = file.read()
        a = {}
        # a['data'] = svg
        a["data"] = X
        a["messageType"] = "loadCircuit"
        a["widget"] = "GFX"
        self.tryComms(a)

    def sendSVG(self, name, file):
        """Sends an SVG to the FBL fileserver. Useful for storing data and using loadSVG.

        # Arguments
            name (str): Name to use when saving the file; '_visual' gets automatically appended to it.
            file (str): Path to the SVG file.
        """
        with open(file, "r") as ifile:
            data = ifile.read()
        data = json.dumps({"name": name, "svg": data})
        self.rpc("ffbo.gfx.sendSVG", data)

    def loadSVG(self, name):
        """Loads an SVG in the FBL fileserver.

        # Arguments
            name (str): Name to use when loading the file.
        """
        self.tryComms({"widget": "GFX", "messageType": "loadCircuit", "data": name})

    def FICurveGenerator(self, model):
        """Sample library function showing how to do automated experimentation using FBL's Notebook features. Takes a simple abstract neuron model and runs experiments on it.

        # Arguments
            model (Neuroballad Model Object): The model object to test.

        # Returns
            numpy array: A tuple of NumPy arrays corresponding to the X and Y of the FI curve.
        """
        del self.data
        self.data = []
        self.sendDataToGFX = False
        del self.C
        self.C = nb.Circuit()

        self.experimentInputs = []
        # self.addInput(
        #    nb.InIStep(0, 5., 0., 1.))
        self.executionSuccessful = True
        circuitName = "FITest"

        for stepAmplitude in range(30):
            idx = self.C.add_cluster(1, model)[0]
            self.addInput(nb.InIStep(idx, float(stepAmplitude), 0.0, 1.0))
        self.sendCircuitPrimitive(self.C, args={"name": circuitName})
        self.log['GFX'].debug("Circuit sent. Queuing execution.")
        # while self.executionSuccessful == False:
        #    sleep(1)
        # self.experimentInputs = []

        #
        a = {}
        a["name"] = "FITest"
        a["experiment"] = self.experimentInputs
        self.experimentQueue.append(a)
        self.executionSuccessful = False
        a = self.experimentQueue.pop(0)
        # self.parseSimResults()
        res = self.rpc("ffbo.gfx.sendExperiment", a)
        res = self.rpc("ffbo.gfx.startExecution", {"name": circuitName})

        return True

    def parseSimResults(self):
        """Parses the simulation results. Deprecated.
        """
        numpyData = {}
        for x in self.data:
            if "data" in x:
                if type(x["data"]) is dict:
                    if "data" in x["data"]:
                        if "ydomain" in x["data"]["data"]:
                            for i in x["data"].keys():
                                if i not in numpyData.keys():
                                    numpyData[i] = x["data"][i]
                                else:
                                    numpyData[i] += x["data"][i]
        self.simData = numpyData

    def getSimResults(self):
        """Computes the simulation results.

        # Returns
            numpy array: A neurons-by-time array of results.
            list: A list of neuron names, sorted according to the data.
        """
        i = -1
        sim_output = json.loads(self.data[-1]["data"]["data"])
        sim_output_new = json.loads(self.data[i]["data"]["data"])
        while True:
            if not i == -1:
                sim_output["data"] = sim_output_new["data"] + sim_output["data"]
            i = i - 1
            try:
                sim_output_new = json.loads(self.data[i]["data"]["data"])
            except:
                break
        sim_output["data"] = json.loads(sim_output["data"])
        bs = []
        keys = []
        for key in sim_output["data"].keys():
            A = np.array(sim_output["data"][key])
            b = A[:, 1]
            keys.append(key)
            bs.append(b)

        B = np.array(bs)
        print("Shape of Results:", B.shape)
        return B, keys

    def plotSimResults(self, B, keys):
        """Plots the simulation results. A simple function to demonstrate result display.

        # Arguments
            model (Neuroballad Model Object): The model object to test.
        """
        plt.figure(figsize=(22, 10))
        plt.title("Sim Results")
        plt.plot(B.T)
        plt.legend(keys)

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
            if "spike" in key:
                a = np.sum(self.simData[key])
                keynums = [float(s) for s in re.findall(r"-?\d+\.?\d*", key)]
                X.append(keynums[0])
                Y.append(a)

        X = np.array(X)
        Y = np.array(Y)
        Y = Y[np.argsort(X)]
        X = np.sort(X)
        plt.plot(np.array(X), np.array(Y))
        plt.xlabel("Input Amplitude (muA)")
        plt.ylabel("Spike Rate (Spikes/Second)")
        plt.title("F-I Curve for the Queried Model")

    def loadCartridge(self, cartridgeIndex=100):
        """Sample library function for loading cartridges, showing how one can build libraries that work with flybrainlab.
        """
        self.executeNAquery(
            {
                "query": [
                    {
                        "action": {"method": {"query": {"name": ["lamina"]}}},
                        "object": {"class": "LPU"},
                    },
                    {
                        "action": {
                            "method": {
                                "traverse_owns": {
                                    "cls": "CartridgeModel",
                                    "name": "cartridge_" + str(cartridgeIndex),
                                }
                            }
                        },
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {"traverse_owns": {"instanceof": "MembraneModel"}}
                        },
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {"traverse_owns": {"instanceof": "DendriteModel"}}
                        },
                        "object": {"memory": 1},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 0}}},
                        "object": {"memory": 1},
                    },
                    {
                        "action": {"method": {"traverse_owns": {"cls": "Port"}}},
                        "object": {"memory": 3},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 0}}},
                        "object": {"memory": 1},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_in": {
                                    "min_depth": 2,
                                    "pass_through": [
                                        ["SendsTo", "SynapseModel", "instanceof"],
                                        ["SendsTo", "MembraneModel", "instanceof"],
                                    ],
                                }
                            }
                        },
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"method": {"has": {"name": "Amacrine"}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_in": {
                                    "min_depth": 2,
                                    "pass_through": [
                                        ["SendsTo", "SynapseModel", "instanceof"],
                                        ["SendsTo", "Aggregator", "instanceof"],
                                    ],
                                }
                            }
                        },
                        "object": {"memory": 2},
                    },
                    {
                        "action": {"method": {"has": {"name": "Amacrine"}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_out": {
                                    "min_depth": 2,
                                    "pass_through": [
                                        ["SendsTo", "SynapseModel", "instanceof"],
                                        ["SendsTo", "MembraneModel", "instanceof"],
                                    ],
                                }
                            }
                        },
                        "object": {"memory": 4},
                    },
                    {
                        "action": {"method": {"has": {"name": "Amacrine"}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_out": {
                                    "min_depth": 2,
                                    "pass_through": [
                                        ["SendsTo", "SynapseModel", "instanceof"],
                                        ["SendsTo", "Aggregator", "instanceof"],
                                    ],
                                }
                            }
                        },
                        "object": {"memory": 6},
                    },
                    {
                        "action": {"method": {"has": {"name": "Amacrine"}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 2}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 6}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 8}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 11}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"method": {"get_connecting_synapsemodels": {}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 1}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"method": {"get_connected_ports": {}}},
                        "object": {"memory": 1},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 1}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"method": {"query": {"name": ["retina-lamina"]}}},
                        "object": {"class": "Pattern"},
                    },
                    {
                        "action": {"method": {"owns": {"cls": "Interface"}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 0}}},
                        "object": {"memory": 1},
                    },
                    {
                        "action": {
                            "op": {"find_matching_ports_from_selector": {"memory": 20}}
                        },
                        "object": {"memory": 1},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 0}}},
                        "object": {"memory": 1},
                    },
                    {
                        "action": {"method": {"query": {"name": ["retina"]}}},
                        "object": {"class": "LPU"},
                    },
                    {
                        "action": {
                            "op": {"find_matching_ports_from_selector": {"memory": 1}}
                        },
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_in": {
                                    "pass_through": [
                                        "SendsTo",
                                        "MembraneModel",
                                        "instanceof",
                                    ]
                                }
                            }
                        },
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 10}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 4}}},
                        "object": {"memory": 0},
                    },
                ],
                "format": "no_result",
            }
        )

        res = self.executeNAquery(
            {
                "query": [{"action": {"method": {"has": {}}}, "object": {"state": 0}}],
                "format": "nx",
            }
        )

        data = []
        for i in res:
            if "data" in i:
                if "data" in i["data"]:
                    if "nodes" in i["data"]["data"]:
                        data.append(i["data"]["data"])
        G = nx.Graph(data[0])
        self.C.G = G
        return True

    def loadExperimentConfig(self, x):
        """Updates the simExperimentConfig attribute using input from the diagram.

        # Arguments
            x (string): A JSON dictionary as a string.

        # Returns
            bool: True.
        """
        print("Obtained Experiment Configuration: ", x)
        self.simExperimentConfig = json.loads(x)
        if self.experimentWatcher is not None:
            self.experimentWatcher.loadExperimentConfig(self.simExperimentConfig)
        return True

    def initiateExperiments(self):
        """Initializes and executes experiments for different LPUs.
        """
        print("Initiating experiments...")
        print("Experiment Setup: ", self.simExperimentConfig)
        for key in self.simExperimentConfig.keys():
            if key in self.simExperimentRunners.keys():
                try:
                    module = importlib.import_module(i)
                    print("Loaded LPU {}.".format(i))
                    self.simExperimentRunners[key] = getattr(module, "sim")
                except:
                    print("Failed to load LPU {}.".format(i))
                run_func = self.simExperimentRunners[key]
                run_func(self.simExperimentConfig)
            else:
                print("No runner(s) were found for Diagram {}.".format(key))
        return True

    def prune_retina_lamina(
        self, removed_neurons=[], removed_labels=[], retrieval_format="nk"
    ):
        """Prunes the retina and lamina circuits.

        # Arguments
            cartridgeIndex (int): The cartridge to load. Optional.

        # Returns
            dict: A result dict to use with the execute_lamina_retina function.

        # Example:
            res = nm[0].load_retina_lamina()
            nm[0].execute_multilpu(res)
        """
        list_of_queries = [
            {
                "command": {"swap": {"states": [0, 1]}},
                "format": "nx",
                "user": self.client._async_session._session_id,
                "server": self.naServerID,
            },
            {
                "query": [
                    {
                        "action": {"method": {"has": {"name": removed_neurons}}},
                        "object": {"state": 0},
                    },
                    {
                        "action": {"method": {"has": {"label": removed_labels}}},
                        "object": {"state": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 1}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"method": {"get_connected_ports": {}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"method": {"has": {"via": ["+removed_via+"]}}},
                        "object": {"state": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 1}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "op": {"find_matching_ports_from_selector": {"memory": 0}}
                        },
                        "object": {"state": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 1}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_in": {
                                    "min_depth": 1,
                                    "pass_through": [
                                        "SendsTo",
                                        "SynapseModel",
                                        "instanceof",
                                    ],
                                }
                            }
                        },
                        "object": {"memory": 0},
                    },
                    {
                        "action": {
                            "method": {
                                "gen_traversal_out": {
                                    "min_depth": 1,
                                    "pass_through": [
                                        "SendsTo",
                                        "SynapseModel",
                                        "instanceof",
                                    ],
                                }
                            }
                        },
                        "object": {"memory": 1},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 1}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__add__": {"memory": 3}}},
                        "object": {"memory": 0},
                    },
                    {
                        "action": {"op": {"__sub__": {"memory": 0}}},
                        "object": {"state": 0},
                    },
                ],
                "format": retrieval_format,
                "user": self.client._async_session._session_id,
                "server": self.naServerID,
            },
        ]
        res = self.rpc(
            "ffbo.processor.neuroarch_query", list_of_queries[0]
        )
        print("Pruning ", removed_neurons)
        print("Pruning ", removed_labels)
        res = self.rpc(
            "ffbo.processor.neuroarch_query",
            list_of_queries[1],
            options=CallOptions(timeout=30000000000),
        )
        return res

    def load_retina_lamina(
        self,
        cartridgeIndex=11,
        removed_neurons=[],
        removed_labels=[],
        retrieval_format="nk",
    ):
        """Loads retina and lamina.

        # Arguments
            cartridgeIndex (int): The cartridge to load. Optional.

        # Returns
            dict: A result dict to use with the execute_lamina_retina function.

        # Example:
            nm[0].getExperimentConfig() # In a different cell
            experiment_configuration = nm[0].load_retina_lamina(cartridgeIndex=126)
            experiment_configuration = experiment_configuration['success']['result']
            nm[0].execute_multilpu(experiment_configuration)
        """

        inp = {
            "query": [
                {
                    "action": {"method": {"query": {"name": ["lamina"]}}},
                    "object": {"class": "LPU"},
                },
                {
                    "action": {
                        "method": {
                            "traverse_owns": {
                                "cls": "CartridgeModel",
                                "name": "cartridge_" + str(cartridgeIndex),
                            }
                        }
                    },
                    "object": {"memory": 0},
                },
                {
                    "action": {
                        "method": {"traverse_owns": {"instanceof": "MembraneModel"}}
                    },
                    "object": {"memory": 0},
                },
                {
                    "action": {
                        "method": {"traverse_owns": {"instanceof": "DendriteModel"}}
                    },
                    "object": {"memory": 1},
                },
                {"action": {"op": {"__add__": {"memory": 0}}}, "object": {"memory": 1}},
                {
                    "action": {"method": {"traverse_owns": {"cls": "Port"}}},
                    "object": {"memory": 3},
                },
                {"action": {"op": {"__add__": {"memory": 0}}}, "object": {"memory": 1}},
                {
                    "action": {
                        "method": {
                            "gen_traversal_in": {
                                "min_depth": 2,
                                "pass_through": [
                                    ["SendsTo", "SynapseModel", "instanceof"],
                                    ["SendsTo", "MembraneModel", "instanceof"],
                                ],
                            }
                        }
                    },
                    "object": {"memory": 0},
                },
                {
                    "action": {"method": {"has": {"name": "Amacrine"}}},
                    "object": {"memory": 0},
                },
                {
                    "action": {
                        "method": {
                            "gen_traversal_in": {
                                "min_depth": 2,
                                "pass_through": [
                                    ["SendsTo", "SynapseModel", "instanceof"],
                                    ["SendsTo", "Aggregator", "instanceof"],
                                ],
                            }
                        }
                    },
                    "object": {"memory": 2},
                },
                {
                    "action": {"method": {"has": {"name": "Amacrine"}}},
                    "object": {"memory": 0},
                },
                {
                    "action": {
                        "method": {
                            "gen_traversal_out": {
                                "min_depth": 2,
                                "pass_through": [
                                    ["SendsTo", "SynapseModel", "instanceof"],
                                    ["SendsTo", "MembraneModel", "instanceof"],
                                ],
                            }
                        }
                    },
                    "object": {"memory": 4},
                },
                {
                    "action": {"method": {"has": {"name": "Amacrine"}}},
                    "object": {"memory": 0},
                },
                {
                    "action": {
                        "method": {
                            "gen_traversal_out": {
                                "min_depth": 2,
                                "pass_through": [
                                    ["SendsTo", "SynapseModel", "instanceof"],
                                    ["SendsTo", "Aggregator", "instanceof"],
                                ],
                            }
                        }
                    },
                    "object": {"memory": 6},
                },
                {
                    "action": {"method": {"has": {"name": "Amacrine"}}},
                    "object": {"memory": 0},
                },
                {"action": {"op": {"__add__": {"memory": 2}}}, "object": {"memory": 0}},
                {"action": {"op": {"__add__": {"memory": 6}}}, "object": {"memory": 0}},
                {"action": {"op": {"__add__": {"memory": 8}}}, "object": {"memory": 0}},
                {
                    "action": {"op": {"__add__": {"memory": 11}}},
                    "object": {"memory": 0},
                },
                {
                    "action": {"method": {"get_connecting_synapsemodels": {}}},
                    "object": {"memory": 0},
                },
                {"action": {"op": {"__add__": {"memory": 1}}}, "object": {"memory": 0}},
                {
                    "action": {"method": {"get_connected_ports": {}}},
                    "object": {"memory": 1},
                },
                {"action": {"op": {"__add__": {"memory": 1}}}, "object": {"memory": 0}},
                {
                    "action": {"method": {"query": {"name": ["retina-lamina"]}}},
                    "object": {"class": "Pattern"},
                },
                {
                    "action": {"method": {"owns": {"cls": "Interface"}}},
                    "object": {"memory": 0},
                },
                {"action": {"op": {"__add__": {"memory": 0}}}, "object": {"memory": 1}},
                {
                    "action": {
                        "op": {"find_matching_ports_from_selector": {"memory": 20}}
                    },
                    "object": {"memory": 1},
                },
                {"action": {"op": {"__add__": {"memory": 0}}}, "object": {"memory": 1}},
                {
                    "action": {"method": {"get_connected_ports": {}}},
                    "object": {"memory": 0},
                },
                {"action": {"op": {"__add__": {"memory": 0}}}, "object": {"memory": 1}},
                {
                    "action": {"method": {"query": {"name": ["retina"]}}},
                    "object": {"class": "LPU"},
                },
                {
                    "action": {
                        "op": {"find_matching_ports_from_selector": {"memory": 1}}
                    },
                    "object": {"memory": 0},
                },
                {
                    "action": {
                        "method": {
                            "gen_traversal_in": {
                                "pass_through": [
                                    "SendsTo",
                                    "MembraneModel",
                                    "instanceof",
                                ]
                            }
                        }
                    },
                    "object": {"memory": 0},
                },
                {
                    "action": {"op": {"__add__": {"memory": 10}}},
                    "object": {"memory": 0},
                },
                {"action": {"op": {"__add__": {"memory": 4}}}, "object": {"memory": 0}},
            ],
            "format": "no_result",
            "user": self.client._async_session._session_id,
            "server": self.naServerID,
        }

        res = self.rpc("ffbo.processor.neuroarch_query", inp)

        inp = {
            "query": [{"action": {"method": {"has": {}}}, "object": {"state": 0}}],
            "format": "nx",
            "user": self.client._async_session._session_id,
            "server": self.naServerID,
        }

        res = self.rpc("ffbo.processor.neuroarch_query", inp)
        #print(res)
        """
        res_info = self.rpc(u'ffbo.processor.server_information')
        msg = {"user": self.client._async_session._session_id,
            "servers": {'na': self.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.rpc(u'ffbo.na.query.' + msg['servers']['na'], {'user': msg['user'],
                                        'command': {"retrieve":{"state":0}},
                                        'format': "nk"}, options=CallOptions(
                                        timeout = 30000000000
                                        ))
        """

        neurons = self.get_current_neurons(res["success"]["result"])
        if "cartridge_" + str(cartridgeIndex) in self.simExperimentConfig:
            if (
                "disabled"
                in self.simExperimentConfig["cartridge_" + str(cartridgeIndex)]
            ):
                removed_neurons = (
                    removed_neurons
                    + self.simExperimentConfig["cartridge_" + str(cartridgeIndex)][
                        "disabled"
                    ]
                )
                print("Updated Disabled Neuron List: ", removed_neurons)
        removed_neurons = self.ablate_by_match(
            res["success"]["result"], removed_neurons
        )

        res = self.prune_retina_lamina(
            removed_neurons=removed_neurons,
            removed_labels=removed_labels,
            retrieval_format=retrieval_format,
        )
        """
        res_info = self.rpc(u'ffbo.processor.server_information')
        msg = {"user": self.client._async_session._session_id,
            "servers": {'na': self.naServerID, 'nk': list(res_info['nk'].keys())[0]}}
        res = self.rpc(u'ffbo.na.query.' + msg['servers']['na'], {'user': msg['user'],
                                'command': {"retrieve":{"state":0}},
                                'format': "nk"})
        """
        # print(res['data']['LPU'].keys())
        print("Retina and lamina circuits have been successfully loaded.")
        return res

    def get_current_neurons(self, res):
        labels = []
        for j in res["data"]["nodes"]:
            if "label" in res["data"]["nodes"][j]:
                label = res["data"]["nodes"][j]["label"]
                if "port" not in label and "synapse" not in label:
                    labels.append(label)
        return labels

    def ablate_by_match(self, res, neuron_list):
        neurons = self.get_current_neurons(res)
        removed_neurons = []
        for i in neuron_list:
            removed_neurons = removed_neurons + [j for j in neurons if i in j]
        removed_neurons = list(set(removed_neurons))
        return removed_neurons

    def execute_multilpu(self, name, inputProcessors = {}, outputProcessors = {},
                         steps= None, dt = None):
        """Executes a multilpu circuit. Requires a result dictionary.

        # Arguments
            res (dict): The result dictionary to use for execution.

        # Returns
            bool: A success indicator.
        """
        # labels = []
        # for i in res['data']['LPU']:
        #     for j in res['data']['LPU'][i]['nodes']:
        #         if 'label' in res['data']['LPU'][i]['nodes'][j]:
        #             label = res['data']['LPU'][i]['nodes'][j]['label']
        #             if 'port' not in label and 'synapse' not in label:
        #                 labels.append(label)

        res = self.rpc(u'ffbo.processor.server_information')
        if len(res['nk']) == 0:
            raise RuntimeError('Neurokernel Server not found. If it halts, please restart it.')
        # TODO: randomly choose from the nk servers that are not busy. If all are busy, randomly choose one.
        msg = {#'neuron_list': labels,
                "user": self.client._async_session._session_id,
                "name": name,
                "servers": {'na': self.naServerID, 'nk': random.choice(list(res['nk'].keys()))}}

        if len(inputProcessors)>0:
            msg['inputProcessors'] = inputProcessors
        if len(outputProcessors)>0:
            msg['outputProcessors'] = outputProcessors
        if dt is not None:
            msg["dt"] = dt
        if steps is not None:
            msg["steps"] = steps

        self.log['NK'].debug('server_info: {}'.format(res))
        res = []

        def on_progress(x, res):
            res.append(x)

        res_list = []
        res = self.rpc(
            "ffbo.processor.nk_execute",
            msg,
            options=CallOptions(
                on_progress=partial(on_progress, res=res_list), timeout=30000000000
            ),
        )
        self.log['NK'].info("Execution request sent. Please wait.")
        if 'success' in res:
            self.log['NK'].info(res['success'])
        else:
            raise RuntimeError('Job not received for unknown reason.')

    def updateSimResultLabel(self, result_name, label_dict):
        result = self.exec_result[result_name]
        input_keys = list(result['input'].keys())
        for k in input_keys:
            if k in label_dict:
                result['input'][label_dict[k]] = result['input'].pop(k)
        output_keys = list(result['output'].keys())
        for k in output_keys:
            if k in label_dict:
                result['output'][label_dict[k]] = result['output'].pop(k)

    def plotExecResult(self, result_name, inputs = None, outputs = None):
        # inputs
        res_input = self.exec_result[result_name]['input']
        if inputs is None:
            inputs = res_input.keys()
            input_vars = list(set(sum([list(v.keys()) for k, v in res_input.items()],[])))
        else:
            input_vars = list(set(sum([list(v.keys()) for k, v in res_input.items() if k in inputs],[])))
        if len(input_vars):
            plt.figure(figsize=(22,10))
            ax = {var: plt.subplot(len(input_vars), 1, i+1) for i, var in enumerate(input_vars)}
            legends = {var: [] for var in input_vars}
            for k, v in res_input.items():
                if k in inputs:
                    for var, d in v.items():
                        ax[var].plot(np.arange(len(d['data']))*d['dt'], d['data'])
                        legends[var].append(k)
            for var in ax:
                ax[var].set_title('{}: Input - {}'.format(result_name, var))
                ax[var].legend(legends[var])
                ax[var].set_xlabel('time (s)')
            plt.show()

        # outputs
        res_output = self.exec_result[result_name]['output']
        if outputs is None:
            outputs = res_output.keys()
            output_vars = list(set(sum([list(v.keys()) for k, v in res_output.items()],[])))
        else:
            output_vars = list(set(sum([list(v.keys()) for k, v in res_output.items() if k in outputs],[])))
        if len(output_vars):
            plt.figure(figsize=(22,10))
            ax = {var: plt.subplot(len(output_vars), 1, i+1) for i, var in enumerate(output_vars)}
            legends = {var: [] for var in output_vars}
            for k, v in res_output.items():
                if k in outputs:
                    for var, d in v.items():
                        ax[var].plot(np.arange(len(d['data']))*d['dt'], d['data'])
                        legends[var].append(k)
            for var in ax:
                ax[var].set_title('{}: Output - {}'.format(result_name, var))
                ax[var].legend(legends[var])
                ax[var].set_xlabel('time (s)')
            plt.show()


    def export_diagram_config(self, res):
        """Exports a diagram configuration from Neuroarch data to GFX.

        # Arguments
            res (dict): The result dictionary to use for export.

        # Returns
            dict: The configuration to export.
        """
        newConfig = {"cx": {"disabled": []}}
        param_names = [
            "reset_potential",
            "capacitance",
            "resting_potential",
            "resistance",
        ]
        param_names_js = [
            "reset_potential",
            "capacitance",
            "resting_potential",
            "resistance",
        ]
        state_names = ["initV"]
        state_names_js = ["initV"]
        for lpu in res["data"]["LPU"].keys():
            for node in res["data"]["LPU"][lpu]["nodes"]:
                if "name" in res["data"]["LPU"][lpu]["nodes"][node]:
                    node_data = res["data"]["LPU"][lpu]["nodes"][node]
                    new_node_data = {"params": {}, "states": {}}
                    for param_idx, param in enumerate(param_names):
                        if param in node_data:
                            new_node_data["params"][
                                param_names_js[param_idx]
                            ] = node_data[param]
                            new_node_data["name"] = "LeakyIAF"
                    for state_idx, state in enumerate(state_names):
                        if state in node_data:
                            new_node_data["states"][
                                state_names_js[state_idx]
                            ] = node_data[state]
                    newConfig["cx"][
                        res["data"]["LPU"][lpu]["nodes"][node]["name"]
                    ] = new_node_data
        newConfig_tosend = json.dumps(newConfig)
        self.JSCall(messageType="setExperimentConfig", data=newConfig_tosend)
        return newConfig

    def import_diagram_config(self, res, newConfig):
        """Imports a diagram configuration from Neuroarch data.

        # Arguments
            res (dict): The result dictionary to update.
            newConfig (dict): The imported configuration from a diagram.

        # Returns
            dict: The updated Neuroarch result dictionary.
        """
        param_names = [
            "reset_potential",
            "capacitance",
            "resting_potential",
            "resistance",
        ]
        param_names_js = [
            "reset_potential",
            "capacitance",
            "resting_potential",
            "resistance",
        ]
        state_names = ["initV"]
        state_names_js = ["initV"]
        for lpu in res["data"]["LPU"].keys():
            for node in res["data"]["LPU"][lpu]["nodes"]:
                if "name" in res["data"]["LPU"][lpu]["nodes"][node]:
                    if (
                        res["data"]["LPU"][lpu]["nodes"][node]["name"]
                        in newConfig["cx"].keys()
                    ):
                        updated_node_data = newConfig["cx"][
                            res["data"]["LPU"][lpu]["nodes"][node]["name"]
                        ]
                        for param_idx, param in enumerate(param_names_js):
                            if param in updated_node_data:
                                res["data"]["LPU"][lpu]["nodes"][node][
                                    param_names[param_idx]
                                ] = updated_node_data[param]
                        for state_idx, state in enumerate(state_names_js):
                            if state in updated_node_data:
                                res["data"]["LPU"][lpu]["nodes"][node][
                                    state_names[state_idx]
                                ] = updated_node_data[state]
        return res

    def get_neuron_graph(self, query_result = None):
        if query_result is None:
            data = self.getConnectivity()
        else:
            #data = self.executeNAquery()
            pass
        nodes = data[-2]['data']['data']['nodes']
        edges = data[-2]['data']['data']['edges']
        neurons = {n: v for n, v in nodes.items() if v['class'] in ['Neuron']}
        synapses = {n: v for n, v in nodes.items() if v['class'] in ['Synapse', 'InferredSynapse']}

        pre_to_synapse_edges = {post:pre for pre, post, prop in edges if prop.get('class', None) == 'SendsTo' and pre in neurons}
        synapse_to_post_edges = {pre:post for pre, post, prop in edges if prop.get('class', None) == 'SendsTo' and post in neurons}

        connections = [(pre, synapse_to_post_edges[syn], synapses[syn]['N']) for syn, pre in pre_to_synapse_edges.items()]
        g = nx.MultiDiGraph()
        g.add_nodes_from( list(neurons.items()))
        g.add_weighted_edges_from(connections)
        return g

    def get_neuron_adjacency_matrix(self, query_result = None, uname_order = None, rid_order = None):
        g = self.get_neuron_graph(query_result = query_result)
        if uname_order is None and rid_order is None:
            order = sorted([(g.nodes[n]['uname'], n) for n in g.nodes()])
            uname_order = [uname for uname, _ in order]
            rid_order = [rid for _, rid in order]
        elif uname_order is None:
            # rid_order
            uname_order = [g.nodes[n]['uname'] for n in rid_order]
        else:
            # uname_order
            order_dict = {g.nodes[n]['uname']: n for n in g.nodes()}
            rid_order = [order_dict[uname] for uname in uname_order]
        M = nx.adj_matrix(g, nodelist = rid_order).todense()
        return M, uname_order

    def select_DataSource(self, name, version):
        uri = "ffbo.na.datasource.{}".format(self.naServerID)
        res = self.rpc(
                uri,
                name, version,
                options=CallOptions(timeout=10000) )
        if 'error' in res:
            raise Error(res['error']['message'] + res['error']['exception'])
        elif 'success' in res:
            self.log['NA'].info(res['success']['message'])

    def add_neuron(self, uname,
                   name,
                   referenceId = None,
                   locality = None,
                   synonyms = None,
                   info = None,
                   morphology = None,
                   arborization = None,
                   neurotransmitters = None):
        uri = "ffbo.na.add_neuron.{}".format(self.naServerID)
        res = self.rpc(
                uri,
                uname, name, referenceId = referenceId, locality = locality,
                synonyms = None,
                info = None,
                morphology = None,
                arborization = None,
                neurotransmitters = None,
                options=CallOptions(timeout=10000) )
        if 'error' in res:
            raise Error(res['error']['message'] + res['error']['exception'])
        elif 'success' in res:
            return res['success']['data']

FBLClient = Client
