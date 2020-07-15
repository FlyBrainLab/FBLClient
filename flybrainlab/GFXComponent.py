import sys
from twisted.internet.defer import inlineCallbacks
from twisted.logger import Logger
import json
import ast
from autobahn.twisted.util import sleep
from autobahn.twisted.wamp import ApplicationSession
from autobahn.wamp.exception import ApplicationError
import networkx as nx
import numpy as np
import h5py
import neuroballad as nb
from time import gmtime, strftime
import os
from os.path import expanduser
import pickle
import math
import time
from pathlib import Path

try:
    from neuroballad import *
    from HDF5toJSON import *
    from diagram_generator import *
    from circuit_execution import *
    import matplotlib.pyplot as plt
    import pygraphviz
except:
    pass


def printHeader(name):
    return "[" + name + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "] "


## Create the home directory
import os
import urllib
import requests
home = str(Path.home())
if not os.path.exists(os.path.join(home, ".ffbolab")):
    os.makedirs(os.path.join(home, ".ffbolab"), mode=0o777)
if not os.path.exists(os.path.join(home, ".ffbolab", "data")):
    os.makedirs(os.path.join(home, ".ffbolab", "data"), mode=0o777)
if not os.path.exists(os.path.join(home, ".ffbolab", "config")):
    os.makedirs(os.path.join(home, ".ffbolab", "config"), mode=0o777)
if not os.path.exists(os.path.join(home, ".ffbolab", "lib")):
    os.makedirs(os.path.join(home, ".ffbolab", "lib"), mode=0o777)

_FFBOLabDataPath = os.path.join(home, ".ffbolab", "data")
_FFBOLabExperimentPath = os.path.join(home, ".ffbolab", "experiments")

def urlRetriever(url, savePath, verify = False):
    """Retrieves and saves a url in Python 3.
    # Arguments:
        url (str): File url.
        savePath (str): Path to save the file to.
    """
    with open(savePath, 'wb') as f:
        resp = requests.get(url, verify=verify)
        f.write(resp.content)

print(os.path.exists(_FFBOLabDataPath))
print(_FFBOLabDataPath)

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
logging.getLogger("twisted").setLevel(logging.CRITICAL)


def loadExperimentSettings(X):
    inList = []
    for a in X:
        if a["name"] == "InIGaussianNoise":
            inList.append(
                InIGaussianNoise(
                    a["node_id"], a["mean"], a["std"], a["t_start"], a["t_end"]
                )
            )
        if a["name"] == "InISinusoidal":
            inList.append(
                InISinusoidal(
                    a["node_id"],
                    a["amplitude"],
                    a["frequency"],
                    a["t_start"],
                    a["t_end"],
                    a["mean"],
                    a["shift"],
                    a["frequency_sweep"],
                    a["frequency_sweep_frequency"],
                    a["threshold_active"],
                    a["threshold_value"],
                )
            )
        if a["name"] == "InIBoxcar":
            inList.append(InIBoxcar(a["node_id"], a["I_val"], a["t_start"], a["t_end"]))
    return inList


class ffbolabComponent:
    def __init__(
        self,
        ssl=True,
        debug=True,
        authentication=True,
        user=u"ffbo",
        secret=u"",
        url=u"wss://neuronlp.fruitflybrain.org:7777/ws",
        realm=u"realm1",
        ca_cert_file="isrgrootx1.pem",
        intermediate_cert_file="letsencryptauthorityx3.pem",
        FFBOLabcomm=None,
    ):
        if os.path.exists(os.path.join(home, ".ffbolab", "lib")):
            print(
                printHeader("FFBOLab Client") + "Downloading the latest certificates."
            )
            # CertificateDownloader = urllib.URLopener()
            if not os.path.exists(os.path.join(home, ".ffbolab", "lib")):
                urlRetriever(
                    "https://data.flybrainlab.fruitflybrain.org/config/FBLClient.ini",
                    os.path.join(home, ".ffbolab", "config", "FBLClient.ini"),
                )
            urlRetriever(
                "https://data.flybrainlab.fruitflybrain.org/lib/isrgrootx1.pem",
                os.path.join(home, ".ffbolab", "lib", "caCertFile.pem"),
            )
            urlRetriever(
                "https://data.flybrainlab.fruitflybrain.org/lib/letsencryptauthorityx3.pem",
                os.path.join(home, ".ffbolab", "lib", "intermediateCertFile.pem"),
            )
            config_file = os.path.join(home, ".ffbolab", "config", "FBLClient.ini")
            ca_cert_file = os.path.join(home, ".ffbolab", "lib", "caCertFile.pem")
            intermediate_cert_file = os.path.join(
                home, ".ffbolab", "lib", "intermediateCertFile.pem"
            )
        config = ConfigParser()
        config.read(config_file)
        user = config["ComponentInfo"]["user"]
        secret = config["ComponentInfo"]["secret"]
        url = config["ComponentInfo"]["url"]
        self.FFBOLabcomm = FFBOLabcomm
        self.NKSimState = 0
        self.executionSettings = []
        extra = {"auth": authentication}
        self.lmsg = 0
        st_cert = open(ca_cert_file, "rt").read()
        c = OpenSSL.crypto
        ca_cert = c.load_certificate(c.FILETYPE_PEM, st_cert)
        st_cert = open(intermediate_cert_file, "rt").read()
        intermediate_cert = c.load_certificate(c.FILETYPE_PEM, st_cert)
        certs = OpenSSLCertificateAuthorities([ca_cert, intermediate_cert])
        ssl_con = CertificateOptions(trustRoot=certs)
        self.log = Logger()
        FFBOLABClient = AutobahnSync()
        self.client = FFBOLABClient

        @FFBOLABClient.on_challenge
        def on_challenge(challenge):
            if challenge.method == u"wampcra":
                print("WAMP-CRA challenge received: {}".format(challenge))
                if u"salt" in challenge.extra:
                    # salted secret
                    salted_key = auth.derive_key(
                        secret,
                        challenge.extra["salt"],
                        challenge.extra["iterations"],
                        challenge.extra["keylen"],
                    )
                    salted_key = (salted_key).decode("utf-8")
                    print(salted_key)
                # if user==u'ffbo':
                # plain, unsalted secret
                #    salted_key = u"kMU73GH4GS1WGUpEaSdDYwN57bdLdB58PK1Brb25UCE="
                # print(salted_key)
                # compute signature for challenge, using the key
                signature = auth.compute_wcs(salted_key, challenge.extra["challenge"])

                # return the signature to the router for verification
                return signature

            else:
                raise Exception("Invalid authmethod {}".format(challenge.method))

        if ssl:
            FFBOLABClient.run(url=url, authmethods=[u'wampcra'], authid=user, ssl=ssl_con)
        else:
            FFBOLABClient.run(url=url, authmethods=[u'wampcra'], authid=user)

        self.client_data = []
        self.data = []

        @FFBOLABClient.register("ffbo.gfx.addMessage")
        def add_message(message_name, message):
            output = json.loads(message)
            output = ast.literal_eval(json.dumps(output))
            # self.log.info("add_message() returns {x}", x=output)
            print("add_message returns", output)
            if message_name == "neurogist":
                pass
            return output

        print("Procedure ffbo.gfx.addMessage Registered...")

        @FFBOLABClient.register("ffbo.gfx.updateFileList")
        def updateFileList():
            self.files = [
                f
                for f in listdir(_FFBOLabDataPath)
                if isfile(join(_FFBOLabDataPath, f))
            ]
            output = ast.literal_eval(json.dumps(self.files))
            print("updateFileList returns", output)
            # self.log.info("return_files() returns {x}", x=output)
            self.data = {}
            self.data["data"] = output
            self.data["messageType"] = "updateFileList"
            return self.data

        print("Procedure ffbo.gfx.updateFileList Registered...")

        registerOptions = RegisterOptions(details_arg="details")

        @FFBOLABClient.register("ffbo.gfx.startExecution", options=registerOptions)
        def start_execution(settings, details=None):
            self.NKSimState = 1
            settings["userID"] = details.caller
            self.executionSettings.append(settings)
            return True


        print("Procedure ffbo.gfx.startExecution Registered...")

        registerOptions = RegisterOptions(details_arg="details")

        @FFBOLABClient.register("ffbo.gfx.loadResults", options=registerOptions)
        def loadResults(details=None):
            userID = details.caller
            filename = "neuroballad_temp_model_output.h5"
            G = nx.read_gexf("neuroballad_temp_model.gexf.gz")
            name_dict = {}
            for n, d in G.nodes(data=True):
                name_dict[n] = d["name"]
            print(name_dict)
            f = h5py.File(filename, "r")
            json_data = {}
            for variable_name in f.keys():
                if variable_name != "metadata":
                    print(variable_name)
                    data = list(f[variable_name].values())[0][:]
                    labels = list(f[variable_name].values())[1][:]
                    print(labels)
                    labels = list(filter(lambda k: b"auto" not in k, labels))
                    labels = list(filter(lambda k: b"input" not in k, labels))
                    print(labels)
                    for i in range(min(5000, len(labels))):
                        str_key = (
                            variable_name + "/" + name_dict[labels[i].decode("utf8")]
                        )
                        str_key = "".join(i for i in str_key if ord(i) < 128)
                        data_to_send = data[:, i]
                        data_to_send[data_to_send >= 1e3] = 0
                        data_to_send[data_to_send <= -1e3] = 0
                        data_to_send = np.nan_to_num(data_to_send)
                        data_to_send = np.clip(data_to_send, -1000.0, 1000.0)
                        json_data[str_key] = data_to_send.astype(np.float32).tolist()
            no_keys = int(math.ceil(float(len(json_data.keys())) / 5.0))
            json_out = json.dumps(json_data, ensure_ascii=True)
            text_file = open("Output.txt", "w")
            text_file.write(json_out)
            text_file.close()

            # Switch with calls
            # yield self.publish(u'com.example.clear_results', [])
            a = {}
            a["widget"] = "GFX"
            a["messageType"] = "clearResults"
            a["data"] = {}
            res = self.client.session.call(u"ffbo.ui.receive_gfx." + str(userID), a)
            for i in range(no_keys):
                dicttosend = dict(
                    (k, json_data[k])
                    for k in list(json_data.keys())[i * 5 : (i + 1) * 5]
                    if k in list(json_data.keys())
                )
                # try:
                #   yield self.publish(u'com.example.load_results', [dicttosend])
                print("Sending data...")
                a = {}
                a["widget"] = "GFX"
                a["messageType"] = "loadResults"
                a["data"] = dicttosend
                res = self.client.session.call(u"ffbo.ui.receive_gfx." + str(userID), a)
                # except:
                #    print('Data sending failed...')
                #    pass

            # Switch with calls
            a = {}
            a["widget"] = "GFX"
            a["messageType"] = "showServerMessage"
            a["data"] = ["[GFX] Simulation results successfully obtained.", 1, 2]
            res = self.client.session.call(u"ffbo.ui.receive_gfx." + str(userID), a)
            # yield self.publish(u'com.example.server_message', ["Results acquired...",1,2])
            print("Results sent...")
            # print(str(data[:,i].shape))
            # print(data)
            # print(str(no_keys))
            # print(str(len(json_data.keys())))
            # print(details.caller)
            NK_sim_state = 0
            return True

        print("Procedure ffbo.gfx.loadSimResults Registered...")

        @FFBOLABClient.register("ffbo.gfx.createDiagram")
        def create_diagram(diagram):
            self.log.info("create_diagram() called with {x}", x=diagram)
            output = diagram_generator(json.loads(diagram), self.log)
            return json.dumps(output)

        print("Procedure ffbo.gfx.createDiagram Registered...")

        @FFBOLABClient.register("ffbo.gfx.sendSVG")
        def send_svg(X):
            self.log.info("send_svg() called with {x}", x=X)
            X = json.loads(X)
            name = X['name']
            G = X['svg']
            with open( os.path.join(_FFBOLabDataPath,  name + '_visual.svg'), "w") as file:
                file.write(G)
            output = {}
            output["success"] = True
            self.log.info("send_svg() responded with {x}", x=output)
            return json.dumps(output)

        print("Procedure ffbo.gfx.sendSVG Registered...")

        @FFBOLABClient.register("ffbo.gfx.sendCircuit")
        def send_circuit(X):
            name = X["name"]
            G = binascii.unhexlify(X["graph"].encode())
            with open(os.path.join(_FFBOLabDataPath, name + ".gexf.gz"), "wb") as file:
                file.write(G)
            G = nx.read_gexf(os.path.join(_FFBOLabDataPath, name + ".gexf.gz"))
            nx.write_gexf(G, os.path.join(_FFBOLabDataPath, name + ".gexf"))
            return True

        print("Procedure ffbo.gfx.sendCircuit Registered...")

        @FFBOLABClient.register("ffbo.gfx.getCircuit")
        def get_circuit(X):
            name = X
            with open(os.path.join(_FFBOLabDataPath, name + ".gexf.gz"), "rb") as file:
                data = file.read()
            a = {}
            a["data"] = binascii.hexlify(data).decode()
            a["name"] = name
            return a

        print("Procedure ffbo.gfx.getCircuit Registered...")

        @FFBOLABClient.register("ffbo.gfx.getExperiment")
        def get_experiment(X):
            name = X
            with open(os.path.join(_FFBOLabDataPath, name + ".json"), "r") as file:
                experimentInputsJson = file.read()
            experimentInputsJson = json.loads(experimentInputsJson)
            a = {}
            a["data"] = experimentInputsJson
            a["name"] = name
            return a

        print("Procedure ffbo.gfx.getExperiment Registered...")

        @FFBOLABClient.register("ffbo.gfx.getSVG")
        def get_svg(X):
            name = X
            with open(os.path.join(_FFBOLabDataPath, name + ".svg"), "rb") as file:
                data = file.read()
            # experimentInputsJson = json.loads(experimentInputsJson)
            a = {}
            a["data"] = binascii.hexlify(data).decode()
            a["name"] = name
            return a

        print("Procedure ffbo.gfx.getSVG Registered...")

        @FFBOLABClient.register("ffbo.gfx.sendExperiment")
        def send_experiment(X):
            print(printHeader("FFBOLab Client GFX") + "sendExperiment called.")
            # X = json.loads(X)
            name = X["name"]
            data = json.dumps(X["experiment"])
            with open(os.path.join(_FFBOLabDataPath, name + ".json"), "w") as file:
                file.write(data)
            output = {}
            output["success"] = True
            print(printHeader("FFBOLab Client GFX") + "Experiment save successful.")
            return True

        print("Procedure ffbo.gfx.ffbolab.sendExperiment Registered...")

        @FFBOLABClient.register("ffbo.gfx.queryDB")
        def query_db(message):
            output = json.loads(message)
            self.log.info("query_db() called with {x}", x=output)
            return output

        print("Procedure ffbo.gfx.queryDB Registered...")

        @FFBOLABClient.register("ffbo.gfx.NKSim")
        def NK_sim(message):
            output = json.loads(message)
            self.log.info("NK_sim() called with {x}", x=output)
            self.NK_sim_state = 1
            return output

        print("Procedure ffbo.gfx.NKSim Registered...")

        @FFBOLABClient.register("ffbo.gfx.loadNeurogist")
        def load_neurogist(message):
            pass
            output = client.command(message)
            self.log.info(str(len(output)))
            # self.log.info(str((output[0].value)))
            self.log.info(str(json.dumps((output[0].oRecordData))))
            all_out = ""
            queries = []
            for i in output:
                if "tags" in i.oRecordData:
                    b = json.dumps(i.oRecordData)
                    queries.append(i.oRecordData)
                    self.log.info("load_neurogist() was called: returns {x}", x=b)
                    all_out = all_out + "\n" + str(b)
            return json.dumps(queries, indent=2)

        print("Procedure ffbo.gfx.loadNeurogist Registered...")

        @FFBOLABClient.register("ffbo.gfx.loadNeurobeholder")
        def load_neurobeholder(message):
            output = client.command(message)
            self.log.info(str(len(output)))
            # self.log.info(str((output[0].value)))
            self.log.info(str(json.dumps((output[0].oRecordData))))
            all_out = ""
            queries = []
            for i in output:
                if "tags" not in i.oRecordData:
                    b = json.dumps(i.oRecordData)
                    queries.append(i.oRecordData)
                    self.log.info("load_neurobeholder() was called: returns {x}", x=b)
                    all_out = all_out + "\n" + str(b)
            return json.dumps(queries, indent=2)

        print("Procedure ffbo.gfx.loadNeurobeholder Registered...")

        @FFBOLABClient.register("ffbo.gfx.loadNeuroscepter")
        def load_neurospecter(message):
            output = json.loads(message)
            self.log.info("load_neurospecter() called with {x}", x=output)
            return output

        print("Procedure ffbo.gfx.loadNeuroscepter Registered...")

        res = FFBOLABClient.session.call(u"ffbo.processor.server_information")

        for i in res["na"]:
            if "na" in res["na"][i]["name"]:
                print("Found working NA Server: " + res["na"][i]["name"])
                self.naServerID = i
        for i in res["nlp"]:
            self.nlpServerID = i


def loadResults(Client, userID=None):
    filename = "neuroballad_temp_model_output.h5"
    G = nx.read_gexf("neuroballad_temp_model.gexf.gz")
    name_dict = {}
    for n, d in G.nodes(data=True):
        name_dict[n] = d["name"]
    print(name_dict)
    f = h5py.File(filename, "r")
    json_data = {}
    for variable_name in f.keys():
        if variable_name != "metadata":
            print(variable_name)
            data = list(f[variable_name].values())[0][:]
            labels = list(f[variable_name].values())[1][:]
            print(labels)
            labels = list(filter(lambda k: b"auto" not in k, labels))
            labels = list(filter(lambda k: b"input" not in k, labels))
            print(labels)
            for i in range(min(5000, len(labels))):
                str_key = variable_name + "/" + name_dict[labels[i].decode("utf8")]
                str_key = "".join(i for i in str_key if ord(i) < 128)
                data_to_send = data[:, i]
                data_to_send[data_to_send >= 1e3] = 0
                data_to_send[data_to_send <= -1e3] = 0
                data_to_send = np.nan_to_num(data_to_send)
                data_to_send = np.clip(data_to_send, -1000.0, 1000.0)
                json_data[str_key] = data_to_send.astype(np.float32).tolist()
    no_keys = int(math.ceil(float(len(json_data.keys())) / 5.0))
    json_out = json.dumps(json_data, ensure_ascii=True)
    text_file = open("Output.txt", "w")
    text_file.write(json_out)
    text_file.close()

    a = {}
    a["widget"] = "GFX"
    a["messageType"] = "clearResults"
    a["data"] = ""
    print(a)
    print(userID)
    res = Client.client.session.call(u"ffbo.ui.receive_gfx." + str(userID), a)
    for i in range(no_keys):
        dicttosend = dict(
            (k, json_data[k])
            for k in list(json_data.keys())[i * 5 : (i + 1) * 5]
            if k in list(json_data.keys())
        )
        # try:
        #   yield self.publish(u'com.example.load_results', [dicttosend])
        print("Sending data...")
        a = {}
        a["widget"] = "GFX"
        a["messageType"] = "loadResults"
        a["data"] = dicttosend
        res = Client.client.session.call(u"ffbo.ui.receive_gfx." + str(userID), a)
        # except:
        #    print('Data sending failed...')
        #    pass

    # Switch with calls
    a = {}
    a["widget"] = "GFX"
    a["messageType"] = "showServerMessage"
    a["data"] = ["[GFX] Simulation results successfully obtained.", 1, 2]
    res = Client.client.session.call(u"ffbo.ui.receive_gfx." + str(userID), a)
    # yield self.publish(u'com.example.server_message', ["Results acquired...",1,2])
    print("Results sent...")
    # print(str(data[:,i].shape))
    # print(data)
    # print(str(no_keys))
    # print(str(len(json_data.keys())))
    # print(details.caller)
    return True


def mainThreadExecute(Component):
    # self.execution_settings = json.loads(settings)
    if len(Component.executionSettings) > 0:
        Component.NKSimState = 2
        print("Starting Neurokernel execution...")
        print(Component.NKSimState)
        print(Component.executionSettings)
        settings = Component.executionSettings[0]
        C = nb.Circuit()
        name = settings["name"]
        experimentInputs = []
        if "dt" in settings.keys():
            dt = settings["dt"]
        if "tmax" in settings.keys():
            dt = settings["tmax"]
        if os.path.isfile(os.path.join(_FFBOLabDataPath, name + ".json")):
            with open(os.path.join(_FFBOLabDataPath, name + ".json"), "r") as file:
                experimentInputsJson = file.read()
            experimentInputsJson = json.loads(experimentInputsJson)
            print(experimentInputsJson)
            experimentInputs = loadExperimentSettings(experimentInputsJson)
        else:
            C_in_ex = nb.InIStep(0, 40.0, 0.20, 0.40)
            experimentInputs.append(C_in_ex)
        C.G = nx.read_gexf(os.path.join(_FFBOLabDataPath, name + ".gexf.gz"))
        nx.write_gexf(C.G, "neuroballad_temp_model.gexf.gz")
        C.sim(1.0, 1e-4, experimentInputs)
        Component.NKSimState = 0
        del Component.executionSettings[0]
        loadResults(Component, settings["userID"])
        # C.G = nx.read_gexf(_FFBOLabDataPath + self.execution_settings['circuit_name'] + '.gexf')
        return True
    else:
        return False
