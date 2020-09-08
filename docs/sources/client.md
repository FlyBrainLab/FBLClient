### Client


```python
flybrainlab.Client.Client(
    ssl=False,
    debug=True,
    authentication=True,
    user="guest",
    secret="guestpass",
    custom_salt=None,
    url="wss://flycircuitdev.neuronlp.fruitflybrain.org/ws",
    realm="realm1",
    ca_cert_file="isrgrootx1.pem",
    intermediate_cert_file="letsencryptauthorityx3.pem",
    FFBOLabcomm=None,
    legacy=False,
    initialize_client=True,
    name=None,
    species="",
    use_config=False,
    custom_config=None,
    widgets=[],
    dataset="default",
)
```


FlyBrainLab Client class. This class communicates with JupyterLab frontend and connects to FFBO components.

__Attributes:__

FFBOLabcomm (obj): The communication object for sending and receiving data.
circuit (obj): A Neuroballad circuit that enables local circuit execution and facilitates circuit modification.
dataPath (str): Data path to be used.
experimentInputs (list of dicts): Inputs as a list of dicts that can be parsed by the GFX component.
compiled (bool): Circuits need to be compiled into networkx graphs before being sent for simulation. This is necessary as circuit compilation is a slow process.
sendDataToGFX (bool): Whether the data received from the backend should be sent to the frontend. Useful for code-only projects.


----

### __init__


```python
Client.__init__(
    ssl=False,
    debug=True,
    authentication=True,
    user="guest",
    secret="guestpass",
    custom_salt=None,
    url="wss://flycircuitdev.neuronlp.fruitflybrain.org/ws",
    realm="realm1",
    ca_cert_file="isrgrootx1.pem",
    intermediate_cert_file="letsencryptauthorityx3.pem",
    FFBOLabcomm=None,
    legacy=False,
    initialize_client=True,
    name=None,
    species="",
    use_config=False,
    custom_config=None,
    widgets=[],
    dataset="default",
)
```


Initialization function for the FBL Client class.


__Arguments__

- __ssl (bool)__: Whether the FFBO server uses SSL.
- __debug (bool) __: Whether debugging should be enabled.
- __authentication (bool)__: Whether authentication is enabled.
- __user (str)__: Username for establishing communication with FFBO components.
- __secret (str)__: Password for establishing communication with FFBO components.
- __url (str)__: URL of the WAMP server with the FFBO Processor component.
- __realm (str)__: Realm to be connected to.
- __ca_cert_file (str)__: Path to the certificate for establishing connection.
- __intermediate_cert_file (str)__: Path to the intermediate certificate for establishing connection.
- __FFBOLabcomm (obj)__: Communications object for the frontend.
- __legacy (bool)__: Whether the server uses the old FFBO server standard or not. Should be False for most cases. Defaults to False.
- __initialize_client (bool)__: Whether to connect to the database or not. Defaults to True.
- __name (str)__: Name for the client. String. Defaults to None.
- __use_config (bool)__: Whether to read the url from config instead of as arguments to the initializer. Defaults to False. False recommended for new users.
- __species (str)__: Name of the species to use for client information. Defaults to ''.
- __custom_config (str)__: A .ini file name to use to initiate a custom connection. Defaults to None. Used if provided.
- __widgets (list)__: List of widgets associated with this client. Optional.
- __dataset (str)__: Name of the dataset to use. Not used right now, but included for future compatibility.


----

### tryComms


```python
Client.tryComms(a)
```


Communication function to communicate with a JupyterLab frontend if one exists.

__Arguments__

- __a (obj)__: Arbitrarily formatted data to be sent via communication.


----

### executeNLPquery


```python
Client.executeNLPquery(query=None, language="en", uri=None, queryID=None, returnNAOutput=False)
```


Execute an NLP query.

__Arguments__

- __query (str)__: Query string.
- __language (str)__: Language to use.
- __uri (str)__: Currently not used; for future NLP extensions.
- __queryID (str)__: Query ID to be used. Generated automatically.
- __returnNAOutput (bool)__: Whether the corresponding NA query should not be executed.

__Returns__

dict: NA output or the NA query itself, depending on the returnNAOutput setting.


----

### executeNAquery


```python
Client.executeNAquery(res, language="en", uri=None, queryID=None, progressive=True, threshold=20)
```


Execute an NA query.

__Arguments__

- __res (dict)__: Neuroarch query.
- __language (str)__: Language to use.
- __uri (str)__: A custom FFBO query URI if desired.
- __queryID (str)__: Query ID to be used. Generated automatically.
- __progressive (bool)__: Whether the loading should be progressive. Needs to be true most of the time for the connection to be stable.
- __threshold (int)__: Data chunk size. Low threshold is required for the connection to be stable.

__Returns__

bool: Whether the process has been successful.


----

### FICurveGenerator


```python
Client.FICurveGenerator(model)
```


Sample library function showing how to do automated experimentation using FFBOLab's Notebook features. Takes a simple abstract neuron model and runs experiments on it.

__Arguments__

- __model (Neuroballad Model Object)__: The model object to test.

__Returns__

numpy array: A tuple of NumPy arrays corresponding to the X and Y of the FI curve.


----

### FICurvePlotSimResults


```python
Client.FICurvePlotSimResults()
```


Plots some result curves for the FI curve generator example.
        


----

### GFXcall


```python
Client.GFXcall(args)
```


Arbitrary call to a GFX procedure in the GFX component format.

__Arguments__

- __args (list)__: A list whose first element is the function name (str) and the following are the arguments.

__Returns__

dict OR string: The call result.


----

### JSCall


```python
Client.JSCall(messageType="getExperimentConfig", data={})
```


----

### ablate_by_match


```python
Client.ablate_by_match(res, neuron_list)
```


----

### addByUname


```python
Client.addByUname(uname, verb="add")
```


Adds some neurons by the uname.

__Returns__

bool: True.


----

### addInput


```python
Client.addInput(x)
```


Adds an input to the experiment settings. The input is a Neuroballad input object.

__Arguments__

- __x (Neuroballad Input Object)__: The input object to append to the list of inputs.

__Returns__

dict: The input object added to the experiment list.


----

### alter


```python
Client.alter(X)
```


Alters a set of models with specified Neuroballad models.

__Arguments__

- __X (list of lists)__: A list of lists. Elements are lists whose first element is the neuron ID (str) and the second is the Neuroballad object corresponding to the model.


----

### autoLayout


```python
Client.autoLayout()
```


Layout raw data from NeuroArch and save results as G_auto.*.
        


----

### createTag


```python
Client.createTag(tagName)
```


Creates a tag.

__Returns__

bool: True.


----

### execute_multilpu


```python
Client.execute_multilpu(name, inputProcessors={}, outputProcessors={}, steps=None, dt=None)
```


Executes a multilpu circuit. Requires a result dictionary.

__Arguments__

- __res (dict)__: The result dictionary to use for execution.

__Returns__

bool: A success indicator.


----

### export_diagram_config


```python
Client.export_diagram_config(res)
```


Exports a diagram configuration from Neuroarch data to GFX.

__Arguments__

- __res (dict)__: The result dictionary to use for export.

__Returns__

dict: The configuration to export.


----

### fetchCircuit


```python
Client.fetchCircuit(X, local=True)
```


Deprecated function that locally saves a circuit file via the backend.
Deprecated because of connectivity issues with large files.


----

### fetchExperiment


```python
Client.fetchExperiment(X, local=True)
```


Deprecated function that locally saves an experiment file via the backend.
Deprecated because of connectivity issues with large files.


----

### fetchSVG


```python
Client.fetchSVG(X, local=True)
```


Deprecated function that locally saves an SVG via the backend.
Deprecated because of connectivity issues with large files.


----

### findServerIDs


```python
Client.findServerIDs(dataset=None)
```


Find server IDs to be used for the utility functions.
        


----

### genNB


```python
Client.genNB(
    nodes, edges, model="auto", config={}, default_neuron=HodgkinHuxley, default_synapse=AlphaSynapse
)
```


Processes the output of processConnectivity to generate a Neuroballad circuit.

__Returns__

tuple: A tuple of the Neuroballad circuit, and a dictionary that maps the neuron names to the uids.


----

### getConnectivity


```python
Client.getConnectivity()
```


Obtain the connectivity matrix of the current circuit in NetworkX format.

__Returns__

dict: The connectivity dictionary.


----

### getConnectivityDendrogram


```python
Client.getConnectivityDendrogram()
```


----

### getConnectivityMatrix


```python
Client.getConnectivityMatrix()
```


----

### getExperimentConfig


```python
Client.getExperimentConfig()
```


----

### getInfo


```python
Client.getInfo(dbid)
```


Get information on a neuron.

__Arguments__

- __dbid (str)__: Database ID of the neuron or node.

__Returns__

dict: NA information regarding the node.


----

### getNeuropils


```python
Client.getNeuropils()
```


Get the neuropils the neurons in the workspace reside in.

__Returns__

list of strings: Set of neuropils corresponding to neurons.


----

### getPostsynapticNeurons


```python
Client.getPostsynapticNeurons(presynapticNeuron)
```


Returns a dictionary of all postsynaptic neurons for a given presynaptic neuron.

__Arguments__

- __presynapticNeuron (str)__: The name of the presynaptic neuron.

__Returns__

dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given presynaptic neuron.


----

### getPresynapticNeurons


```python
Client.getPresynapticNeurons(postsynapticNeuron)
```


Returns a dictionary of all presynaptic neurons for a given postsynaptic neuron.

__Arguments__

- __postsynapticNeuron (str)__: The name of the postsynaptic neuron.

__Returns__

dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given postsynaptic neuron.


----

### getSimData


```python
Client.getSimData(url)
```


----

### getSimResults


```python
Client.getSimResults()
```


Computes the simulation results.

__Returns__

numpy array: A neurons-by-time array of results.
list: A list of neuron names, sorted according to the data.


----

### getSlowConnectivity


```python
Client.getSlowConnectivity()
```


Obtain the connectivity matrix of the current circuit in a custom dictionary format. Necessary for large circuits.

__Returns__

dict: The connectivity dictionary.


----

### getStats


```python
Client.getStats(neuron_name)
```


----

### getSynapses


```python
Client.getSynapses(presynapticNeuron, postsynapticNeuron)
```


Returns the synapses between a given presynaptic neuron and a postsynaptic neuron.

__Arguments__

- __presynapticNeuron (str)__: The name of the presynaptic neuron.
- __postsynapticNeuron (str)__: The name of the postsynaptic neuron.

__Returns__

float: The number of synapses.


----

### get_client_info


```python
Client.get_client_info(fbl=None)
```


Receive client data for this client only.

__Arguments__

- __fbl (Object)__: MetaClient object. Optional. Gives us.

__Returns__

dict: dict of dicts with client name as key and widgets, name and species as keys of the value.


----

### get_current_neurons


```python
Client.get_current_neurons(res)
```


----

### import_diagram_config


```python
Client.import_diagram_config(res, newConfig)
```


Imports a diagram configuration from Neuroarch data.

__Arguments__

- __res (dict)__: The result dictionary to update.
- __newConfig (dict)__: The imported configuration from a diagram.

__Returns__

dict: The updated Neuroarch result dictionary.


----

### init_client


```python
Client.init_client(ssl, user, secret, custom_salt, url, ssl_con, legacy)
```


----

### initiateExperiments


```python
Client.initiateExperiments()
```


Initializes and executes experiments for different LPUs.
        


----

### listInputs


```python
Client.listInputs()
```


Sends the current experiment settings to the frontend for displaying in an editor.
        


----

### loadCartridge


```python
Client.loadCartridge(cartridgeIndex=100)
```


Sample library function for loading cartridges, showing how one can build libraries that work with flybrainlab.
        


----

### loadExperimentConfig


```python
Client.loadExperimentConfig(x)
```


Updates the simExperimentConfig attribute using input from the diagram.

__Arguments__

- __x (string)__: A JSON dictionary as a string.

__Returns__

bool: True.


----

### loadSVG


```python
Client.loadSVG(name)
```


Loads an SVG in the FBL fileserver.

__Arguments__

- __name (str)__: Name to use when loading the file.


----

### loadSWC


```python
Client.loadSWC(file_name, scale_factor=1.0, uname=None)
```


Loads a neuron skeleton stored in the .swc format.

__Arguments__

- __file_name (str)__: Database ID of the neuron or node.
- __scale_factor (float)__: A scale factor to scale the neuron's dimensions with. Defaults to 1.
- __uname (str)__: Unique name to use in the frontend. Defaults to the file_name.


----

### loadTag


```python
Client.loadTag(tagName)
```


Loads a tag.

__Returns__

bool: True.


----

### load_retina_lamina


```python
Client.load_retina_lamina(
    cartridgeIndex=11, removed_neurons=[], removed_labels=[], retrieval_format="nk"
)
```


Loads retina and lamina.

__Arguments__

- __cartridgeIndex (int)__: The cartridge to load. Optional.

__Returns__

dict: A result dict to use with the execute_lamina_retina function.

__Example:__

nm[0].getExperimentConfig() # In a different cell
experiment_configuration = nm[0].load_retina_lamina(cartridgeIndex=126)
experiment_configuration = experiment_configuration['success']['result']
nm[0].execute_multilpu(experiment_configuration)


----

### parseSimResults


```python
Client.parseSimResults()
```


Parses the simulation results. Deprecated.
        


----

### plotExecResult


```python
Client.plotExecResult(result_name, inputs=None, outputs=None)
```


----

### plotSimResults


```python
Client.plotSimResults(B, keys)
```


Plots the simulation results. A simple function to demonstrate result display.

__Arguments__

- __model (Neuroballad Model Object)__: The model object to test.


----

### prepareCircuit


```python
Client.prepareCircuit(model="auto")
```


Prepares the current circuit for the Neuroballad format.
        


----

### processConnectivity


```python
Client.processConnectivity(connectivity)
```


Processes a Neuroarch connectivity dictionary.

__Returns__

tuple: A tuple of nodes, edges and unique edges.


----

### prune_retina_lamina


```python
Client.prune_retina_lamina(removed_neurons=[], removed_labels=[], retrieval_format="nk")
```


Prunes the retina and lamina circuits.

__Arguments__

- __cartridgeIndex (int)__: The cartridge to load. Optional.

__Returns__

dict: A result dict to use with the execute_lamina_retina function.

__Example:__

res = nm[0].load_retina_lamina()
nm[0].execute_multilpu(res)


----

### removeByUname


```python
Client.removeByUname(uname)
```


Removes some neurons by the uname.

__Returns__

bool: True.


----

### runLayouting


```python
Client.runLayouting(type="auto", model="auto")
```


Sends a request for the running of the layouting algorithm.

__Returns__

bool: True.


----

### sendCircuit


```python
Client.sendCircuit(name="temp")
```


Sends a circuit to the backend.

__Arguments__

- __name (str)__: The name of the circuit for the backend.


----

### sendCircuitPrimitive


```python
Client.sendCircuitPrimitive(C, args={"name": "temp"})
```


Sends a NetworkX graph to the backend.
        


----

### sendExecuteReceiveResults


```python
Client.sendExecuteReceiveResults(
    circuitName="temp", dt=1e-05, tmax=1.0, inputProcessors=[], compile=False
)
```


Compiles and sends a circuit for execution in the GFX backend.

__Arguments__

- __circuitName (str)__: The name of the circuit for the backend.
- __compile (bool)__: Whether to compile the circuit first.

__Returns__

bool: Whether the call was successful.


----

### sendNeuropils


```python
Client.sendNeuropils()
```


Pack the list of neuropils into a GFX message.

__Returns__

bool: Whether the messaging has been successful.


----

### sendSVG


```python
Client.sendSVG(name, file)
```


Sends an SVG to the FBL fileserver. Useful for storing data and using loadSVG.

__Arguments__

- __name (str)__: Name to use when saving the file; '_visual' gets automatically appended to it.
- __file (str)__: Path to the SVG file.


----

### updateBackend


```python
Client.updateBackend(type="Null", data="Null")
```


Updates variables in the backend using the data in the Editor.

__Arguments__

- __type (str)__: A string, either "WholeCircuit" or "SingleNeuron", specifying the type of the update.
- __data (str)__: A stringified JSON

__Returns__

bool: Whether the update was successful.


----

### updateSimResultLabel


```python
Client.updateSimResultLabel(result_name, label_dict)
```


----

