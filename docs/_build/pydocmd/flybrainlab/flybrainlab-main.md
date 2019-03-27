<h1 id="flybrainlab">flybrainlab</h1>


<h2 id="flybrainlab.Client">Client</h2>

```python
Client(self, ssl=True, debug=True, authentication=True, user='guest', secret='guestpass', custom_salt=None, url='wss://neuronlp.fruitflybrain.org:7777/ws', realm='realm1', ca_cert_file='isrgrootx1.pem', intermediate_cert_file='letsencryptauthorityx3.pem', FFBOLabcomm=None, legacy=False)
```
FlyBrainLab Client class. This class communicates with JupyterLab frontend and connects to FFBO components.

__Attributes:__

    FFBOLabcomm (obj): The communication object for sending and receiving data.
    circuit (obj): A Neuroballad circuit that enables local circuit execution and facilitates circuit modification.
    dataPath (str): Data path to be used.
    experimentInputs (list of dicts): Inputs as a list of dicts that can be parsed by the GFX component.
    compiled (bool): Circuits need to be compiled into networkx graphs before being sent for simulation. This is necessary as circuit compilation is a slow process.
    sendDataToGFX (bool): Whether the data received from the backend should be sent to the frontend. Useful for code-only projects.

<h3 id="flybrainlab.Client.tryComms">tryComms</h3>

```python
Client.tryComms(self, a)
```
Communication function to communicate with a JupyterLab frontend if one exists.

__Arguments:__

    a (obj): Arbitrarily formatted data to be sent via communication.

<h3 id="flybrainlab.Client.findServerIDs">findServerIDs</h3>

```python
Client.findServerIDs(self)
```
Find server IDs to be used for the utility functions.

<h3 id="flybrainlab.Client.executeNLPquery">executeNLPquery</h3>

```python
Client.executeNLPquery(self, query=None, language='en', uri=None, queryID=None, returnNAOutput=False)
```
Execute an NLP query.

__Arguments:__

    query (str): Query string.
    language (str): Language to use.
    uri (str): Currently not used; for future NLP extensions.
    queryID (str): Query ID to be used. Generated automatically.
    returnNAOutput (bool): Whether the corresponding NA query should not be executed.

__Returns:__

    dict: NA output or the NA query itself, depending on the returnNAOutput setting.

<h3 id="flybrainlab.Client.executeNAquery">executeNAquery</h3>

```python
Client.executeNAquery(self, res, language='en', uri=None, queryID=None, progressive=True, threshold=20)
```
Execute an NA query.

__Arguments:__

    res (dict): Neuroarch query.
    language (str): Language to use.
    uri (str): A custom FFBO query URI if desired.
    queryID (str): Query ID to be used. Generated automatically.
    progressive (bool): Whether the loading should be progressive. Needs to be true most of the time for the connection to be stable.
    threshold (int): Data chunk size. Low threshold is required for the connection to be stable.

__Returns:__

    bool: Whether the process has been successful.

<h3 id="flybrainlab.Client.createTag">createTag</h3>

```python
Client.createTag(self, tagName)
```
Creates a tag.

__Returns:__

    bool: True.

<h3 id="flybrainlab.Client.loadTag">loadTag</h3>

```python
Client.loadTag(self, tagName)
```
Loads a tag.

__Returns:__

    bool: True.

<h3 id="flybrainlab.Client.addByUname">addByUname</h3>

```python
Client.addByUname(self, uname, verb='add')
```
Adds some neurons by the uname.

__Returns:__

    bool: True.

<h3 id="flybrainlab.Client.removeByUname">removeByUname</h3>

```python
Client.removeByUname(self, uname)
```
Removes some neurons by the uname.

__Returns:__

    bool: True.

<h3 id="flybrainlab.Client.runLayouting">runLayouting</h3>

```python
Client.runLayouting(self, type='auto', model='auto')
```
Sends a request for the running of the layouting algorithm.

__Returns:__

    bool: True.

<h3 id="flybrainlab.Client.getNeuropils">getNeuropils</h3>

```python
Client.getNeuropils(self)
```
Get the neuropils the neurons in the workspace reside in.

__Returns:__

    list of strings: Set of neuropils corresponding to neurons.

<h3 id="flybrainlab.Client.sendNeuropils">sendNeuropils</h3>

```python
Client.sendNeuropils(self)
```
Pack the list of neuropils into a GFX message.

__Returns:__

    bool: Whether the messaging has been successful.

<h3 id="flybrainlab.Client.getInfo">getInfo</h3>

```python
Client.getInfo(self, args)
```
Get information on a neuron.

__Arguments:__

    args (str): Database ID of the neuron or node.

__Returns:__

    dict: NA information regarding the node.

<h3 id="flybrainlab.Client.GFXcall">GFXcall</h3>

```python
Client.GFXcall(self, args)
```
Arbitrary call to a GFX procedure in the GFX component format.

__Arguments:__

    args (list): A list whose first element is the function name (str) and the following are the arguments.

__Returns:__

    dict OR string: The call result.

<h3 id="flybrainlab.Client.updateBackend">updateBackend</h3>

```python
Client.updateBackend(self, type='Null', data='Null')
```
Updates variables in the backend using the data in the Editor.

__Arguments:__

    type (str): A string, either "WholeCircuit" or "SingleNeuron", specifying the type of the update.
    data (str): A stringified JSON

__Returns:__

    bool: Whether the update was successful.

<h3 id="flybrainlab.Client.getConnectivity">getConnectivity</h3>

```python
Client.getConnectivity(self)
```
Obtain the connectivity matrix of the current circuit in NetworkX format.

__Returns:__

    dict: The connectivity dictionary.

<h3 id="flybrainlab.Client.sendExecuteReceiveResults">sendExecuteReceiveResults</h3>

```python
Client.sendExecuteReceiveResults(self, circuitName='temp', dt=1e-05, tmax=1.0, inputProcessors=[], compile=False)
```
Compiles and sends a circuit for execution in the GFX backend.

__Arguments:__

    circuitName (str): The name of the circuit for the backend.
    compile (bool): Whether to compile the circuit first.

__Returns:__

    bool: Whether the call was successful.

<h3 id="flybrainlab.Client.prepareCircuit">prepareCircuit</h3>

```python
Client.prepareCircuit(self, model='auto')
```
Prepares the current circuit for the Neuroballad format.

<h3 id="flybrainlab.Client.getSlowConnectivity">getSlowConnectivity</h3>

```python
Client.getSlowConnectivity(self)
```
Obtain the connectivity matrix of the current circuit in a custom dictionary format. Necessary for large circuits.

__Returns:__

    dict: The connectivity dictionary.

<h3 id="flybrainlab.Client.sendCircuit">sendCircuit</h3>

```python
Client.sendCircuit(self, name='temp')
```
Sends a circuit to the backend.

__Arguments:__

    name (str): The name of the circuit for the backend.

<h3 id="flybrainlab.Client.processConnectivity">processConnectivity</h3>

```python
Client.processConnectivity(self, connectivity)
```
Processes a Neuroarch connectivity dictionary.

__Returns:__

    tuple: A tuple of nodes, edges and unique edges.

<h3 id="flybrainlab.Client.getSynapses">getSynapses</h3>

```python
Client.getSynapses(self, presynapticNeuron, postsynapticNeuron)
```
Returns the synapses between a given presynaptic neuron and a postsynaptic neuron.

__Arguments:__

    presynapticNeuron (str): The name of the presynaptic neuron.
    postsynapticNeuron (str): The name of the postsynaptic neuron.

__Returns:__

    float: The number of synapses.

<h3 id="flybrainlab.Client.getPresynapticNeurons">getPresynapticNeurons</h3>

```python
Client.getPresynapticNeurons(self, postsynapticNeuron)
```
Returns a dictionary of all presynaptic neurons for a given postsynaptic neuron.

__Arguments:__

    postsynapticNeuron (str): The name of the postsynaptic neuron.

__Returns:__

    dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given postsynaptic neuron.

<h3 id="flybrainlab.Client.getPostsynapticNeurons">getPostsynapticNeurons</h3>

```python
Client.getPostsynapticNeurons(self, presynapticNeuron)
```
Returns a dictionary of all postsynaptic neurons for a given presynaptic neuron.

__Arguments:__

    presynapticNeuron (str): The name of the presynaptic neuron.

__Returns:__

    dict: A dictionary whose keys are the presynaptic neurons and whose values are numbers of synapses for the given presynaptic neuron.

<h3 id="flybrainlab.Client.genNB">genNB</h3>

```python
Client.genNB(self, nodes, edges, model='auto', config={}, default_neuron=<neuroballad.neuroballad.MorrisLecar object at 0x0000024E4696BCC0>, default_synapse=<neuroballad.neuroballad.AlphaSynapse object at 0x0000024E44278B00>)
```
Processes the output of processConnectivity to generate a Neuroballad circuit.

__Returns:__

    tuple: A tuple of the Neuroballad circuit, and a dictionary that maps the neuron names to the uids.

<h3 id="flybrainlab.Client.sendCircuitPrimitive">sendCircuitPrimitive</h3>

```python
Client.sendCircuitPrimitive(self, C, args={'name': 'temp'})
```
Sends a NetworkX graph to the backend.

<h3 id="flybrainlab.Client.alter">alter</h3>

```python
Client.alter(self, X)
```
Alters a set of models with specified Neuroballad models.

__Arguments:__

     X (list of lists): A list of lists. Elements are lists whose first element is the neuron ID (str) and the second is the Neuroballad object corresponding to the model.

<h3 id="flybrainlab.Client.addInput">addInput</h3>

```python
Client.addInput(self, x)
```
Adds an input to the experiment settings. The input is a Neuroballad input object.

__Arguments:__

    x (Neuroballad Input Object): The input object to append to the list of inputs.

__Returns:__

    dict: The input object added to the experiment list.

<h3 id="flybrainlab.Client.listInputs">listInputs</h3>

```python
Client.listInputs(self)
```
Sends the current experiment settings to the frontend for displaying in an editor.

<h3 id="flybrainlab.Client.fetchCircuit">fetchCircuit</h3>

```python
Client.fetchCircuit(self, X, local=True)
```
Deprecated function that locally saves a circuit file via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.Client.fetchExperiment">fetchExperiment</h3>

```python
Client.fetchExperiment(self, X, local=True)
```
Deprecated function that locally saves an experiment file via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.Client.fetchSVG">fetchSVG</h3>

```python
Client.fetchSVG(self, X, local=True)
```
Deprecated function that locally saves an SVG via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.Client.sendSVG">sendSVG</h3>

```python
Client.sendSVG(self, name, file)
```
Sends an SVG to the FBL fileserver. Useful for storing data and using loadSVG.

__Arguments:__

    name (str): Name to use when saving the file; '_visual' gets automatically appended to it.
    file (str): Path to the SVG file.

<h3 id="flybrainlab.Client.loadSVG">loadSVG</h3>

```python
Client.loadSVG(self, name)
```
Loads an SVG in the FBL fileserver.

__Arguments:__

    name (str): Name to use when loading the file.

<h3 id="flybrainlab.Client.FICurveGenerator">FICurveGenerator</h3>

```python
Client.FICurveGenerator(self, model)
```
Sample library function showing how to do automated experimentation using FFBOLab's Notebook features. Takes a simple abstract neuron model and runs experiments on it.

__Arguments:__

    model (Neuroballad Model Object): The model object to test.

__Returns:__

    numpy array: A tuple of NumPy arrays corresponding to the X and Y of the FI curve.

<h3 id="flybrainlab.Client.parseSimResults">parseSimResults</h3>

```python
Client.parseSimResults(self)
```
Parses the simulation results. Deprecated.

<h3 id="flybrainlab.Client.getSimResults">getSimResults</h3>

```python
Client.getSimResults(self)
```
Computes the simulation results.

__Returns:__

    numpy array: A neurons-by-time array of results.
    list: A list of neuron names, sorted according to the data.

<h3 id="flybrainlab.Client.plotSimResults">plotSimResults</h3>

```python
Client.plotSimResults(self, B, keys)
```
Plots the simulation results. A simple function to demonstrate result display.

__Arguments:__

    model (Neuroballad Model Object): The model object to test.

<h3 id="flybrainlab.Client.FICurvePlotSimResults">FICurvePlotSimResults</h3>

```python
Client.FICurvePlotSimResults(self)
```
Plots some result curves for the FI curve generator example.

<h3 id="flybrainlab.Client.loadCartridge">loadCartridge</h3>

```python
Client.loadCartridge(self, cartridgeIndex=100)
```
Sample library function for loading cartridges, showing how one can build libraries that work with flybrainlab.

<h3 id="flybrainlab.Client.loadExperimentConfig">loadExperimentConfig</h3>

```python
Client.loadExperimentConfig(self, x)
```
Updates the simExperimentConfig attribute using input from the diagram.

__Arguments:__

    x (string): A JSON dictionary as a string.

__Returns:__

    bool: True.

<h3 id="flybrainlab.Client.initiateExperiments">initiateExperiments</h3>

```python
Client.initiateExperiments(self)
```
Initializes and executes experiments for different LPUs.

<h3 id="flybrainlab.Client.prune_retina_lamina">prune_retina_lamina</h3>

```python
Client.prune_retina_lamina(self, removed_neurons=[], removed_labels=[], retrieval_format='nk')
```
Prunes the retina and lamina circuits.

__Arguments:__

    cartridgeIndex (int): The cartridge to load. Optional.

__Returns:__

    dict: A result dict to use with the execute_lamina_retina function.

__Example:__

    res = nm[0].load_retina_lamina()
    nm[0].execute_multilpu(res)

<h3 id="flybrainlab.Client.load_retina_lamina">load_retina_lamina</h3>

```python
Client.load_retina_lamina(self, cartridgeIndex=11, removed_neurons=[], removed_labels=[], retrieval_format='nk')
```
Loads retina and lamina.

__Arguments:__

    cartridgeIndex (int): The cartridge to load. Optional.

__Returns:__

    dict: A result dict to use with the execute_lamina_retina function.

__Example:__

    nm[0].getExperimentConfig() # In a different cell
    experiment_configuration = nm[0].load_retina_lamina(cartridgeIndex=126)
    experiment_configuration = experiment_configuration['success']['result']
    nm[0].execute_multilpu(experiment_configuration)

<h3 id="flybrainlab.Client.execute_multilpu">execute_multilpu</h3>

```python
Client.execute_multilpu(self, res, inputProcessors=[], steps=None, dt=None)
```
Executes a multilpu circuit. Requires a result dictionary.

__Arguments:__

    res (dict): The result dictionary to use for execution.

__Returns:__

    bool: A success indicator.

<h3 id="flybrainlab.Client.export_diagram_config">export_diagram_config</h3>

```python
Client.export_diagram_config(self, res)
```
Exports a diagram configuration from Neuroarch data to GFX.

__Arguments:__

    res (dict): The result dictionary to use for export.

__Returns:__

    dict: The configuration to export.

<h3 id="flybrainlab.Client.import_diagram_config">import_diagram_config</h3>

```python
Client.import_diagram_config(self, res, newConfig)
```
Imports a diagram configuration from Neuroarch data.

__Arguments:__

    res (dict): The result dictionary to update.
    newConfig (dict): The imported configuration from a diagram.

__Returns:__

    dict: The updated Neuroarch result dictionary.

