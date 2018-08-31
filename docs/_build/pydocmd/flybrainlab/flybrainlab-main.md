<h1 id="flybrainlab">flybrainlab</h1>


<h2 id="flybrainlab.ffbolabClient">ffbolabClient</h2>

```python
ffbolabClient(self, ssl=True, debug=True, authentication=True, user='guest', secret='guestpass', url='wss://neuronlp.fruitflybrain.org:7777/ws', realm='realm1', ca_cert_file='isrgrootx1.pem', intermediate_cert_file='letsencryptauthorityx3.pem', FFBOLabcomm=None)
```
FFBOLab Client class. This class communicates with JupyterLab frontend and connects to FFBO components.

__Attributes:__

    FFBOLabcomm (obj): The communication object for sending and receiving data.
    circuit (obj): A Neuroballad circuit that enables local circuit execution and facilitates circuit modification.
    dataPath (str): Data path to be used.
    experimentInputs (list of dicts): Inputs as a list of dicts that can be parsed by the GFX component.
    compiled (bool): Circuits need to be compiled into networkx graphs before being sent for simulation. This is necessary as circuit compilation is a slow process.
    sendDataToGFX (bool): Whether the data received from the backend should be sent to the frontend. Useful for code-only projects.

<h3 id="flybrainlab.ffbolabClient.tryComms">tryComms</h3>

```python
ffbolabClient.tryComms(self, a)
```
Communication function to communicate with a JupyterLab frontend if one exists.

__Arguments:__

    a (obj): Arbitrarily formatted data to be sent via communication.

<h3 id="flybrainlab.ffbolabClient.findServerIDs">findServerIDs</h3>

```python
ffbolabClient.findServerIDs(self)
```
Find server IDs to be used for the utility functions.

<h3 id="flybrainlab.ffbolabClient.executeNLPquery">executeNLPquery</h3>

```python
ffbolabClient.executeNLPquery(self, query=None, language='en', uri=None, queryID=None, returnNAOutput=False)
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

<h3 id="flybrainlab.ffbolabClient.executeNAquery">executeNAquery</h3>

```python
ffbolabClient.executeNAquery(self, res, language='en', uri=None, queryID=None, progressive=True, threshold=5)
```
Execute an NA query.

__Arguments:__

    res (dict): Neuroarch query.
    language (str): Language to use.
    uri (str): A custom FFBO query URI if desired.
    queryID (str): Query ID to be used. Generated automatically.
    progressive (bool): Whether the loading should be progressive. Needs to be true most of the time for connection to be stable.
    threshold (int): Data chunk size. Low threshold is required for the connection to be stable.

__Returns:__

    bool: Whether the process has been successful.

<h3 id="flybrainlab.ffbolabClient.getNeuropils">getNeuropils</h3>

```python
ffbolabClient.getNeuropils(self)
```
Get the neuropils the neurons in the workspace reside in.

__Returns:__

    list of strings: Set of neuropils corresponding to neurons.

<h3 id="flybrainlab.ffbolabClient.sendNeuropils">sendNeuropils</h3>

```python
ffbolabClient.sendNeuropils(self)
```
Pack the list of neuropils into a GFX message.

__Returns:__

    bool: Whether the messaging has been successful.

<h3 id="flybrainlab.ffbolabClient.getInfo">getInfo</h3>

```python
ffbolabClient.getInfo(self, args)
```
Get information on a neuron.

__Arguments:__

    args (str): Database ID of the neuron or node.

__Returns:__

    dict: NA information regarding the node.

<h3 id="flybrainlab.ffbolabClient.GFXcall">GFXcall</h3>

```python
ffbolabClient.GFXcall(self, args)
```
Arbitrary call to a GFX procedure in the GFX component format.

__Arguments:__

    args (list): A list whose first element is the function name (str) and the following are the arguments.

__Returns:__

    dict OR string: The call result.

<h3 id="flybrainlab.ffbolabClient.updateBackend">updateBackend</h3>

```python
ffbolabClient.updateBackend(self, type='Null', data='Null')
```
Updates variables in the backend using the data in the Editor.

__Arguments:__

    type (str): A string, either "WholeCircuit" or "SingleNeuron", specifying the type of the update.
    data (str): A stringified JSON

__Returns:__

    bool: Whether the update was successful.

<h3 id="flybrainlab.ffbolabClient.getConnectivity">getConnectivity</h3>

```python
ffbolabClient.getConnectivity(self)
```
Obtain the connectivity matrix of the current circuit in NetworkX format.

__Returns:__

    dict: The connectivity dictionary.

<h3 id="flybrainlab.ffbolabClient.sendExecuteReceiveResults">sendExecuteReceiveResults</h3>

```python
ffbolabClient.sendExecuteReceiveResults(self, circuitName='temp', dt=1e-05, tmax=1.0, compile=False)
```
Compiles and sends a circuit for execution in the GFX backend.

__Arguments:__

    circuitName (str): The name of the circuit for the backend.
    compile (bool): Whether to compile the circuit first.

__Returns:__

    bool: Whether the call was successful.

<h3 id="flybrainlab.ffbolabClient.prepareCircuit">prepareCircuit</h3>

```python
ffbolabClient.prepareCircuit(self, model='auto')
```
Prepares the current circuit for the Neuroballad format.

<h3 id="flybrainlab.ffbolabClient.sendCircuit">sendCircuit</h3>

```python
ffbolabClient.sendCircuit(self, name='temp')
```
Sends a circuit to the backend.

__Arguments:__

    name (str): The name of the circuit for the backend.

<h3 id="flybrainlab.ffbolabClient.processConnectivity">processConnectivity</h3>

```python
ffbolabClient.processConnectivity(self, connectivity)
```
Processes a Neuroarch connectivity dictionary.

__Returns:__

    tuple: A tuple of nodes, edges and unique edges.

<h3 id="flybrainlab.ffbolabClient.GenNB">GenNB</h3>

```python
ffbolabClient.GenNB(self, nodes, edges, model='auto', config={}, default_neuron=<neuroballad.neuroballad.MorrisLecar object at 0x000002453D7F55F8>, default_synapse=<neuroballad.neuroballad.AlphaSynapse object at 0x000002454452CC88>)
```
Processes the output of processConnectivity to generate a Neuroballad circuit

__Returns:__

    tuple: A tuple of the Neuroballad circuit, and a dictionary that maps the neuron names to the uids.

<h3 id="flybrainlab.ffbolabClient.sendCircuitPrimitive">sendCircuitPrimitive</h3>

```python
ffbolabClient.sendCircuitPrimitive(self, C, args={'name': 'temp'})
```
Sends a NetworkX graph to the backend.

<h3 id="flybrainlab.ffbolabClient.alter">alter</h3>

```python
ffbolabClient.alter(self, X)
```
Alters a set of models with specified Neuroballad models.

__Arguments:__

     X (list of lists): A list of lists. Elements are lists whose first element is the neuron ID (str) and the second is the Neuroballad object corresponding to the model.

<h3 id="flybrainlab.ffbolabClient.addInput">addInput</h3>

```python
ffbolabClient.addInput(self, x)
```
Adds an input to the experiment settings. The input is a Neuroballad input object.

__Arguments:__

    x (Neuroballad Input Object): The input object to append to the list of inputs.

__Returns:__

    dict: The input object added to the experiment list.

<h3 id="flybrainlab.ffbolabClient.listInputs">listInputs</h3>

```python
ffbolabClient.listInputs(self)
```
Sends the current experiment settings to the frontend for displaying in the JSONEditor.

<h3 id="flybrainlab.ffbolabClient.fetchCircuit">fetchCircuit</h3>

```python
ffbolabClient.fetchCircuit(self, X, local=True)
```
Deprecated function that locally saves a circuit file via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.ffbolabClient.fetchExperiment">fetchExperiment</h3>

```python
ffbolabClient.fetchExperiment(self, X, local=True)
```
Deprecated function that locally saves an experiment file via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.ffbolabClient.fetchSVG">fetchSVG</h3>

```python
ffbolabClient.fetchSVG(self, X, local=True)
```
Deprecated function that locally saves an SVG via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.ffbolabClient.sendSVG">sendSVG</h3>

```python
ffbolabClient.sendSVG(self, X)
```
Deprecated function that loads an SVG via the backend.
Deprecated because of connectivity issues with large files.

<h3 id="flybrainlab.ffbolabClient.FICurveGenerator">FICurveGenerator</h3>

```python
ffbolabClient.FICurveGenerator(self, model)
```
Sample library function showing how to do automated experimentation using FFBOLab's Notebook features. Takes a simple abstract neuron model and runs experiments on it.

__Arguments:__

    model (Neuroballad Model Object): The model object to test.

__Returns:__

    numpy array: A tuple of NumPy arrays corresponding to the X and Y of the FI curve.

<h3 id="flybrainlab.ffbolabClient.loadCartridge">loadCartridge</h3>

```python
ffbolabClient.loadCartridge(self, cartridgeIndex=100)
```
Sample library function for loading cartridges, showing how one can build libraries that work with FFBOLab.

