# FlyBrainLab Client

FlyBrainLab Client is a user-side backend implemented in Python that connects to the FFBO processor and accesses services provided by the connected backend servers. FlyBrainLab Client provides program execution APIs for handling requests to the server-side components and parsing of data coming from backend servers. The FlyBrainLab Client also exhibits a number of high-level APIs for processing data collected from the backend servers, such as computing the adjacency matrix from retrieved connectivity data or retrieving morphometrics data. In addition, it handles the communication with the frontend through the Jupyter kernel.

## Installation

To install FlyBrainLab Client, In a terminal, execute the following:

```python
pip install txaio twisted autobahn crochet service_identity autobahn-sync matplotlib h5py seaborn fastcluster networkx
git clone https://github.com/FlyBrainLab/FBLClient.git
cd ./FBLClient
python setup.py develop
```

## Tutorials

Tutorials for learning about how to use FlyBrainLab Client with NeuroMynerva are given in [FlyBrainLab Tutorials](https://github.com/FlyBrainLab/Tutorials).
