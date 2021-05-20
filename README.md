# FlyBrainLab Client

FlyBrainLab Client is a user-side backend implemented in Python that connects to the FFBO processor and accesses services provided by the connected backend servers. FlyBrainLab Client provides program execution APIs for handling requests to the server-side components and parsing of data coming from backend servers.

## Documentation

FlyBrainLab Client documentation is available at https://flybrainlab.github.io/FBLClient/.

## Installation

### Quick Installation

Up-to-date installation instructions for the whole FlyBrainLab ecosystem are available at https://github.com/FlyBrainLab/FlyBrainLab/blob/master/README.md. Follow the steps below for a manual installation of the front-end that may not be up-to-date.

### Manual Installation

To install FlyBrainLab Client, in a terminal window, execute the following:

```python
pip install git+https://github.com/mkturkcan/autobahn-sync.git git+https://github.com/FlyBrainLab/Neuroballad.git flybrainlab
```

To install FlyBrainLab with all optional dependencies for the utilities library, execute the following:

```python
pip install git+https://github.com/mkturkcan/autobahn-sync.git git+https://github.com/FlyBrainLab/Neuroballad.git git+https://github.com/palash1992/GEM.git git+https://github.com/mkturkcan/nxcontrol flybrainlab[full]
```

To install FlyBrainLab Client from a clone of this repository, execute:

```python
pip install .[full]
```

## Tutorials

Tutorials for learning about how to use FlyBrainLab Client with NeuroMynerva are given in [FlyBrainLab Tutorials](https://github.com/FlyBrainLab/Tutorials).
