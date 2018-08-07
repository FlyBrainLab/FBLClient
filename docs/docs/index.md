# Neuroballad: <small>A Neural Circuit Simulation Library</small>
## Introduction
Neuroballad is a high level API for simulation of biological neural networks on GPU's, built to facilitate development of circuit-level implementations. Written in Python and on top of [Neurokernel](https://github.com/neurokernel/neurokernel)/[Neurodriver](https://github.com/neurokernel/neurodriver), Neuroballad focuses on user experience and attempts to minimize the time required to execute a desired circuit design.

Neuroballad is in its early stages of development at this time and therefore its use in most circumstances is not recommended.
## Installation
Neuroballad is built on and requires a working [Neurokernel](https://github.com/neurokernel/neurokernel)/[Neurodriver](https://github.com/neurokernel/neurodriver) installation. For simplicity, we will focus on a Conda-based installation procedure here; other installation methods can be found on the [Neurokernel](https://github.com/neurokernel/neurokernel) page. To start with Neurokernel, first, add the neurokernel channel to your ~/.condarc file:
``` bash
channels:
- https://conda.binstar.org/neurokernel/channel/ubuntu1404
- defaults
```
Afterwards, you can install Neurokernel and Neurodriver as follows:
``` bash
conda create -n NK neurokernel_deps
source activate NK
# Remember to edit the following line depending on your CUDA version
conda install pycuda=2015.1.3=np110py27_cuda75_0
cd ~
git clone https://github.com/neurokernel/neurokernel.git
cd ~/neurokernel
python setup.py develop
cd ~
git clone https://github.com/neurokernel/neurodriver
cd ~/neurodriver
python setup.py develop
```
Finally, install Neuroballad:
``` shell
cd ~
git clone https://github.com/mkturkcan/neuroballad
cd ~/neuroballad
python setup.py develop
```
## Getting Started
It is easy to start working with Neuroballad; the following example utilizes three Hodgkin-Huxley neurons connected circularly via alpha synapses:
``` python
# Import Neuroballad
from neuroballad import *
# Create a circuit
C = Circuit()
# Create three neurons
C.add([0, 2, 4], HodgkinHuxley())
# Create three synapses
C.add([1, 3, 5], AlphaSynapse())
# Join the nodes together, using an edge list notation
# (i.e. [0,1] implies a directed edge from node 0 to node 1)
C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
# Create a current input for node 0 from t=0.25s to t=0.50s with an amplitude of 40mA.
C_in_a = InIStep(0, 40., 0.20, 0.40)
# Create inputs, similarly, for nodes 2 and 4.
C_in_b = InIStep(2, 40., 0.40, 0.60)
C_in_c = InIStep(4, 40., 0.60, 0.80)
# Use the three inputs and simulate the circuit for a second, in 1e-4s time steps.
C.sim(1., 1e-4, [C_in_a, C_in_b, C_in_c])
```
Neuroballad includes a number of additional methods for easily dealing with larger systems, the details of which you can find in the upcoming sections.
