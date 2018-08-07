# Tutorial

This tutorial will introduce you to Neuroballad by showing a number of examples demonstrating its (current) capabilities.

## Hello World
Let us start by injecting constant current to a neuron for a set amount of time.

We will utilize a Hudgkin-Huxley neuron and feed it with constant current using a "step function", usually referred to as a [boxcar function](https://en.wikipedia.org/wiki/Boxcar_function). Such a function is defined by two parameters indicating the start and the end times of the pulse and as well as an another parameter specifying the amplitude.

Specification and simulation of such a neuron can be accomplished in 5 lines with Neuroballad:
``` python
# Import Neuroballad
from neuroballad import *
# Create a circuit
C = Circuit()
# Create a neuron
C.add([0], HodgkinHuxley())
# Create a current input for node 0 from t=0.25s to t=0.75s with an amplitude of 40mA.
C_in = InIStep(0, 40., 0.25, 0.75)
# Simulate the circuit for a second, in 1e-4s time steps.
C.sim(1., 1e-4, [C_in])
```
## Three Neurons

Coming soon!

## Densely Connected Clusters of Neurons

Coming soon!

## Neuromodulation

Coming soon!
