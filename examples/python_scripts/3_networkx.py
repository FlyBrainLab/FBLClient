# For using both FFBO and FBL efficiently and correctly, you should learn
# to use NetworkX, a graph library for Python.
# You can access the documentation here:
# https://networkx.github.io/documentation/stable/

# Let us start with a basic query and get the connectivity dendrogram:
nm[0].executeNLPquery("show GABAergic neurons in the ellipsoid body")
nm[0].getConnectivityDendrogram()
# We can now access the circuit graph as follows:
circuit_graph = nm[0].C.G
# We can for example look at a list of nodes:
print(circuit_graph.nodes(data=True))
# Or the synapses:
print(circuit_graph.edges(data=True))
# This circuit is automatically generated in a Neurokernel-executable format.

# For simulations, it is recommended that you start to look at Neurodriver:
# https://github.com/neurokernel/neurodriver
