# In this example, we show how you can build a dendrogram of your current circuit.
# First, let us start by querying a sufficiently small circuit:
nm[0].executeNLPquery("show GABAergic neurons in the ellipsoid body")
# We can then get a connectivity dendrogram simply by running:
nm[0].getConnectivityDendrogram()
# How does this work? This function runs several functions in order.
# First, it runs:
nm[0].prepareCircuit()
# This generates a connectivity matrix by querying FFBO and
# builds an internal representation of the circuit.
# Secondly, it generates a connectivity matrix:
M = nm[0].getConnectivityMatrix()
# Finally, it uses the seaborn library to generate a nice visualization:
import pandas as pd
import seaborn as sns

M = pd.DataFrame(M, index=nm[0].out_nodes, columns=nm[0].out_nodes)
sns.clustermap(M)
