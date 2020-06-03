# This example demonstrates the workflow for FBL scripts.
# FBL uses Python to interface with an FFBO backend.
# By default, each FBL kernel connects to two backends:
# - A default backend for the adult fly.
# - A second backend for the larva.
# A Client object is generated for each.

# You can address the two clients independently as follows:
client_adult = nm[0]  # This is the adult client
client_larva = nm[1]  # This is the larva client
# Execute an NLP query directly:
nm[0].executeNLPquery("show neurons in the ellipsoid body")
# Auto-completion in JupyterLab will be helpful.
# Sometimes you may need to import FBL directly:
import flybrainlab as fbl

# And create a new client:
my_client = fbl.Client()
# You can write scripts to override the default client:
nm[0] = my_client
