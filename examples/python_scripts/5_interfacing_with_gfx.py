# In this demo we will show the basics of interfacing with GFX. 

# Enter into lamina (LAM), then double-click one of the circles ('cartridges').
# Click on some neurons to enable/disable them.
# You can look at what your changes have caused by printing 'simExperimentConfig':
print(nm[0].simExperimentConfig)
# You can use the following two functions to update results from NeuroArch:
# nm[0].export_diagram_config
# nm[0].import_diagram_config

# Your queries will be reflected on the diagram if possible:
nm[0].executeNLPquery('remove L1')