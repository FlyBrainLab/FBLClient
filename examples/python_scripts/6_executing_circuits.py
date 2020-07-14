# In this demo, we show how one can perform circuit execution.
# We have an example script for the execution of a cartridge:
# load_retina_lamina

# We can use it as follows:
experiment_configuration = nm[0].load_retina_lamina(cartridgeIndex=126)
experiment_configuration = experiment_configuration["success"]["result"]
nm[0].execute_multilpu(experiment_configuration)
# You will be notified in the front end when the execution results are obtained.
# Simulation data is kept in a stack:
my_sim_data = nm[0].data[-1]
# We can plot our results as follows:
B, keys = nm[0].getSimResults()
nm[0].plotSimResults(B, keys)
# Note that these scripts only provide a bare-bones implementation.
