"""
This file contains some utilities from Neurowatch for visualizing local files of neurons and meshes in FlyBrainLab.
"""

import json
import pandas as pd

def loadJSON(client, file_name, uname=None, mesh_class = 'Neuropil'):
    """Loads a mesh stored in the .json format.
    # Arguments
        client (FlyBrainLab Client): A FlyBrainLab Client.
        file_name (str): Database ID of the neuron or node.
        scale_factor (float): A scale factor to scale the neuron's dimensions with. Defaults to 1.
        uname (str): Unique name to use in the frontend. Defaults to the file_name.
        mesh_class (str): Mesh class to use. Can be "Neuropil" or "MorphologyData". Defaults to "Neuropil".
    # Examples:
        loadJSON(client, 'cloud.json', uname = 'my_neuropil')

    """
    with open(file_name) as f:
        data = json.load(f)
    if uname == None:
        uname = file_name.split('.')[0]
    rid = '#'+uname
    mesh_data = {'data': {'data': {rid: {'name': uname,
                    'uname': uname,
                    'morph_type': 'mesh',
                    'faces': data['faces'],
                    'vertices': data['vertices'],
                    'class': mesh_class}},
                  'queryID': '0-0'},
                 'messageType': 'Data',
                 'widget': 'NLP'}
    client.tryComms(mesh_data)

def loadSWC(client, file_name, scale_factor=1., uname=None):
    """Loads a neuron skeleton stored in the .swc format.
    # Arguments
        client (FlyBrainLab Client): A FlyBrainLab Client.
        file_name (str): Database ID of the neuron or node.
        scale_factor (float): A scale factor to scale the neuron's dimensions with. Defaults to 1.
        uname (str): Unique name to use in the frontend. Defaults to the file_name.
    """
    neuron_pd = pd.read_csv(file_name,
                    names=['sample','identifier','x','y','z','r','parent'],
                    comment='#',
                    delim_whitespace=True)
    if uname == None:
        uname = file_name.split('.')[0]
    rid = '#'+file_name
    neuron_data = {'data': {'data': {rid: {'name': file_name,
            'uname': uname,
            'morph_type': 'swc',
            'x': list(scale_factor * neuron_pd['x']),
            'y': list(scale_factor * neuron_pd['y']),
            'z': list(scale_factor * neuron_pd['z']),
            'r': list(scale_factor * neuron_pd['r']),
            'parent': list(neuron_pd['parent']),
            'identifier': list(neuron_pd['identifier']),
            'sample': list(neuron_pd['sample']),
            'class': 'MorphologyData'}},
            'queryID': '0-0'},
            'messageType': 'Data',
            'widget': 'NLP'}
    client.tryComms(neuron_data)

    return True