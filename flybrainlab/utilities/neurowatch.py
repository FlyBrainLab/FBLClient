"""
This file contains some utilities from Neurowatch for visualizing local files of neurons and meshes in FlyBrainLab.
"""

import json
import time
import matplotlib.cm as cm
import matplotlib.colors as mc
import pandas as pd
import numpy as np

DEFAULT_COLORS = [mc.rgb2hex(n) for n in cm.hsv([0, 27, 53, 80, 107, 133, 160, 187, 213, 240])]


def loadJSON(client, file_name, scale_factor=1., name=None, mesh_class = 'Neuropil'):
    """Loads a mesh stored in the .json format.
    # Arguments
        client (FlyBrainLab Client): A FlyBrainLab Client.
        file_name (str): Database ID of the neuron or node.
        scale_factor (float): A scale factor to scale the neuron's dimensions with. Defaults to 1.
        name (str): Unique name to use in the frontend. Defaults to the file_name.
        mesh_class (str): Mesh class to use. Can be "Neuropil" or "MorphologyData". Defaults to "Neuropil".
    # Examples:
        loadJSON(client, 'cloud.json', uname = 'my_neuropil')

    """
    with open(file_name) as f:
        data = json.load(f)
    if name == None:
        name = file_name.split('.')[0]
    rid = '#'+name
    visualize_mesh(client, data, rid, name = name, mesh_class = mesh_class, 
                   scale_factor = scale_factor)
    return rid
    
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
    visualize_neuron(client, neuron_pd, rid, uname = uname, scale_factor = scale_factor)
    return rid


def send_data_to_NLP(client, data):
    a = {'messageType': 'Data', 'widget': 'NLP',
         'data': {'data': data, "queryID": '0-0'}}
    client.tryComms(a)


def color_by_rids(client, rids, rgb):
    if isinstance(rids, str):
        rids = [rids]
    a = {'messageType': 'Command', 'widget': 'NLP',
         'data': {'commands': {'setcolor': [rids, rgb]}}}
    client.tryComms(a)


def color_group(client, rids, user_color = None, colormap = None, colors = None):
    if user_color is None or len(user_color) == 0:
        color_type = 'list'
        if colors is None:
            colors = DEFAULT_COLORS
    else:
        color_types = [isinstance(n, str) for n in user_color.values()]
        if any(color_types):
            if all(color_types):
                color_type = 'rgb'
            else:
                raise ValueError('The values of colors are not all in hex code RGB or all in float')
        else:
            if not all(color_types):
                color_type = 'colormap'
            else:
                raise ValueError('The values of colors are not all in hex code RGB or all in float')        
    
    if color_type == 'list':
        rid_list = list(rids.keys())
        for i, rgb in enumerate(colors):
            color_by_rids(client, [rids[n] for n in rid_list[i::len(colors)]], rgb)
            if i%500 == 0:
                time.sleep(0.5)
    elif color_type == 'rgb':
        for i, rid in enumerate(user_color):
            color_by_rids(client, rid, user_color[rid])
            if i%500 == 0:
                time.sleep(0.5)
    else: #color_type == 'colormap'
        min_value = min(list(user_color.values()))
        max_value = max(list(user_color.values()))
        if colormap is None:
            colormap = cm.gnuplot
        for i, (rid, v) in enumerate(user_color.items()):
            if min_value == max_value:
                c = 1.0
            else:
                c = (v-min_value)/(max_value-min_value)
            color_by_rids(client, rid, mc.rgb2hex(colormap(c)))
            if i%500 == 0:
                time.sleep(0.5)


def remove_by_rids(client, rids):
    if isinstance(rids, str):
        rids = [rids]
    else:
        rids = list(rids)
    a = {'messageType': 'Command', 'widget': 'NLP',
         'data': {'commands': {'remove': [rids, []]}}}
    client.tryComms(a)
    return rids


def points_to_morphology(points, r = 0.2, scale_factor = None,
                         x_scale = 1.0, y_scale = 1.0,
                         z_scale = 1.0, r_scale = 1.0, 
                         x_shift = 0.0, y_shift = 0.0,
                         z_shift = 0.0, r_shift = 0.0):
    if scale_factor is not None:
        x_scale = scale_factor
        y_scale = scale_factor
        z_scale = scale_factor
        r_scale = scale_factor
    morphology_data = {'morph_type': 'swc', 'class': 'Synapse'}
    length = len(points)
    morphology_data['x'] = [n[0]*x_scale+x_shift for n in points]
    morphology_data['y'] = [n[1]*y_scale+y_shift for n in points]
    morphology_data['z'] = [n[2]*z_scale+z_shift for n in points]
    morphology_data['r'] = [r*r_scale+r_shift]*length
    morphology_data['identifier'] = [7]*length
    morphology_data['sample'] = list(range(1, length+1))
    morphology_data['parent'] = [-1]*length
    return morphology_data


def neuron_data_to_morphology(data, scale_factor = None,
                              x_scale = 1.0, y_scale = 1.0,
                              z_scale = 1.0, r_scale = 1.0, 
                              x_shift = 0.0, y_shift = 0.0,
                              z_shift = 0.0, r_shift = 0.0):
    if scale_factor is not None:
        x_scale = scale_factor
        y_scale = scale_factor
        z_scale = scale_factor
        r_scale = scale_factor
    morphology_data = {'morph_type': 'swc', 'class': 'Neuron'}
    assert isinstance(data, (dict, pd.DataFrame)), 'data must be either dict or a pandas DataFrame containing the field "sample/index", "identifier", "x", "y", "z", "r" and "parent"'
    morphology_data['x'] = [n*x_scale+x_shift for n in data['x']]
    morphology_data['y'] = [n*y_scale+y_shift for n in data['y']]
    morphology_data['z'] = [n*z_scale+z_shift for n in data['z']]
    morphology_data['r'] = [n*r_scale+r_shift for n in data['r']]
    morphology_data['identifier'] = list(data['identifier'])
    morphology_data['parent'] = list(data['parent'])
    if 'sample' in data:
        morphology_data['sample'] = list(data['sample'])
    else: # assume the sample is the index
        morphology_data['sample'] = list(data.index)
    return morphology_data


def visualize_synapses(client, points, rid, 
                       uname = None, r = 0.2, color = None,
                       scale_factor = None,
                       x_scale = 1.0, y_scale = 1.0, z_scale = 1.0, r_scale = 1.0, 
                       x_shift = 0.0, y_shift = 0.0, z_shift = 0.0, r_shift = 0.0):
    if uname is None:
        uname = 'synapses_{}'.format(rid)
    morphology_data = points_to_morphology(
                        points, r = r, 
                        scale_factor = scale_factor,
                        x_scale = x_scale, y_scale = y_scale,
                        z_scale = z_scale, r_scale = r_scale,
                        x_shift = x_shift, y_shift = y_shift,
                        z_shift = z_shift, r_shift = r_shift)
    morphology_data['uname'] = uname
    data = {rid: morphology_data}
    send_data_to_NLP(client, data)

    if color is not None:
        color_by_rids(client, rid, color)
    return rid


def visualize_neuron(client, data, rid, uname = None, color = None,
                     scale_factor = None,
                     x_scale = 1.0, y_scale = 1.0, z_scale = 1.0, r_scale = 1.0, 
                     x_shift = 0.0, y_shift = 0.0, z_shift = 0.0, r_shift = 0.0):
    if uname is None:
        uname = 'neuron_{}'.format(rid)
    morphology_data = neuron_data_to_morphology(
                        data, scale_factor = scale_factor,
                        x_scale = x_scale, y_scale = y_scale,
                        z_scale = z_scale, r_scale = r_scale,
                        x_shift = x_shift, y_shift = y_shift,
                        z_shift = z_shift, r_shift = r_shift)
    morphology_data['uname'] = uname
    data = {rid: morphology_data}

    send_data_to_NLP(client, data)
    if color is not None:
        color_by_rids(client, rid, color)
    return rid


def visualize_mesh(client, data, rid, name = None, 
                   mesh_class = 'Neuropil', color = None,
                   scale_factor = None,
                   x_scale = 1.0, y_scale = 1.0,
                   z_scale = 1.0, x_shift = 0.0,
                   y_shift = 0.0, z_shift = 0.0):
    if scale_factor is not None:
        x_scale = scale_factor
        y_scale = scale_factor
        z_scale = scale_factor
    scale = [x_scale, y_scale, z_scale]
    shift = [x_shift, y_shift, z_shift]
    if name is None:
        name = '{}_{}'.format(mesh_class, rid)
    morphology_data = {'morph_type': 'mesh', 'class': mesh_class,
                       'name': name}
    assert isinstance(data, (dict, pd.DataFrame)), 'data must be either dict or a pandas DataFrame containing the field "faces" and "vertices"'
    
    morphology_data['faces'] = list(data['faces'])
    morphology_data['vertices'] = [n*scale[i%3]+shift[i%3] for i, n in enumerate(data['vertices'])]
    data = {rid: morphology_data}

    send_data_to_NLP(client, data)
    if color is not None:
        color_by_rids(client, rid, color)
    return rid


def visualize_synapses_group(client, synapses_group,
                             r = 0.2, colormap = None, 
                             colors = None, scale_factor = None,
                             x_scale = 1.0, y_scale = 1.0,
                             z_scale = 1.0, r_scale = 1.0, 
                             x_shift = 0.0, y_shift = 0.0,
                             z_shift = 0.0, r_shift = 0.0):
    rids = {}
    user_color = {}
    all_data = {}
    for uname, data in synapses_group.items():
        rid = data.get('rid', '#{}'.format(uname))
        morphology_data = points_to_morphology(
                            data['points'], r = r,
                            scale_factor = scale_factor,
                            x_scale = x_scale, y_scale = y_scale,
                            z_scale = z_scale, r_scale = r_scale,
                            x_shift = x_shift, y_shift = y_shift,
                            z_shift = z_shift, r_shift = r_shift)
        morphology_data['uname'] = uname
        all_data[rid] = morphology_data
        if 'color' in data:
            user_color[rid] = data['color']
        rids[uname] = rid
    send_data_to_NLP(client, all_data)

    if len(user_color) and len(user_color) != len(all_data):
        if isinstance(user_color[user_color.keys()[0]], str):
            color_type == 'rgb'
        else:
            color_type = 'colormap'
        for rid in all_data:
            if rid not in user_color:
                if color_type == 'rgb':
                    user_color[rid] = '#FFFFFF'
                else:
                    user_color[rid] = 0.0
    color_group(client, rids, user_color = user_color, colormap = colormap, colors = colors)
    return rids


def visualize_neuron_group(client, neuron_group,
                           colormap = None, colors = None,
                           scale_factor = None,
                           x_scale = 1.0, y_scale = 1.0,
                           z_scale = 1.0, r_scale = 1.0, 
                           x_shift = 0.0, y_shift = 0.0,
                           z_shift = 0.0, r_shift = 0.0):
    """
    Visualize a group of neurons.

    # Arguments
    neuron_group (dict):
        Specifying the morphology of the neuron keyed by its uname, in the form of:
        neuron_group = {'morphology': pandas.DataFrame or dict in swc format,
                        'rid': rid of the neuron (optional),
                        'color': hex code starting with '#' for RGB value or float (optional)}
        If color is a hex code, the color will be used to color.
        If color is a float, color will use a colors as a colormap with the mininum of all color value
        mapped to the lowest value of the colormap, and maximum mapped to the highest value of colormap.
    colormap (matplotlib.colormap.cm):
        A colormap to use when the color are specified in float
    colors (list):
        A list of hex code of colors to use in round robin fashion.
    scale_factor (float):
        If specified, scaling of the x, y, z and r of the neuron will all be overwritten by this value.
    x_scale, y_scale, z_scale, r_scale (float):
        scaling of the x, y, z and r of the neuron will all be overwritten by this value.
    x_shift, y_shift, z_shift, r_shift (float):
        shifting of the x, y, z position values and a bias for r.
    """
    rids = {}
    user_color = {}
    all_data = {}
    for uname, data in neuron_group.items():
        rid = data.get('rid', '#{}'.format(uname))
        morphology_data = neuron_data_to_morphology(
                            data['morphology'],
                            scale_factor = scale_factor,
                            x_scale = x_scale, y_scale = y_scale,
                            z_scale = z_scale, r_scale = r_scale,
                            x_shift = x_shift, y_shift = y_shift,
                            z_shift = z_shift, r_shift = r_shift)
        morphology_data['uname'] = uname
        all_data[rid] = morphology_data
        if 'color' in data:
            user_color[rid] = data['color']
        rids[uname] = rid
    send_data_to_NLP(client, all_data)

    if len(user_color) and len(user_color) != len(all_data):
        if isinstance(user_color[user_color.keys()[0]], str):
            color_type == 'rgb'
        else:
            color_type = 'colormap'
        for rid in all_data:
            if rid not in user_color:
                if color_type == 'rgb':
                    user_color[rid] = '#FFFFFF'
                else:
                    user_color[rid] = 0.0
    color_group(client, rids, user_color = user_color, colormap = colormap, colors = colors)
    return rids



    