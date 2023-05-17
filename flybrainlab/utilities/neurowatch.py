"""
This file contains some utilities from Neurowatch for visualizing local files of neurons and meshes in FlyBrainLab.
"""

import os
import json
import time
import random
import matplotlib.cm as cm
import matplotlib.colors as mc
import pandas as pd
import numpy as np

DEFAULT_COLORS = [mc.rgb2hex(n) for n in cm.hsv([0, 27, 53, 80, 107, 133, 160, 187, 213, 240])]
DEFAULT_MESH_COLOR = '#260226'
DEFAULT_NEURON_COLORS = [mc.rgb2hex(n) for n in cm.Set2.colors]
DEFAULT_SYNAPSE_COLORS = [mc.rgb2hex(n) for n in cm.hsv([0, 27, 53, 80, 107, 133, 160, 187, 213, 240])]

class NeuroWatch(object):
    """
    Visualization handle
    
    # Arguments
        client (flybrainlab.Client.Client)
            Client object to specify the neu3d widget to use.
            
    # Examples
        import fruitflybrain.utilities.neurowatch as nw
        watch = nw.NeuroWatch(fbl.get_client())
        # see docstring for argument format.
        watch.add_mesh(...)
        watch.add_neuron(...)
        watch.add_synapses(...)
        watch.visualize()
        watch.hide([uname_of_neurons/mesh/synapses,...])
        watch.pin([uname_of_neurons/mesh/synapses,...])
        watch.remove() 
    """
    def __init__(self, client):
        self.client = client
        self.meshes = {}
        self.neurons = {}
        self.synapses = {}
        self.colors = {}
        self._uname_to_rid = {}
        self._rids = set()

    def add_mesh(self, data, name = None, 
                 mesh_class = 'Neuropil', color = None,
                 scale_factor = None,
                 x_scale = 1.0, y_scale = 1.0,
                 z_scale = 1.0, x_shift = 0.0,
                 y_shift = 0.0, z_shift = 0.0):
        """
        Add mesh to the visualization.
        
        # Arguments
        data (dict)
            A dictionary containing 'faces', 'vertices' as usually
            used in the trimesh specification.
            values should be a single, rastered list of values,
            with every 3 consecutive entries of 'vertices' representing
            the x,y,z coordinates of a vertex, and every 3 consecutive
            entries of 'faces' specifying the 3 vertices to triangulate.
        name (str)
            name to be used for the mesh.
            This name will be displayed in the meshes list.
        mesh_class (str)
            Either 'Neuropil' or 'Subregion'.
        color (str)
            RGB Hex code for the color to be used for these points.
        scale_factor (float or None)
            A single scaling factor for x, y, z and r.
            If scales are different for each axis,
            use x_scale, y_scale, z_scale and r_scale individually.
        x_scale (float)
            scaling factor for x coordinate.
        y_scale (float)
            scaling factor for y coordinate.
        z_scale (float)
            scaling factor for z coordinate.
        x_shift (float)
            shifting x coordinate by this amount.
        y_shift (float)
            shifting y coordinate by this amount.
        z_shift (float)
            shifting z coordinate by this amount.
        """
        rid = self._get_random_rid()
        if scale_factor is not None:
            x_scale = scale_factor
            y_scale = scale_factor
            z_scale = scale_factor
        scale = [x_scale, y_scale, z_scale]
        shift = [x_shift, y_shift, z_shift]
        if name is None:
            name = '{}_{}'.format(mesh_class, rid)
        morphology_data = {'morph_type': 'mesh',
                           'class': mesh_class,
                           'name': name}
        assert isinstance(data, (dict, pd.DataFrame)), 'data must be either dict or a pandas DataFrame containing the field "faces" and "vertices"'
        
        morphology_data['faces'] = list(data['faces'])
        morphology_data['vertices'] = [n*scale[i%3]+shift[i%3] for i, n in enumerate(data['vertices'])]
        self.meshes[rid] = morphology_data
        self.colors[rid] = DEFAULT_MESH_COLOR if color is None else color
        self._uname_to_rid[name] = rid
        self._rids.add(rid)
        
    def add_neuron(self, data, uname, color = None,
                   scale_factor = None,
                   x_scale = 1.0, y_scale = 1.0, z_scale = 1.0, r_scale = 1.0, 
                   x_shift = 0.0, y_shift = 0.0, z_shift = 0.0, r_shift = 0.0):
        """
        Add neuron to visualization.
        
        # Arguments
        data (dict)
            A dictionary containing 'x', 'y', 'z', 'r', 'identifier', 'parent'
            and optionally 'sample' as keys, similar to the specification of
            swc files.
        uname (str)
            uname to be used for these points.
            This uname will be displayed in the neuron/synapse list.
        color (str)
            RGB Hex code for the color to be used for these points.
        scale_factor (float or None)
            A single scaling factor for x, y, z and r.
            If scales are different for each axis,
            use x_scale, y_scale, z_scale and r_scale individually.
        x_scale (float)
            scaling factor for x coordinate.
        y_scale (float)
            scaling factor for y coordinate.
        z_scale (float)
            scaling factor for z coordinate.
        r_scale (float)
            scaling factor for radius.
        x_shift (float)
            shifting x coordinate by this amount.
        y_shift (float)
            shifting y coordinate by this amount.
        z_shift (float)
            shifting z coordinate by this amount.
        r_shift (float)
            adding a constant value to all radius.
        """
        rid = self._get_random_rid()
        morphology_data = neuron_data_to_morphology(
                            data, scale_factor = scale_factor,
                            x_scale = x_scale, y_scale = y_scale,
                            z_scale = z_scale, r_scale = r_scale,
                            x_shift = x_shift, y_shift = y_shift,
                            z_shift = z_shift, r_shift = r_shift)
        morphology_data['uname'] = uname
        self.neurons[rid] = morphology_data
        self.colors[rid] = random.choice(DEFAULT_NEURON_COLORS) if color is None else color
        self._uname_to_rid[uname] = rid
        self._rids.add(rid)
    
    def add_synapses(self, points, uname,
                     r = 0.2, color = None, scale_factor = None,
                     x_scale = 1.0, y_scale = 1.0, z_scale = 1.0, r_scale = 1.0, 
                     x_shift = 0.0, y_shift = 0.0, z_shift = 0.0, r_shift = 0.0):
        """
        add synapses to the visualization
        
        # Arguments
        point (list)
            A list of 3-tuples of the coordinates of the points.
        uname (str)
            uname to be used for these points.
            This uname will be displayed in the neuron/synapse list.
        r (float)
            The radius of the sphere for the points
        color (str)
            RGB Hex code for the color to be used for these points.
        scale_factor (float or None)
            A single scaling factor for x, y, z and r.
            If scales are different for each axis,
            use x_scale, y_scale, z_scale and r_scale individually.
        x_scale (float)
            scaling factor for x coordinate.
        y_scale (float)
            scaling factor for y coordinate.
        z_scale (float)
            scaling factor for z coordinate.
        r_scale (float)
            scaling factor for radius.
        x_shift (float)
            shifting x coordinate by this amount.
        y_shift (float)
            shifting y coordinate by this amount.
        z_shift (float)
            shifting z coordinate by this amount.
        r_shift (float)
            adding a constant value to all radius.
        """
        rid = self._get_random_rid()
        morphology_data = points_to_morphology(
                            points, r = r, 
                            scale_factor = scale_factor,
                            x_scale = x_scale, y_scale = y_scale,
                            z_scale = z_scale, r_scale = r_scale,
                            x_shift = x_shift, y_shift = y_shift,
                            z_shift = z_shift, r_shift = r_shift)
        morphology_data['uname'] = uname
        self.synapses[rid] = morphology_data
        self.colors[rid] = random.choice(DEFAULT_SYNAPSE_COLORS) if color is None else color
        self._uname_to_rid[uname] = rid
        self._rids.add(rid)
        
    def add_mesh_group(self, mesh_group, mesh_class = 'Neuropil',
                       colormap = None, colors = None,
                       scale_factor = None,
                       x_scale = 1.0, y_scale = 1.0, z_scale = 1.0, 
                       x_shift = 0.0, y_shift = 0.0, z_shift = 0.0):
        """
        Add several meshes to the visualization at once.
        
        # Arguments
            mesh_group (dict)
                A dictionary with name of each mesh as keys,
                and values as a dictionary with the following keys:
                "vertices", "faces", (optional) "color".
                So a typical dict can look like 
                {"meshA": {"vertices": [x1,y1,z1, x2,y2,z2, ....],
                           "faces": [0,1,2, 1,2,3, 1,3,5, ...]
                           "color": "#FF0000"},
                 "meshB": {"vertices": [x3,y3,z3, x4,y4,z4, ....],
                           "faces": [0,1,2, 2,4,11, 12,31,15, ...]
                           "color": "#00FFFF"},
                }
                If color is a hex code, the color will be used to color.
                If color is a float, must be between 0 and 1 and color will be determined by the colormap at this value.
            mesh_class (str)
                Either "Neuropil" or "Subregion"
            colormap (matplotlib.colors.Colormap)
                Colormap to use if "color" in synapses_group is specified by floating point values.
            colors (list)
                A list of colors to use when user_color and colormap are not supplied.
                The list of colors does not need to have the same length as rids.
                It will be used in a round-robin fashion. See also color_group.
            scale_factor (float or None)
                A single scaling factor for x, y, z and r.
                If scales are different for each axis,
                use x_scale, y_scale, z_scale and r_scale individually.
            x_scale (float)
                scaling factor for x coordinate.
            y_scale (float)
                scaling factor for y coordinate.
            z_scale (float)
                scaling factor for z coordinate.
            r_scale (float)
                scaling factor for radius.
            x_shift (float)
                shifting x coordinate by this amount.
            y_shift (float)
                shifting y coordinate by this amount.
            z_shift (float)
                shifting z coordinate by this amount.
            r_shift (float)
                adding a constant value to all radius.
        """
        for i, (name, data) in enumerate(mesh_group.items()):
            if 'color' in data:
                if isinstance(data['color'], str):
                    color = data['color'],
                elif isinstance(data['color'], float):
                    if colormap is None:
                        colormap = cm.gnuplot
                    color = mc.rgb2hex(colormap(data['color']))
            else:
                if colors is not None:
                    color = colors[i%len(colors)]
                elif colormap is not None:
                    color = mc.rgb2hex(colormap(0.0))
                else:
                    color = None
            self.add_mesh({"vertices": data["vertices"], "faces": data["faces"]},
                          name, mesh_class, color = color,
                          scale_factor = scale_factor,
                          x_scale = x_scale, y_scale = y_scale,
                          z_scale = z_scale,
                          x_shift = x_shift, y_shift = y_shift,
                          z_shift = z_shift)
        
    def add_neuron_group(self, neuron_group,
                            colormap = None, colors = None,
                            scale_factor = None,
                            x_scale = 1.0, y_scale = 1.0,
                            z_scale = 1.0, r_scale = 1.0, 
                            x_shift = 0.0, y_shift = 0.0,
                            z_shift = 0.0, r_shift = 0.0):
        """
        Add a group of neurons to the visualization.

        # Arguments
        neuron_group (dict):
            Specifying the morphology of the neuron keyed by its uname, in the form of:
            neuron_group = {'morphology': pandas.DataFrame or dict in swc format,
                            'color': hex code starting with '#' for RGB value or float (optional)}
            If color is a hex code, the color will be used to color.
            If color is a float, must be between 0 and 1 and color will be determined by the colormap at this value.
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
        for i, (uname, data) in enumerate(neuron_group.items()):
            if 'color' in data:
                if isinstance(data['color'], str):
                    color = data['color'],
                elif isinstance(data['color'], float):
                    if colormap is None:
                        colormap = cm.gnuplot
                    color = mc.rgb2hex(colormap(data['color']))
            else:
                if colors is not None:
                    color = colors[i%len(colors)]
                elif colormap is not None:
                    color = mc.rgb2hex(colormap(0.0))
                else:
                    color = None
            self.add_neuron(data["morphology"],
                            uname, color = color,
                            scale_factor = scale_factor,
                            x_scale = x_scale, y_scale = y_scale,
                            z_scale = z_scale, r_scale = r_scale,
                            x_shift = x_shift, y_shift = y_shift,
                            z_shift = z_shift, r_shift = r_shift)

    def add_synapses_group(self, synapses_group,
                            r = 0.2, colormap = None, 
                            colors = None, scale_factor = None,
                            x_scale = 1.0, y_scale = 1.0,
                            z_scale = 1.0, r_scale = 1.0, 
                            x_shift = 0.0, y_shift = 0.0,
                            z_shift = 0.0, r_shift = 0.0):
        """
        Add a group of synapse sets to the visualization.
        
        # Arguments 
        synapses_group (dict)
            A dictionary with uname of each set of synapses as keys,
            and values as a dictionary with the following keys:
            "points", (optional) "color".
            So a typical dict can look like 
            {
                "synapse_A_to_B": {"points": [(x1,y1,z1), (x2,y2,z2), ....],
                                    "color": "#FF0000"},
                "synapse_C_to_D": {"points": [(x3,y3,z3), (x4,y4,z4), ....],
                                    "color": "#00FFFF"},
            }
            If color is a hex code, the color will be used to color.
            If color is a float, must be between 0 and 1 and color will be determined by the colormap at this value.
        r (float)
            The radius of the sphere for the points
        colormap (matplotlib.colors.Colormap)
            Colormap to use if "color" in synapses_group is specified by floating point values.
        colors (list)
            A list of colors to use when user_color and colormap are not supplied.
            The list of colors does not need to have the same length as rids.
            It will be used in a round-robin fashion. See also color_group.
        """
        for i, (uname, data) in enumerate(synapses_group.items()):
            if 'color' in data:
                if isinstance(data['color'], str):
                    color = data['color'],
                elif isinstance(data['color'], float):
                    if colormap is None:
                        colormap = cm.gnuplot
                    color = mc.rgb2hex(colormap(data['color']))
            else:
                if colors is not None:
                    color = colors[i%len(colors)]
                elif colormap is not None:
                    color = mc.rgb2hex(colormap(0.0))
                else:
                    color = None
            self.add_synapses(data["points"],
                                uname, r = r, color = color,
                                scale_factor = scale_factor,
                                x_scale = x_scale, y_scale = y_scale,
                                z_scale = z_scale, r_scale = r_scale,
                                x_shift = x_shift, y_shift = y_shift,
                                z_shift = z_shift, r_shift = r_shift)
    
    def visualize(self):
        """
        Visualize the data within this handle.
        Objects will be displayed in Neu3D window after this method is called.
        """
        _send_data_to_NLP(self.client, self.meshes)
        _send_data_to_NLP(self.client, self.neurons)
        _send_data_to_NLP(self.client, self.synapses)
        color_group(self.client, list(self._rids), user_color = self.colors)
    
    def _get_rids_from_items(self, items = None):
        if items is None:
            rids = list(self._rids)
        else:
            if isinstance(items, str):
                if items == 'mesh':
                    items = [v['name'] for v in self.meshes.values()]
                elif items == 'neuron':
                    items = [v['uname'] for v in self.neurons.values()]
                elif items == 'synapse':
                    items = [v['uname'] for v in self.synapses.values()]
                else:
                    items = [items]
            rids = []
            for k in items:
                rids.append(self._uname_to_rid[k])
        return rids
        
    def hide(self, items = None):
        """
        Hide objects.
        
        # Arguments
            items (list)
                A list of uname or name of the objects to hide.
                If None, all will be hidden.
                If 'mesh', all meshes will be hidden.
                If 'neuron', all neurons will be hidden.
                If 'synapse', all synapses will be hidden.
        """
        rids = self._get_rids_from_items(items)
        _command_by_rids(self.client, rids, 'hide')
    
    def remove(self, items = None):
        """
        Remove objects.
        
        # Arguments
            items (list)
                A list of uname or name of the objects to remove.
                If None, all will be removed.
                If 'mesh', all meshes will be removed.
                If 'neuron', all neurons will be removed.
                If 'synapse', all synapses will be removed.
        """
        rids = self._get_rids_from_items(items)
        _command_by_rids(self.client, rids, 'remove')
        
    def show(self, items = None):
        """
        Show objects.
        
        # Arguments
            items (list)
                A list of uname or name of the objects to show.
                If None, all will be shown.
                If 'mesh', all meshes will be shown.
                If 'neuron', all neurons will be shown.
                If 'synapse', all synapses will be shown.
        """
        rids = self._get_rids_from_items(items)
        _command_by_rids(self.client, rids, 'show')
        
    def pin(self, items = None):
        """
        Pin objects.
        
        # Arguments
            items (list)
                A list of uname or name of the objects to pin.
                If None, all will be pinned.
                If 'mesh', all meshes will be pinned.
                If 'neuron', all neurons will be pinned.
                If 'synapse', all synapses will be pinned.
        """
        rids = self._get_rids_from_items(items)
        _command_by_rids(self.client, rids, 'pin')
    
    def unpin(self, items = None):
        """
        Unpin objects.
        
        # Arguments
            items (list)
                A list of uname or name of the objects to unpin.
                If None, all will be unpinned.
                If 'mesh', all meshes will be unpinned.
                If 'neuron', all neurons will be unpinned.
                If 'synapse', all synapses will be unpinned.
        """
        rids = self._get_rids_from_items(items)
        _command_by_rids(self.client, rids, 'unpin')
    
    def color(self, color, items = None):
        """
        Color objects.
        
        # Arguments
            color (str)
                A hex code for color.
            items (list)
                A list of uname or name of the objects to color.
                If None, all will be colored.
                If 'mesh', all meshes will be colored.
                If 'neuron', all neurons will be colored.
                If 'synapse', all synapses will be colored.
        """
        rids = self._get_rids_from_items(items)
        color_by_rids(self.client, rids, color)
        for rid in rids:
            self.colors[rid] = color
    
    def _get_random_rid(self):
        """
        Get a random rid '#??????:??????'
        """
        rid = "#{}:{}".format(str(random.randint(0,999999)).zfill(6),
                              str(random.randint(0,999999)).zfill(6))
        while rid in self._rids:
            rid = "#{}:{}".format(str(random.randint(0,999999)).zfill(6),
                                  str(random.randint(0,999999)).zfill(6))
        return rid
    
    def update(self, other):
        """
        Update current object by adding objects from other.
        """
        for rid in other._rids:
            if rid in self._rids:
                new_rid = self._get_random_rid()
            else:
                new_rid = rid
            if rid in other.meshes:
                self.meshes[new_rid] = other.meshes[rid]
                name = other.meshes[rid]['name']
            elif rid in other.neurons:
                self.neurons[new_rid] = other.neurons[rid]
                name = other.neurons[rid]['uname']
            elif rid in other.synapses:
                self.synapses[new_rid] = other.synapses[rid]
                name = other.synapses[rid]['uname']
            self.colors[new_rid] = other.colors[rid]
            self._rids.add(new_rid)
            self._uname_to_rid[name] = new_rid


def loadJSON(client, file_name, scale_factor=1., name=None, mesh_class = 'Neuropil'):
    """Loads a mesh stored in the .json format.
    # Arguments
        client (flybrainlab.Client.Client or flybrainlab.utilities.NeuroWatch)
            Client object to specify the neu3d widget to use.
            If a NeuroWatch object, visualization will be added to this object.
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
        name = os.path.splitext(os.path.split(file_name)[-1])[0]
    if isinstance(client, NeuroWatch):
        watch = client
        client = watch.client
    else:
        watch = NeuroWatch(client)
        
    watch.add_mesh(data, name, mesh_class = mesh_class, 
                   scale_factor = scale_factor)
    watch.visualize()
    return watch
    
def loadSWC(client, file_name, scale_factor=1., uname=None):
    """Loads a neuron skeleton stored in the .swc format.
    # Arguments
        client (flybrainlab.Client.Client or flybrainlab.utilities.NeuroWatch)
            Client object to specify the neu3d widget to use.
            If a NeuroWatch object, visualization will be added to this object.
        file_name (str): Database ID of the neuron or node.
        scale_factor (float): A scale factor to scale the neuron's dimensions with. Defaults to 1.
        uname (str): Unique name to use in the frontend. Defaults to the file_name.
    """
    neuron_pd = pd.read_csv(file_name,
                    names=['sample','identifier','x','y','z','r','parent'],
                    comment='#',
                    delim_whitespace=True)
    if uname == None:
        uname = os.path.splitext(os.path.split(file_name)[-1])[0]
    if isinstance(client, NeuroWatch):
        watch = client
        client = watch.client
    else:
        watch = NeuroWatch(client)
    
    data = neuron_pd.to_dict(orient='list')
    watch.add_neuron(data, uname, scale_factor = scale_factor)
    watch.visualize()
    return watch


def _send_data_to_NLP(client, data):
    """
    Utility to send data to NLP for visualization
    """
    a = {'messageType': 'Data', 'widget': 'NLP',
         'data': {'data': data, "queryID": '0-0'}}
    client.tryComms(a)


def color_by_rids(client, rids, rgb):
    """
    Utility to color items in Neu3D widget with rids to rgb color.
    
    # Arguments
    client (flybrainlab.Client.Client)
        Client object to specify the neu3d widget to use.
    rids (list)
        A list of rids for the items already in the neu3d widget to be colored.
    rgb (str)
        RGB Hex string, starting with '#'.
    """
    if isinstance(rids, str):
        rids = [rids]
    a = {'messageType': 'Command', 'widget': 'NLP',
         'data': {'commands': {'setcolor': [rids, rgb]}}}
    client.tryComms(a)
    
def _command_by_rids(client, rids, command):
    """
    Utility to hide items in Neu3D widget with rids.
    
    # Arguments
    client (flybrainlab.Client.Client)
        Client object to specify the neu3d widget to use.
    rids (list)
        A list of rids for the items already in the neu3d widget to be hidden.
    command (str)
        The command to execute: "hide", "show", "pin", "unpin", "keep", "remove"
    """
    assert command in ["hide", "show", "pin", "unpin", "keep", "remove"]
    if isinstance(rids, str):
        rids = [rids]
    a = {'messageType': 'Command', 'widget': 'NLP',
         'data': {'commands': {command: [rids, []]}}}
    client.tryComms(a)


def color_group(client, rids, user_color = None, colormap = None, colors = None):
    """
    Color a group of items in the neu3d widget with different colors.
    
    # Arguments
    client (flybrainlab.Client.Client)
        Client object to specify the neu3d widget to use.
    rids (list)
        A list of rids for the items already in the neu3d widget to be colored
    user_color (dict or None)
        A dict of color for each rid using rid as key.
        Values must be either of the following: 
        1. all values are RGB Hex strings starting with '#'.
        2. all values are floating point values.
    colormap (matplotlib.colors.Colormap)
        A color map to map floating point values into color.
        The floating point values are supplied by user_color
    colors (list)
        A list of colors to use when user_color and colormap are not supplied.
        The list of colors does not need to have the same length as rids.
        It will be used in a round-robin fashion.
    """
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
    """
    Remove items in neu3d widget with rids.
    
    # Arguments
    client (flybrainlab.Client.Client)
        Client object to specify the neu3d widget to use.
    rids (list)
        A list of rids for the items to be removed from neu3d widget.
        
    # Returns
        list : rids of the items removed.
    """
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
    """
    Utility to reorganize point data into morphology data that
    can be passed to neu3d widget.
    
    # Arguments
    point (list)
        A list of 3-tuples of the coordinates of the points.
    r (float)
        The radius of the sphere for the points
    scale_factor (float or None)
        A single scaling factor for x, y, z and r.
        If scales are different for each axis,
        use x_scale, y_scale, z_scale and r_scale individually.
    x_scale (float)
        scaling factor for x coordinate.
    y_scale (float)
        scaling factor for y coordinate.
    z_scale (float)
        scaling factor for z coordinate.
    r_scale (float)
        scaling factor for radius.
    x_shift (float)
        shifting x coordinate by this amount.
    y_shift (float)
        shifting y coordinate by this amount.
    z_shift (float)
        shifting z coordinate by this amount.
    r_shift (float)
        adding a constant value to all radius.
        
    # Returns
        dict : A dictionary that can be passed to neu3d for visualization.
    """
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
    """
    Utility to reorganize neuron skeleton data into morphology data that
    can be passed to neu3d widget.
    
    # Arguments
    data (dict)
        A dictionary containing 'x', 'y', 'z', 'r', 'identifier', 'parent'
        and optionally 'sample' as keys, similar to the specification of
        swc files.
    scale_factor (float or None)
        A single scaling factor for x, y, z and r.
        If scales are different for each axis,
        use x_scale, y_scale, z_scale and r_scale individually.
    x_scale (float)
        scaling factor for x coordinate.
    y_scale (float)
        scaling factor for y coordinate.
    z_scale (float)
        scaling factor for z coordinate.
    r_scale (float)
        scaling factor for radius.
    x_shift (float)
        shifting x coordinate by this amount.
    y_shift (float)
        shifting y coordinate by this amount.
    z_shift (float)
        shifting z coordinate by this amount.
    r_shift (float)
        adding a constant value to all radius.
        
    # Returns
        dict : A dictionary that can be passed to neu3d for visualization.
    """
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




    