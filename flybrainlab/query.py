
import time
from functools import partial

import numpy as np
import networkx as nx
import pandas as pd
from autobahn.wamp.types import CallOptions
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from .exceptions import FlyBrainLabNAserverException


class NeuroArch_Mirror(object):
    def __init__(self, client):
        self.client = client
        self.NeuroArchWrite_rpc = partial(client.rpc,
                                          'ffbo.na.NeuroArch.write.{}'.format(client.naServerID),
                                          options=CallOptions(timeout=10000))
        self.NeuroArchQuery_rpc = partial(client.rpc,
                                          'ffbo.na.NeuroArch.query.{}'.format(client.naServerID),
                                          options=CallOptions(timeout=10000))

    def update_client(self, client):
        self.client = client

    def _check_result(self, res):
        if 'success' in res:
            if 'message' in res['success']:
                self.client.raise_message(res['success']['message'])
                self.client.log['NA'].info(res['success']['message'])
            return res['success'].get('data', None)
        elif 'error' in res:
            self.client.raise_error(
                FlyBrainLabNAserverException(res['error']['exception']),
                res['error']['message'])

    def available_DataSources(self):
        res = self.NeuroArchQuery_rpc('available_DataSources')
        result = self._check_result(res)
        return result

    def select_DataSource(self, name, version = None):
        uri = "ffbo.na.datasource.{}".format(self.client.naServerID)
        res = self.client.rpc(
                uri,
                name, version = version,
                options=CallOptions(timeout=10000) )
        result = self._check_result(res)
        return result

    def add_Species(self, name, stage, sex, synonyms = None):
        """
        Add a Species.

        Parameters
        ----------
        name : str
            Name of the species.
        stage : str
            Development stage of the species.
        synonyms : list of str
            Other names used by the species.

        Returns
        -------
        species : models.Species
            The created species record.
        """
        res = self.NeuroArchWrite_rpc('add_Species', name, stage, sex, synonyms = synonyms)
        result = self._check_result(res)
        return result

    def add_DataSource(self, name, version,
                       url = None, description = None,
                       species = None):
        """
        Add a DataSource.

        Parameters
        ----------
        name : str
            Name of the DataSource.
        version : str
            Version of the Dataset.
        url : str
            Web URL describing the origin of the DataSource
        description : str
            A brief description of the DataSource
        species : dict or models.Species
            The species the added DataSource is for.
            If species is a dict, it must be contain the following keys:
                {'name': str,
                 'stage': str,
                 'synonyms': list of str (optional)
                }

        Returns
        -------
        datasource : models.DataSource
            created DataSource object
        """
        res = self.NeuroArchWrite_rpc(
                         'add_DataSource', name, version, url = url,
                         description = description, species = species)
        result = self._check_result(res)
        return result

    def add_Subsystem(self, name, synonyms = None,
                      morphology = None, data_source = None):
        """
        Create a Subsystem record and link it to related node types.

        Parameters
        ----------
        name : str
            Name of the subsystem
            (abbreviation is preferred, full name can be given in the synonyms)
        synonyms : list of str
            Synonyms of the subsystem.
        morphology : dict (optional)
            Morphology of the neuropil boundary specified with a triangulated mesh,
            with fields
                'vertices': a single list of float, every 3 entries specify (x,y,z) coordinates.
                'faces': a single list of int, every 3 entries specify samples of vertices.
            Or, specify the file path to a json file that includes the definition of the mesh.
            Or, specify only a url which can be readout later on.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        neuroarch.models.Subsystem
            Created Subsystem object
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Subsystem', name, synonyms = synonyms,
                         morphology = morphology, data_source = data_source)
        result = self._check_result(res)
        return result

    def add_Neuropil(self, name,
                     synonyms = None,
                     subsystem = None,
                     morphology = None,
                     data_source = None):
        """
        Create a Neuropil record and link it to related node types.

        Parameters
        ----------
        name : str
            Name of the neuropil
            (abbreviation is preferred, full name can be given in the synonyms)
        synonyms : list of str
            Synonyms of the neuropil.
        subsystem : str or neuroarch.models.Subsystem (optional)
            Subsystem that owns the neuropil. Can be specified either by its name
            or the Subsytem object instance.
        morphology : dict (optional)
            Morphology of the neuropil boundary specified with a triangulated mesh,
            with fields
                'vertices': a single list of float, every 3 entries specify (x,y,z) coordinates.
                'faces': a single list of int, every 3 entries specify samples of vertices.
            Or, specify the file path to a json file that includes the definition of the mesh.
            Or, specify only a url which can be readout later on.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        neuroarch.models.Neuropil
            Created Neuropil object
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Neuropil', name, synonyms = synonyms,
                         subsystem = subsystem, morphology = morphology,
                         data_source = data_source)
        result = self._check_result(res)
        return result

    def add_Subregion(self, name,
                      synonyms = None,
                      neuropil = None,
                      morphology = None,
                      data_source = None):
        """
        Create a Subregion record and link it to related node types.

        Parameters
        ----------
        name : str
            Name of the subregion
            (abbreviation is preferred, full name can be given in the synonyms)
        synonyms : list of str
            Synonyms of the synonyms.
        neuropil : str or neuroarch.models.Neuropil (optional)
            Neuropil that owns the subregion. Can be specified either by its name
            or the Neuropil object instance.
        morphology : dict (optional)
            Morphology of the neuropil boundary specified with a triangulated mesh,
            with fields
                'vertices': a single list of float, every 3 entries specify (x,y,z) coordinates.
                'faces': a single list of int, every 3 entries specify samples of vertices.
            Or, specify the file path to a json file that includes the definition of the mesh.
            Or, specify only a url which can be readout later on.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        dict : a dictionary with rid of creaed node as key and its properties as values
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Subregion', name, synonyms = synonyms,
                         neuropil = neuropil, morphology = morphology,
                         data_source = data_source)
        result = self._check_result(res)
        return result

    def add_Tract(self, name,
                  synonyms = None,
                  morphology = None,
                  data_source = None):
        """
        Create a Subregion record and link it to related node types.

        Parameters
        ----------
        name : str
            Name of the tract
            (abbreviation is preferred, full name can be given in the synonyms)
        synonyms : list of str
            Synonyms of the synonyms.
        morphology : dict (optional)
            Morphology of the neuropil boundary specified with a triangulated mesh,
            with fields
                'vertices': a single list of float, every 3 entries specify (x,y,z) coordinates.
                'faces': a single list of int, every 3 entries specify samples of vertices.
            Or, specify the file path to a json file that includes the definition of the mesh.
            Or, specify only a url which can be readout later on.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        neuroarch.models.Tract
            Created Tract object
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Tract', name, synonyms = synonyms,
                         morphology = morphology, data_source = data_source)
        result = self._check_result(res)
        return result

    def add_Circuit(self, name, circuit_type, neuropil = None, data_source =  None):
        """
        Create a Subregion record and link it to related node types.

        Parameters
        ----------
        name : str
            Name of the circuit
        neuropil : str or neuroarch.models.Neuropil (optional)
            Neuropil that owns the subregion. Can be specified either by its name
            or the Neuropil object instance.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        neuroarch.models.Circuit
            Created Circuit object
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Circuit', name, circuit_type,
                         neuropil = neuropil, data_source = data_source)
        result = self._check_result(res)
        return result

    def add_Neuron(self, uname,
                   name,
                   referenceId = None,
                   locality = None,
                   synonyms = None,
                   info = None,
                   morphology = None,
                   arborization = None,
                   neurotransmitters = None,
                   neurotransmitters_datasources = None,
                   data_source = None,
                   circuit = None):
        """
        Parameters
        ----------
        uname : str
            A unqiue name assigned to the neuron, must be unique within the DataSource
        name : str
            Name of the neuron, typically the cell type.
        referenceId : str (optional)
            Unique identifier in the original data source
        locality : bool (optional)
            Whether or not the neuron is a local neuron
        synonyms : list of str (optional)
            Synonyms of the neuron
        info : dict (optional)
            Additional information about the neuron, values must be str
        morphology : list of dict (optional)
            Each dict in the list defines a type of morphology of the neuron.
            Must be loaded from a file.
            The dict must include the following key to indicate the type of morphology:
                {'type': 'swc'/'obj'/...}
            Additional keys must be provides, either 'filename' with value
            indicating the file to be read for the morphology,
            or a full definition of the morphology according the schema.
            For swc, required fields are ['sample', 'identifier', 'x', 'y, 'z', 'r', 'parent'].
            More formats pending implementation.
        arborization : list of dict (optional)
            A list of dictionaries define the arborization pattern of
            the neuron in neuropils, subregions, and tracts, if applicable, with
            {'type': 'neuropil' or 'subregion' or 'tract',
             'dendrites': {'EB': 20, 'FB': 2},
             'axons': {'NO': 10, 'MB': 22}}
            Name of the regions must already be present in the database.
        neurotransmitters : str or list of str (optional)
            The neurotransmitter(s) expressed by the neuron
        neurotransmitters_datasources : neuroarch.models.DataSource or list of neuroarch.models.DataSource (optional)
            The datasource of neurotransmitter data.
            If None, all neurotransmitter will have the same datasource of the Neuron.
            If specified, the size of the list must be the same as the size of
            neurotransmitters, and have one to one corresponsdence in the same order.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.
        circuit : 

        Returns
        -------
        neuron : neuroarch.models.Neuron
            Created Neuron object
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Neuron', uname, name,
                         referenceId = referenceId,
                         locality = locality,
                         synonyms = synonyms,
                         info = info,
                         morphology = morphology,
                         arborization = arborization,
                         neurotransmitters = neurotransmitters,
                         neurotransmitters_datasources = neurotransmitters_datasources,
                         data_source = data_source,
                         circuit = circuit)
        result = self._check_result(res)
        return result

    def add_neurotransmitter(self, neuron, neurotransmitters, data_sources = None):
        """
        Parameters
        ----------
        neuron : neuroarch.models.Neuron subclass
            An instance of Neuron class to which the neurotransmitters will be associated to
        neurotransmitters : str or list of str
            The neurotransmitter(s) expressed by the neuron
        data_sources : neuroarch.models.DataSource or list of neuroarch.models.DataSource (optional)
            The datasource of neurotransmitter data.
            If None, all neurotransmitter will have the same datasource of the Neuron.
            If specified, the size of the list must be the same as the size of
            neurotransmitters, and have one to one corresponsdence in the same order.
        """
        res = self.NeuroArchWrite_rpc(
                         'add_neurotransmitter', neuron, neurotransmitters,
                         data_source = data_source)
        result = self._check_result(res)
        return result

    def add_morphology(self, obj, morphology, data_source = None):
        """
        Add a morphology to a node, e.g., a neuropil, or a neuron.

        Parameters
        ----------
        obj : neuroarch.models.BioNode subclass
            An instance of BioNode class, e.g., Neuropil, Neuron, etc...
            to which the morphology will be associated to
        morphology : list of dict (optional)
            Each dict in the list defines a type of morphology of the neuron.
            Must be loaded from a file.
            The dict must include the following key to indicate the type of morphology:
                {'type': 'swc'/'obj'/...}
            Additional keys must be provides, either 'filename' with value
            indicating the file to be read for the morphology,
            or a full definition of the morphology according the schema.
            For swc, required fields are ['sample', 'identifier', 'x', 'y, 'z', 'r', 'parent'].
            For mesh, requires an obj file or ['faces', 'vertices'] defined as rastered list of a wavefront obj file
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.
        """
        res = self.NeuroArchWrite_rpc(
                         'add_morphology', neuron, neurotransmitters,
                         data_source = data_source)
        result = self._check_result(res)
        return result

    def add_neuron_arborization(self, neuron, arborization, data_source = None):
        res = self.NeuroArchWrite_rpc(
                         'add_neuron_arborization', neuron, arborization,
                         data_source = data_source)
        result = self._check_result(res)
        return result

    def add_Synapse(self, pre_neuron, post_neuron,
                    N = None, NHP = None,
                    morphology = None,
                    arborization = None,
                    data_source = None):
        """
        Add a Synapse from pre_neuron and post_neuron.
        The Synapse is typicall a group of synaptic contact points.

        parameters
        ----------
        pre_neuron : str or models.Neuron
            The neuron that is presynaptic in the synapse.
            If str, must be the uname or rid of the presynaptic neuron.
        post_neuron : str or models.Neuron
            The neuron that is postsynaptic in the synapse.
            If str, must be the uname or rid of the postsynaptic neuron.
        N : int (optional)
            The number of synapses from pre_neuron to the post_neuron.
        NHP : int (optional)
            The number of synapses that can be confirmed with a high probability
        morphology : list of dict (optional)
            Each dict in the list defines a type of morphology of the neuron.
            Must be loaded from a file.
            The dict must include the following key to indicate the type of morphology:
                {'type': 'swc'}
            For swc, required fields are ['sample', 'identifier', 'x', 'y, 'z', 'r', 'parent'].
            For synapses, if both postsynaptic and presynaptic sites are available,
            x, y, z, r must each be a list where the first half indicate the
            locations/radii of postsynaptic sites (on the presynaptic neuron),
            and the second half indicate the locations/radii of the presynaptic
            sites (on the postsynaptic neuron). There should be a one-to-one relation
            between the first half and second half.
            parent must be a list of -1.
        arborization : list of dict (optional)
            A list of dictionaries define the arborization pattern of
            the neuron in neuropils, subregions, and tracts, if applicable, with
            {'type': 'neuropil' or 'subregion' or 'tract',
             'synapses': {'EB': 20, 'FB': 2}}
            Name of the regions must already be present in the database.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        synapse : models.Synapse
            The created synapse object.
        """
        res = self.NeuroArchWrite_rpc(
                         'add_Synapse', pre_neuron, post_neuron,
                             N = N, NHP = NHP,
                             morphology = morphology,
                             arborization = arborization,
                             data_source = data_source)
        result = self._check_result(res)
        return result

    def add_synapse_arborization(self, synapse, arborization, data_source = None):
        res = self.NeuroArchWrite_rpc(
                         'add_synapse_arborization', synapse, arborization,
                             data_source = data_source)
        result = self._check_result(res)
        return result

    def add_InferredSynapse(self, pre_neuron, post_neuron,
                            N = None, NHP = None,
                            morphology = None,
                            arborization = None,
                            data_source = None):
        """
        Add an InferredSynapse from pre_neuron and post_neuron.
        The Synapse is typicall a group of synaptic contact points.

        parameters
        ----------
        pre_neuron : str or models.Neuron
            The neuron that is presynaptic in the synapse.
            If str, must be the uname of the presynaptic neuron.
        post_neuron : str or models.Neuron
            The neuron that is postsynaptic in the synapse.
            If str, must be the uname of the postsynaptic neuron.
        N : int (optional)
            The number of synapses from pre_neuron to the post_neuron.
        morphology : list of dict (optional)
            Each dict in the list defines a type of morphology of the neuron.
            Must be loaded from a file.
            The dict must include the following key to indicate the type of morphology:
                {'type': 'swc'}
            For swc, required fields are ['sample', 'identifier', 'x', 'y, 'z', 'r', 'parent'].
            For synapses, if both postsynaptic and presynaptic sites are available,
            x, y, z, r must each be a list where the first half indicate the
            locations/radii of postsynaptic sites (on the presynaptic neuron),
            and the second half indicate the locations/radii of the presynaptic
            sites (on the postsynaptic neuron). There should be a one-to-one relation
            between the first half and second half.
            parent must be a list of -1.
        arborization : list of dict (optional)
            A list of dictionaries define the arborization pattern of
            the neuron in neuropils, subregions, and tracts, if applicable, with
            {'type': 'neuropil' or 'subregion' or 'tract',
             'synapses': {'EB': 20, 'FB': 2}}
            Name of the regions must already be present in the database.
        data_source : neuroarch.models.DataSource (optional)
            The datasource. If not specified, default DataSource will be used.

        Returns
        -------
        synapse : models.Synapse
            The created synapse object.
        """
        res = self.NeuroArchWrite_rpc(
                         'add_InferredSynapse', pre_neuron, post_neuron,
                             N = N, NHP = NHP,
                             morphology = morphology,
                             arborization = arborization,
                             data_source = data_source)
        result = self._check_result(res)
        return result
