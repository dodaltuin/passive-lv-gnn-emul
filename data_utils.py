"""
File: data_utils.py
Author: David Dalton
Description: Data loading utils for Passive LV Simulation Data
"""

import os
import numpy as np
import jax.numpy as jnp
from jax import device_put

class DataLoader:
    """Data loader for graph-formatted input-output data with common, fixed topology"""

    def __init__(self, data_path: str, data_type: str, n_shape_coeff:int, fixed_geom: bool):
        """
        Parameters
        ----------
        data_path: str
            Name of the subdirectory within "/data" where the data is stored
        data_type: str
            The type of data to be loaded, one of "train", "validation", "test"
        n_shape_coeff: int >= 0
            The number of shape coefficients to input the the GNN emulator
        fixed_geom: bool
            True if dataset corresponds to a fixed LV geometry, False otherwise
        """

        full_data_path = f'data/{data_path}'
        if not os.path.isdir(full_data_path):
            raise NotADirectoryError(f'No directory at: {full_data_path}')

        data_save_dir = f'{full_data_path}/processedData/{data_type}'
        topology_save_dir = f'{full_data_path}/topologyData'
        stats_save_dir = f'{full_data_path}/normalisationStatistics'

        # boolean stores whether or not this data is for a fixed LV geometry (with varying values of theta)
        self._fixed_geom=fixed_geom

        # load mesh topology (assumed fixed for each graph)
        sparse_topology =  jnp.load(f'{topology_save_dir}/sparse-topology.npy').astype(jnp.int32)
        self._senders = sparse_topology[:,0]
        self._receivers = sparse_topology[:,-1]

        # node features
        self._nodes = np.load(f'{data_save_dir}/augmented-node-features.npy').astype('float32')

        # edge features
        self._edges = np.load(f'{data_save_dir}/edge-features.npy').astype(np.float32)

        # array holding the displacement between end and start diastole (normalised for training data, un-normalised for validation and test data)
        self._displacement = np.load(f'{data_save_dir}/real-node-displacement.npy').astype(np.float32)

        # summary statistics for displacement values calculated on training data
        self._displacement_mean = np.load(f'{stats_save_dir}/real-node-displacement-mean.npy').astype(np.float32)
        self._displacement_std = np.load(f'{stats_save_dir}/real-node-displacement-std.npy').astype(np.float32)

        # the co-ordinates of the LV geometry in the reference configuration
        self._real_node_coords = np.load(f'{data_save_dir}/real-node-coords.npy').astype(np.float32)

        node_layer_labels = np.load(f'{topology_save_dir}/node-layer-labels.npy')

        self._real_node_indices = (node_layer_labels == 0)

        self._data_size = self._displacement.shape[0]
        self._output_dim = self._displacement.shape[-1]
        self._n_total_nodes= int(self._nodes.shape[1])
        self._n_real_nodes = self._real_node_indices.sum()
        self._n_edges= int(self._edges.shape[1])
        self._n_shape_coeff = n_shape_coeff

        # array of data point indices that can be iterated over during each epoch
        self._epoch_indices = np.arange(self._data_size)

        if self._n_shape_coeff > 0:

            # load shape coefficients
            shape_coeffs = np.load(f'{data_save_dir}/shape-coeffs.npy').astype(np.float32)

            assert shape_coeffs.shape[-1] >= n_shape_coeff, \
                f"Number of shape coefficients to retain ({n_shape_coeff}) must be <= number of columns in 'shape-coeffs.npy' ({shape_coeffs.shape[-1]})"

            # retain n_shape_coeff of these to input to the emulator
            self._shape_coeffs = shape_coeffs[:,:n_shape_coeff]

        else:
            self._shape_coeffs = [None]*self._data_size

        # global variables are the same for each node in the graph (e.g. global material stiffness parameters)
        self._theta_vals = np.load(f'{data_save_dir}/global-features.npy').astype(np.float32)

        # summary statistics for global variables, calculated on training data
        self._theta_mean = np.load(f'{stats_save_dir}/global-features-mean.npy').astype(np.float32)
        self._theta_std = np.load(f'{stats_save_dir}/global-features-std.npy').astype(np.float32)


    def get_graph(self, data_idx):
        """Returns input/output values for specified data point and places on GPU

        If the dataset is for a fixed LV geometry, the only input is the global
        parameter vector (theta) and output is the nodal displacement array (U)

        If the dataset is for varying LV geometries, inputs are node features (V),
        edge features (E), parameter vector (theta) and optionally global shape
        embedding vector (z_global). Again, the output is the displacement array (U)
        """

        if self._fixed_geom:
            # for fixed_geom emulator, (input/output) data has format (theta / U)
            return device_put(self._theta_vals[data_idx:(data_idx+1)]), device_put(self._displacement[data_idx])
        else:
            # for varying_geom emulator, (input/output) data has format ([V, E, theta, z_global] / U)"
            return device_put(self._nodes[data_idx]), device_put(self._edges[data_idx]), device_put(self._theta_vals[data_idx:(data_idx+1)]), device_put(self._shape_coeffs[data_idx]), device_put(self._displacement[data_idx])

    def return_index_0(self):
        """Returns input/output data ([V, E, theta, z_global]/ U) for first data point and places on GPU

        This method is used when initialising the parameters of the varying geometry GNN emulator
        """

        return device_put(self._nodes[0]), device_put(self._edges[0]), device_put(self._theta_vals[0:1]), device_put(self._shape_coeffs[0]), device_put(self._displacement[0])

    def shuffle_epoch_indices(self, seed_idx=476):
        """Shuffles the order in which the dataset is cycled through

        This is called at the start of each training epoch to randomise the order
        in which the training data points are seen
        """

        np.random.seed(seed_idx)
        self._epoch_indices = np.random.choice(self._data_size, self._data_size, False)

