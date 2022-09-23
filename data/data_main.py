"""
File: data_main.py
Author: David Dalton
Description: Data processing script

Processes data from raw format into augmented graph format,
with node and edge features assigned

Example usage: the following command can be used to process
the raw beam displacement dataset beamData:

python -m data_main --mode="run_all" --data_dir="beamData" --n_nearest_nbrs=4 --n_leaves=4 --min_root_nodes=2

And for lvData:

python -m data_main --mode="run_all" --data_dir="lvData" --n_nearest_nbrs=5 --n_leaves=8 --min_root_nodes=3

Once data processing is complete, emulator training and validation
can be performed from the main directory
"""

import data_process_utils as utils

from typing import Sequence

from absl import app
from absl import flags
from absl import logging

## shell input variables
flags.DEFINE_enum('mode', 'run_all', ['generate_topology', 'generate_nodes', 'generate_edges', 'normalise_data', 'run_all'], help = 'Select which data processing function to run')
flags.DEFINE_string('data_dir', 'beamData', help='Path to directory where raw data is stored')
flags.DEFINE_string('existing_topology_dir', "None", help='Path to directory where raw data is stored')
flags.DEFINE_string('existing_stats_dir', "None", help='(optional) Path to directory where normalisation summary stats are saved')
flags.DEFINE_integer('n_nearest_nbrs', 4, lower_bound=1, help='number of nearest neighbours to consider when defining root-root topology (called $\kappa$ in the manuscript')
flags.DEFINE_integer('n_leaves', 4, lower_bound=2, help='(approximate) number of leaves each root node should have')
flags.DEFINE_integer('min_root_nodes', 2, lower_bound=1, help='minimum number of nodes to have final virtual layer')
FLAGS = flags.FLAGS

## Hard coded input variables

# (optional) list of decreasing integers specifying the number of virtual nodes to create at each layer
# (referred to as $\boldsymbol{\eta}$ in the manuscript). If set to None, it will be calculated automatically
# using the shell input variables "n_leaves" and "min_root_nodes" defined above
n_nodes_per_layer: Sequence[int] = None

# data directories inside data_dir/rawData:
DATA_TYPES: Sequence[str] = ['train', 'validation', 'test']

# data which simply needs to be copied to final processes data dir
COPY_FILENAMES: Sequence[str] = ['real-node-coords', 'augmented-node-coords']

# data which needs to be normalised before saving to final processed data dir
NORMALISE_FILENAMES: Sequence[str] = ['augmented-node-features', 'global-features', 'edge-features', 'shape-coeffs', 'real-node-displacement']


def main(_):

    if not utils.os.path.isdir(FLAGS.data_dir):
        raise NotADirectoryError(f'No directory at: {FLAGS.data_dir}')

    if FLAGS.mode in ['run_all', 'generate_topology']:
        utils.generate_augmented_topology(FLAGS.data_dir,
                                          FLAGS.existing_topology_dir,
                                          FLAGS.n_nearest_nbrs,
                                          n_nodes_per_layer,
                                          FLAGS.n_leaves,
                                          FLAGS.min_root_nodes)

    if FLAGS.mode in ['run_all', 'generate_nodes']:
        utils.generate_augmented_nodes(FLAGS.data_dir,
                                       DATA_TYPES)

    if FLAGS.mode in ['run_all', 'generate_edges']:
        utils.generate_edge_features(FLAGS.data_dir,
                                     DATA_TYPES)

    if FLAGS.mode in ['run_all', 'normalise_data']:
        utils.generate_normalised_data(FLAGS.data_dir,
                                       FLAGS.existing_stats_dir,
                                       DATA_TYPES,
                                       COPY_FILENAMES,
                                       NORMALISE_FILENAMES)

if __name__ == "__main__":
    app.run(main)


