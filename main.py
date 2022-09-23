"""
File: train.py
Author: David Dalton
Description: Training and Evaluation script for LV Emulation with GNNs

The objective is to predict the nodal displacement of the LV geometry,
given the specific anatomy and material stiffness parameters (theta)
of the LV. Prediction is performed using a Graph Neural Network (GNN)
emulator called DeepGraphEmulator, which is implemented in models.py

Three datasets are considered "beamData", "lvData" and "lvDataFixedGeom",
all stored in the "/data" subdirectory.

The first dataset contains simulation results for the textbook example
of a beam, clamped at one end, deforming under its own weight due to
gravity. The dataset consists of 225 train, 25 validation and 50 test
simulations. Each simulation was run with different values of the
Lame parameters, and for different beam geometries that varied in
width and length

The second dataset contains simulation data for passive mechanics of the
left ventricle (LV), where each data point is for a different LV geometry
and material parameter configuration. There are 3000 simulation in
total, split between training/validation/test as 2250/150/600

The final dataset contains additional simulation results for one of the
test geometries in "lvData". All simulations are run for this fixed
LV geometry, with varying material parameters. The train/validation/test
split is 85/15/250

The GNN architecture is designed so that the message passing stage
of the model can be precomputed in the case of a fixed LV geomet.ry.
This means that predictions are made differently based on whether
we consider data with varying LV geometries or a fixed geometry.
This difference is reflected in a number of functions below, suffixed
with "_fg" which indicates that they are used for fixed geometry data.

Example usage: the following command can be used to train an emulator
on the beam data:

python -m train --mode="train" --data_path=beamData --n_epochs=300 --K=5 --n_shape_coeff=2

Once the emulator is trained, change --mode="evaluate" above and re-run
to evaluate the out of sample predictive performance of the emulator
"""

#############################################################################
## Package imports
#############################################################################

import jax
from jax import random, jit, numpy as jnp
import numpy as np

import flax
from flax import optim
from flax.training import train_state
from flax.metrics import tensorboard

import data_utils as data_utils
import utils

import functools

from absl import app
from absl import flags
from absl import logging

#############################################################################
## Set up shell input variables
#############################################################################

flags.DEFINE_enum('mode', 'train', ['train', 'evaluate'], help = 'Fit model to training data or evaluate on test data')
flags.DEFINE_integer('n_epochs', 300, lower_bound = 1, help = 'Number of epochs to train the model for')
flags.DEFINE_string('data_path', 'beamData4NonAugmented', help = 'Name of sub-directory in "/data" where simulation data is stored')
flags.DEFINE_string('trained_params_dir', "None", help='Path to directory with pre-trained network parameters')
flags.DEFINE_boolean('fixed_geom', False, help='Set whether the dataset is for varying or fixed geometries')
flags.DEFINE_float('lr', 5e-5, lower_bound=0, help='Learning rate for training the network')
flags.DEFINE_integer('n_shape_coeff', 2, lower_bound=0, help='Number of shape coefficients to input to the emulator')
flags.DEFINE_integer('K', 2, lower_bound=1, help='Number of message passing steps to perform')
flags.DEFINE_string('dir_label', '', help='Optional label to append to end of results save directory')
FLAGS = flags.FLAGS

#############################################################################
## Set up hard-coded input variables
#############################################################################

# MLP OPTIONS
MLP_WIDTH = 128
MLP_DEPTH = 2
MLP_FEATURES = [MLP_WIDTH] * MLP_DEPTH

# MeshGraphNets hyper-parameter values:
LOCAL_EMBED_DIM = 40

# Random seed for initialising network parameters
RNG_SEED = 753

# function to create subdir to save emulation results
create_savedir = functools.partial(utils.create_savedir,
                                   local_embedding_dim=LOCAL_EMBED_DIM,
                                   mlp_width=MLP_WIDTH,
                                   mlp_depth=MLP_DEPTH,
                                   rng_seed=RNG_SEED)

# function to create configuration dictionary of emulator hyper-parameters
create_config_dict = functools.partial(utils.create_config_dict,
                                       local_embed_dim=LOCAL_EMBED_DIM,
                                       mlp_features=MLP_FEATURES,
                                       rng_seed=RNG_SEED)

#############################################################################
## Define loss functions
#############################################################################

def rmse(true, pred):
    return (((true - pred)**2).sum(-1))**.5

def mean_rmse(true, pred):
    return jnp.mean(rmse(true, pred))

#############################################################################
## Prediction / training / evaluation functions for varying-geometry emulator
## Training is done on normalised scale, evaluation on un-normalised scale (cm)
#############################################################################

def predict_norm_displacement(V, E, theta, z_global, params, net):
    return net.apply(params, V, E, theta, z_global)

def predict_displacement(V, E, theta, z_global, params, Umean, Ustd, net):
    return predict_norm_displacement(V, E, theta, z_global, params, net) * Ustd + Umean

def compute_train_loss(params, net, input_output_tuple):
    """Compute (mean) MSE between true and predicted displacement (normalised)"""
    V, E, theta, z_global, U = input_output_tuple
    Upred = predict_norm_displacement(V, E, theta, z_global, params, net)
    return mean_rmse(U, Upred)

def compute_test_loss(params, input_output_tuple, Umean, Ustd, net):
    """Compute RMSE loss between true and predicted displacement (un-normalised)"""
    V, E, theta, z_global, U = input_output_tuple
    Upred = predict_displacement(V, E, theta, z_global, params, Umean, Ustd, net)
    return mean_rmse(U, Upred)

#############################################################################
## Prediction / training / evaluation functions for fixed-geometry (fg) emulator
## Training is done on normalised scale, evaluation on un-normalised scale (cm)
#############################################################################

def predict_norm_displacement_fg(theta, params, net):
    return net.apply(params, theta)

def predict_displacement_fg(theta, params, Umean, Ustd, net):
    return predict_norm_displacement_fg(theta, params, net) * Ustd + Umean

def compute_train_loss_fg(params, input_output_tuple, net):
    """Compute (mean) MSE between true and predicted displacement (normalised)"""
    theta, U = input_output_tuple
    Upred = predict_norm_displacement_fg(theta, params, net)
    return mean_rmse(U, Upred)

def compute_test_loss_fg(params, input_output_tuple, Umean, Ustd, net):
    """Compute RMSE loss between true and predicted displacement (un-normalised)"""
    theta, U = input_output_tuple
    Upred = predict_displacement_fg(theta, params, Umean, Ustd, net)
    return mean_rmse(U, Upred)

#############################################################################
## Batch / Epoch training and evaluation functions
#############################################################################

def select_loss_fns(fixed_geom: bool):
    """Selects the loss function used to train the network

    The loss function used depends whether we consider
    varying geometry data (fixed_geom=False) or fixed
    geometry data (fixed_geom=True)
    """
    if fixed_geom:
        return compute_train_loss_fg, compute_test_loss_fg
    return compute_train_loss, compute_test_loss

def train_step(state, input_output_tuple, net, train_loss_fn):
    """Train network for a single input/output ([V,E,H]/U) pair"""
    partial_loss_fn = functools.partial(
        train_loss_fn, net=net, input_output_tuple=input_output_tuple)
    grad_fn = jax.value_and_grad(partial_loss_fn)
    loss, grads = grad_fn(state.target)
    state = state.apply_gradient(grads=grads)
    return state, loss

def train_epoch(state, data_loader, train_step_fn, epoch_idx):
    """Train network for one epoch"""
    data_loader.shuffle_epoch_indices(epoch_idx)
    loss = 0.
    for graph_idx in data_loader._epoch_indices:
        input_output_tuple_idx = data_loader.get_graph(graph_idx)
        state, loss_idx = train_step_fn(state, input_output_tuple_idx)
        loss += loss_idx
    return state, loss/data_loader._data_size

def validate_epoch(network_params, data_loader, loss_fn):
    """Find mean-RMSE loss all data points in data_loader"""
    loss_epoch = 0.
    for graph_idx in data_loader._epoch_indices:
        input_output_tuple_idx = data_loader.get_graph(graph_idx)
        loss_epoch += loss_fn(network_params, input_output_tuple_idx)
    return loss_epoch/data_loader._data_size

#############################################################################
## Functions for prediction of entire test dataset
#############################################################################

def emul_pred_fn_vg(input_output_tuple, params, Umean, Ustd, net):
    # for varying-geom (vg) data, node features (V), edge features (E)
    # and global features (globals) are inputted to the emulator
    V, E, theta, z_global, _ = input_output_tuple
    return predict_displacement(V, E, theta, z_global, params, Umean, Ustd, net)

def emul_pred_fn_fg(input_output_tuple, params, Umean, Ustd, net):
    # for fixed-geom (fg) data, only theta needs to be inputted
    theta, _ = input_output_tuple
    return predict_displacement_fg(theta, params, Umean, Ustd, net)

def select_pred_fn(fixed_geom: bool):
    """Selects the function used to predict test points

    The prediction function used depends whether we
    consider varying geometry data (fixed_geom=False)
    or fixed geometry data (fixed_geom=True)
    """
    if fixed_geom:
        return emul_pred_fn_fg
    return emul_pred_fn_vg

def predict_dataset(data_loader, pred_fn):
    """Make predictions for entire dataset using (jitted) prediction function"""
    Upred = np.zeros_like(data_loader._displacement)
    for graph_idx in data_loader._epoch_indices:
        input_output_tuple = data_loader.get_graph(graph_idx)
        Upred[graph_idx] = np.array(pred_fn(input_output_tuple))
    return Upred


#############################################################################
## Final training and evaluation functions
#############################################################################

def train(data_path, K, n_shape_coeff, n_epochs, lr, trained_params_dir, fixed_geom, dir_label):

    logging.info(f'Beginning Training')
    logging.info(f'Data path: {data_path}')
    logging.info(f'Message passing steps (K): {K}')
    logging.info(f'Num. shape coeffs: {n_shape_coeff}')
    logging.info(f'Training epochs: {n_epochs}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Trained Params Dir: {trained_params_dir}')
    logging.info(f'Fixed LV geom: {fixed_geom}\n')

    # load train data
    train_data = data_utils.DataLoader(data_path, 'train', n_shape_coeff, fixed_geom)
    logging.info(f'Number of train data points: {train_data._data_size}')

    # load validation data
    valid_data = data_utils.DataLoader(data_path, 'validation', n_shape_coeff, fixed_geom)
    logging.info(f'Number of validation data points: {valid_data._data_size}\n')

    # create directory to store the trained parameters of the network
    results_save_dir = create_savedir(data_path, K, n_shape_coeff, n_epochs, lr, dir_label)
    logging.info(f'Results save directory: {results_save_dir}\n')

    # create configuration dictionary of hyperparameters of the GNN emulator
    config_dict = create_config_dict(K, n_shape_coeff, n_epochs, lr, train_data._output_dim)

    # write configuration dictionary to a text file in results_save_dir
    with open(f'{results_save_dir}/config_dict.txt', 'w') as f: print(config_dict, file=f)

    # initialise GNN emulator and trainable emulator parameters
    model, params = utils.initialise_emulator(config_dict, train_data, results_save_dir, fixed_geom, trained_params_dir)

    # initialise Adam optimiser
    state = optim.Adam(learning_rate=lr).create(params)

    # create tensorboard SummaryWriter to monitor training
    summary_writer = tensorboard.SummaryWriter(results_save_dir)
    summary_writer.hparams(config_dict)

    # placehold variable to track lowest validation set prediction error
    min_valid_loss = 1e7

    # select training and evaluation loss functions for fixed / varying LV geometry emulator
    train_loss_fn, compute_test_loss_fn = select_loss_fns(fixed_geom)

    # jit train and evaluation functions for faster execution
    train_step_jit = jit(functools.partial(train_step,
                                           net=model,
                                           train_loss_fn=train_loss_fn))

    valid_step_jit = jit(functools.partial(compute_test_loss_fn,
                                           net=model,
                                           Umean=valid_data._displacement_mean,
                                           Ustd=valid_data._displacement_std))

    logging.info(f'Beginning training for {n_epochs} epochs')
    for epoch_idx in range(n_epochs):

        # train network for one epoch
        state, train_loss = train_epoch(state, train_data, train_step_jit, epoch_idx)

        # calculate mean node-wise RMSE on validation data
        valid_loss = validate_epoch(state.target, valid_data, valid_step_jit)

        # write loss values to tensorboard summary_writer
        summary_writer.scalar('train_loss', train_loss, epoch_idx)
        summary_writer.scalar('valid_loss', valid_loss, epoch_idx)

        if (epoch_idx % 250 == 0) or (epoch_idx < 50):
            logging.info(f'({epoch_idx}): train_loss={train_loss:.3e}, valid_loss={valid_loss:.3e}')

        # save trained network parameters based on validation set prediction error
        min_valid_loss = utils.save_trained_params(valid_loss, min_valid_loss, state.target, epoch_idx, n_epochs, results_save_dir)

    summary_writer.flush()

def evaluate(data_path, K, n_shape_coeff, n_epochs, lr, trained_params_dir, fixed_geom, dir_label):

    logging.info('Beginning Evaluation')
    logging.info(f'Data path: {data_path}')
    logging.info(f'Message passing steps (K): {K}')
    logging.info(f'Num. shape coeffs: {n_shape_coeff}')
    logging.info(f'Training epochs: {n_epochs}')
    logging.info(f'Learning rate: {lr}')
    logging.info(f'Trained Params Dir: {trained_params_dir}')
    logging.info(f'Fixed LV geom: {fixed_geom}\n')

    # load test data
    test_data = data_utils.DataLoader(data_path, 'test', n_shape_coeff, fixed_geom)
    logging.info(f'Number of test data points: {test_data._data_size}')

    # create dictionary of hyperparameters of the GNN emulator
    config_dict = create_config_dict(K, n_shape_coeff, n_epochs, lr, test_data._output_dim)

    # create directory to store the trained parameters of the network
    results_save_dir = create_savedir(data_path, K, n_shape_coeff, n_epochs, lr, dir_label)
    logging.info(f'Results save directory: {results_save_dir}\n')

    # if trained_params_dir is not set, parameters are read from results_save_dir
    if trained_params_dir == "None": trained_params_dir = results_save_dir

    # initialise GNN emulator and read trained network parameters
    model, trained_params = utils.initialise_emulator(config_dict, test_data, results_save_dir, fixed_geom, trained_params_dir)

    # select the emulator predict function based on whether we consider fixed or varying LV geometry data
    emul_pred_fn = select_pred_fn(fixed_geom)

    # jit prediction for faster execution
    prediction_fn_jit = jit(functools.partial(emul_pred_fn,
                                              net=model,
                                              params=trained_params,
                                              Umean=test_data._displacement_mean,
                                              Ustd=test_data._displacement_std))

    logging.info('Predicting on test data set using trained emulator')
    Upred = predict_dataset(test_data, prediction_fn_jit)

    logging.info('Saving Results')
    np.save(f'{results_save_dir}/predDisplacement.npy', Upred)
    np.save(f'{results_save_dir}/trueDisplacement.npy', test_data._displacement)

    # save coordinates of test geometries in reference configuration
    np.save(f'{results_save_dir}/referenceCoords.npy', test_data._real_node_coords)

    # print prediction error statistics to console
    utils.print_error_statistics(test_data._displacement, Upred, logging)


def main(_):

    # select function to either train or evaluate emulator
    run_fn = train if FLAGS.mode == 'train' else evaluate

    run_fn(FLAGS.data_path,
           FLAGS.K,
           FLAGS.n_shape_coeff,
           FLAGS.n_epochs,
           FLAGS.lr,
           FLAGS.trained_params_dir,
           FLAGS.fixed_geom,
           FLAGS.dir_label
           )

if __name__ == "__main__":
    app.run(main)

