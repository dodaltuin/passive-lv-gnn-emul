"""
File: utils.py
Author: David Dalton
Description: Utility functions for initialising emulators and saving results
"""

import os

import pathlib
import pickle

from jax import random
import jax.numpy as jnp
import flax

import models

def create_config_dict(K, n_shape_coeff, n_epochs, lr, output_dim,  local_embed_dim, mlp_features, rng_seed):
    """Creates dictionary of configuration details for the GNN emulator"""

    return {'K': K,
            'n_shape_coeff': n_shape_coeff,
            'n_train_epochs': n_epochs,
            'learning_rate': lr,
            'output_dim': output_dim,
            'local_embedding_dim': local_embed_dim,
            'mlp_features': mlp_features,
            'rng_seed': rng_seed
            }


def save_trained_params(epoch_loss, min_loss, params, epoch_idx, epochs_count, save_dir):
    """Saves the trained parameters of the GNN based on loss value"""

    if epoch_loss < min_loss:
        min_loss = epoch_loss
        with pathlib.Path(save_dir, f'trainedNetworkParams.pkl').open('wb') as fp:
            pickle.dump(params, fp)

    return min_loss


def create_savedir(data_path, K, n_shape_coeff, n_epochs, lr, dir_label, local_embedding_dim, mlp_width, mlp_depth, rng_seed):
    """Create directory where emulation resuls are saved

    The emulator's configuration details are written to the directory name for ease of reference
    """

    save_dir = f'emulationResults/{data_path}/dge_{K}_{n_shape_coeff}_{n_epochs}_{lr:.1e}_{local_embedding_dim}_{mlp_width}_{mlp_depth}_{rng_seed}{dir_label}/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    return save_dir


def load_trained_params(params_save_dir, params_filename="trainedNetworkParams.pkl"):
    """Load trained parameters of GNN emulator from params_save_dir"""

    params_filename_full = params_save_dir + params_filename
    if not os.path.isfile(params_filename_full):
        raise FileNotFoundError(f'No file at: {params_filename_full}')

    with pathlib.Path(params_filename_full).open('rb') as fp:
        params_load = pickle.load(fp)

    return params_load


def initialise_network_params(data_loader, model, trained_params_dir: str, rng_seed: int):
    """Initialise the parameters of the GNN emulator

    If initialising from scratch, use the ".init" method from Flax

    If initialising from earlier training results, simply read these parameters
    from trained_params_dir
    """

    if trained_params_dir == "None":
        key = random.PRNGKey(rng_seed)
        V_init, E_init, theta_init, z_global_init,  _ = data_loader.return_index_0()
        params = model.init(key, V_init, E_init, theta_init, z_global_init)
        return params
    else:
        trained_params = load_trained_params(trained_params_dir)
        return trained_params


def init_varying_geom_emulator(config_dict: dict, data_loader, fixed_geom: bool, trained_params_dir: str):
    """Initialise GNN emulator (varying geometry data)

    Initialises GNN architecture and trainable paramters for prediction of varying LV geom data

    If trained_params_dir is "None", the parameters are initialised randomly
    If trained_params_dir is a directory path, pre-trained parameters are read from there
    """

    # initialise GNN architecture based on configuration details
    model =  models.DeepGraphEmulator(mlp_features=config_dict['mlp_features'],
                                      latent_size=[config_dict['local_embedding_dim']],
                                      K = config_dict['K'],
                                      receivers = data_loader._receivers,
                                      senders = data_loader._senders,
                                      n_total_nodes= data_loader._n_total_nodes,
                                      output_dim= [config_dict['output_dim']],
                                      real_node_indices = data_loader._real_node_indices)

    # initialise trainable emulator parameters (either randomly or read from trained_params_dir)
    params = initialise_network_params(data_loader, model, trained_params_dir, config_dict['rng_seed'])

    return model, params


# When emulating data for a fixed LV geometry, the message passing stage
# of the GNN is pre-computed once when the emulator is initialised.
# For this reason, initialiation of the fixed LV geometry emulator differs
# depending on whether the initialiation uses the pre-trained parameters
# from a varying geometry emulator (in which case the message-passing
# pre-computation must be performed first before initialiation) or from
# the pre-trained parameters of a fixed geometry emulator (in which case
# the pre-computation has already been performed, so these parameters can
# simply be read from the save directory)
# The former case is performed by "init_fixed_geom_emulator_from_varying",
# and the latter by "init_fixed_geom_emulator_from_fixed", both of which
# are defined below

def init_fixed_geom_emulator_from_varying(results_save_dir, model, params, data_loader):
    """Initialises fixed-geom emulator network parameters

    Initialises the parameters of a GNN emulator for fixed LV geometry data,
    where the pre-trained parameters come from a varying geometry emulator

    In this case, the message passing stage for the fixed geometry must
    be pre-compututed before the emulator (DeepGraphEmulatorFixedGeom) is
    initialised
    """

    # extract the node features (V), edge features (E), global parameters (theta)
    # and global shape embedding coefficients (z_global)
    V, E, theta, z_global, _ = data_loader.return_index_0()

    # sow_latents=True returns the local learned reprentation from the message
    # passing stage or each node in the geometry
    latent_nodal_values = model.apply(params, V, E, theta, z_global, sow_latents=True)

    # If including shape coefficients, make a copy for each individual node and append to latent_nodal_values
    if z_global is not None:
        #repeated_z_global = jnp.repeat(data_loader._shape_coeffs, data_loader._n_real_nodes, axis=0).reshape(data_loader._n_real_nodes, data_loader._n_shape_coeff)
        repeated_z_global = jnp.tile(data_loader._shape_coeffs, (data_loader._n_real_nodes, 1))
        latent_nodal_values = jnp.hstack((repeated_z_global, latent_nodal_values))

    # Flax names the internal MLPs of the GNN 'FlaxMLP_i', where i ranges over the
    # number of MLPs in the network, and is numbered in order of initialisation
    # There are 2 (the two encode MLPs) + K*2 (two MLPs for each message passing step)
    # + 1 (the theta encode MLP) + D decoder MLPs = 3 + K*2 + D MLPs in the DeepGraphEmulator
    # GNN architecture. We want the index of the theta encoder MLP for use in the fixed
    # geometry emulator, which was the (2*K + 3)th MLP to be initialised. Therefore its
    # index is given as follows (Python uses 0-based indexing)
    theta_encode_mlp_index = model.K*2 + 2

    # extract theta encoder MLP params from params dictionary
    theta_encode_mlp_params = params['params'][f'FlaxMLP_{theta_encode_mlp_index}']
    theta_encode_mlp_params_dict = {'params': theta_encode_mlp_params.unfreeze()}
    theta_encode_mlp_params_dict = flax.core.frozen_dict.freeze(theta_encode_mlp_params_dict)

    # define a function to map theta to z_theta given value for theta encode MLP parameters
    theta_encode_mlp = models.FlaxMLP(model.mlp_features + model.latent_size, True)
    theta_encode_mlp_fn = lambda x: theta_encode_mlp.apply(theta_encode_mlp_params_dict, x)

    # initialise the fixed geometry (fg) emulator
    model_fg = models.DeepGraphEmulatorFixedGeom(mlp_features = model.mlp_features,
                                                 output_dim= [data_loader._output_dim],
                                                 n_real_nodes= data_loader._n_real_nodes,
                                                 latent_nodal_values= latent_nodal_values,
                                                 theta_encode_mlp_fn=theta_encode_mlp_fn)

    # index for first decoder MLP is one after the theta_encode_mlp
    node_decode_first_mlp_index = theta_encode_mlp_index + 1
    node_decode_last_mlp_index = node_decode_first_mlp_index + model.output_dim[0]

    # extract parameters for "D" node decode MLPs to a list
    decoder_mlps_params_list = [params['params'][f'FlaxMLP_{index}'] for index in range(node_decode_first_mlp_index, node_decode_last_mlp_index)]

    # convert parameters list to a frozen_dict suitable for use with Flax
    decoder_mlps_params_dict = {'params': {f'FlaxMLP_{i}':params_i for i, params_i in enumerate(decoder_mlps_params_list)}}
    params_fg = flax.core.frozen_dict.freeze(decoder_mlps_params_dict)

    # save calculate values so that next time they can be simply loaded in using
    # "init_fixed_geom_emulator_from_fixed" below
    jnp.save(results_save_dir + 'latent_nodal_values.npy', latent_nodal_values)
    with pathlib.Path(results_save_dir, f'theta_encode_params.pkl').open('wb') as fp:
        pickle.dump(theta_encode_mlp_params_dict, fp)

    return model_fg, params_fg


def init_fixed_geom_emulator_from_fixed(trained_params_dir, results_save_dir, model, data_loader):
    """Initialises fixed-geom emulator network parameters

    Initialises the parameters of a GNN emulator for fixed LV geometry data,
    where the pre-trained parameters come from a fixed geometry emulator

    In this case, the message passing precomputation has already been performed
    and is saved in results_save_dir as "latent_nodal_values.npy", and
    the trainable parameters of the fixed geom emulator (from the node decode
    MLP) are also saved in results_save_dir
    """

    # load pre-computed learned node reprentations from the message passing stage
    latent_nodal_values = jnp.load(trained_params_dir + 'latent_nodal_values.npy')

    # load pre-trained theta_encode mlp params
    theta_encoder_params_load = load_trained_params(trained_params_dir, 'theta_encode_params.pkl')
    theta_encode_mlp = models.FlaxMLP(model.mlp_features + model.latent_size, True)
    theta_encode_mlp_fn = lambda x: theta_encode_mlp.apply(theta_encoder_params_load, x)

    # if we are loading from a different directory (trained_params_dir) to whre we are saving rsults
    # results_save_dir, write the nodal valeus and theta_encoder mlp params to results_save_dir so
    # they can be loaded later from there if needed
    if trained_params_dir != results_save_dir:
        jnp.save(results_save_dir + 'latent_nodal_values.npy', latent_nodal_values)
        with pathlib.Path(results_save_dir, f'theta_encode_params.pkl').open('wb') as fp:
            pickle.dump(theta_encoder_params_load, fp)

    # initialise the fixed geometry emulator
    model_fg = models.DeepGraphEmulatorFixedGeom(mlp_features = model.mlp_features,
                                                 output_dim= [data_loader._output_dim],
                                                 n_real_nodes= data_loader._n_real_nodes,
                                                 latent_nodal_values= latent_nodal_values,
                                                 theta_encode_mlp_fn=theta_encode_mlp_fn)

    # load the pre-trained parameters of the node-decode MLP
    decoder_params_load = load_trained_params(trained_params_dir)

    return model_fg, decoder_params_load


def init_fixed_geom_emulator(model, params, data_loader, results_save_dir, trained_params_dir):
    """Initialises fixed-geom emulator network parameters

    If "latent_nodal_values.npy" exists in results_save_dir, this means
    the pre-computation of the message passing stage of the GNN has already
    been performed, in which case "init_fixed_geom_emulator_from_fixed" can be
    used to initialise the fixed-geom emulator parameters

    If this file doesn't exist, it means that the pre-computation has not yet
    been performed, in which case "init_fixed_geom_emulator_from_varying" is
    used to initialise the fixed-geometry emulator
    """

    if os.path.isfile(trained_params_dir + 'latent_nodal_values.npy'):
        model_fg, params_fg = init_fixed_geom_emulator_from_fixed(trained_params_dir, results_save_dir, model, data_loader)
    else:
        model_fg, params_fg = init_fixed_geom_emulator_from_varying(results_save_dir, model, params, data_loader)

    return model_fg, params_fg


def initialise_emulator(emulator_config_dict, data_loader, results_save_dir, fixed_geom=False, trained_params_dir="None"):
    """Initialises GNN emulator

    Initialises GNN emulator and parameters, given specifications given in emulator_config_dict

    If fixed_geom=False, returns instance of models.DeepGraphEmulator (varying geometry emulator)
    If fixed_geom=True, returns instance of models.DeepGraphEmulatorFixedGeom (fixed geometry emulator)
    """

    # check input conflict
    if fixed_geom:
        assert trained_params_dir != "None", \
               'If using fixed_geometry emulator, must initialise from pre-trained network parameters'

    # initialise varying geometry emulator (models.DeepGraphEmulator) and parameters
    model, params = init_varying_geom_emulator(emulator_config_dict, data_loader, fixed_geom, trained_params_dir)

    # if using fixed geometry data, use the above model and parameters to initialise
    # a fixed geometry emulator (models.DeepGraphEmulatorFixedGeom) by pre-computing
    # the message passing stage of the emulation architecture
    if fixed_geom:
        model, params = init_fixed_geom_emulator(model, params, data_loader, results_save_dir, trained_params_dir)

    return model, params

def print_error_statistics(Utrue, Upred, logging):
    """Prints prediction error statistics to console
    """

    # calculate point-wise RMSE between true and predicted values
    def rmse(true, pred=0):
        return (((true - pred)**2).sum(-1))**.5

    # find the average magnitude of the true displacement vectors
    mean_norm = rmse(Utrue).mean()

    # rmse errors between true and predicted displacements
    prediction_errors = rmse(Utrue, Upred)

    # find 25th, 50th (median) and 75th percentile values of the prediction errors
    error_quantiles = jnp.percentile(prediction_errors, jnp.array([25., 50., 75.]))

    # print results to console
    logging.info(f'Mean Displacement Vector Norm: {mean_norm:.2f}')
    logging.info(f'Prediction Error Percentiles: 25%:{error_quantiles[0]:.2e}, 50%:{error_quantiles[1]:.2e}, 75%:{error_quantiles[2]:.2e}')


