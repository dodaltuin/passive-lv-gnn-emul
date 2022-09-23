"""
File: models_test.py
Author: David Dalton
Description: Tests GNN Implementation in models.py
"""

import jax.numpy as jnp
from jax import random
import flax

import models
import pytest

# input variables
N_TEST_GRAPHS = 2
OUTPUT_DIMENSION_LIST = [2, 3]

MLP_FEATURES = [16, 16]
LATENT_SIZE = 8
K = 5

RNG_SEED = 44

# simple node and edge upate functions to test message passing with
node_update_fn = lambda arr: arr.sum(-1).reshape(-1,1)
edge_update_fn = lambda arr: arr.min(-1).reshape(-1,1)

def generate_graph_0():
    """
    Generates first synthetic graph to run tests on
    """

    # initialise graph 0
    n_nodes = 4
    output_dim = OUTPUT_DIMENSION_LIST[0]

    # generate (sparse) graph topology
    senders =   jnp.array([0, 1, 0, 2, 1, 2, 2, 3], dtype=jnp.int32)
    receivers = jnp.array([1, 0, 2, 0, 2, 1, 3, 2], dtype=jnp.int32)
    sparse_indices = senders > receivers
    topology = (senders[sparse_indices], receivers[sparse_indices])

    # edge features
    E = jnp.array([1, -1, 2, -2., 3, -3, 5, -5], dtype=jnp.float32).reshape(-1,1)[sparse_indices]
    ones = jnp.ones((E.shape[0],1), dtype=jnp.float32)
    E = jnp.concatenate((E, ones), axis=1)

    # node features
    V = jnp.array([0., 0., 1., 0.]).reshape(-1,1)

    # global variable
    theta = jnp.array([[3., 4.]])
    z_global = jnp.array([-1., 6., 9.])
    inputs = (V, E, theta, z_global)

    # true message aggregation value
    agg_results_true = jnp.array([[-3, 2], [-3 ,1], [-5, 1], [0, 0]], dtype=jnp.float32)

    # true updated edges / node features (conditional on above node and edge update functions)
    E_update_true = jnp.array([[-2, 0], [-4, -1], [-6, -2], [-10, -4]], dtype=jnp.float32)
    V_update_true = jnp.array([-3, -2, 2, 5], dtype=jnp.float32).reshape(-1,1)

    # tuple to hold the true outputs to test against
    true_values = (agg_results_true, E_update_true, V_update_true)

    return n_nodes, output_dim, topology, inputs, true_values

def generate_graph_1():
    """
    Generates second synthetic graph to run tests on
    """

    # initialise graph 1
    n_nodes = 4
    output_dim = OUTPUT_DIMENSION_LIST[1]

    # generate (sparse) graph topology
    senders =   jnp.array([0, 1, 0, 3, 1, 2, 2, 3], dtype=jnp.int32)
    receivers = jnp.array([1, 0, 3, 0, 2, 1, 3, 2], dtype=jnp.int32)
    sparse_indices = senders > receivers
    topology = (senders[sparse_indices], receivers[sparse_indices])

    # edge features
    E = jnp.array([[1, 0], [-1, 0], [0, -1], [0, 1], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.float32)[sparse_indices]

    # node features
    V = jnp.array([0, 1, 1, 0], dtype=jnp.float32).reshape(-1,1)

    # global variable
    theta = jnp.array([[5., 6.]])
    z_global = jnp.array([-1., 6., 9.])
    inputs = (V, E, theta, z_global)

    # true message aggregation value
    agg_results_true = jnp.array([[-1, 1], [0 ,1], [1, 0], [0, 0]], dtype=jnp.float32)

    # true updated edges / node features (conditional on above node and edge update functions)
    E_update_true = jnp.array([[-2, -1], [0, 1], [0, 1], [1, 0]], dtype=jnp.float32)
    V_update_true = jnp.array([-1, 3, 2, 0], dtype=jnp.float32).reshape(-1,1)

    # tuple to hold the true outputs to test against
    true_values = (agg_results_true, E_update_true, V_update_true)

    return n_nodes, output_dim, topology, inputs, true_values


def generate_graph(idx):
    assert idx == 0 or idx == 1, 'Only two test example graphs implemented'
    if idx == 0:
        return generate_graph_0()
    return generate_graph_1()


####################
### TEST ONE
### Test that the function "aggregate_incoming_messages"
### performs the correct sum aggregation
###################

def test_agg_incoming_messages():

    for i in range(N_TEST_GRAPHS):

        n_nodes, _, topology, inputs, true_values  = generate_graph(i)
        _, receivers = topology
        _, E, _, _ = inputs

        agg_results_true, _, _ = true_values
        agg_results_pred = models.aggregate_incoming_messages(E, receivers, n_nodes)

        assert jnp.array_equal(agg_results_pred, agg_results_true)


####################
### TEST TWO
### Test that the node/edge message-passing update mechanism
### follows the Algorithm presented in paper
###################

def test_message_passing_step():

    for i in range(N_TEST_GRAPHS):

        n_nodes, _, topology, inputs, true_values = generate_graph(i)
        V, E, _, _  = inputs
        _, E_update_true, V_update_true = true_values

        message_pass_fn = models.MessagePassingStep(node_update_fn, edge_update_fn, topology[0], topology[1], n_nodes)
        V_update_pred, E_update_pred = message_pass_fn(V, E)

        assert jnp.array_equal(V_update_pred, V_update_true)
        assert jnp.array_equal(E_update_pred, E_update_true)


####################
### TEST THREE
### Test that the fixed / varying geom emulators predict
### the correct output shape
###################

def predict_with_emulator(index, emulator_type):
    """
    Makes a forward prediction for graph index = 0/1,
    and emulator type = DeepGraphEmulator or DeepGraphEmulatorFixedGeom
    """
    # load test data
    n_nodes, output_dim, topology, inputs, _ = generate_graph(index)
    V, E, theta, z_global = inputs

    # initialise parameters
    key = random.PRNGKey(RNG_SEED)

    # initialise emulator
    if emulator_type == "fixed_geom":
        theta_encoder = models.FlaxMLP(MLP_FEATURES, True)
        encoder_params = theta_encoder.init(key, theta)
        theta_encoder_fn = lambda x: theta_encoder.apply(encoder_params, x)
        model = models.DeepGraphEmulatorFixedGeom(MLP_FEATURES, [output_dim], n_nodes, latent_nodal_values=V, theta_encode_mlp_fn=theta_encoder_fn)
        params = model.init(key, theta)
        # predict on test case
        pred = model.apply(params, theta)
    else:
        model = models.DeepGraphEmulator(MLP_FEATURES, [LATENT_SIZE], K, topology[1], topology[0], n_nodes, [output_dim], jnp.arange(n_nodes))
        params = model.init(key, V, E, theta, z_global)
        # predict on test case
        pred = model.apply(params, V, E, theta, z_global)

    assert pred.shape == (n_nodes, output_dim)


def test_prediction_shape():
    for emulator_type in ["varying_geom", "fixed_geom"]:
        for index in range(N_TEST_GRAPHS):
            predict_with_emulator(index, emulator_type)


####################
### TEST FOUR
### Check that DeepGraphEmulator and DeepGraphEmulatorFixedGeom
### make the same predictions given identical network parameters
### and input graph
###################

def predict_fixed_and_varying_emulator(index):
    """
    Compares the prediction of the varying geometry emulator (DeepGraphEmulator)
    against the fixed-geometry emulator (DeepGraphEmulatorFixedGeom) given the
    same input graph
    """
    # load test data
    n_nodes, output_dim, topology, inputs, _ = generate_graph(index)
    V, E, theta, z_global = inputs

    # initialise model and parameters
    key = random.PRNGKey(RNG_SEED)
    model = models.DeepGraphEmulator(MLP_FEATURES, [LATENT_SIZE], K, topology[1], topology[0], n_nodes, [output_dim], jnp.arange(n_nodes))
    params = model.init(key, V, E, theta, z_global)

    # predict on test case
    pred = model.apply(params, V, E, theta, z_global)

    # extract z_local
    z_local = model.apply(params, V, E, theta, z_global, sow_latents=True)

    assert z_local.shape == (n_nodes, LATENT_SIZE*2)

    # append z_global to z_local
    repeated_z_global = jnp.tile(z_global, (n_nodes, 1))
    latent_nodal_values = jnp.hstack((repeated_z_global, z_local))

    # extract theta-encoder mlp parameters
    theta_encode_mlp_index = (model.K+1)*2
    theta_encode_mlp_params = params['params'][f'FlaxMLP_{theta_encode_mlp_index}']
    theta_encode_mlp_params_dict = {'params': theta_encode_mlp_params.unfreeze()}
    theta_params = flax.core.frozen_dict.freeze(theta_encode_mlp_params_dict)

    # create that encoder function
    theta_mlp = models.FlaxMLP(MLP_FEATURES + [LATENT_SIZE] , True)
    theta_encoder_fn = lambda x: theta_mlp.apply(theta_params, x)

    # initialise fixed geometry emulator
    model_fg = models.DeepGraphEmulatorFixedGeom(MLP_FEATURES, [output_dim], n_nodes,
                                                 latent_nodal_values=latent_nodal_values,
                                                 theta_encode_mlp_fn=theta_encoder_fn)

    # index for first decoder MLP is one after the theta_encode_mlp
    node_decode_first_mlp_index = theta_encode_mlp_index + 1
    node_decode_last_mlp_index = node_decode_first_mlp_index + output_dim

    # extract parameters for "D" node decode MLPs to a list
    decoder_mlps_params_list = [params['params'][f'FlaxMLP_{index}'] for index in range(node_decode_first_mlp_index, node_decode_last_mlp_index)]

    # convert parameters list to a frozen_dict suitable for use with Flax
    decoder_mlps_params_dict = {'params': {f'FlaxMLP_{i}':params_i for i, params_i in enumerate(decoder_mlps_params_list)}}
    params_fg = flax.core.frozen_dict.freeze(decoder_mlps_params_dict)

    # predict on test case
    pred_fg = model_fg.apply(params_fg, theta)

    assert jnp.array_equal(pred_fg, pred)


def test_fixed_against_varying_emulator():
    """
    Test that fixed and varying geometry emulators give
    the same prediction for both test graphs
    """
    for index in range(N_TEST_GRAPHS):
        predict_fixed_and_varying_emulator(index)


