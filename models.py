"""
File: models.py
Author: David Dalton
Description: Implements DeepGraphEmulator GNN Architecture
"""

import jax
import jax.numpy as jnp
import jax.tree_util as tree

from flax import linen as nn

from typing import Sequence, Callable

DTYPE = jnp.float32

class FlaxMLP(nn.Module):
  """Implements an MLP in Flax"""
  features: Sequence[int]
  layer_norm: bool

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, dtype=DTYPE)(x)
      if i != len(self.features) - 1:
        x = nn.tanh(x)
    if self.layer_norm:
        x = nn.LayerNorm()(x)
    return x

def make_mlp(features: Sequence[int]):
    """Makes standard MLP

    With hidden layers defined by features
    """
    def update_fn(inputs):
        return FlaxMLP(features, False)(inputs)
    return update_fn

def make_layernorm_mlp(features: Sequence[int]):
    """Makes MLP followed by LayerNorm

    With hidden layers specified by features
    """
    def update_fn(inputs):
        return (FlaxMLP(features, True)(inputs))
    return update_fn

def aggregate_incoming_messages(messages: jnp.ndarray, receivers: Sequence[int], n_nodes: int):
    """Sum aggregates incoming messages to each node

    Performs the sum over incoming messages $\sum_{j \in \mathcal{N}_i} m_{ij}^k$
    from the processor stage of Algorithm 2 of the manuscript, for all nodes $i$ similtaneously
    """
    return jax.ops.segment_sum(messages, receivers, n_nodes)

def MessagePassingStep(node_update_fn: FlaxMLP, edge_update_fn: FlaxMLP, senders: Sequence[int], receivers: Sequence[int], n_real_nodes: int):
    """Returns function to perform one message passing step

    Function _ApplyMessagePassingStep performs one step of message passing $k$ as
    in the for loop in Algorithm 2 of the manuscript.
    """

    def _ApplyMessagePassingStep(Vold: jnp.ndarray, Eold: jnp.ndarray):

        # calculate messages along each directed edge with an edge feature vector assigned
        messages = edge_update_fn(jnp.hstack((Eold, Vold[receivers], Vold[senders])))

        # aggregate incoming messages m_{ij} from nodes i to j where i > j
        received_messages_ij = aggregate_incoming_messages(messages, receivers, n_real_nodes)

        # aggregate incoming messages m_{ij} from nodes i to j where i < j
        # m_{ij} = -m_{ji} where i < j (momentum conservation property of the message passing)
        received_messages_ji = aggregate_incoming_messages(-messages, senders, n_real_nodes)

        # concatenate node representation with incoming messages and then update node representation
        V = node_update_fn(jnp.hstack((Vold, received_messages_ij + received_messages_ji)))

        # return updated node and edge representations with residual connection
        return Vold + V, Eold + messages

    return _ApplyMessagePassingStep


class DeepGraphEmulator(nn.Module):
    """DeepGraphEmulator (varying geometry data)"""
    mlp_features: Sequence[int]
    latent_size: Sequence[int]
    K: int
    receivers: Sequence[int]
    senders: Sequence[int]
    n_total_nodes: int
    output_dim: int
    real_node_indices: Sequence[bool]

    @nn.compact
    def __call__(self, V: jnp.ndarray, E: jnp.ndarray, theta: jnp.ndarray, z_global: jnp.ndarray = None, sow_latents: bool = False):
        """Implements Algorithm 2 of the manuscript: forward pass of DeepGraphEmulator

        Inputs:
        ---------
        V: jnp.ndarray
            Array giving feature vectors of each node (real and virtual)
        E: jnp.ndarray
            Array giving feature vectors of each edge
        theta: jnp.ndarray
            Vector of global graph parameters
        z_global: jnp.ndarray (Optional)
            Vector embedding of the global shape of the geometry
        sow_latents: bool
            Boolean controlling the output returned: see below for details

        Outputs:
        ---------
        U: jnp.ndarray (if sow_latents = False)
            Array of displacement predictions for each real node in V
        z_local: jnp.ndarray (if sow_latents = True)
            Array of latent embeddings $z_i^{local}$ for each real node $i$ in V,
            found by the Processor stage of the emulator.
        """

        ## Initialise internal MLPs:
        # 2 encoder MLPs
        node_encode_mlp=make_layernorm_mlp(self.mlp_features + self.latent_size)
        edge_encode_mlp=make_layernorm_mlp(self.mlp_features + self.latent_size)

        # 2K processor MLPs
        message_passing_blocks = [MessagePassingStep(make_layernorm_mlp(self.mlp_features + self.latent_size),
                                                     make_layernorm_mlp(self.mlp_features + self.latent_size),
                                                     self.senders, self.receivers, self.n_total_nodes) for i in range(self.K)]

        # 1 + D decoder MLPs
        theta_encode_mlp=make_layernorm_mlp(self.mlp_features + self.latent_size)
        node_decode_mlps = [make_mlp(self.mlp_features + (1,)) for i in range(self.output_dim[0])]

        ## Encoder:
        # encode vertices and edges
        V = node_encode_mlp(V)
        E = edge_encode_mlp(E)

        ## Processor:
        # perform K rounds of message passing
        for message_pass_block_i in message_passing_blocks:
            V, E = message_pass_block_i(V, E)

        # aggregate incoming messages to each node
        incoming_messages = aggregate_incoming_messages(E, self.receivers, self.n_total_nodes)

        # final local learned representation is a concatenation of vector embedding and incoming messages
        z_local = jnp.hstack((V, incoming_messages))

        # only need local representation for real nodes
        z_local = z_local[self.real_node_indices]

        # save value of final learned representation if required for fixed geometry emulator
        if sow_latents:
            return z_local

        ## Decoder:
        # encode global parameters theta
        z_theta = theta_encode_mlp(theta)

        # tile global values (z_theta and optionally z_global) to each individual real node
        if z_global is None:
            globals_array = jnp.tile(z_theta, (z_local.shape[0], 1))
        else:
            # stack z_global with z_theta if z_global is inputted
            global_embedding = jnp.hstack((z_theta, z_global.reshape(1,-1)))
            globals_array = jnp.tile(global_embedding, (z_local.shape[0],1))

        # final learned representation is (z_theta, z_local) or (z_theta, z_global, z_local)
        final_representation = jnp.hstack((globals_array, z_local))

        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [decode_mlp(final_representation) for decode_mlp in node_decode_mlps]

        # concatenate the predictions of each individual decoder mlp
        Upred = jnp.hstack(individual_mlp_predictions)

        # return displacment prediction array
        return Upred


class DeepGraphEmulatorFixedGeom(nn.Module):
    """DeepGraphEmulator (fixed geometry data)"""
    mlp_features: Sequence[int]
    output_dim: Sequence[int]
    n_real_nodes: int
    latent_nodal_values: jnp.ndarray
    theta_encode_mlp_fn: Callable

    @nn.compact
    def __call__(self, theta: jnp.ndarray):
        """Implements Algorithm 2 of the manuscript for a fixed input geometry

        Input:
        ---------
        theta: jnp.array
            Vector giving the global parameters $\theta$ for the fixed geom
            being modelled

        Output:
        ---------
        U: jnp.ndarray
            Array of displacement predictiosn for each real node
        """


        # initialise node-decode MLP
        node_decode_mlps = [make_mlp(self.mlp_features + (1,)) for i in range(self.output_dim[0])]

        # embed theta to higher dim space using pre-trained theta_encode_mlp
        z_theta = self.theta_encode_mlp_fn(theta)

        # make a copy of the theta embedding vector for each real (finite-element) node
        z_theta_array = jnp.tile(z_theta, (self.n_real_nodes,1))

        # latent_nodal_values contains the precomputed values of z^{global} and z^{local}
        final_representation = jnp.hstack((z_theta_array, self.latent_nodal_values))

        # make prediction for forward displacement using different decoder mlp for each dimension
        individual_mlp_predictions = [decode_mlp(final_representation) for decode_mlp in node_decode_mlps]

        # concatenate the predictions of each individual decoder mlp
        Upred = jnp.hstack(individual_mlp_predictions)

        # return displacment prediction array
        return Upred
