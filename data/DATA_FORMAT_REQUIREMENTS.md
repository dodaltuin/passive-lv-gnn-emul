# Data Format Requirements

## Varying Geometry Data

Within the [``/beamData``](/beamData) and ``/lvData`` directories, there are two inital subdirectories required before data processing begins; ``/topologyData`` and ``/rawData``. We refer to these datasets as "varying geometry data" as each simulation is for a different initial geometry.


### topologyData

 Let *Nv* denote the number of real nodes used by the simulator to represent each geometry (*Nv*=96 for example for ``beamData``), *Ne* denote the number of edges and *D* be the dimensionality of the system (*D*=2 for ``beamData``, and *D*=3 for ``lvData``). Within ``/topologyData``, there are two files:
 
 ``representative-nodes.npy``: (*Nv* * *D*) array, which gives the nodal coordinates of the given representative geometry that is used to generate the augmented graph topology as described in Algorithm 1 of the manuscript.

 ``real-node-topology.npy``: (*Ne* * 2) array of integers. The first column gives the indices of the senders of each directed edge, and the second column gives the receiver node indices.


### rawData

Within ``/rawData``, there are three further subdirectories, for training / validation and test data. Let *Ntrain* denote the number of training simulations, *Dv* / *Dg* / *Ds* the dimensionality of the node features / global features / shape coefficients. Then within ``/rawData/train``, there are five initial raw data files:

```real-node-displacement.npy```: (*Ntrain* * *Nv* * *D*) array, where for each index *i=1, ... Ntrain*, the *ith* value of the array gives the (*Nv* x *D*) array of nodal displacements between the reference geometry configuration and the results of the simulator

```real-node-coords.npy```: (*Ntrain* * *Nv* * *D*) array, where for each index *i=1, ... Ntrain*, the *ith* value of the array gives the (*Nv* x *D*) array of nodal coordinates of the geometry in reference configuration

```real-node-features.npy```: (*Ntrain* * *Nv* * *Dv*) array, where for each index *i=1, ... Ntrain*, the *ith* value of the array gives the (*Nv* x *Dv*) array of node features of the geometry in reference configuration

```global-features.npy ```: (*Ntrain* x *Dg*) array, where for each index *i=1, ... Ntrain*, the *ith* value of the array gives the *Dg*-dimensional vector of global features, referred to as theta in the manuscript, for the corresponding simulation *i*

(OPTIONAL) ```shape-coeffs.npy```: (*Ntrain* * *Ds*) array, where for each index *i=1, ... Ntrain*, the *ith* value of the array gives the *Ds*-dimensional vector of global shape coefficients, referred to as z^{global} in the manuscript, for the corresponding simulation *i*

For validation and test data, the format is the same as above, replacing *Ntrain* above with *Nvalid* and *Ntest* respectively.


## Fixed Geometry Data

For [``/lvDataFixedGeom``](/lvDataFixedGeom), all simulations are run for a single, fixed geometry. Because of this, many of the files above do not need to be repeated for each different simulation. In particular, the first index of each of ```real-node-coords.npy```, ```real-node-features.npy``` and  ```shape-coeffs.npy``` only needs to be 1 and not *Ntrain* above.

In addition, applying a transfer learned emulator to the fixed geometry data, it is essential that the same augmented graph topology and data normalisation statistics that were used when training on the inital data should be used for the fixed geometry data - details of how this is done are given in the [``README.md``](README.md) in this subdirectory.
