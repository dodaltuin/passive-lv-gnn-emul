# Emulation of Cardiac Mechanics using Graph Neural Networks  

This repository contains Python scripts to perform the Graph Neural Network (GNN) emulation experiments as described in the paper Emulation of Cardiac Mechanics using Graph Neural Networks. Please cite the paper if you use this code.

## Environment Setup 

Experiments were performed with ``Python`` version 3.9.7, ``JAX`` version 0.3.16 and ``Flax`` version 0.3.6. The module pytest is required to run the test file [``models_test.py``](models_test.py), while tensorboard is also required to monitor training. To set up a virtual environment using [conda](https://www.anaconda.com/products/distribution), run the following commands in sequence once the repo has been cloned:

```
conda create --name gnnEmulEnv python=3.9.7
conda activate gnnEmulEnv
pip install "jax[cuda]==0.3.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade -r requirements.txt
```

### Import Error
Note if this error message appears when running experiments "ImportError: cannot import name 'isin' from 'jax._src.numpy.lax_numpy'", it can be resolved by changing the import on line 31 of ``~/anaconda3/envs/gnnEmulEnv/lib/python3.9/site-packages/flax/linen/module.py`` to
```
from jax.numpy import isin
```
The error occurs because  ``isin`` is no longer part of the private ``jax._src.numpy.lax_numpy`` submodule.

## Running Experiments
### Beam Data

The beam data is included in the repository, inside [``data/beamData``](/data/beamData). A GNN emulator with *K*=2 message passing steps for the beam data can be trained for 300 epochs as follows:
```
python -m main --mode="train" --n_epochs=300 --K=2 --data_path="beamData" --n_shape_coeff=2
```
The trained emulator can then be used to predict using the same command as above with ``"train"`` replaced with ``"evaluate"``.

### Left Ventricle Data (varying LV geometries)

The varying-geometry LV emulation data set is too large to be included in the repository - an external download link is available [here](https://zenodo.org/record/7075055). Assuming the data has been downloaded to ``data/lvData``, training can be performed as:

```
python -m main --mode="train" --n_epochs=3000 --K=5 --data_path="lvData" --n_shape_coeff=32
```

Again, the trained emulator can then be used to predict using the same command as above with ``"train"`` replaced with ``"evaluate"``.

### Left Ventricle Data (fixed LV geometry)

The emulation dataset for the fixed LV geometry is included in the repository, inside [``data/lvDataFixedGeom``](/data/lvDataFixedGeom). Training can be performed as follows:
```
python -m main --mode="train" --n_epochs=1000 --K=5 --lr=1e-5 --data_path="lvDataFixedGeom" --fixed_geom=True --n_shape_coeff=32 --trained_params_dir="emulationResults/trainedParameters/lvData/"
```

Note how we initialise training based on the pre-trained parameters from the varying LV geometry data (``/lvData``). Once training is complete, the emulator can be used to predict on the test data as follows:
```
python -m main --mode="evaluate" --n_epochs=600 --K=5 --lr=1e-5 --data_path="lvDataFixedGeom" --fixed_geom=True --n_shape_coeff=32
```

## Replicating Paper Results

Trained GNN parameters for each of the three datasets are stored in [``emulationResults/trainedParameters``](emulationResults/trainedParameters). Detailed instructions on how to to use these parameters to replicate paper results in given in [``PAPER_REPLICATION.md``](PAPER_REPLICATION.md)

## Monitoring Training

[Tensorboard](https://www.tensorflow.org/tensorboard) can be used to monitor training, for example for the beam data by running:
 ```
tensorboard --logdir=emulationResults/beamData
```
and then following the instructions printed to the console.


## Running Tests

Tests for the GNN implementation in [``models.py``](models.py) can be run as follows:
 ```
pytest models_test.py -v
```
## Additional Comments

1. **Optimisation:** is performed using ``flax.optim``, which is [now deprecated](https://github.com/google/flax/blob/main/docs/flip/1009-optimizer-api.md) in favour of [``Optax``](https://github.com/deepmind/optax)
2. **Batching:** a batch size of one is used in all examples - the code can easily extended to larger batches by "stacking" the graphs of multiple data points into one large graph (and shifting the sender/receiver indices accordingly), on which the existing emulators defined in [``models.py``](models.py) can then be applied
3. **Applying to other datasets:** the emulation framework can be applied to other datasets beyond those provided in this repository. See the file [``DATA_FORMAT_REQUIREMENTS.md``](/data/DATA_FORMAT_REQUIREMENTS.md) inside the subdirectory [``/data``](/data) for details of the required data format

## Directory Layout

## Files:

### [``main.py``](main.py)

Main script for training and evaluating emulators

### [``models.py``](models.py)

Implements *DeepGraphEmulator* GNN emulation architecture

### [``data_utils.py``](data_utils.py)

Contains a data loader utility class

### [``utils.py``](utils.py)

Contains utility functions for emulator training and evaluation

### [``requirements.txt``](requirements.txt)

Contains the packages in addition to ``JAX`` that are required for experiments to be run

### [``PAPER_REPLICATION.md``](PAPER_REPLICATION.md)

Details how the results from the paper can be reproduced

## Subdirectories:

### [``/data``](/data)

Stores simulation data for training and testing of the GNN emulator. Also contains scripts to process raw simulation data into the augmented graph format described in the manuscript: see the [``README.md``](data/README.md) file inside [``/data``](/data) for more details.

### [``/emulationResults``](/emulationResults)

Stores the trained emulator parameters and predictions.


