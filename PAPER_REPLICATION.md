# Replicating Paper Results

The subdirectory [``emulationResults/trainedParameters``](emulationResults/trainedParameters) stores  trained parameters for one emulator for each dataset respectively. Configuration information for each emulator is given in its respective ``config_dict.txt`` file.

## Beam Data

The beam data paper emulation results (for ``K=2`` and augmented-graph representation) can be replicated as follows:

```
python -m main --mode="evaluate" --data_path="beamData" --K=2 --n_shape_coeff=2 --dir_label="_paperReplication" --trained_params_dir="emulationResults/trainedParameters/beamData/"
```

## LV Data (varying geometries)

The varying LV geometry data paper emulation results (for ``K=5`` and ``n_shape_coeff=32``) can be replicated as follows:

```
python -m main --mode="evaluate" --data_path="lvData" --K=5 --n_shape_coeff=32 --dir_label="_paperReplication" --trained_params_dir="emulationResults/trainedParameters/lvData/"               
```
### Note

In order for the varying LV geometry data results to replicate, the raw data in ``data/lvData`` must be processed as described under ``Replicating Paper Results`` in [``data/README.md``](data/README.md).

## LV Data (fixed geometry)

And finally the fixed LV geometry data results (for ``K=5`` and ``n_shape_coeff=32``) can be replicated as follows:

```
python -m main --mode="evaluate" --data_path="lvDataFixedGeom" --fixed_geom=True --K=5 --n_shape_coeff=32 --dir_label="_paperReplication" --trained_params_dir="emulationResults/trainedParameters/lvDataFixedGeom/"
```

