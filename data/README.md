# data

This directory contains raw data from the simulator and code to process data for use by the emulator.

Processed data ready to perform emulation is provided for [``/beamData``](beamData), ``/lvData`` and [``/lvDataFixedGeom``](lvDataFixedGeom). Emulation details are given in the parent directory.

``/lvData`` is too large to be uploaded to GitHub - the raw simulation data can be downloaded instead [at this link](https://zenodo.org/record/7075055).

Processing code is provided to process the raw ``/lvData`` and to allow the emulation framework to be easily applied on other datasets. This code is contained inside [``data_processing_utils.py``](data_processing_utils.py), which can be called using [``data_main.py``](data_main.py) as is detailed below. The requirement formats for the raw data to allow the above scripts to run are detailed in [``DATA_FORMAT_REQUIREMENTS.md``](DATA_FORMAT_REQUIREMENTS.md).

## Processing Data

### beamData

The processed data in [``/beamData``](beamData) was generated with the following command:

```
python -m data_main --mode="run_all" --data_dir="beamData" --n_nearest_nbrs=4 --n_leaves=4 --min_root_nodes=2
```
### lvData

The varying geometry ``/lvData`` can be processed as follows:

```
python -m data_main --mode="run_all" --data_dir="lvData" --n_nearest_nbrs=5 --n_leaves=8 --min_root_nodes=3
```

once it has been downloaded.

### lvDataFixedGeom

For the simulation data set run with a single fixed LV geometry [``/lvDataFixedGeom``](lvDataFixedGeom), we apply an emulator that is pretrained on the above ``/lvData`` dataset. To ensure consistency between the processed datasets, we require that that the fixed geometry data is processed into the same augmented graph topology and that the data is then normalised using the same summary statistics as ``/lvData``. This is achieved by setting the paths to the already existing topology and normalisation statistics subdirectories in ``/lvData`` as follows:
```
python -m data_main --mode="run_all" --data_dir="lvDataFixedGeom" --existing_topology_dir="lvData/topologyData"  --existing_stats_dir="lvData/normalisationStatistics"
```

Note that ``/lvData`` must be downloaded before the above can run.

## Replicating Paper Results
The above augmentation generation commands may produce different results when run on a different machine. The trained GNN parameters provided in the parent directory however were trained conditional on a specific set of augmentation results, meaning that the paper results will not necessarily replicate.

For the [``/beamData``](beamData) and [``/lvDataFixedGeom``](lvDataFixedGeom), the processed data used for the experiments presented in the paper are saved here, so replicating the results is not a problem. For the ``/lvData``, only the *unprocessed* data is provided [at this link](https://zenodo.org/record/7075055). After processing with the command given above, the results may not exactly match the processed data used originally and so the paper results will not replicate. To obtain processed ``/lvData`` so that the results will replicate, use the following command:
```
python -m data_main --mode="run_all" --data_dir="lvData" --existing_topology_dir="lvDataFixedGeom/topologyData"  --existing_stats_dir="lvDataFixedGeom/normalisationStatistics"
```

## Directory Layout

## Files:

### [``data_main.py``](data_main.py)

Main script for processing raw data into format suitable for emulation

### [``data_process_utils.py``](data_process_utils.py)

Utility functions for processing raw data

### [``DATA_FORMAT_REQUIREMENTS.md``](DATA_FORMAT_REQUIREMENTS.md)

Details of the required format for raw data

## Subdirectories:

### [``/beamData``](beamData)

Stores raw and processed simulation data for the beam experiment

### ``/lvData``

Stores raw and processed simulation data for the varying geometry LV experiment - note the raw data can be downloaded [from here](https://zenodo.org/record/7075055)

### [``/lvDataFixedGeom``](lvDataFixedGeom)

Stores raw and processed simulation data for the fixed geometry LV experiment
