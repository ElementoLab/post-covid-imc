# Multiplexed imaging of patients with post-acute COVID-19

[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.7060271.svg)](https://doi.org/10.5281/zenodo.7060271)

[![medRxiv badge](https://zenodo.org/badge/doi/10.1101/2022.11.28.22282811.svg)](https://doi.org/10.1101/2022.11.28.22282811) ⬅️ read the preprint here

This repository contains source code used to analyze the data in the above manuscript.

## Organization

- The [metadata](metadata) directory contains metadata relevant to annotate the samples
- [This CSV file](metadata/samples.csv) is the master record of all analyzed samples
- The [src](src) directory contains source code used to analyze the data
- Raw data (i.e. MCD files) will be under the `data` directory.
- Processing of the data will create files under the `processed`  directory.
- Outputs from the analysis will be present in a `results` directory, with subfolders pertaining to each part of the analysis as described below.

## Reproducibility

### Requirements

- Python 3.10+
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.

### Running

To see all available steps type:
```bash
$ make
```

Steps used for the initiall processing of raw data are marked with the `[dev]` label.
```
Makefile for the covid-pasc-imc project/package.
Available commands:
help                Display help and quit
requirements        Install Python requirements
download_data       Download processed data from Zenodo (for reproducibility)
analysis            Run the actual analysis
```

To reproduce analysis using the pre-preocessed data, one would so:

```bash
$ make help
$ make requirements   # install python requirements using pip
$ make download_data  # download processed from Zenodo
$ make analysis       # run the analysis scripts
```
