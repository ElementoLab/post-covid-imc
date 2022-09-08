# Multiplexed imaging of patients with post-acute COVID-19

[![Zenodo badge](https://zenodo.org/badge/doi/___doi1___.svg)](https://doi.org/___doi1___)

[![medRxiv badge](https://zenodo.org/badge/doi/__doi1___.svg)](https://doi.org/__doi1___) ⬅️ read the preprint here

Project description

## Organization

- The [metadata](metadata) directory contains metadata relevant to annotate the samples
- [This CSV file](metadata/samples.csv) is the master record of all analyzed samples
- The [src](src) directory contains source code used to analyze the data
- Raw data (i.e. MCD files) will be under the `data` directory.
- Processing of the data will create files under the `processed`  directory.
- Outputs from the analysis will be present in a `results` directory, with subfolders pertaining to each part of the analysis as described below.

## Reproducibility

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
transfer            [dev] Transfer data from wcm.box.com to local environment
process             [dev] Process raw data into processed data types
sync                [dev] Sync data/code to SCU server
upload_data         [dev] Upload processed files to Zenodo
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

#### Requirements

- Python 3.10+
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.
