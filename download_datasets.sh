#!/bin/bash

if [[ -n "$VIRTUAL_ENV" || -n "$CONDA_PREFIX" ]]; then
    echo "INFO: running on Environment"
else
    echo "WARNING: running on system Python"
fi

# CPSMI2025
kaggle datasets download suyash1567/cpsmi2025

# Medneley LBC
kaggle datasets download blank1508/mendeley-lbc-cervical-cancer

# SIPaKMeD
kaggle datasets download prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed

# Herlev
kaggle datasets download yuvrajsinhachowdhury/herlev-dataset

# BMT
mkdir -p ./BTM
cd ./BTM
synapse get -r syn55262661
cd ..

# HiCervix
mkdir -p ./HiCervix
cd ./HiCervix
zenodo_get -o . 11087263 -a $ZENODO_PAT
cd ..
