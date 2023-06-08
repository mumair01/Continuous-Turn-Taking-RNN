#!/bin/bash 

# Sets the paths required by the Tufts University HPC specifically. 
# NOTE: This should not be run, and will not work for, for non-Tufts clusters. 

# ------ These can be changed to construct the exported paths

## Root user path 
# NOTE: This should not be loaded using dirnmae due to HPC constraints

## Conda env.
CONDA_ENV_DIR_REL_PATH=condaenv 
CONDA_ENV_NAME=ctt

## Project paths 
PROJECT_DIR_REL_PATH=projects
PROJECT_DIR_NAME=Continuous-Turn-Taking-RNN

# ## Scripts paths 
SCRIPT_DIR_REL_PATH=src/scripts

## Constructed paths
export USER_PATH=/cluster/tufts/deruiterlab/mumair01 
export PROJECT_PATH=${USER_PATH}/${PROJECT_DIR_REL_PATH}/${PROJECT_DIR_NAME}
export PYTHON_ENV_PATH=${USER_PATH}/${CONDA_ENV_DIR_REL_PATH}/${CONDA_ENV_NAME}
export EXPERIMENT_41_SCRIPT_PATH=${PROJECT_PATH}/${SCRIPT_DIR_REL_PATH}/experiment_4_1.py

## 
echo Paths set for Tufts HPC env:
echo USER_PATH=${USER_PATH}
echo PROJECT_PATH=${PROJECT_PATH}
echo PYTHON_ENV_PATH=${PYTHON_ENV_PATH}

## HPC Modules 
export ANACONDA_MOD=anaconda/2021.05
export CUDA_MODS="cuda/10.2 cudnn/7.1"

