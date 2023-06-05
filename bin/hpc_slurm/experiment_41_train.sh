#!/bin/bash
#SBATCH -J experiment_4_1 #job name
#SBATCH --time=07-00:00:00 # maximum duration is 7 days
#SBATCH -p preempt #in 'preempt'
#SBATCH -N 1  #1 nodes
#SBATCH -n 1  #30 number of cores (number of threads)
#SBATCH --gres=gpu:p100:1 # one P100 GPU, can request up to 6 on one node, total of 10, a100
#SBATCH --exclude=c1cmp[025-026]
#SBATCH -c 1 #1 cpu per task - leave this!
#SBATCH --mem=2g #requesting 60GB of RAM total
#SBATCH --output=./%x.%j.%N.out #saving standard output to file
#SBATCH --error=./%x.%j.%N.err # saving standard error to file
#SBATCH --mail-type=ALL # email optitions
#SBATCH --mail-user=muhammad.umair@tufts.edu


# Path to the set_env script is required
ENV_SCRIPT_PATH=./set_hpc_env.sh 
source $ENV_SCRIPT_PATH

#load anaconda module
module load ${ANACONDA_MOD}

# NOTE: If not using a100 GPU, load the appropriate cuda version.
module load ${CUDA_MODS}

# module load cuda/11.0 cudnn/8.0.4-11.0

# Get the GPU details
nvidia-smi

#activate conda environment
source activate $PYTHON_ENV_PATH

echo $EXPERIMENT_41_SCRIPT_PATH

python $EXPERIMENT_41_SCRIPT_PATH

conda deactivate
