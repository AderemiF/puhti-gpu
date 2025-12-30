#!/bin/bash

#SBATCH --job-name=pipeline
#SBATCH --account=project_2001220
#SBATCH --partition=gputest
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --hint=nomultithread


module load pytorch/2.7
pip install -U --user pandas ollama

# Download ollama models to scratch rather than the home directory
OLLAMA_SCRATCH=/scratch/project_2001220/fayoyiwa/ollama
export OLLAMA_MODELS=${OLLAMA_SCRATCH}/models

# Add ollama installation dir to PATH
export PATH=/scratch/project_2001220/fayoyiwa/ollama/bin:$PATH

# Simple way to start ollama. All the server outputs will appear in
# the slurm log mixed with everything else.
#ollama serve &

# If you want to direct ollama server's outputs to a separate log file
# you can start it like this instead
mkdir -p ${OLLAMA_SCRATCH}/logs
ollama serve > ${OLLAMA_SCRATCH}/logs/${SLURM_JOB_ID}.log 2>&1 &

# Capture process id of ollama server
OLLAMA_PID=$!

# Wait to make sure Ollama has started properly
sleep 15

# After this you can use ollama normally in this session

MODEL=ministral-3:3b

# Example: use ollama commands
ollama pull $MODEL
ollama list

srun --cpu-bind=none python pipeline.py images.txt model.csv

# At the end of the job, stop the ollama server
kill $OLLAMA_PID

