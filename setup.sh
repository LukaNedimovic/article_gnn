#!/bin/bash

# ---------------------------------------------------------------------
# This script sets up project environment.
# This project uses Conda for package management.
# Remember to run the script using: `source ./setup.sh`.
# ---------------------------------------------------------------------

# Project root is assumed to be the place where this script is ran
export PROJECT_ROOT=$(pwd)

# Activate Conda virtual environment
CONDA_ENV_NAME="article_gnn_env"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

# Check if the conda environment exists
if ! conda info --envs | grep -w "$CONDA_ENV_NAME" > /dev/null; then
  echo "Conda environment '$CONDA_ENV_NAME' does not exist. Creating it..."
  conda create --name "$CONDA_ENV_NAME" python=3.12.1 -y
else
  echo "Conda environment '$CONDA_ENV_NAME' already exists."
fi

# Activate the conda environment
echo "Activating the conda environment..."
conda activate "$CONDA_ENV_NAME"

# Add channels to successfully download packages
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia
conda config --add channels defaults
conda config --add channels huggingface
conda config --add channels pyg

# Install or update dependencies using Conda
if [[ -f "$REQUIREMENTS_FILE" ]]; then
  echo "Installing/updating dependencies from '$REQUIREMENTS_FILE'..."
  conda install --file "$REQUIREMENTS_FILE" -y
else
  echo "Requirements file '$REQUIREMENTS_FILE' not found. Skipping dependencies installation."
fi
echo "Environment setup and dependencies installation completed!"

echo "Environment setup completed!"

# Environment variables
export DATA_DIR="$PROJECT_ROOT/data"
export TRAIN_DIR="$PROJECT_ROOT/train"

export DATASET_PATH="$DATA_DIR/dataset.csv"

# Set to current directory, i.e. PROJECT_ROOT, for absolute imports
export PYTHONPATH=$PROJECT_ROOT
echo "PYTHONPATH set to $PYTHONPATH."