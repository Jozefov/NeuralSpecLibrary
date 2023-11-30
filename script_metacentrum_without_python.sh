#!/bin/bash
#PBS -N Train_test
#PBS -l select=1:ncpus=4:mem=40gb:scratch_local=40gb
#PBS -l walltime=3:00:00
#PBS -m ae

# Set up the Python environment variables
export PYTHONUSERBASE="/storage/projects/msml/NeuralSpecLib/python310"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.10/site-packages:${PYTHONPATH}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

# Path to the requirements.txt file and the main Python script
REQUIREMENTS_PATH="/storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/requirements.txt"
MAIN_PY_PATH="/storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/main.py"

# Function to install PIP
#install_pip() {
#    wget 'https://bootstrap.pypa.io/get-pip.py'
#    python3 "/storage/projects/msml/NeuralSpecLib/get-pip.py"
#}

# Function to install Python packages locally
install_packages() {
    "${PYTHONUSERBASE}/bin/pip3" install -r "${REQUIREMENTS_PATH}"
}

# Function to run the main Python script
run_script() {
    python3 "${MAIN_PY_PATH}"
}

# Execute the functions
#install_pip
install_packages
run_script
