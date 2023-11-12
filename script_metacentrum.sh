#!/bin/bash
#PBS -N Train_test
#PBS -l select=1:ncpus=4:mem=40gb:scratch_local=40gb
#PBS -l walltime=3:00:00 
#PBS -m ae

# Define the desired Python version and installation paths
DESIRED_PYTHON_VERSION="3.10.12"
PYTHON_INSTALL_PATH="/storage/brno2/home/jozefov_147/python/${DESIRED_PYTHON_VERSION}"
VENV_NAME="venv"
REQUIREMENTS_PATH="/storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/requirements.txt"
MAIN_PY_PATH="/storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/main.py"

# Prepare environment
export PYTHONUSERBASE="${PYTHON_INSTALL_PATH}/.local"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.10/site-packages:${PYTHONPATH}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

# Function to install Python from source
install_python_from_source() {
    # Define Python source URL
    local python_source_url="https://www.python.org/ftp/python/${DESIRED_PYTHON_VERSION}/Python-${DESIRED_PYTHON_VERSION}.tgz"
    
    # Download Python source
    wget "$python_source_url" -O Python.tgz
    
    # Extract the source code
    tar -xzf Python.tgz
    cd "Python-${DESIRED_PYTHON_VERSION}"
    
    # Configure and compile Python
    ./configure --prefix="${PYTHON_INSTALL_PATH}"
    make
    make install
    
    # Update the PATH
    export PATH="${PYTHON_INSTALL_PATH}/bin:${PATH}"
}

# Function to install or update pip
install_or_update_pip() {
    wget 'https://bootstrap.pypa.io/get-pip.py'
    python3 get-pip.py --user
}

# Check if the desired Python version is installed
if [ ! -d "${PYTHON_INSTALL_PATH}" ]; then
    install_python_from_source
fi

# Install or update pip
install_or_update_pip

# Create a virtual environment with the desired Python version
python3 -m venv "${VENV_NAME}"

# Activate the virtual environment
source "${VENV_NAME}/bin/activate"

# Install requirements from the requirements.txt file
pip install -r "${REQUIREMENTS_PATH}"

# Run the main.py script
python "${MAIN_PY_PATH}"

# Deactivate the virtual environment
deactivate
