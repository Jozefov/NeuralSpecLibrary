#!/bin/bash
#PBS -N Train_test
#PBS -l select=1:ncpus=4:mem=40gb:scratch_local=40gb
#PBS -l walltime=3:00:00 
#PBS -m ae

# Define the desired Python version
DESIRED_PYTHON_VERSION="3.10.12"
VENV_NAME="venv_$DESIRED_PYTHON_VERSION"
REQUIREMENTS_PATH="/storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/requirements.txt"
MAIN_PY_PATH="/storage/projects/msml/NeuralSpecLib/NeuralSpecLibrary/main.py"

# Function to check if Python version is installed
check_python_version() {
    if ! pyenv versions | grep -q "$DESIRED_PYTHON_VERSION"; then
        echo "Python $DESIRED_PYTHON_VERSION is not installed."
        echo "Installing Python $DESIRED_PYTHON_VERSION..."
        pyenv install "$DESIRED_PYTHON_VERSION"
    else
        echo "Python $DESIRED_PYTHON_VERSION is already installed."
    fi
}

# Function to create a virtual environment with the desired Python version
create_virtualenv() {
    # Set the local Python version
    pyenv local "$DESIRED_PYTHON_VERSION"

    # Create virtual environment
    python -m venv "$VENV_NAME"

    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
}

# Install requirements
install_requirements() {
    if [ -f "$REQUIREMENTS_PATH" ]; then
        pip install -r "$REQUIREMENTS_PATH"
    else
        echo "Requirements file not found at $REQUIREMENTS_PATH"
        exit 1
    fi
}

# Run main.py
run_main_py() {
    if [ -f "$MAIN_PY_PATH" ]; then
        python "$MAIN_PY_PATH"
    else
        echo "main.py not found at $MAIN_PY_PATH"
        exit 1
    fi
}

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "pyenv could not be found. Please install pyenv first."
    exit 1
fi

# Main execution
check_python_version
create_virtualenv
install_requirements
run_main_py

# Deactivate virtual environment after the script is done
deactivate
