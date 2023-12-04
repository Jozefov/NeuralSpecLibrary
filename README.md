
# Mass Spectrum Prediction Framework

## Overview

This repository hosts a mass spectrum prediction framework, designed to facilitate the analysis and prediction of mass spectrometry data. The framework is structured on a divisional approach, consisting of two main components:

1. **Part Embedding**: This component takes an input molecule and generates an embedding representation of it.
2. **Prediction Head**: The head component utilizes the embedding to produce the final prediction.

These components are seamlessly integrated through a `combine_model` function. The framework is designed to be modular, allowing users to easily add new embeddings or modify the prediction head as per their requirements.

## Getting Started

### Prerequisites

Ensure you have [Anaconda](https://www.anaconda.com/products/individual) installed on your system to manage virtual environments and dependencies.

### Installation

1. **Clone the Repository**

   ```
   git clone [Your Repository URL]
   cd [Your Repository Name]
   ```

2. **Create a Conda Virtual Environment**

   Using the `requirements.txt` file provided in the repository, you can create a Conda environment with all necessary dependencies:

   ```
   conda create --name myenv --file requirements.txt
   ```

   Replace `myenv` with your preferred environment name.

3. **Activate the Virtual Environment**

   ```
   conda activate myenv
   ```

4. **Running the Framework**

   After activating the environment, you can run the framework:

   ```
   python main.py
   ```

   Make sure to modify `main.py` as per your configuration needs.

## Customization

To customize the framework:

- **Adding a New Embedding Part**: Insert your code into the embedding module. This allows for new ways to process and embed input molecules.
- **Modifying the Prediction Head**: Implement your changes in the head module to alter how predictions are made based on the embeddings.
- **Configuring the Model**: Construct new instances or modify the model configuration in `config_model.py`.

## Contribution

Contributions to enhance or extend the framework's capabilities are welcome. Please submit your pull requests or open issues to discuss proposed changes.
