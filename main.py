# main.py
import torch
from torch_geometric.data import DataLoader
import pickle

from config import DEVICE
import config_model
from train import train_model


def main():

    # Set model we are working with
    model_configuration = config_model.cnn_model

    # Set path to datasets
    train_dataset_path = model_configuration["training"]["train_dataset_path"]
    validation_dataset_path = model_configuration["training"]["validation_dataset_path"]
    test_validation_path = model_configuration["training"]["test_dataset_path"]

    with open(validation_dataset_path, 'rb') as handle:
        validation_dataset = pickle.load(handle)
    with open(train_dataset_path, 'rb') as handle:
        train_dataset = pickle.load(handle)
    with open(test_validation_path, 'rb') as handle:
        test_dataset = pickle.load(handle)

    # Pick first 5000 from test dataset during training, to see validation
    test_dataset = test_dataset[:5000]

    from torch.utils.data import Subset
    import random
    def create_subset_indices(dataset, num_instances):
        return random.sample(range(len(dataset)), min(len(dataset), num_instances))

    # Create subset indices for each dataset
    train_indices = create_subset_indices(train_dataset, 5000)
    validation_indices = create_subset_indices(validation_dataset, 5000)

    # Create subsets from the datasets using the generated indices
    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(validation_dataset, validation_indices)

    train_loader = DataLoader(train_subset, batch_size=model_configuration["training"]["batch_size"], shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=model_configuration["training"]["batch_size"], shuffle=True)


    # Initialize data loaders
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=model_configuration["training"]["batch_size"], shuffle=True)
    #
    # validation_loader = DataLoader(validation_dataset,
    #                                batch_size=model_configuration["training"]["batch_size"], shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=model_configuration["training"]["batch_size"], shuffle=True)

    # Initialize model
    model = model_configuration["model"].to(DEVICE)

    # Initialize loss function
    loss_fn = model_configuration["training"]["loss_fun"]

    # Initialize optimizer (and scheduler if needed)
    learning_rate = model_configuration["training"]["learning_rate"]
    optimizer = model_configuration["training"]["optimizer"](model.parameters(), lr=learning_rate)

    # Assume scheduler is optional, hence the use of get to return None if not found
    step_size = model_configuration["training"]["step_size"]
    gamma = model_configuration["training"]["gamma"]
    scheduler = model_configuration["training"]["scheduler"](optimizer, step_size, gamma)

    # Training loop
    train_model(
        model=model,
        train_loader=train_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=model_configuration["training"]["epochs"],
        device=DEVICE,
        val_loader=validation_loader,
        test_loader=test_loader,
        save_every=model_configuration["training"]["save_every"],
        report_every=model_configuration["training"]["report_every"],
        save_path=model_configuration["training"]["save_path"],
        scheduler=scheduler
    )
    # Any post-training logic could be added here


if __name__ == "__main__":
    main()
