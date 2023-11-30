import torch
import os

from config import DEVICE
from metrics import validate_similarities_torch, validate_regression


def train_model(
        model,  # The PyTorch model you want to train.
        model_config,   # Python dictionary, where model parameters are stored.
        train_loader,  # DataLoader object for your training dataset.
        loss_fn,  # The loss function used for training (e.g., MSE, CrossEntropy).
        optimizer,  # The optimization algorithm (e.g., Adam, SGD).
        scheduler=None,  # (Optional) Learning rate scheduler that adjusts the learning rate during training.
        epochs=300,  # Number of training epochs. Default is 300.
        device=DEVICE,  # Device to which the model should be moved. Use 'cuda' for GPU and 'cpu' for CPU.
        val_loader=None,
        # (Optional) DataLoader object for the validation dataset. If provided, the function will evaluate the model on this dataset.
        test_loader=None,
        # (Optional) DataLoader object for the test dataset. If provided, the function will evaluate the model on this dataset.
        save_every=10,  # Interval (in terms of epochs) at which the model should be saved.
        report_every=1,
        # Interval (in terms of epochs) at which the training and validation performance should be reported.
        # intensity_power=0.5,    # Coefficient by witch intensity is scaled during validation
        # mass_power=1.0, # Coefficient by witch mass is scaled during validation
        save_path="",  # Directory where the trained model checkpoints will be saved.
        metadata={},  # Dictionary, additional metadata to save with the checkpoint.
):

    def train(loader):
        # Enumerate over the data

        model.train()
        total_loss = 0
        for batch in loader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss
        return total_loss / len(loader)

    def train_fingerprints(loader):
        # Enumerate over the data

        model.train()
        total_loss = 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch["output_tensor"])
            loss.backward()
            optimizer.step()
            total_loss += loss
        return total_loss / len(loader)

    loss_list = []

    for epoch in range(epochs):
        if scheduler:
            scheduler.step()

        if model_config["training"]["training_method"] == "fingerprints":
            train_loss = train_fingerprints(train_loader)
        else:
            train_loss = train(train_loader)

        loss_list.append(train_loss)

        if epoch % report_every == 0:
            report_message = f"Epoch {epoch} | Train Loss {train_loss} \n"
            report_status(report_message, save_to_file=model_config["training"]["print_to_file"],
                          directory_path=save_path)

            if test_loader:
                if model_config["training"]["training_method"] == "regression":
                    regression_metrics = validate_regression(test_loader, model)
                    report_message = (f"Epoch {epoch} | Test rmse {regression_metrics['rmse']} \n"
                                      f"Epoch {epoch} | Test mae {regression_metrics['mae']} \n"
                                      f"Epoch {epoch} | Test r2 {regression_metrics['r2']} \n")

                else:
                    test_similarity = validate_similarities_torch(test_loader, model)
                    report_message = f"Epoch {epoch} | Test DotSimilarity is {test_similarity}"

                report_status(report_message, save_to_file=model_config["training"]["print_to_file"],
                              directory_path=save_path)

            if val_loader:
                if model_config["training"]["training_method"] == "regression":
                    regression_metrics = validate_regression(val_loader, model)
                    report_message = (f"Epoch {epoch} | Validation rmse {regression_metrics['rmse']} \n"
                                      f"Epoch {epoch} | Validation {regression_metrics['mae']} \n"
                                      f"Epoch {epoch} | Validation r2 {regression_metrics['r2']} \n")

                else:
                    val_similarity = validate_similarities_torch(val_loader, model)
                    report_message = f"Epoch {epoch} | Validation DotSimilarity is {val_similarity}"

                report_status(report_message, save_to_file=model_config["training"]["print_to_file"],
                              directory_path=save_path)

            report_status("",  save_to_file=model_config["training"]["print_to_file"],
                          directory_path=save_path)

        if epoch % save_every == 0:
            metadata["loss_all"] = loss_list

            save_model(model, epoch, train_loss, save_path, optimizer=optimizer, scheduler=scheduler, metadata=metadata)


def report_status(message, save_to_file=True, directory_path=None):
    if save_to_file and directory_path is not None:
        # Ensure the directory exists, and if not, create it
        os.makedirs(directory_path, exist_ok=True)

        # Define the full file path
        file_path = os.path.join(directory_path, 'model_training_report.txt')

        # Open the file in append mode. If it doesn't exist, it will be created.
        with open(file_path, 'a') as file:
            file.write(message + '\n')
    else:
        # Just print the message to the console
        print(message)


def save_model(model, epoch, loss, save_path, optimizer=None, scheduler=None, metadata=None):
    """
    Save the model state, optimizer state, and scheduler state to a checkpoint file.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - epoch (int): The current epoch number.
    - loss (float): The loss value to save.
    - save_path (str): The path where the checkpoint will be saved.
    - optimizer (torch.optim.Optimizer, optional): The optimizer to save.
    - scheduler (torch.optim.lr_scheduler, optional): The scheduler to save.
    - metadata (dict, optional): Additional metadata to save with the checkpoint.
    """

    if (optimizer is not None) and (scheduler is not None):
        checkpoint = {
            'epoch': epoch,
            'model_body_state_dict': model.model_body.state_dict(),
            'model_head_state_dict': model.model_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }

    elif (optimizer is not None) and (scheduler is None):
        checkpoint = {
            'epoch': epoch,
            'model_body_state_dict': model.model_body.state_dict(),
            'model_head_state_dict': model.model_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

    else:
        checkpoint = {
            'epoch': epoch,
            'model_body_state_dict': model.model_body.state_dict(),
            'model_head_state_dict': model.model_head.state_dict(),
            'loss': loss,
        }

    # If metadata is provided, add it to the checkpoint dictionary
    if metadata is not None:
        checkpoint['metadata'] = metadata

    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the checkpoint
    torch.save(checkpoint, os.path.join(save_path, f"model_epoch_{epoch}.pt"))

def to_device(batch, device):
    """
    Move the entire batch to the specified device.
    :param batch: Batch of data (as a dictionary).
    :param device: Target device.
    :used for molecular fingerprints
    """
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch