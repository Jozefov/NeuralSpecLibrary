import torch
import os

from config import DEVICE
from metrics import validate_similarities_torch


def train_model(
        model,  # The PyTorch model you want to train.
        # model_config,   # Python dictionary, where model parameters are stored.
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

    test_similarity_list = []
    val_similarity_list = []
    loss_list = []

    for epoch in range(epochs):
        if scheduler:
            scheduler.step()

        train_loss = train(train_loader)
        loss_list.append(train_loss)

        if epoch % report_every == 0:
            print(f"Epoch {epoch} | Train Loss {train_loss}")

            if test_loader:
                test_similarity = validate_similarities_torch(test_loader, model)
                test_similarity_list.append(test_similarity)
                print(f"Epoch {epoch} | Test DotSimilarity is {test_similarity}")

            if val_loader:
                val_similarity = validate_similarities_torch(val_loader, model)
                val_similarity_list.append(val_similarity)
                print(f"Epoch {epoch} | Validation DotSimilarity is {val_similarity}")

            print()

        if epoch % save_every == 0:
            metadata["test_similarity"] = test_similarity_list
            metadata["validation_similarity"] = val_similarity_list
            metadata["loss_all"] = loss_list

            save_model(model, epoch, train_loss, save_path, optimizer=optimizer, scheduler=scheduler, metadata=metadata)


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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }

    elif (optimizer is not None) and (scheduler is None):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }

    # If metadata is provided, add it to the checkpoint dictionary
    if metadata is not None:
        checkpoint['metadata'] = metadata

    # Make sure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the checkpoint
    torch.save(checkpoint, os.path.join(save_path, f"model_epoch_{epoch}.pt"))