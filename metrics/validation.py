import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DEVICE


def dot_product_torch(true, pred, mass_pow=3.0, intensity_pow=0.6):
    # Ensure input tensors are 1D
    assert true.dim() == pred.dim() == 1, "Both tensors should be 1-dimensional"

    length = true.size(0)
    mass = torch.arange(length, dtype=torch.float64).to(DEVICE)

    wl = (mass ** mass_pow * pred**intensity_pow)
    wu = (mass ** mass_pow * true**intensity_pow)

    pred_weighted_norm = torch.sqrt(torch.sum(wl**2))
    true_weighted_norm = torch.sqrt(torch.sum(wu**2))

    result = torch.sum(wl * wu) / (pred_weighted_norm * true_weighted_norm)

    # Handling potential NaN values
    if torch.isnan(result):
        result = torch.tensor(0.0, dtype=torch.float64)

    return result


def validate_similarities_torch(loader, model, mass_pow=1.0,  intensity_pow=0.5):
    # validate model during training
    # return: similarity, float

    total_similarity = 0
    count = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)
            pred = model(batch)

            similarities = []

            for true_instance, pred_instance in zip(batch.y, pred):
                sim = dot_product_torch(true_instance, pred_instance, mass_pow=mass_pow, intensity_pow=intensity_pow)
                similarities.append(sim)

            similarities_torch = torch.tensor(similarities, dtype=torch.float64)

            total_similarity += torch.sum(similarities_torch).item()
            count += len(similarities_torch)

    mean_similarity = total_similarity / count if count != 0 else 0

    return mean_similarity


def validate_regression(loader, model):
    # validate model during training on regression task
    # return: dictionary with measured values

    true_values = []
    predicted_values = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)

            predictions = model(batch)
            true_values.extend(batch.y.numpy())
            predicted_values.extend(predictions.numpy())

    # Convert lists to arrays for metric calculation
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)


    # Calculate metrics
    rmse = calculate_rmse(true_values, predicted_values)
    mae = calculate_mae(true_values, predicted_values)
    r2 = calculate_r2(true_values, predicted_values)

    result = {"rmse": rmse,
              "mae": mae,
              "r2": r2}

    return result



def calculate_rmse(true_values, predicted_values):
    # calculate RMSE on scalar

    mse = mean_squared_error(true_values, predicted_values)
    rmse = mse ** 0.5
    return rmse


def calculate_mae(true_values, predicted_values):
    # calculate mean absolute error

    mae = mean_absolute_error(true_values, predicted_values)
    return mae


def calculate_r2(true_values, predicted_values):
    # calculate R-squared
    r2 = r2_score(true_values, predicted_values)
    return r2



