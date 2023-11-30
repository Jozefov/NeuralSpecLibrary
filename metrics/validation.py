import torch

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
