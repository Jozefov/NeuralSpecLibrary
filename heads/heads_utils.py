import torch
from config import DEVICE
import torch.nn as nn

def mask_prediction_by_mass(total_mass, raw_prediction, index_shift):
    # Zero out predictions to the right of the maximum possible mass.
    # input
    # anchor_indices: shape (,batch_size) = ex [3,4,5]
    #     total_mass = Weights of whole molecule, not only fragment
    # data: shape (batch_size, embedding), embedding from GNN in our case
    # index_shift: int constant how far can the heaviest fragment differ from weight of original molecule


    data = raw_prediction.type(torch.float64)

    total_mass = torch.round(total_mass).type(torch.int64)
    indices = torch.arange(data.shape[-1])[None, ...].to(DEVICE)

    right_of_total_mass = indices > (
            total_mass[..., None] +
            index_shift)
    return torch.where(right_of_total_mass, torch.zeros_like(data), data)


def scatter_by_anchor_indices(anchor_indices, data, index_shift):
    # reverse vector by anchor_indices and rest set to zero
    # input
    # anchor_indices: shape (,batch_size) = ex [3,4,5]
    #     total_mass = Weights of whole molecule, not only fragment
    # data: shape (batch_size, embedding), embedding from GNN in our case
    # index_shift: int constant how far can the heaviest fragment differ from weight of original molecule

    index_shift = index_shift
    anchor_indices = anchor_indices
    data = data.type(torch.float64)
    batch_size = data.shape[0]

    num_data_columns = data.shape[-1]
    indices = torch.arange(num_data_columns)[None, ...].to(DEVICE)
    shifted_indices = anchor_indices[..., None] - indices + index_shift
    valid_indices = shifted_indices >= 0

    batch_indices = torch.tile(torch.arange(batch_size)[..., None], [1, num_data_columns]).to(DEVICE)
    shifted_indices += batch_indices * num_data_columns

    shifted_indices = torch.reshape(shifted_indices, [-1])
    num_elements = data.shape[0] * data.shape[1]
    row_indices = torch.arange(num_elements).to(DEVICE)
    stacked_indices = torch.stack([row_indices, shifted_indices], axis=1)


    lower_batch_boundaries = torch.reshape(batch_indices * num_data_columns, [-1])
    upper_batch_boundaries = torch.reshape(((batch_indices + 1) * num_data_columns),
                                          [-1])

    valid_indices = torch.logical_and(shifted_indices >= lower_batch_boundaries,
                                     shifted_indices < upper_batch_boundaries)

    stacked_indices = stacked_indices[valid_indices]

    dense_shape = torch.tile(torch.tensor(num_elements)[..., None], [2]).type(torch.int32)

    scattering_matrix = torch.sparse.FloatTensor(stacked_indices.type(torch.int64).T,
                                                 torch.ones_like(stacked_indices[:, 0]).type(torch.float64),
                                                dense_shape.tolist())

    flattened_data = torch.reshape(data, [-1])[..., None]
    flattened_output = torch.sparse.mm(scattering_matrix, flattened_data)
    return torch.reshape(torch.transpose(flattened_output, 0, 1), [-1, num_data_columns])


def reverse_prediction(total_mass, raw_prediction, index_shift):
    # reverse vector by anchor_indices and rest set to zero and make preprocessing
    # input
    # total_mass: shape (,batch_size) = ex [3,4,5]
    #     total_mass = Weights of whole molecule, not only fragment
    # raw_prediction: shape (batch_size, embedding), embedding from GNN in our case
    # index_shift: int constant how far can the heaviest fragment differ from weight of original molecule
    #     total_mass = feature_dict[fmap_constants.MOLECULE_WEIGHT][..., 0]

    total_mass = torch.round(total_mass).type(torch.int32)
    return scatter_by_anchor_indices(
        total_mass, raw_prediction, index_shift)


class SKIPblock(nn.Module):
    def __init__(self, in_features, hidden_features, bottleneck_factor=0.5, use_dropout=True, dropout_rate=0.2):
        super().__init__()
        # module for building skip connection with bottleneck and one skip connection

        self.batchNorm1 = nn.BatchNorm1d(in_features)
        self.relu1 = nn.ReLU()
        if use_dropout:
            self.dropout1 = nn.Dropout(dropout_rate)
        self.hidden1 = nn.utils.weight_norm(nn.Linear(in_features, int(hidden_features * bottleneck_factor)),
                                            name='weight', dim=0)

        self.batchNorm2 = nn.BatchNorm1d(int(hidden_features * bottleneck_factor))
        self.relu2 = nn.ReLU()
        if use_dropout:
            self.dropout2 = nn.Dropout(dropout_rate)
        self.hidden2 = nn.utils.weight_norm(nn.Linear(int(hidden_features * bottleneck_factor), in_features),
                                            name='weight', dim=0)

    def forward(self, x):

        hidden = self.batchNorm1(x)
        hidden = self.relu1(hidden)
        hidden = self.dropout1(hidden)
        hidden = self.hidden1(hidden)

        hidden = self.batchNorm2(hidden)
        hidden = self.relu2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.hidden2(hidden)

        hidden = hidden + x

        return hidden
