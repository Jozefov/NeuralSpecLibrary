import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap

from .heads_utils import mask_prediction_by_mass
from .heads_utils import reverse_prediction
from .heads_utils import SKIPblock


class CONV_HEAD(torch.nn.Module):
    def __init__(self, embedding_size_gnn, embedding_size, output_size):
        # Init parent
        # embedding_size_gnn: int, size of embedding vector, usually vector size for vertex node
        #
        # embedding_size: int, size of hidden layer vector,
        # output_size: int, size of output vector, resolution of spectrum

        super(CONV_HEAD, self).__init__()
        torch.manual_seed(42)

        self.bottleneck = nn.Linear(embedding_size_gnn, embedding_size)

        self.skip1 = SKIPblock(embedding_size, embedding_size)
        self.skip2 = SKIPblock(embedding_size, embedding_size)
        self.skip3 = SKIPblock(embedding_size, embedding_size)
        self.skip4 = SKIPblock(embedding_size, embedding_size)
        self.skip5 = SKIPblock(embedding_size, embedding_size)
        self.skip6 = SKIPblock(embedding_size, embedding_size)
        self.skip7 = SKIPblock(embedding_size, embedding_size)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = nn.Linear(embedding_size, output_size)
        self.backward_prediction = nn.Linear(embedding_size, output_size)
        self.gate = nn.Linear(embedding_size, output_size)

        self.relu_out = nn.ReLU()

    def forward(self, x, batch, mass_shift):

        batch_index = batch.batch
        total_mass = batch.molecular_weight

        hidden = gap(x, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)

        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, mass_shift)

        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, mass_shift)

        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = nn.functional.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)

        out = out.type(torch.float64)
        return out

class TANFORMER_CONV_HEAD(torch.nn.Module):
    def __init__(self, embedding_size_gnn, embedding_size, output_size):
        # Init parent
        # embedding_size_gnn: int, size of embedding vector, usually vector size for vertex node
        #
        # embedding_size: int, size of hidden layer vector,
        # output_size: int, size of output vector, resolution of spectrum

        super(TANFORMER_CONV_HEAD, self).__init__()
        torch.manual_seed(42)

        self.bottleneck = nn.Linear(embedding_size_gnn, embedding_size)

        self.skip1 = SKIPblock(embedding_size, embedding_size)
        self.skip2 = SKIPblock(embedding_size, embedding_size)
        self.skip3 = SKIPblock(embedding_size, embedding_size)
        self.skip4 = SKIPblock(embedding_size, embedding_size)
        self.skip5 = SKIPblock(embedding_size, embedding_size)
        self.skip6 = SKIPblock(embedding_size, embedding_size)
        self.skip7 = SKIPblock(embedding_size, embedding_size)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = nn.Linear(embedding_size, output_size)
        self.backward_prediction = nn.Linear(embedding_size, output_size)
        self.gate = nn.Linear(embedding_size, output_size)

        self.relu_out = nn.ReLU()

    def forward(self, x, batch, mass_shift):

        batch_index = batch.batch
        total_mass = batch.molecular_weight

        hidden = gap(x, batch_index)
        hidden = self.bottleneck(hidden)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)

        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, mass_shift)

        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, mass_shift)

        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = nn.functional.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)

        out = out.type(torch.float64)
        return out


class BIDIRECTIONAL_HEAD(torch.nn.Module):
    def __init__(self, embedding_size_gnn, embedding_size, output_size, use_graph=True):
        # Init parent
        # embedding_size_gnn: int, size of embedding vector, usually vector size for vertex node
        #
        # embedding_size: int, size of hidden layer vector,
        # output_size: int, size of output vector, resolution of spectrum

        super(BIDIRECTIONAL_HEAD, self).__init__()
        torch.manual_seed(42)

        self.use_graph = use_graph

        self.bottleneck = nn.Linear(embedding_size_gnn, embedding_size)

        self.skip1 = SKIPblock(embedding_size, embedding_size)
        self.skip2 = SKIPblock(embedding_size, embedding_size)
        self.skip3 = SKIPblock(embedding_size, embedding_size)
        self.skip4 = SKIPblock(embedding_size, embedding_size)
        self.skip5 = SKIPblock(embedding_size, embedding_size)
        self.skip6 = SKIPblock(embedding_size, embedding_size)
        self.skip7 = SKIPblock(embedding_size, embedding_size)
        self.relu_out_resnet = nn.ReLU()

        self.forward_prediction = nn.Linear(embedding_size, output_size)
        self.backward_prediction = nn.Linear(embedding_size, output_size)
        self.gate = nn.Linear(embedding_size, output_size)

        self.relu_out = nn.ReLU()

    def forward(self, x, batch, mass_shift):

        if self.use_graph:
            total_mass = batch.molecular_weight
            batch_index = batch.batch
            x = gap(x, batch_index)
        else:
            total_mass = batch["molecular_weight"]

        hidden = self.bottleneck(x)

        hidden = self.skip1(hidden)
        hidden = self.skip2(hidden)
        hidden = self.skip3(hidden)
        hidden = self.skip4(hidden)
        hidden = self.skip5(hidden)
        hidden = self.skip6(hidden)
        hidden = self.skip7(hidden)

        hidden = self.relu_out_resnet(hidden)

        # Bidirectional layer
        # Forward prediction
        forward_prediction_hidden = self.forward_prediction(hidden)
        forward_prediction_hidden = mask_prediction_by_mass(total_mass, forward_prediction_hidden, mass_shift)

        # # Backward prediction
        backward_prediction_hidden = self.backward_prediction(hidden)
        backward_prediction_hidden = reverse_prediction(total_mass, backward_prediction_hidden, mass_shift)

        # # Gate
        gate_hidden = self.gate(hidden)
        gate_hidden = nn.functional.sigmoid(gate_hidden)

        # # Apply a final (linear) classifier.
        out = gate_hidden * forward_prediction_hidden + (1. - gate_hidden) * backward_prediction_hidden
        out = self.relu_out(out)

        out = out.type(torch.float64)
        return out

class RegressionHead(torch.nn.Module):
    def __init__(self, input_size, hidden_size, use_graph=True, dropout_rate=0.15):
        # Init parent
        # input_size: int, size of embedding vector, usually vector size for vertex node
        #
        # hidden_size: int, size of hidden layer vector,

        super(RegressionHead, self).__init__()

        torch.manual_seed(42)
        self.use_graph = use_graph
        self.dropout_rate = dropout_rate

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc2 = torch.nn.Linear(input_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc3 = torch.nn.Linear(input_size, hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(hidden_size, 1)  # Output is a single scalar

    def forward(self, x, batch, mass_shift):

        if self.use_graph:
            batch_index = batch.batch
            x = gap(x, batch_index)

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        out = x.type(torch.float64)
        return out

