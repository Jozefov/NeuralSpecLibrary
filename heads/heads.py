import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap

from heads_utils import mask_prediction_by_mass
from heads_utils import reverse_prediction
from heads_utils import SKIPblock


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

    def forward(self, x, total_mass, batch_index, mass_shift):

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
