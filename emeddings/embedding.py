import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv


class CONV_GNN(torch.nn.Module):
    def __init__(self, node_features, embedding_size_reduced):
        # Init parent
        # node_features: int, number of features each atom has
        # embedding_size_reduced: int, size of vector for each vertex in graph

        super(CONV_GNN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = GCNConv(node_features, embedding_size_reduced)
        self.reluinit = nn.ReLU()
        self.conv1 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv1 = nn.ReLU()
        self.conv2 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv2 = nn.ReLU()
        self.conv3 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv3 = nn.ReLU()
        self.conv4 = GCNConv(embedding_size_reduced, embedding_size_reduced)
        self.reluconv4 = nn.ReLU()

    def forward(self, batch):
        x = batch.x.float()
        edge_index = batch.edge_index

        hidden = self.initial_conv(x, edge_index)
        hidden = self.reluinit(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = self.reluconv1(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = self.reluconv2(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.reluconv3(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = self.reluconv4(hidden)

        return hidden


class TRANSFORMER_CONV(torch.nn.Module):
    def __init__(self, node_features, edge_features, embedding_size_reduced, heads_num=4, dropout=0.1):
        # Init parent
        # node_features: int, number of features each atom has
        # edge_features: int, number of features each bond has
        # embedding_size_reduced: int, size of vector for each vertex in graph
        # heads_num: int, number of transformer heads
        # dropout: float, dropout rate

        super(TRANSFORMER_CONV, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        self.initial_conv = TransformerConv(node_features, embedding_size_reduced, heads=heads_num, beta=True,
                                            dropout=dropout, edge_dim=edge_features)
        self.reluinit = nn.ReLU()
        self.conv1 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                     beta=True, dropout=dropout, edge_dim=edge_features)
        self.reluconv1 = nn.ReLU()
        self.conv2 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                     beta=True, dropout=dropout, edge_dim=edge_features)
        self.reluconv2 = nn.ReLU()
        self.conv3 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, heads=heads_num,
                                     beta=True, dropout=dropout)
        self.reluconv3 = nn.ReLU()
        self.conv4 = TransformerConv(embedding_size_reduced * heads_num, embedding_size_reduced, concat=False,
                                     heads=heads_num, beta=True, dropout=dropout)
        self.reluconv4 = nn.ReLU()

    def forward(self, batch):

        x = batch.x.float()
        edge_index = batch.edge_index
        edge_attribute = batch.edge_attr

        hidden = self.initial_conv(x, edge_index, edge_attribute)
        hidden = self.reluinit(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index, edge_attribute)
        hidden = self.reluconv1(hidden)
        hidden = self.conv2(hidden, edge_index, edge_attribute)
        hidden = self.reluconv2(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = self.reluconv3(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = self.reluconv4(hidden)

        return hidden
    