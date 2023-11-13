import torch.nn as nn
from torch_geometric.nn import GATConv


class SKIPGAT(nn.Module):
    def __init__(self, in_features, hidden_features, heads, edge_features=None, use_dropout=True, dropout_rate=0.1):
        super().__init__()
        # [bs,in_c,seq_length]->conv(1,in_c,out_c)->[bs,out_c,seq_length]

        self.relu1 = nn.ReLU()
        if use_dropout:
            self.conv1 = GATConv(in_features, hidden_features, heads=heads, dropout=dropout_rate, edge_dim=edge_features)
        else:
            self.conv1 = GATConv(in_features, hidden_features, heads=heads, edge_dim=edge_features)

        self.relu2 = nn.ReLU()
        if use_dropout:
            self.conv2 = GATConv(hidden_features * heads, int(in_features / heads), heads=heads, dropout=dropout_rate,
                                 edge_dim=edge_features)
        else:
            self.conv2 = GATConv(hidden_features * heads, int(in_features / heads), heads=heads,
                                 edge_dim=edge_features)

    def forward(self, x, edge_index, edge_attribute=None):

        hidden = self.relu1(x)
        hidden = self.conv1(hidden, edge_index, edge_attribute)

        hidden = self.relu2(hidden)
        hidden = self.conv2(hidden, edge_index, edge_attribute)
        hidden = hidden + x

        return hidden
