import torch
from heads import CONV_HEAD, BIDIRECTIONAL_HEAD
from emeddings import CONV_GNN, TRANSFORMER_CONV


class CombinedModelCNN(torch.nn.Module):
    # This module is used when a new CNN is build

    def __init__(self, node_features, embedding_size_reduced, embedding_size, output_size, mass_shift):
        super(CombinedModelCNN, self).__init__()
        self.model_body = CONV_GNN(node_features, embedding_size_reduced)
        self.model_head = CONV_HEAD(embedding_size_reduced, embedding_size, output_size)
        self.mass_shift = mass_shift

    def forward(self, batch, return_embedding=False):
        # Get the embeddings from the model_body (GNN)
        embeddings = self.model_body(batch)

        # If only the embeddings are needed
        if return_embedding:
            return embeddings

        # Pass the embeddings to the model_head for final prediction
        out = self.model_head(embeddings, batch, self.mass_shift)
        return out


class CombinedTransformerConvolutionModel(torch.nn.Module):
    # This module is used when we build a model with attribute features included
    def __init__(self, node_features, edge_features, embedding_size_reduced, embedding_size, output_size, mass_shift):
        super(CombinedTransformerConvolutionModel, self).__init__()
        self.model_body = TRANSFORMER_CONV(node_features, edge_features, embedding_size_reduced)
        self.model_head = BIDIRECTIONAL_HEAD(embedding_size_reduced, embedding_size, output_size)
        self.mass_shift = mass_shift

    def forward(self, batch, return_embedding=False):
        # Get the embeddings from the model_body (GNN)
        embeddings = self.model_body(batch)

        # If only the embeddings are needed
        if return_embedding:
            return embeddings

        # Pass the embeddings to the model_head for final prediction
        out = self.model_head(embeddings, batch, self.mass_shift)
        return out
