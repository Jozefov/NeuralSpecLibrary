import numpy as np

import train.optimizer
import train.scheduler
import train.loss_fun
from emeddings import CONV_GNN, TRANSFORMER_CONV, GAT, NEIMS
from heads import CONV_HEAD, BIDIRECTIONAL_HEAD, RegressionHead
from combine_models import CombinedModelCNN, CombineGeneral, CombineMolecularFingerPrint

cnn_model = {
    "node_features": 50,
    "spectrum_preprocessing": "pow",
    "embedding_size_reduced": int(2000*0.15),
    "embedding_size": 2000,
    "output_size": 1000,
    "intensity_power": 0.5,
    "mass_power": 1.0,
    "mass_shift": 5,
    "embedding": CONV_GNN,
    "head": CONV_HEAD,
    "model": CombinedModelCNN(50, int(2000*0.15), 2000, 1000, 5),
    "training": {
        "training_method": "graph",
        "loss_fun": train.loss_fun.loss_fun_dict["Huber"],
        "learning_rate": 0.0005,
        "optimizer": train.optimizer.optimizer_dic["Adam"],
        "scheduler": train.scheduler.scheduler_dic["lr_scheduler"],
        "step_size": 100,
        "gamma": 0.5,
        "epochs": 300,
        "batch_size": 64,
        "train_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/train_subset_pow.pkl",
        "validation_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/validation_subset_pow.pkl",
        "test_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/Preprocessed_test_pow_preparation_no_sparse_small.output",
        "save_every": 1,
        "report_every": 1,
        "print_to_file": True,
        "save_path": "/home/michpir/Documents/PROJECTS/output",
    }
}

transformer_cnn_model = {
    "name": "conv_transformers",
    "node_features": 50,
    "spectrum_preprocessing": "pow",
    "embedding_size_reduced": int(2000*0.1),
    "embedding_size": 2000,
    "output_size": 1000,
    "number_of_att_heads": 4,
    "dropout": 0.1,
    "intensity_power": 0.5,
    "mass_power": 1.0,
    "mass_shift": 5,
    "embedding": TRANSFORMER_CONV(50, 10, int(200*0.1), 4, 0.1),
    "head": BIDIRECTIONAL_HEAD(int(200*0.1), 200, 1000),
    "combine": CombineGeneral,
    "model": CombineGeneral(TRANSFORMER_CONV(50, 10, int(200*0.1), 4, 0.1),
                            BIDIRECTIONAL_HEAD(int(200 * 0.1), 200, 1000),
                            5),
    "training": {
        "training_method": "graph",
        "loss_fun": train.loss_fun.loss_fun_dict["Huber"],
        "learning_rate": 0.0005,
        "optimizer": train.optimizer.optimizer_dic["Adam"],
        "scheduler": train.scheduler.scheduler_dic["lr_scheduler"],
        "step_size": 100,
        "gamma": 0.5,
        "epochs": 300,
        "batch_size": 64,
        "train_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/train_graph_pow.output",
        "validation_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/validation_graph_pow.output",
        "test_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/test_graph_pow.output",
        "save_every": 1,
        "report_every": 1,
        "print_to_file": True,
        "save_path": "/home/michpir/Documents/PROJECTS/output",
    }
}

gat_model = {
    "name": "gat_model",
    "node_features": 50,
    "spectrum_preprocessing": "pow",
    "embedding_size_reduced": int(2000*0.1),
    "embedding_size": 2000,
    "output_size": 1000,
    "number_of_att_heads": 4,
    "dropout": 0.1,
    "intensity_power": 0.5,
    "mass_power": 1.0,
    "mass_shift": 5,
    "embedding": GAT(50, 10, int(200*0.1), 4, 0.1),
    "head": BIDIRECTIONAL_HEAD(int(200*0.1), 200, 1000),
    "model": CombineGeneral(GAT(50, 10, int(200*0.1), 4, 0.1),
                            BIDIRECTIONAL_HEAD(int(200 * 0.1), 200, 1000),
                            5),
    "training": {
        "training_method": "graph",
        "loss_fun": train.loss_fun.loss_fun_dict["Huber"],
        "learning_rate": 0.0005,
        "optimizer": train.optimizer.optimizer_dic["Adam"],
        "scheduler": train.scheduler.scheduler_dic["lr_scheduler"],
        "step_size": 100,
        "gamma": 0.5,
        "epochs": 300,
        "batch_size": 64,
        "train_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/train_graph_pow.output",
        "validation_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/validation_graph_pow.output",
        "test_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/test_graph_pow.output",
        "save_every": 1,
        "report_every": 1,
        "print_to_file": True,
        "save_path": "/home/michpir/Documents/PROJECTS/output",
    }
}


neims_model = {
    "name": "neims_model",
    "input_features": 4096,
    "spectrum_preprocessing": "pow",
    "embedding_size": 2000,
    "output_size": 1000,
    "intensity_power": 0.5,
    "mass_power": 1.0,
    "mass_shift": 5,
    "embedding": NEIMS(),
    "head": BIDIRECTIONAL_HEAD(4096, 2000, 1000, use_graph=False),
    "model": CombineMolecularFingerPrint(NEIMS(),
                            BIDIRECTIONAL_HEAD(4096, 2000, 1000, use_graph=False),
                            5),
    "training": {
        "training_method": "fingerprints",
        "loss_fun": train.loss_fun.loss_fun_dict["Huber"],
        "learning_rate": 0.0005,
        "optimizer": train.optimizer.optimizer_dic["Adam"],
        "scheduler": train.scheduler.scheduler_dic["lr_scheduler"],
        "step_size": 100,
        "gamma": 0.5,
        "epochs": 300,
        "batch_size": 64,
        "train_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/train_small_ecfp_pow.pkl",
        "validation_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/train_small_ecfp_pow.pkl",
        "test_dataset_path": "/home/michpir/Documents/PROJECTS/dataset/test_small_ecfp_pow.pkl",
        "save_every": 1,
        "report_every": 1,
        "print_to_file": True,
        "save_path": "/home/michpir/Documents/PROJECTS/output",
    },
    "preprocessing": {
        "method": "molecular fingerprints",
        "radius": 2,
        "fingerprint_length": 4096,
        "use_features": False,
        "use_chirality": False,
        "intensity_power": 0.5,
        "output_size": 1000,
        "operation": "pow"
    }
}

transformer_cnn_regression_model = {
    "name": "regression",
    "node_features": 50,
    "embedding_size_reduced": int(2000*0.1),
    "embedding_size": 2000,
    "output_size": 1,
    "hidden_size": 100,
    "number_of_att_heads": 4,
    "dropout": 0.1,
    "intensity_power": 0.5,
    "embedding": TRANSFORMER_CONV(50, 10, int(2000*0.15), 4, 0.1),
    "head": RegressionHead(int(2000*0.15), 400),
    "combine": CombineGeneral,
    "model": CombineGeneral(TRANSFORMER_CONV(50, 10, int(2000*0.15), 4, 0.1),
                            RegressionHead(int(2000*0.15), 400),
                            np.Inf),
    "training": {
        "training_method": "regression",
        "loss_fun": train.loss_fun.loss_fun_dict["MSE"],
        "learning_rate": 0.001,
        "optimizer": train.optimizer.optimizer_dic["Adam"],
        "scheduler": train.scheduler.scheduler_dic["lr_scheduler"],
        "step_size": 50,
        "gamma": 0.5,
        "epochs": 100,
        "batch_size": 64,
        "train_dataset_path": "/storage/projects/msml/NeuralSpecLib/dataset/train_homo_lumo_graph.output",
        "validation_dataset_path": "/storage/projects/msml/NeuralSpecLib/dataset/validation_homo_lumo_graph.output",
        "test_dataset_path": "/storage/projects/msml/NeuralSpecLib/dataset/test_challenge_homo_lumo_graph.output",
        "save_every": 1,
        "report_every": 1,
        "print_to_file": True,
        "save_path": "/storage/projects/msml/NeuralSpecLib/output/homo_lumo_transformer/",
    }
}