import train.optimizer
import train.scheduler
import train.loss_fun
from emeddings import CONV_GNN, TRANSFORMER_CONV
from heads import CONV_HEAD, BIDIRECTIONAL_HEAD
from combine_models import CombinedModelCNN, CombinedTransformerConvolutionModel

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
    "intensity_power": 0.5,
    "mass_power": 1.0,
    "mass_shift": 5,
    "embedding": TRANSFORMER_CONV,
    "head": BIDIRECTIONAL_HEAD,
    "model": CombinedTransformerConvolutionModel(50, 10, int(200*0.1), 200, 1000, 5),
    "training": {
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
