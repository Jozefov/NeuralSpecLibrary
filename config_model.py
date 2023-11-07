import train.optimizer
import train.scheduler
import train.loss_fun
from emeddings import CONV_GNN
from heads import CONV_HEAD
from combine_models import CombinedModelCNN

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
        "train_dataset_path": "",
        "validation_dataset_path": "",
        "test_dataset_path": "",
        "save_every": 1,
        "report_every": 1,
        "save_path": "",
    }
}

