import sys
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append("src/")

from utils import load, params
from auto_encoder import AutoEncoder


def load_dataloader():
    config = params()

    if os.path.exists(config["path"]["processed_path"]):
        train_dataloader = load(
            filename=os.path.join(
                config["path"]["processed_path"], "train_dataloader.pkl"
            )
        )
        test_dataloader = load(
            filename=os.path.join(
                config["path"]["processed_path"], "test_dataloader.pkl"
            )
        )

        return {
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
        }

    else:
        raise Exception("Could not find the file".capitalize())


def helpers(**kwargs):
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    lr_scheduler = kwargs["lr_scheduler"]

    config = params()

    model = AutoEncoder(in_channels=3)

    if adam:
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=lr,
            betas=(config["model"]["beta1"], config["model"]["beta2"]),
        )

    elif SGD:
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=lr,
            momentum=config["model"]["momentum"],
        )

    dataloader = load_dataloader()
    criterion = nn.MSELoss(reduction="mean")

    return {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
        "criterion": criterion,
    }


if __name__ == "__main__":
    helpers()
