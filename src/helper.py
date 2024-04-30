import sys
import os
import torch
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
                config["path"]["processed_path"], "train_dataloader.pth"
            )
        )
        test_dataloader = load(
            filename=os.path.join(
                config["path"]["processed_path"], "test_dataloader.pth"
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

    model = AutoEncoder(in_channels=3)

    if adam:
        optimizer = optim.Adam(params=model.parameters(), lr=lr, betas=(0.5, 0.999))

        if lr_scheduler:
            scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    elif SGD:
        optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)

        if lr_scheduler:
            scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    dataloader = load_dataloader()

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_dataloader": dataloader["train_dataloader"],
        "test_dataloader": dataloader["test_dataloader"],
    }


if __name__ == "__main__":
    helpers()
