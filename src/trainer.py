import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

sys.path.append("src/")


from utils import device_init, params
from helper import helpers


class Trainer:
    def __init__(
        self,
        epochs=100,
        lr=0.0002,
        adam=True,
        SGD=False,
        lr_scheduler=False,
        device="mps",
        is_l1=False,
        is_l2=False,
        is_elastic_net=False,
        is_display=True,
        **kwargs,
    ):
        self.epochs = epochs
        self.lr = lr
        self.adam = adam
        self.SGD = SGD
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.is_l1 = is_l1
        self.is_l2 = is_l2
        self.is_elastic_net = is_elastic_net
        self.is_display = is_display
        self.kwargs = kwargs

        self.device = device_init(self.device)
        self.config = params()

        init = helpers(
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
            lr_scheduler=self.lr_scheduler,
        )

        self.train_dataloader = init["train_dataloader"]
        self.test_dataloader = init["test_dataloader"]

        self.model = init["model"].to(self.device)

        self.optimizer = init["optimizer"]
        self.criterion = init["criterion"]

        self.total_train_loss = []
        self.total_test_loss = []

        self.loss = float("inf")

    def saved_best_model(self, **kwargs):
        if os.path.exists(self.config["path"]["best_model"]):
            if self.loss > kwargs["loss"]:
                self.loss = kwargs["loss"]

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "loss": kwargs["loss"],
                        "epoch": kwargs["epoch"],
                    },
                    os.path.join(self.config["path"]["best_model"], "best_model.pth"),
                )

        else:
            raise FileNotFoundError(
                "Best model folder does not exist. Please create a folder."
            )

    def saved_checkpoints(self, **kwargs):
        if os.path.exists(self.config["path"]["train_models"]):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.config["path"]["train_models"],
                    "model{}.pth".format(kwargs["epoch"] + 1),
                ),
            )

        else:
            raise FileNotFoundError(
                "Checkpoints folder does not exist. Please create a folder."
            )

    def update_train_model(self, **kwargs):
        self.optimizer.zero_grad()

        loss = self.criterion(self.model(kwargs["blurred"]), kwargs["sharp"])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def show_progress(self, **kwargs):
        if self.is_display:
            print(
                "Epochs - [{}/{}] - train_loss: {:.4f} - test_loss: {:.4f}".format(
                    kwargs["epoch"],
                    kwargs["epochs"],
                    kwargs["train_loss"],
                    kwargs["test_loss"],
                )
            )

        else:
            print(
                "Epochs - [{}/{}] is completed.".capitalize().format(
                    kwargs["epoch"], kwargs["epochs"]
                )
            )

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            train_loss = []
            test_loss = []

            for _, (blurred, sharp) in enumerate(self.train_dataloader):
                blurred = blurred.to(self.device)
                sharp = sharp.to(self.device)

                train_loss.append(self.update_train_model(blurred=blurred, sharp=sharp))

            for _, (blurred, sharp) in enumerate(self.test_dataloader):
                blurred = blurred.to(self.device)
                sharp = sharp.to(self.device)

                test_loss.append(self.criterion(self.model(blurred), sharp).item())

            try:
                self.show_progress(
                    epoch=epoch + 1,
                    epochs=self.epochs,
                    train_loss=np.mean(train_loss),
                    test_loss=np.mean(test_loss),
                )

            except Exception as e:
                print("The exception was: %s" % e)

            else:
                self.saved_checkpoints(epoch=epoch)
                self.saved_best_model(epoch=epoch + 1, loss=np.mean(test_loss))


if __name__ == "__main__":
    trainer = Trainer(epochs=2)

    trainer.train()
