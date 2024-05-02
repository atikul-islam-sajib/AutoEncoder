import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

sys.path.append("src/")


from utils import dump, load, params, device_init, weight_init
from helper import helpers
from vgg16 import VGG16


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
        weight_clip=False,
        content_loss=False,
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
        self.weight_clip = weight_clip
        self.content_loss = content_loss

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

        self.model.apply(weight_init)

        self.optimizer = init["optimizer"]
        self.criterion = init["criterion"]

        if self.lr_scheduler:
            self.scheduler = StepLR(
                optimizer=self.optimizer,
                step_size=self.config["model"]["step_size"],
                gamma=self.config["model"]["gamma"],
            )

        self.total_train_loss = []
        self.total_test_loss = []

        self.loss = float("inf")

        self.vgg16 = VGG16().to(self.device)
        self.content_vgg_loss = nn.L1Loss()

    def l1(self, model=None):
        if model is not None:
            return sum(torch.norm(params, 1) for params in model.parameters())

        else:
            raise ValueError("The model must not be None.".capitalize())

    def l2(self, model=None):
        if model is not None:
            return sum(torch.norm(params, 2) for params in model.parameters())

        else:
            raise ValueError("The model must not be None.".capitalize())

    def elastic_net(self, model=None):
        if model is not None:
            return self.l1(model=self.model) + self.l2(model=self.model)

        else:
            raise ValueError("The model must not be None.".capitalize())

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

        if self.is_l1:
            loss += 0.001 * self.l1(model=self.model)

        elif self.is_l2:
            loss += 0.001 * self.l2(model=self.model)

        elif self.is_elastic_net:
            loss += 0.001 * self.elastic_net(model=self.model)

        if self.weight_clip:
            for params in self.model.parameters():
                params.data.clamp_(
                    -self.config["model"]["weight_clip"],
                    self.config["model"]["weight_clip"],
                )

        if self.content_loss:
            loss += 0.001 * self.content_vgg_loss(
                self.vgg16(kwargs["blurred"]), self.vgg16(self.model(kwargs["blurred"]))
            )

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

    def saved_train_images(self, **kwargs):
        blurred, _ = next(iter(self.test_dataloader))

        predicted_sharp = self.model(blurred.to(self.device))

        if os.path.exists(self.config["path"]["train_images"]):
            save_image(
                predicted_sharp,
                os.path.join(
                    self.config["path"]["train_images"],
                    "images{}.jpg".format(kwargs["epoch"] + 1),
                ),
                nrow=4,
                normalize=True,
            )

        else:
            raise FileNotFoundError(
                "Train images folder does not exist. Please create a folder.".capitalize()
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

            if self.lr_scheduler:
                self.scheduler.step()

            try:
                self.show_progress(
                    epoch=epoch + 1,
                    epochs=self.epochs,
                    train_loss=np.mean(train_loss),
                    test_loss=np.mean(test_loss),
                )

                self.saved_checkpoints(epoch=epoch)
                self.saved_best_model(epoch=epoch + 1, loss=np.mean(test_loss))
                self.saved_train_images(epoch=epoch)

            except Exception as e:
                print("The exception was: %s" % e)

            else:
                self.total_test_loss.append(np.mean(test_loss))
                self.total_train_loss.append(np.mean(train_loss))

        try:
            if os.path.exists(self.config["path"]["train_history"]):
                pd.DataFrame(
                    {
                        "train_loss": self.total_train_loss,
                        "test_loss": self.total_test_loss,
                    }
                ).to_csv(
                    os.path.join(
                        self.config["path"]["train_history"], "model_history.csv"
                    )
                )

                dump(
                    value=self.total_train_loss,
                    filename=os.path.join(
                        self.config["path"]["train_history"], "train_loss.pkl"
                    ),
                )

                dump(
                    value=self.total_test_loss,
                    filename=os.path.join(
                        self.config["path"]["train_history"], "test_loss.pkl"
                    ),
                )
            else:
                raise FileNotFoundError(
                    "Train history folder does not exist. Please create a folder.".capitalize()
                )

        except Exception as e:
            print("The exception was: %s" % e)

    @staticmethod
    def plot_history():
        config = params()

        if os.path.exists(config["path"]["train_history"]):

            plt.figure(figsize=(20, 10))

            train_loss = load(
                filename=os.path.join(config["path"]["train_history"], "train_loss.pkl")
            )
            test_loss = load(
                filename=os.path.join(config["path"]["train_history"], "test_loss.pkl")
            )

            plt.plot(train_loss, label="Train Loss")
            plt.plot(test_loss, label="Test Loss")

            plt.legend()
            plt.xlabel("Epochs".capitalize())
            plt.ylabel("Loss".capitalize())
            plt.title("Train and Test Loss".title())
            plt.tight_layout()

            plt.savefig(
                os.path.join(config["path"]["file_path"], "train_test_loss.png")
            )

            plt.show()

        else:
            raise FileNotFoundError(
                "Train history folder does not exist. Please create a folder.".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer code for Auto Encoder".title()
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for training".capitalize(),
    )
    parser.add_argument(
        "--adam", type=bool, default=True, help="Use adam optimizer or not".capitalize()
    )
    parser.add_argument(
        "--SGD", type=bool, default=False, help="Use SGD optimizer or not".capitalize()
    )
    parser.add_argument(
        "--l1",
        type=bool,
        default=False,
        help="Use l1 regularization or not".capitalize(),
    )
    parser.add_argument(
        "--l2",
        type=bool,
        default=False,
        help="Use l2 regularization or not".capitalize(),
    )
    parser.add_argument(
        "--elastic_net",
        type=bool,
        default=False,
        help="Use elastic net regularization or not".capitalize(),
    )
    parser.add_argument(
        "--is_display",
        type=bool,
        default=True,
        help="Display progress or not".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for training".capitalize()
    )
    parser.add_argument(
        "--weight_clip",
        type=bool,
        default=True,
        help="Use weight clipping or not".capitalize(),
    )

    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=True,
        help="Use learning rate scheduler or not".capitalize(),
    )
    parser.add_argument(
        "--content_loss",
        type=bool,
        default=True,
        help="Use content loss or not".capitalize(),
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        adam=args.adam,
        SGD=args.SGD,
        is_l1=args.l1,
        is_l2=args.l2,
        is_elastic_net=args.elastic_net,
        is_display=args.is_display,
        device=args.device,
        lr_scheduler=args.lr_scheduler,
        content_loss=args.content_loss,
    )

    trainer.train()

    trainer.plot_history()
