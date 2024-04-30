import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")


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
        **kwargs
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

        init = helpers(
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
            lr_scheduler=self.lr_scheduler,
        )

        self.train_dataloader = init["train_dataloader"]
        self.test_dataloader = init["test_dataloader"]
        self.model = init["model"]
        self.optimizer = init["optimizer"]
        self.scheduler = init["scheduler"]
