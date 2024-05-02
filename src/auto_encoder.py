import sys
import os
import argparse
import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import params
from encoder import EncoderBlock
from decoder import DecoderBlock


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super(AutoEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 64
        self.num_repetitive = 1

        self.layers = []

        for _ in range(self.num_repetitive):
            self.layers.append(
                EncoderBlock(
                    in_channels=self.in_channels, out_channels=self.out_channels
                )
            )
            self.in_channels = self.out_channels
            self.out_channels *= 2

        self.out_channels = self.in_channels // 2

        for idx in range(self.num_repetitive):
            self.layers.append(
                DecoderBlock(
                    in_channels=self.in_channels,
                    out_channels=(
                        3 if idx == self.num_repetitive - 1 else self.out_channels
                    ),
                    is_last=True if idx == self.num_repetitive - 1 else False,
                )
            )

            self.in_channels = self.out_channels
            self.out_channels //= 2

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if x is not None:
            return self.model(x)

        else:
            raise ValueError("The input must not be None.".capitalize())

    @staticmethod
    def total_params(model=None):
        if model is not None:
            return sum(params.numel() for params in model.parameters())

        else:
            raise ValueError("The model must not be None.".capitalize())

    @staticmethod
    def model_architecture(model=None):
        config = params()
        if model is not None:
            if os.path.exists(config["path"]["file_path"]):

                draw_graph(
                    model=model, input_size=(1, 3, 256, 256)
                ).visual_graph.render(
                    filename=os.path.join(config["path"]["file_path"], "autoencoder"),
                    format="jpg",
                )

            else:
                raise FileNotFoundError(
                    "The processed data folder does not exist. Please check the path and try again."
                )

        else:
            raise ValueError("The model must not be None.".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoEncoder model".title())

    parser.add_argument(
        "--in_channels", default=3, type=int, help="Input channels".capitalize()
    )

    args = parser.parse_args()

    model = AutoEncoder(in_channels=args.in_channels)

    print(summary(model=model, input_size=(3, 256, 256)))

    AutoEncoder.model_architecture(model=model)

    print(model(torch.randn(1, 3, 256, 256)).size())
