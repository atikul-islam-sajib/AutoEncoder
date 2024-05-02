import sys
import os
import argparse
from collections import OrderedDict
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import params


class EncoderBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = 3
        self.stride = 1
        self.padding = 1

        self.layers = OrderedDict()
        self.config = params()
        self.encoder = self.block()

    def block(self):
        self.layers["conv"] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )

        self.layers["relu"] = nn.ReLU(inplace=True)

        # self.layers["conv_1"] = nn.Conv2d(
        #     in_channels=self.out_channels,
        #     out_channels=self.out_channels,
        #     kernel_size=self.kernel,
        #     stride=self.stride,
        #     padding=self.padding,
        # )

        # self.layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)

        # self.layers["relu_1"] = nn.ReLU(inplace=True)

        self.layers["max_pool"] = nn.MaxPool2d(
            kernel_size=(self.kernel // self.kernel) * 2, stride=self.stride * 2
        )

        return nn.Sequential(self.layers)

    def forward(self, x):
        if x is not None:
            return self.encoder(x)

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
                    filename=os.path.join(config["path"]["file_path"], "encoder_block"),
                    format="jpg",
                )

            else:
                raise FileNotFoundError(
                    "The processed data folder does not exist. Please check the path and try again."
                )

        else:
            raise ValueError("The model must not be None.".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder Block for AutoEncoder".title()
    )

    parser.add_argument(
        "--in_channels", default=3, type=int, help="Input channels".capitalize()
    )
    parser.add_argument(
        "--out_channels", default=64, type=int, help="Output channels".capitalize()
    )

    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels
    num_repetitive = 5

    layers = []

    for _ in range(num_repetitive):
        layers.append(EncoderBlock(in_channels=in_channels, out_channels=out_channels))

        in_channels = out_channels
        out_channels *= 2

    model = nn.Sequential(*layers)

    print(EncoderBlock.total_params(model=model))

    print(summary(model=model, input_size=(3, 128, 128)))

    EncoderBlock.model_architecture(model=model)
