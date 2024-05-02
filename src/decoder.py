import sys
import os
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("src/")

from utils import params


class DecoderBlock(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512, is_last=False):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_last = is_last
        self.kernel = 2
        self.stride = 2
        self.padding = 0

        self.layers = OrderedDict()
        self.decoder = self.block()

    def block(self):
        self.layers["convTranspose"] = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
        )

        if self.is_last:
            self.layers["Tanh"] = nn.Tanh()
        # else:
        # self.layers["conv"] = nn.Conv2d(
        #     in_channels=self.out_channels,
        #     out_channels=self.out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )
        # self.layers["relu"] = nn.ReLU(inplace=True)

        # self.layers["conv_1"] = nn.Conv2d(
        #     in_channels=self.out_channels,
        #     out_channels=self.out_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )
        # self.layers["batch_norm"] = nn.BatchNorm2d(num_features=self.out_channels)
        # self.layers["relu_1"] = nn.ReLU(inplace=True)

        return nn.Sequential(self.layers)

    def forward(self, x):
        if x is not None:
            return self.decoder(x)

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

                draw_graph(model=model, input_size=(1, 1024, 8, 8)).visual_graph.render(
                    filename=os.path.join(config["path"]["file_path"], "decoder_block"),
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
        description="Decoder Block for AutoEncoder".title()
    )

    parser.add_argument(
        "--in_channels", default=1024, type=int, help="Input channels".capitalize()
    )

    parser.add_argument(
        "--out_channels", default=512, type=int, help="Output channels".capitalize()
    )

    args = parser.parse_args()

    in_channels = args.in_channels
    out_channels = args.out_channels
    num_repetitive = 5

    layers = []

    for idx in range(num_repetitive):
        layers.append(
            DecoderBlock(
                in_channels=in_channels,
                out_channels=3 if idx == num_repetitive - 1 else out_channels,
                is_last=True if idx == num_repetitive - 1 else False,
            )
        )

        in_channels = out_channels
        out_channels //= 2

    model = nn.Sequential(*layers)

    print(model(torch.randn(1, 1024, 8, 8)).size())

    print(summary(model=model, input_size=(1024, 8, 8)))

    print(DecoderBlock.total_params(model=model))

    DecoderBlock.model_architecture(model=model)
