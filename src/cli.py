import sys
import os
import argparse
import torch
import torch.nn as nn

sys.path.append("src/")

from dataloader import Loader
from trainer import Trainer
from test import TestModel


def cli():
    parser = argparse.ArgumentParser(description="Dataloader for AutoEncoder".title())

    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the folder containing the images".capitalize(),
    )

    parser.add_argument(
        "--image_size", default=256, type=int, help="Image size".capitalize()
    )

    parser.add_argument(
        "--split_size", default=0.20, type=float, help="Split size".capitalize()
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size".capitalize()
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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the best model".capitalize(),
    )
    parser.add_argument(
        "--gif",
        type=bool,
        default=False,
        help="Create a gif of the test images".capitalize(),
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model".capitalize()
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model".capitalize()
    )

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            split_size=args.split_size,
            batch_size=args.batch_size,
        )

        # loader.unzip_folder()
        loader.create_dataloader()

        loader.plot_images()
        print(loader.dataset_details())

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

    elif args.test:
        test = TestModel(device=args.device, gif=args.gif, model=args.model)
        test.test()


if __name__ == "__main__":
    cli()
