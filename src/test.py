import sys
import os
import argparse
import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append("src/")

from utils import load, params, device_init
from auto_encoder import AutoEncoder


class TestModel:
    def __init__(self, model=None, device="mps", **kwargs):
        self.model = model
        self.device = device_init(device=device)
        self.kwargs = kwargs

        self.netAE = AutoEncoder(in_channels=3)
        self.config = params()

        self.netAE.to(self.device)

    def select_best_model(self):
        if self.model:
            if os.path.exists(self.config["path"]["train_models"]):
                user_defined_model = self.model.split("/")[-1]
                if user_defined_model in os.listdir(
                    self.config["path"]["train_models"]
                ):

                    self.netAE.load_state_dict(torch.load(self.model))

                else:
                    raise Exception("Model cannot be found".capitalize())

            else:
                raise FileNotFoundError(
                    "Train models folder does not exist. Please create a folder.".capitalize()
                )

        else:
            if os.path.exists(self.config["path"]["best_model"]):

                load_state = torch.load(
                    os.path.join(self.config["path"]["best_model"], "best_model.pth")
                )
                self.netAE.load_state_dict(load_state["model"])
            else:
                raise FileNotFoundError("Model cannot be found".capitalize())

    def image_normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def plot(self):
        if os.path.exists(self.config["path"]["test_image"]):

            test_dataloader = load(
                filename=os.path.join(
                    self.config["path"]["processed_path"], "test_dataloader.pkl"
                )
            )

            blurred, sharp = next(iter(test_dataloader))
            blurred, sharp = blurred[:16], sharp[:16]
            predicted_sharp = self.netAE(blurred.to(self.device))

            plt.figure(figsize=(40, 25))

            for index, image in enumerate(predicted_sharp):
                image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                image = self.image_normalize(image=image)

                sharp_image = (
                    sharp[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )
                sharp_image = self.image_normalize(image=sharp_image)

                blurred_image = (
                    blurred[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )
                blurred_image = self.image_normalize(image=blurred_image)

                plt.subplot(3 * 4, 3 * 4, 3 * index + 1)
                plt.imshow(blurred_image, cmap="gray")
                plt.title("Blurred".title())
                plt.axis("off")

                plt.subplot(3 * 4, 3 * 4, 3 * index + 2)
                plt.imshow(sharp_image, cmap="gray")
                plt.title("Sharp".title())
                plt.axis("off")

                plt.subplot(3 * 4, 3 * 4, 3 * index + 3)
                plt.imshow(image)
                plt.title("Pred_sharp".title())
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        else:
            raise FileNotFoundError(
                "Test image folder does not exist. Please create a folder.".capitalize()
            )

    def create_gif(self):
        if self.kwargs["gif"]:
            if os.path.exists(self.config["path"]["train_gif"]):
                self.images = [
                    imageio.imread(
                        os.path.join(self.config["path"]["train_images"], image)
                    )
                    for image in os.listdir(self.config["path"]["train_images"])
                ]

                imageio.mimsave(
                    os.path.join(self.config["path"]["train_gif"], "train_gif.gif"),
                    self.images,
                    "GIF",
                )

            else:
                raise FileNotFoundError(
                    "Train gif folder does not exist. Please create a folder.".capitalize()
                )
        else:
            pass

    def test(self):
        try:
            self.select_best_model()

        except FileNotFoundError as e:
            print("The exception is raised {}".format(e))

        except Exception as e:
            print("The exception is raised {}".format(e))

        else:
            self.plot()

        finally:
            self.create_gif()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Model for Auto Encoder".title())

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
        "--device", type=str, default="mps", help="Device to run the model".capitalize()
    )

    args = parser.parse_args()

    test = TestModel(device=args.device, gif=args.gif, model=args.model)
    test.test()
