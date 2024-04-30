import sys
import os
import argparse
import zipfile
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append("src/")

from utils import params, load, dump


class Loader:
    def __init__(self, image_path=None, image_size=128, batch_size=1, split_size=0.20):
        self.image_path = image_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.split_size = split_size

        self.sharp_images = []
        self.blurred_images = []

        try:
            self.config = params()

        except Exception as e:
            print(
                "The config file is not valid. Please check the file and try again.".capitalize()
            )

    def unzip_folder(self):
        if os.path.exists(self.config["path"]["raw_path"]):
            with zipfile.ZipFile(self.image_path, "r") as zip_ref:
                zip_ref.extractall(
                    os.path.join(self.config["path"]["raw_path"], "dataset")
                )

        else:
            raise FileNotFoundError(
                "The raw data folder does not exist. Please check the path and try again."
            )

    def split_images(self, **kwargs):
        X_train, X_test, y_train, y_test = train_test_split(
            kwargs["X"], kwargs["y"], test_size=self.split_size, random_state=42
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.CenterCrop((self.image_size, self.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def feature_extractor(self):
        if os.path.exists(self.config["path"]["raw_path"]):
            self.directory = os.path.join(self.config["path"]["raw_path"], "dataset")
            self.sharp = ["sharp"]

            blurred_image = os.path.join(self.directory, "blurred")

            for category in self.sharp:
                path = os.path.join(self.directory, category)

                for image in os.listdir(path):
                    sharp_image_number = image.split("_")[0]

                    for blur_image in os.listdir(blurred_image):
                        blurred_image_number = blur_image.split("_")[0]

                        if sharp_image_number == blurred_image_number:
                            sharp_image = cv2.imread(os.path.join(path, image))
                            blur_image = cv2.imread(
                                os.path.join(blurred_image, blur_image)
                            )

                            sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)
                            blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)

                            self.sharp_images.append(
                                self.transforms()(Image.fromarray(sharp_image))
                            )
                            self.blurred_images.append(
                                self.transforms()(Image.fromarray(blur_image))
                            )

                        else:
                            continue

            dataset = self.split_images(X=self.blurred_images, y=self.sharp_images)

            return {
                "data": dataset,
                "sharp": self.sharp_images,
                "blurred": self.blurred_images,
            }

        else:
            raise FileNotFoundError(
                "The raw data folder does not exist. Please check the path and try again."
            )

    def create_dataloader(self):
        if os.path.exists(self.config["path"]["processed_path"]):

            data = self.feature_extractor()

            train_dataloader = DataLoader(
                dataset=list(zip(data["data"]["X_train"], data["data"]["y_train"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            test_dataloader = DataLoader(
                dataset=list(zip(data["data"]["X_test"], data["data"]["y_test"])),
                batch_size=self.batch_size * 16,
                shuffle=True,
            )

            dataloader = DataLoader(
                dataset=list(zip(data["sharp"], data["blurred"])),
                batch_size=self.batch_size,
                shuffle=True,
            )

            dump(
                value=dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "dataloader.pkl"
                ),
            )

            dump(
                value=train_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "train_dataloader.pkl"
                ),
            )

            dump(
                value=test_dataloader,
                filename=os.path.join(
                    self.config["path"]["processed_path"], "test_dataloader.pkl"
                ),
            )

        else:
            raise FileNotFoundError(
                "The processed data folder does not exist. Please check the path and try again."
            )

    @staticmethod
    def plot_images():
        config = params()

        if os.path.exists(config["path"]["processed_path"]):
            plt.figure(figsize=(40, 15))

            test_dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "test_dataloader.pkl"
                )
            )

            blurred, sharp = next(iter(test_dataloader))

            for index, image in enumerate(sharp):
                sharp_image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                blurred_image = (
                    blurred[index].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )

                sharp_image = (sharp_image - sharp_image.min()) / (
                    sharp_image.max() - sharp_image.min()
                )
                blurred_image = (blurred_image - blurred_image.min()) / (
                    blurred_image.max() - blurred_image.min()
                )

                plt.subplot(2 * 4, 2 * 4, 2 * index + 1)
                plt.imshow(sharp_image)
                plt.title("Sharp")
                plt.axis("off")

                plt.subplot(2 * 4, 2 * 4, 2 * index + 2)
                plt.imshow(blurred_image)
                plt.title("Blurred")
                plt.axis("off")

            plt.tight_layout()

            if os.path.exists(config["path"]["file_path"]):
                plt.savefig(os.path.join(config["path"]["file_path"], "images.jpg"))

            plt.show()

        else:
            raise FileNotFoundError(
                "The processed data folder does not exist. Please check the path and try again."
            )

    @staticmethod
    def dataset_details():
        config = params()

        if os.path.exists(config["path"]["processed_path"]):

            dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "dataloader.pkl"
                )
            )

            test_dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "test_dataloader.pkl"
                )
            )

            train_dataloader = load(
                filename=os.path.join(
                    config["path"]["processed_path"], "train_dataloader.pkl"
                )
            )

            train_data, train_label = next(iter(train_dataloader))
            test_data, test_label = next(iter(test_dataloader))

            pd.DataFrame(
                {
                    "total_data": str(sum(sharp.size(0) for sharp, _ in dataloader)),
                    "total_train_data": str(
                        sum(sharp.size(0) for sharp, _ in train_dataloader)
                    ),
                    "total_test_data": str(
                        sum(sharp.size(0) for sharp, _ in test_dataloader)
                    ),
                    "train_data_shape": str(train_data.size()),
                    "train_label_shape": str(train_label.size()),
                    "test_data_shape": str(test_data.size()),
                    "test_label_shape": str(test_label.size()),
                },
                index=["quantity".title()],
            ).T.to_csv(os.path.join(config["path"]["file_path"], "dataset_details.csv"))
        else:
            raise FileNotFoundError(
                "The processed data folder does not exist. Please check the path and try again."
            )


if __name__ == "__main__":
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
    args = parser.parse_args()

    if args.image_path:
        loader = Loader(
            image_path=args.image_path,
            image_size=args.image_size,
            split_size=args.split_size,
        )

        loader.unzip_folder()
        loader.create_dataloader()

        loader.plot_images()
        print(loader.dataset_details())

    else:
        print("Image path should be defined.".capitalize())
