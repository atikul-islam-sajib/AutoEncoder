import yaml
import joblib
import torch
import torch.nn as nn


def params():
    with open("./default_params.yml", "r") as file:
        return yaml.safe_load(file)


def dump(value=None, filename=None):
    if value is not None:
        joblib.dump(value=value, filename=filename)

    else:
        raise ValueError("The value is empty. Please check the value and try again.")


def load(filename):
    if filename is not None:
        return joblib.load(filename=filename)

    else:
        raise ValueError(
            "The filename is empty. Please check the filename and try again."
        )


def device_init(device="mps"):
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        return torch.device("cuda")


def weight_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
