import yaml
import joblib


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
