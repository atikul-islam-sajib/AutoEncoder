# Auto Encoder Implementation

This project provides a complete framework for training and testing an AutoEncoder. It includes functionality for data preparation, model training, testing, and inference as well.

<img src="https://github.com/atikul-islam-sajib/Research-Assistant-Work-/blob/main/images.jpg">

#### Auto Encoder Architecture

<img src="https://images.squarespace-cdn.com/content/v1/62f1caa0a2cb083186ccce31/c3350b48-ff66-4e04-aa79-56cec8350ca2/autoencoder_architecture.png">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/AutoEncoder.git** |
| 2    | Navigate into the project directory.         | **cd AutoEncoder**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure

This project is thoughtfully organized to support the development, training, and evaluation of the AutoEncoder model efficiently. Below is a concise overview of the directory structure and their specific roles:

- **checkpoints/**
  - Stores model checkpoints during training for later resumption.
- **best_model/**

  - Contains the best-performing model checkpoints as determined by validation metrics.

- **train_models/**

  - Houses all model checkpoints generated throughout the training process.

- **data/**

  - **processed/**: Processed data ready for modeling, having undergone normalization, augmentation, or encoding.
  - **raw/**: Original, unmodified data serving as the baseline for all preprocessing.

- **logs/**

  - **Log** files for debugging and tracking model training progress.

- **metrics/**

  - Files related to model performance metrics for evaluation purposes.

- **outputs/**

  - **test_images/**: Images generated during the testing phase, including segmentation outputs.
  - **train_gif/**: GIFs compiled from training images showcasing the model's learning progress.
  - **train_images/**: Images generated during training for performance visualization.

- **research/**

  - **notebooks/**: Jupyter notebooks for research, experiments, and exploratory analyses conducted during the project.

- **src/**

  - Source code directory containing all custom modules, scripts, and utility functions for the U-Net model.

- **unittest/**
  - Unit tests ensuring code reliability, correctness, and functionality across various project components.

### Dataset Organization for AutoEncoder

The dataset is organized into three categories for CycleGAN. Each category directly contains unpaired images and their corresponding  images stored.

#### Directory Structure:

```
dataset/
├── sharp/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── blurred/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

For detailed documentation on the dataset visit the [Dataset - Kaggle](https://www.kaggle.com/datasets/kwentar/blur-dataset).


<!-- ### Model History

For detailed documentation on the implementation and usage, visit the -> [CycleGAN Model History](https://github.com/atikul-islam-sajib/SRGAN/blob/main/checkpoints/model_history/history.csv). -->

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--split_size`| Whether to split the dataset             | float   | 0.20   |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--l1`          | Use l1                            | bool   | True    |
| `--l2`          | Use l2                           | bool   | True    |
| `--elastic_net`          | Use elastic_net                           | bool   | True    |
| `--content_loss`          | Use content_loss                           | bool   | True    |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--test`          | Flag to initiate testing mode                | action | N/A     |

### CLI Command Examples

| Task                     | CUDA Command                                                                                                              | MPS Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python cli.py --train --image_path "/path/to/dataset" --batch_size 1 --image_size 256 --epochs 1000 --lr 0.0002 --adam True --device "cuda"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 1 --image_size 256 --epochs 1000 --lr 0.0002 --adam True --device "mps"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 1 --image_size 256 --epochs 1000 --lr 0.0002 --content_loss 0.01 --adam True --device "cpu"` |
| **Testing a Model**      | `python cli.py --test --model "/path/to/saved_model.pth" --device "cuda"`                                              | `python cli.py --test --model "/path/to/saved_model.pth" --device "mps"`                                              | `python main.py --test --model "/path/to/saved_model.pth" --device "cpu"`                                              |                                           |

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **MPS Command**: For Apple Silicon (M1, M2 chips), using the `mps` device can provide optimized performance.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.


#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=1, image_size=256, split_size=0.2)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
print(Loader.plot_images())
print(Loader.dataset_details())        # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer      
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    is_display=True            # Display training progress and statistics
    l1 = True                  # l1, l2, elastic_net, content loss
)

# Start training
trainer.train()
```

##### Training Performances
```python
print(trainer.plot_history())    # It will plot the netD and netG losses for each epochs
```

#### Testing the Model
```python
test = TestModel(device="cuda", gif=False, model_path="path/to/saved_model.pth") # use mps, cpu
test.test()
```

## Contributing
Contributions to improve this implementation of CycleGAN are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).

