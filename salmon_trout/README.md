# RESNET Image Classification

## Install

1. Install `pipenv` (OS-dependent)
2. `$ pipenv shell`
3. `$ pipenv install`
4. Aquire a dataset. The dataset folder (defaults to `.`, ie `resnet_image_classification`) needs to contain a `train` and `val` folder. Each of those needs to contain a folder for each class in the dataset, ie. "salmon" and "trout". So `<path_to_dataset_folder>/train/salmon` would be the salmon images to use for training, and `<path_to_dataset_folder>/val/trout` would be the trout images used for validation.

## Train

Training can be done either locally or in the cloud. Specifically, Files for training in AWS Sagemaker and on Azure are included.

Regardless of where you train, there are several options and hyperparamters available for tuning:

- `epochs`: how long to train,
- Several network hyperparameters:
  - `learning_rate`,
  - `gamma`,
  - `step-size`
- `model-dir`: the folder where output model should be saved,
- `data-dir`: The folder containing the dataset.

Note that these final two directories should not be supplied when using AWS Sagemaker, as the environment variables `SM_MODEL_DIR` and `SM_CHANNEL_TRAINING` are used instead.

### Training locally

1. `$ python train.py`

It's as simple as that. All of the options above are also available, passed in as such:
`$ python train.py --epochs 20`.

## Model Evaluation

The `evaluation` folder includes files for determinging statistics, generatic graphs, and otherwise provide insight into the performance of the model. `test.py` simply runs the model on a given folder with test data, still structured as `<BASE_FOLDER>/val/salmon`. `evaluate_model` is more comprehensive, generating the aforementioned statistics etc.

## Interpretability

TODO
