# RESNET Image Classification

## Install

1. Install `pipenv` (OS-dependent)
2. `$ pipenv shell`
3. `$ pipenv install`
4. Aquire a dataset. The dataset folder (which defaults to `.`, ie `<BASE_FOLDER>/salmon_trout/`) needs to contain a `train` and `val` folder. Each of those needs to contain a folder for each class in the dataset, ie. "salmon" and "trout". So `<path_to_dataset_folder>/train/salmon` would be the salmon images to use for training, and `<path_to_dataset_folder>/val/trout` would be the trout images used for validation. Placing a `train` and a `val` folder in the same folder as this README will let you run all scripts without specifying dataset-folder.

## Training

Training can be done either locally or in the cloud. Specifically, files for training in AWS Sagemaker and on Azure are included.

Regardless of where you train, there are several options and hyperparamters available for tuning:

- `epochs`: how long to train;
- Several neural network hyperparameters:
  - `learning_rate`,
  - `gamma`,
  - `step-size`;
- `model-dir`: the folder where output model should be saved;
- `data-dir`: the folder containing the dataset.

Note that these final two directories should not be supplied when using AWS Sagemaker, as the environment variables `SM_MODEL_DIR` and `SM_CHANNEL_TRAINING` are used instead.

For explanations of the neural network hyperparameters, see for example RESNET's documentation.

### Training locally

1. `$ python train.py`

It's as simple as that. All of the options above are also available, passed in as such:
`$ python train.py --epochs 20`.

### Training in the Cloud

#### Sagemaker

1. Create a Sagemaker Notebook. The easiest way to include the project files is to add them as a Git Repository when creating the Notebook.
1. Put your dataset in S3. The preffered location is in the Sagemaker Notebook's supplementary bucket, in a folder named `/salmon_trout/`, under a `data` folder. So the training data's path should be something like `sagemaker-<REGION>-<USERID>/salmon_trout/data/train/`
1. Add the project files to the notebook if you didn't do so in step 1.

At this point you have two options for training.

- You can train "directly" by opening and running the `pytorch_image_transfer_learning.ipynb` file.
- Alternatively, you can run the project _containerized_ through Docker by opening and running the `build_and_train_docker.ipynb` file. This will automatically create a runnable Docker project for you and upload it to Amazon Elastic Container Registry (ECR). It will then run that image.

In either case, you can modify the training hyperparameters, dataset folder, etc by editing your Jupyter Notebook file. Perhaps _most important_ to note is that the `train_instance_type` is set to `'local'` for several of the notebooks. This means the code will be run on hardware that is likely no better than your local machine. In order to get the benefit of cloud computing, change this to something like `ml.p3.2xlarge`. For a full list of available instance types, see [here](https://aws.amazon.com/sagemaker/pricing/).

The resuling trained model will be located in the Sagemaker Notebook's S3 bucket.

#### Azure

TODO

## Model Evaluation

The `evaluation` folder includes files for determinging statistics, generatic graphs, and otherwise provide insight into the performance of the model.

`run_model.py` simply runs the model on a given folder with test data, still structured as `<BASE_FOLDER>/val/salmon`. Its options include `--model` and `--datadir`, which simply let you decide which model to run and with which data. It also has the `--url` option, letting you supply an url to an image which is then run through the model for an easy way to test a single image. This is not fully functional yet.

`evaluate_model.py` is more comprehensive, generating the aforementioned statistics etc. Its options include `--model` and `--datadir`, which simply let you decide which model and which data to use for evaluation.

## Interpretability

The `interpetation` folder contains files for explaining and analyzing the model. `shap_explainer.py` uses the [Shap](https://github.com/slundberg/shap) library. Make sure to provide dataset/images and update the paths accordingly in order to run.

