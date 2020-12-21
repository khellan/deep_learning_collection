# Autoencoders

## Install

These notebooks are meant for running on either Google Colab (simple Autoencoder) or Sagemaker (Convolutional Autoencoder). Running locally requires some adaptation.

## WandB

These notebooks use Weights and Biases (wandb.ai) for logging during training. This can be skipped, but it is a nice to tool to have in your ML toolbox. Create an account if you don't have one and create a project. The project name must be the same as the one stated in the wandb.init call later.

## Training

### Simple Autoencoder

This Autoencoder should be trained in **Google Colab**. It is trained as a 
classical Autoencoder. The encoder of the network is then used to 
generate embedding vectors (codes) for each image. These codes are
inserted into a lookup table.

A k-nearest neighbour classificer is trained on these codes. 

These hyperparameters are available for tuning:

- `epochs`: how long to train (it is stated in the train function);
- `input_shape`: the number of pixels in the input images
- `BASE_PATH`: the base path of the data and model 
- `model-dir`: the folder where output model should be saved;
- `data-dir`: the folder containing the dataset.

### Convolutional Autoencoder

This Autoencoder should be trained in **Sagemaker**. The connection to Google 
Drive from Google Colab is brittle and I never managed to train in Colab.

#### Sagemaker

##### Training the Autoencoder

1. Open **Sagemaker Studio**
2. Create a directory with a suitable name and upload *autoencoder.convolutional.market1501.ipynb* to this directory in Sagemaker Studio
3. Create a directory called source under the directory where the notebook was uploaded
4. Upload the files from *source* to the *source* directory
5. Download the [Market1501](https://www.kaggle.com/pengcw1/market-1501/data) from Kaggle, uncompress it and put it in S3. Update the BUCKET and DATA_PATH constants in the notebook.   

Now you can train the convolutional Autoencoder in the notebook. 

The resuling trained model will be located in the S3 bucket pointed to in the BUCKET constant.

##### Training the classifier

Identify the training jobname and Update MODEL_PATH to reflect that.
Run the rest of the Notebook to see how it will work. 

As a learning experience, it is recommeneded to step through and run each step individually. When appropriate, inspect the results to understand what is going on.
