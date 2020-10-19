import torch.nn as nn
import torchvision
import os
import torch
from torch.nn.utils.weight_norm import weight_norm
from sklearn import preprocessing
import shap
import numpy as np
import joblib
from train import load_datasets


if __name__ == "__main__":

    default_data_dir = os.environ["SM_CHANNEL_TRAINING"] if "SM_CHANNEL_TRAINING" in os.environ else '.'
    dataloaders, dataset_sizes, class_names = load_datasets(default_data_dir)

    model = torchvision.models.resnet18(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load('./models/model.pth'))

    batch = next(iter(dataloaders['val']))
    images, _ = batch

    no_images = 8
    background = images[:no_images]

    e = shap.DeepExplainer(model, background)

    n_test_images = 2
    test_images = images[no_images:no_images+n_test_images]
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2)
                  for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy)
