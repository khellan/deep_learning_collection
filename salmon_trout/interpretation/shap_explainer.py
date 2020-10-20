import os
import glob

from torchvision import models, transforms
from PIL import Image
import joblib
import numpy as np
import shap
from sklearn import preprocessing
from torch.nn.utils.weight_norm import weight_norm
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets


def get_data_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        ),
    }
    return data_transforms


def load_datasets(data_dir, shuffle=True, batch_size=4):
    data_transforms = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=4
        )
        for x in ["train", "val"]
    }
    return dataloaders

def get_model(path, object_categories=['salmon', 'trout']):
    classifier = models.resnet18()
    num_features = classifier.fc.in_features
    classifier.fc = torch.nn.Linear(num_features, len(object_categories))
    classifier.load_state_dict(torch.load(path))
    classifier.eval()
    return classifier


def numpy_transform_test_images(test_images):
    """
    Transforming every element to PIL image format to get proper format of RGB values([0,255], [0,255], [0,255])\n
    Then transforming everything to numpy, which is the format the plot_image of shap expects
    """
    pil_transformer = transforms.Compose([transforms.ToPILImage()])
    return np.asarray([np.array(pil_transformer(test_image)).astype(np.float32) for test_image in test_images])


def get_shap_values(explainer, test_images):
    return explainer.shap_values(test_images)

def get_shap_numpy(shap_values):
    return [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]


if __name__ == "__main__":

    model_path = './models/model.pth'
    data_path = '../data'

    batch_size = 10
    #Number of images in dataset + test images cannot currently exceed batch size 
    no_images = 6
    no_test_images = 4
    dataloaders = load_datasets(data_path, batch_size=batch_size)

    batch = next(iter(dataloaders['train']))
    images, _ = batch

    model = get_model(model_path)
    background = images[:no_images]
    explainer = shap.DeepExplainer(model, background)

    test_images = images[no_images:no_images+no_test_images]
    shap_values = get_shap_values(explainer, test_images)

    shap_numpy = get_shap_numpy(shap_values)
    np_transformed_test_images = numpy_transform_test_images(test_images)

    labels = no_test_images*[['Salmon', 'Trout']]

    shap.image_plot(shap_numpy, np_transformed_test_images, labels=labels)
