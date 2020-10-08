import argparse
import torch
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from train import visualize_model, get_data_transforms, load_datasets
from PIL import Image
import requests
from torch.autograd import Variable
import os.path
import json
from sklearn import metrics


def test_single(model, test_input, class_names):
    model.eval()
    output = model(test_input)

    _, prediction_tensor = torch.max(output, 1)

    prediction = class_names[prediction_tensor[0]]
    return prediction


def load_model(model_path):
    model = models.resnet18(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load('./models/model.pth'))
    model.eval()
    return model


def download_image(url):
    local_path = './test_images/%s' % url.split("/")[-1]
    # local_path = './test_images/img.jpg'
    if not os.path.isfile(local_path):
        resp = requests.get(url, headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'})
        if not resp.ok:
            resp.raise_for_status()
        img_data = resp.content
        with open(local_path, 'wb') as handler:
            handler.write(img_data)
    return local_path


def load_image_torch(image_name):
    imsize = 256
    loader = transforms.Compose(
        [transforms.Scale(imsize), transforms.ToTensor()])

    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return image.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default='')
    parser.add_argument("--dataset", type=str, default='.')
    parser.add_argument("--model", type=string, default='./models/model.pth')

    args = parser.parse_args()

    model = load_model(args.model)

    if args.url != '':
        classnames = ['salmon', 'trout']
        local_image_path = download_image(args.url)
        image = load_image_torch(local_image_path)
        prediction = test_single(model, image, classnames)
        print('Predicted:', prediction)
    else:
        dataloaders, _, class_names = load_datasets(args.dataset)
        # '~/Documents/ml/datasets/salmon-trout/output')
        visualize_model(model, dataloaders, class_names)
        time.sleep(100)
