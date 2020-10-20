import argparse
import torch
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
from PIL import Image
import requests
from torch.autograd import Variable
import os.path
import json
from sklearn import metrics
# Required to import from containing folder in python
sys.path.insert(0, '..')


def test_single(model, test_input, class_names):
    model.eval()
    output = model(test_input)

    _, prediction_tensor = torch.max(output, 1)

    prediction = class_names[prediction_tensor[0]]
    return prediction


def load_model(model_path):
    model = models.resnet18(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def download_image(url):
    local_path = f'./test_images/{url.split("/")[-1]}'
    # local_path = './test_images/img.jpg'
    if not os.path.isfile(local_path):
        # You might need the commented-out http header (updated to fit your machine)
        # to successfully download images from the given url.
        resp = requests.get(url,
                            # headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
                            )
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


def visualize_model(model, dataloaders, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(f"predicted: {class_names[preds[j]]}")
                imshow(inputs.cpu().data[j])
                time.sleep(1)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def show_examples(dataloaders, class_names):
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    input("Press Enter to continue...")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, aspect='equal', resample=True)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", type=str, default='')
    parser.add_argument("--dataset", type=str, default='..')
    parser.add_argument("--model", type=str, default='../models/model.pth')

    args = parser.parse_args()

    model = load_model(args.model)

    if args.url != '':
        classnames = ['salmon', 'trout']
        local_image_path = download_image(args.url)
        image = load_image_torch(local_image_path)
        prediction = test_single(model, image, classnames)
        print('Predicted:', prediction)
    else:
        from train import load_datasets

        dataloaders, _, class_names = load_datasets(args.dataset)
        visualize_model(model, dataloaders, class_names)
        time.sleep(100)
