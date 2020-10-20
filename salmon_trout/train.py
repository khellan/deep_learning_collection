import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")
            if run:
                run.log(f"{phase} Accuracy", np.float(epoch_acc))
                run.log(f"{phase} Loss", np.float(epoch_loss))
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
    print(f"Best val Acc: {best_acc}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
                ax.set_title("predicted: {}".format(class_names[preds[j]]))
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


def fine_tune_train(device, num_epochs, dataloaders, dataset_sizes, num_classes):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_ft, step_size=7, gamma=0.1)
    model_ft, best_accuracy = train_model(
        model_ft,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=num_epochs,
    )
    return model_ft


def final_layer_train(device, num_epochs, dataloaders, dataset_sizes, num_classes):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, num_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(
        model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(
        model_conv,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        num_epochs=num_epochs,
    )
    return model_conv


def get_data_transforms():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                    0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        ),
    }
    return data_transforms


def load_datasets(data_dir, shuffle=True):
    data_transforms = get_data_transforms()
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "val"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=4, shuffle=shuffle, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes
    return dataloaders, dataset_sizes, class_names


def save_model(modle, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--step-size", type=int, default=7)
    parser.add_argument(
        "--action",
        type=str,
        const="final_layer",
        default="final_layer",
        nargs="?",
        choices=["final_layer", "fine_tune", "show_examples"],
    )
    parser.add_argument("--environment", type=str, default="aws")

    default_model_dir = os.environ["SM_MODEL_DIR"] if "SM_MODEL_DIR" in os.environ else './models'
    parser.add_argument("--model-dir", type=str, default=default_model_dir)

    default_data_dir = os.environ["SM_CHANNEL_TRAINING"] if "SM_CHANNEL_TRAINING" in os.environ else '.'
    parser.add_argument("--data-dir", type=str, default=default_data_dir)

    args = parser.parse_args()

    run = None  # Used in Azure only
    if args.environment == 'azure':
        from azureml.core import Run

    args = parser.parse_args()
    dataloaders, dataset_sizes, class_names = load_datasets(args.data_dir)
    [print(f"Dataset {x} has {dataset_sizes[x]} images.")
     for x in dataset_sizes]
    [print(f"Class {x}.") for x in class_names]

    print(f"Action argument: {args.action}")
    if args.action == "fine_tune":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = fine_tune_train(
            device, args.epochs, dataloaders, dataset_sizes, len(class_names)
        )
        save_model(model, args.model_dir)
    elif args.action == "final_layer":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = final_layer_train(
            device, args.epochs, dataloaders, dataset_sizes, len(class_names)
        )
        save_model(model, args.model_dir)
    elif args.action == "show_examples":
        show_examples(dataloaders, class_names)
    else:
        print(f"Unknown argument {args.action}")
        exit(-1)
