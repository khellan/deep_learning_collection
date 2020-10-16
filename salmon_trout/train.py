import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import io
from PIL import Image
import time
import os
import copy


def train_model(
    model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, run
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
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

            print("{} Loss: {:.4f} Acc: {:.4f}".format(
                phase, epoch_loss, epoch_acc))
            if run:
                run.log(f"{phase} Accuracy", np.float(epoch_acc))
                run.log(f"{phase} Loss", np.float(epoch_loss))
            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def replace_final_layer(model, num_classes):
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def fine_tune_train(
    model_ft, device, num_epochs, dataloaders, dataset_sizes, num_classes, run
):
    model_ft = replace_final_layer(model_ft, num_classes)

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
        run=run,
    )
    return model_ft, best_accuracy


def final_layer_train(
    model_conv, device, num_epochs, dataloaders, dataset_sizes, num_classes, run
):
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    model_conv = replace_final_layer(model_conv, num_classes)

    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(
        model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer_conv, step_size=7, gamma=0.1)

    model_conv, best_accuracy = train_model(
        model_conv,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        num_epochs=num_epochs,
        run=run,
    )
    return model_conv, best_accuracy


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


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18()
    model = replace_final_layer(model, 2)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def input_fn(request_body, content_type):
    if content_type != 'application/x-image':
        raise Eexception(f'Unknown content type {content_type}')
    print(request_body[:100])
    image_array = np.array(Image.open(io.BytesIO(request_body)))
    image = Image.fromarray(image_array.astype('uint8'), 'RGB')
    transform = get_data_transforms()['val']
    transformed_image = transform(image)
    return torch.unsqueeze(transformed_image, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=1)
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
        choices=["final_layer", "fine_tune", "show_examples", "eval"],
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

        run = Run.get_context()

    pretrained_model = models.resnet18(pretrained=True)
    dataloaders, dataset_sizes, class_names = load_datasets(args.data_dir)
    [print(f"Dataset {x} has {dataset_sizes[x]} images.")
     for x in dataset_sizes]
    [print(f"Class {x}.") for x in class_names]

    print(f"Action argument: {args.action}")
    if args.action == "eval":
        print(pretrained_model.eval())
    elif args.action == "fine_tune":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, best_accuracy = fine_tune_train(
            pretrained_model,
            device,
            args.epochs,
            dataloaders,
            dataset_sizes,
            len(class_names),
            run,
        )
        if run:
            run.log("Accuracy", np.float(best_accuracy))
        save_model(model, args.model_dir)
    elif args.action == "final_layer":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, best_accuracy = final_layer_train(
            pretrained_model,
            device,
            args.epochs,
            dataloaders,
            dataset_sizes,
            len(class_names),
            run,
        )
        if run:
            run.log("Best Accuracy", np.float(best_accuracy))
        save_model(model, args.model_dir)
    else:
        print(f"Unknown argument {args.action}")
        exit(-1)
    if run:
        run.complete()
