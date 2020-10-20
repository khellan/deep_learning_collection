import shutil
import pandas as pd
from datetime import datetime
from run_model import load_model, visualize_model
from sklearn import metrics
import json
import argparse
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
import sys
# Required to import from containing folder in python
sys.path.insert(0, '..')


def evaluate_model(model, dataset_folder):
    from train import load_datasets

    correct = 0
    total = 0
    dataloaders, _, class_names = load_datasets(
        dataset_folder, shuffle=False)

    all_guesses = []
    incorrect_guesses = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        print(
            f'Beginning evauluation using {len(dataloaders["val"].dataset.targets)} validation samples.')
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            if i % 10 == 0:
                print(f'Sample {i * 4}...')
            inputs = inputs.to(device)

            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_guesses += list(predicted.numpy())
            # for o in outputs.data:
            #     all_guesses += [max(list(o.numpy()))]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for j in range(inputs.size()[0]):
                if labels[j] != predicted[j]:
                    absolute_index = i * 4 + j
                    filename, _ = dataloaders['val'].dataset.samples[absolute_index]
                    incorrect_guess = {'index': absolute_index,
                                       'guessed': class_names[predicted[j]],
                                       'was': class_names[labels[j]],
                                       'filename': filename}
                    incorrect_guesses += [incorrect_guess]

    accuracy = correct / total
    print('\nIncorrect guesses: \n', json.dumps(incorrect_guesses, indent=2))
    print(
        f'Accuracy of the network on the {total} validation images: {accuracy}')

    dirpath = 'error_images'
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(f'./{dirpath}')
    os.system(f"mkdir {dirpath}")
    for incorrect_guess in incorrect_guesses:
        orig_path = incorrect_guess['filename']
        new_path = f'error_images/{orig_path.split("/")[-2]}_{orig_path.split("/")[-1]}'
        os.system(f'cp {orig_path} {new_path}')

    actual_labels = dataloaders['val'].dataset.targets

    # Uncomment this code (and comment out the following save_path = ...)
    # to make a separate folder for the graphs produced by each evaluation run.

    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    # save_path = 'graphs/%s/' % dt_string
    save_path = './graphs'
    if not (os.path.exists(save_path) and os.path.isdir(save_path)):
        os.system(f'mkdir {save_path}')

    # Uncomment this to get a json file containing the raw data of
    # all true values and all guesses.

    # with open('guesses.json', 'w') as f:
    #     f.write('{\n\t"actual_labels": %s,\n\t"predicted": %s\n}' %
    #             (actual_labels, all_guesses))

    y = np.array(actual_labels)
    y_pred = np.array(all_guesses)

    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y, y_pred, pos_label=0, average='binary')

    plot_confusion_matrix(y, y_pred, class_names, save_path)
    plot_stats(accuracy, recall, precision, f1, save_path)
    plot_roc(y, y_pred, save_path)


def plot_stats(accuracy, recall, precision, f1, save_path):
    x = np.arange(4)
    fig, ax = plt.subplots()

    plt.grid(zorder=0)
    stats = [accuracy, recall, precision, f1]
    colors = ['#003f5c',
              '#58508d',
              '#bc5090',
              '#ff6361',
              '#ffa600']
    plt.bar(x, stats, zorder=3, color=colors)
    minLim = min(stats) - 0.05
    plt.ylim([minLim, 1.0])
    plt.xticks(x, ('Accuracy', 'Recall', 'Precision', 'F1 Score'))
    # plt.show()
    plt.savefig(f'{save_path}/stats.png')


def plot_confusion_matrix(y, y_pred, class_names, save_path):
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    tn, fp, fn, tp = confusion_matrix.ravel()
    confusion_matrix_with_actual_top = [[tp, fp], [fn, tn]]

    df_cm = pd.DataFrame(confusion_matrix_with_actual_top,
                         index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(6, 4))

    # Importing seaborn in the beginning of the file was causing a strange bug
    # where the loop crashed due to dataloaders['val'] changing size
    # mid-iteration. How that has anything to do with seaborn I don't know.
    import seaborn as sn
    ax = sn.heatmap(df_cm, annot=True, fmt="d")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position('top')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(f'{save_path}/confusion.png')


def plot_roc(y, y_pred, save_path):
    """
    compute ROC curve and ROC area for each class in each fold
    """

    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print(f'AUC score: {roc_auc}')

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)  # roc_auc_score

    plt.plot([0, 1], [0, 1], 'k--')
    # plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC')
    plt.legend(loc="lower right")
    # plt.tight_layout()
    plt.savefig(f'{save_path}/roc.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default='../models/model.pth')
    parser.add_argument("--model", type=str, default='./model.pth')
    parser.add_argument("--dataset", type=str, default='..')
    args = parser.parse_args()

    model = load_model(args.model)
    dataset_folder = args.dataset
    evaluate_model(model, dataset_folder)
