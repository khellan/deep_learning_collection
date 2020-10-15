

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
        print('Beginning evauluation using %s validation samples.' %
              len(dataloaders["val"].dataset.targets))
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            if i % 10 == 0:
                print('Sample %s...' % (i * 4))
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
                                       'guessed': class_names[predicted[j]], 'was': class_names[labels[j]], 'filename': filename}
                    incorrect_guesses += [incorrect_guess]

    accuracy = correct / total
    print('\nIncorrect guesses: \n', json.dumps(incorrect_guesses, indent=2))
    print('Accuracy of the network on the %s validation images: %s%%' % (
        total, accuracy))

    dirpath = 'error_images'
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree('./%s' % dirpath)
    os.system("mkdir %s" % dirpath)
    for incorrect_guess in incorrect_guesses:
        orig_path = incorrect_guess['filename']
        new_path = 'error_images/%s_%s' % (orig_path.split(
            '/')[-2], orig_path.split('/')[-1])
        os.system('cp %s %s' % (orig_path, new_path))

    actual_labels = dataloaders['val'].dataset.targets

    save_path = './graphs'
    if not (os.path.exists(save_path) and os.path.isdir(save_path)):
        os.system('mkdir %s' % save_path)
    # now = datetime.now()
    # dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    # save_path = 'graphs/%s/' % dt_string
    # os.system('mkdir %s' % save_path)

    with open('guesses.json', 'w') as f:
        f.write('{\n\t"actual_labels": %s,\n\t"predicted": %s\n}' %
                (actual_labels, all_guesses))

    confusion_matrix = generate_confusion_matrix(
        actual_labels, incorrect_guesses)
    recall, precision, f1 = generate_stats(confusion_matrix)

    plot_confusion_matrix(confusion_matrix, class_names, save_path)
    plot_stats(accuracy, recall, precision, f1, save_path)
    plot_roc(actual_labels, all_guesses, save_path)


def generate_confusion_matrix(actual_labels, incorrect_guesses):
    # We define 'true positive' as guessed salmon correctly
    confusion_matrix = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    total_salmon_pics = len([x for x in actual_labels if x == 0])
    total_trout_pics = len(actual_labels) - total_salmon_pics
    assert total_trout_pics == len([x for x in actual_labels if x == 1])

    was_salmon_guessed_trout = len(
        [guess for guess in incorrect_guesses if guess['guessed'] == 'trout'])
    was_trout_guessed_salmon = len(
        incorrect_guesses) - was_salmon_guessed_trout
    # was_trout_guessed_salmon = len(
    #     [guess for guess in incorrect_guesses if guess['guessed'] == 'salmon'])

    confusion_matrix['tp'] = total_salmon_pics - was_salmon_guessed_trout
    confusion_matrix['fp'] = was_trout_guessed_salmon
    confusion_matrix['fn'] = was_salmon_guessed_trout
    confusion_matrix['tn'] = total_trout_pics - was_trout_guessed_salmon

    return confusion_matrix


def generate_stats(confusion_matrix):
    recall = confusion_matrix['tp'] / \
        (confusion_matrix['tp'] + confusion_matrix['fn'])
    precision = confusion_matrix['tp'] / \
        (confusion_matrix['tp'] + confusion_matrix['fp'])
    false_pos_rate = confusion_matrix['fp'] / \
        (confusion_matrix['fp'] + confusion_matrix['tn'])
    f1 = 2 * (recall * precision) / (recall + precision)

    print('Recall: %s' % recall)
    print('Precision: %s' % precision)
    print('F1 score: %s' % f1)
    return recall, precision, f1


def plot_stats(accuracy, recall, precision, f1, save_path):
    x = np.arange(4)
    fig, ax = plt.subplots()

    plt.grid(zorder=0)
    stats = [accuracy, recall, precision, f1]
    plt.bar(x, stats, zorder=3, color=['#003f5c',
                                       '#58508d',
                                       '#bc5090',
                                       '#ff6361',
                                       '#ffa600'])
    minLim = min(stats) - 0.05
    plt.ylim([minLim, 1.0])
    plt.xticks(x, ('Accuracy', 'Recall', 'Precision', 'F1 Score'))
    # plt.show()
    plt.savefig('%s/stats.png' % save_path)


def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    confusion_array = [[confusion_matrix['tp'], confusion_matrix['fp']], [
        confusion_matrix['fn'], confusion_matrix['tn']]]
    df_cm = pd.DataFrame(confusion_array, index=[i for i in class_names],
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
    plt.savefig('%s/confusion.png' % save_path)


def plot_roc(actual_labels, all_guesses, save_path):
    """
    compute ROC curve and ROC area for each class in each fold
    """

    y = np.array(actual_labels)
    y_pred = np.array(all_guesses)

    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print('AUC score: %s' % roc_auc)

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
    plt.savefig('%s/roc.png' % save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='../models/model.pth')
    parser.add_argument("--dataset", type=str, default='..')
    args = parser.parse_args()

    model = load_model(args.model)
    dataset_folder = args.dataset
    evaluate_model(model, dataset_folder)
