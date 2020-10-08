import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from train import get_data_transforms, load_datasets
from test_model import load_model
import json
from sklearn import metrics
import seaborn as sn
import pandas as pd


def evaluate_model(model):
    correct = 0
    total = 0
    dataloaders, _, class_names = load_datasets(
        # './finn_test')
        '.', shuffle=False)

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
            for o in outputs.data:
                all_guesses += [max(list(o.numpy()))]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for j in range(inputs.size()[0]):
                if labels[j] != predicted[j]:
                    filename, _ = dataloaders['val'].dataset.samples[counter]
                    incorrect_guess = {'index': counter,
                                       'guessed': class_names[predicted[j]], 'was': class_names[labels[j]], 'filename': filename}
                    incorrect_guesses += [incorrect_guess]

    accuracy = correct / total
    print('Accuracy of the network on the %s validation images: %s%%' % (
        total, accuracy * 100))
    print('\nIncorrect guesses: \n', json.dumps(incorrect_guesses, indent=2))

    os.system("rm -rf error_images && mkdir error_images")
    for incorrect_guess in incorrect_guesses:
        orig_path = incorrect_guess['filename']
        new_path = 'error_images/%s_%s' % (orig_path.split(
            '/')[-2], orig_path.split('/')[-1])
        os.system('cp %s %s' % (orig_path, new_path))

    actual_labels = dataloaders['val'].dataset.targets

    os.system('mkdir graphs')
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = 'graphs/%s/' % dt_string
    os.system('mkdir %s' % save_path)

    confusion_matrix = generate_confusion_matrix(
        actual_labels, incorrect_guesses, save_path)
    recall, precision, f1 = generate_stats(confusion_matrix)

    plot_confusion_matrix(confusion_matrix, class_names, save_path)
    plot_stats(accuracy, recall, precision, f1)
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

    plt.bar(x, [accuracy, recall, precision, f1], color=['#003f5c',
                                                         '#58508d',
                                                         '#bc5090',
                                                         '#ff6361',
                                                         '#ffa600'])
    plt.ylim([0.9, 1.0])
    plt.xticks(x, ('Accuracy', 'Recall', 'Precision', 'F1 Score'))
    plt.grid()
    # plt.show()
    plt.savefig('%s/stats.png' % save_path)


def plot_confusion_matrix(confusion_matrix, class_names, save_path):
    confusion_array = [[confusion_matrix['tp'], confusion_matrix['fp']], [
        confusion_matrix['fn'], confusion_matrix['tn']]]
    df_cm = pd.DataFrame(confusion_array, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
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

    # plt.figure(figsize=(6,6))
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
    parser.add_argument("--model", type=str, default='./models/model.pth')
    args = parser.parse_args()

    model = load_model(args.model)
    evaluate_model(model)
