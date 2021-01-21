import argparse
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from simpletransformers.classification import ClassificationModel
import torch
#import wandb

def balance_subset(df):
    negative_samples = df[df['label'] == 0]
    negative_length = len(negative_samples)
    positive_samples = df[df['label'] == 1]
    positive_length = len(positive_samples)
    shortest_lenght = min(negative_length, positive_length)
    negative_samples = negative_samples.sample(n = shortest_lenght)
    positive_samples = positive_samples.sample(n = shortest_lenght)
    balanced_df = pd.concat([negative_samples, positive_samples]).sample(frac=1.0)
    return balanced_df

def get_training_data(data_path: Path, balance: bool):
    SUBSET_NAMES = ['train', 'dev']
    subsets = {name: pd.read_pickle(data_path / f'norsk_kategori_4_{name}.pkl') for name in SUBSET_NAMES}
    subsets = {name: subsets[name].rename(columns={'rating': 'label'}) for name in SUBSET_NAMES}
    if balance:
        subsets['train'] = balance_subset(subsets['train'])
    return subsets

def get_model(num_labels: int, label_weights):
    use_cuda = True if torch.cuda.is_available() else False
    model = ClassificationModel(
        'distilbert', 
        'distilbert-base-multilingual-cased', 
        num_labels=num_labels,
        use_cuda=use_cuda,
        weight=label_weights,
        args={
            'wandb_project': 'capra_simple_transformers',
            'use_cached_eval_features': False,
        }
    )
    return model

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--overwrite-output-dir", type=bool, default=False)
    parser.add_argument("--evaluate-during-training", type=bool, default=False)
    parser.add_argument("--save-model-every-epoch", type=bool, default=False)
    parser.add_argument("--balance-training-set", type=bool, default=False)
    parser.add_argument("--label-weights", type=str, default='1 1')

    default_model_dir = os.environ["SM_MODEL_DIR"] if "SM_MODEL_DIR" in os.environ else './models'
    parser.add_argument("--model-dir", type=str, default=default_model_dir)

    default_data_dir = os.environ["SM_CHANNEL_TRAINING"] if "SM_CHANNEL_TRAINING" in os.environ else '.'
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)

    args = parser.parse_args()
    print(f'Data directory: {args.data_dir}')

    config = {
        'balance_training_set': args.balance_training_set,
        'data_dir': args.data_dir,
        'eval_batch_size': args.batch_size,
        'evaluate_during_training': args.evaluate_during_training,
        'label_weights': [float(i) for i in args.label_weights.split()],
        'learning_rate': args.learning_rate,
        'model_dir': args.model_dir,
        'num_train_epochs': args.num_epochs,
        'overwrite_output_dir': args.overwrite_output_dir,
        'save_model_every_epoch': args.save_model_every_epoch,
        'train_batch_size': args.batch_size
    }

    subsets = get_training_data(config['data_dir'], config['balance_training_set'])
    model = get_model(2, config['label_weights'])

    model.train_model(
        subsets['train'],
        eval_df=subsets['dev'],
        output_dir=config['model_dir'],
        args=config
    )
    result, model_outputs, wrong_predictions = model.eval_model(subsets['dev'])
