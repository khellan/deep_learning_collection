# Text Classification with NoReC

## Install

1. Install `pipenv` (OS-dependent)
2. `$ pipenv shell`
3. `$ pipenv install`
4. Aquire a dataset. The dataset folder (which defaults to `.`, ie `<BASE_FOLDER>/salmon_trout/`) needs to contain a `train` and `val` folder. Each of those needs to contain a folder for each class in the dataset, ie. "salmon" and "trout". So `<path_to_dataset_folder>/train/salmon` would be the salmon images to use for training, and `<path_to_dataset_folder>/val/trout` would be the trout images used for validation. Placing a `train` and a `val` folder in the same folder as this README will let you run all scripts without specifying dataset-folder.

## Training

There are several notebooks for training with different machine learning frameworks.

* [Simple Transformers](norec_simple_transformers.ipynb)
* [spaCy](norec_spacy.ipynb)
* [XGBoost](norec_xgb.ipynb)
* [Hyperparameter tuning for XGBoost](norec_xgb%20hyperparam_tuning.ipynb)
* [XGBoost with Tokenizers](norec_xgb%20tokenizers.ipynb)

Simple transformers don't work well on this imbalanced dataset. Despite using class weighting, it ends up only declaring one class.
spaCy by itself suffers from the same neural network issues.
XGBoost performs well
Hyperparameter tuning shows that the default arguments for XGBoost are sane.
XGBoost with Tokenizers improve marginally on pure XGBoost.