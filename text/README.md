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

Simple transformers don't work well on this imbalanced dataset. It is possible to get decent results by tweaking class weights, but balancing the dataset by using the same number of positive and negative samples yields the best result. It is also important to note that fine-tuning of transformers is vulnerable to high learning rates and should not be trained too much.
spaCy by itself suffers from the same neural network issues.
XGBoost performs well right out of the box. Hyperparameter tuning shows that the default arguments for XGBoost are sane. XGBoost with Tokenizers improve marginally over pure XGBoost.
Even if not using spaCy for classification or advanced NLP, there are benefits to using spaCy's parser when processing text. Depending on the application, it might be desirable to filtering term classes or perform some normalisation of the text.
Typical filtering are:
* Stop word removal. Stop words are common words that may prolong training of classifiers, increase search index while not providing any value for the result.
* Removal of punctuation, ordinals (numbers in text)
Typical normalisation:
* Lowercasing - this is common since case doesn't carry much meaning for search or machine learning tasks. The exception is named entities where casing is a strong positive signal. 
* Lemmatisation or stemming - trasnforming term into their base form. Lemmatisation is exactly this, transform i.e. a verb like does into the base form do. Stemming is a simpler approach which tries to do the same by chopping of the last part of terms. Stemming is quick and often works well, but there are issues with stemming where terms that have different meaning have the same stem. An example of the latter is meanness and meaning which both are stemmed to mean.
