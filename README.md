<div align="center">
  <img src="https://github.com/denver1117/classificator/blob/master/doc/logo/main_logo.png"><br>
</div>

-----------------

# classificator : a convenience wrapper for classification methods in Python's scikit-learn

### About
The Classificator is a wrapper around scikit-learn’s classification suite, whose primary function is to normalize and automate the process by which common machine learning classification models are trained and validated. This is specifically in the context of text feature spaces, which require NLP techniques for preprocessing and vectorization. There is an additional focus on group cross validation, where labeled groups exist as data features, and variance is known to be smaller within groups. 

### Features
Load a dataset 
- User points to a dataset locally or in Amazon S3 
- User defines simple data specs 
- User defines feature columns, label column and group columns 

Load a configuration 
- User specifies preprocessing options, model choices, and hyperparameter grids per model choice 
- User specifies training session meta parameters 

Train and validate various model choices 
- Builds pre-processing pipeline to vectorize or encode text and standardize and impute numeric feautres
- Uses common cross validation and grid search techniques, and chooses the best model based on a user supplied scoring option 
- User defines output location for training and validation reports and pickled model object 

### Installation

Source Code: https://github.com/denver1117/classificator <br>
Package Index: https://pypi.python.org/pypi?name=classificator&version=0.1&:action=display

Binary installers for the latest released version are available at the Python package index:

```
# PyPI
pip install classificator
```

### Quick Start

- See `scripts/test.py` for a simple example with a stock data set and configuration
- Here is a sample configuration (`config.json`):
```
{
    "data_specs": {
        "loc": "doc/data/tickers.tsv", 
        "label_column": "CapBucket_Label", 
        "feature_columns": ["Industry", "Sector"], 
        "feature_methods": ["vectorize", "encode"], 
        "group_columns": [], "sep": "\t",
        "out_loc": "tmp",
        "model_name": "model1.obj",
        "run_name": "my_model_run"
    },
    "meta_specs": {
        "n_jobs": 1, "bypass": 0, "split_ratio": 0.9, 
        "random_seed": 15, "k": 5, "score": "Accuracy", 
        "train_final": 1
    },
    "vectorizer": {
        "model": "TF-IDF Vectorizer", 
        "args": {
            "ngram_range": [1,2], "min_df": 0.001, 
            "decode_error": "ignore", 
            "analyzer": "word"
        }
    },
    "selector": {
        "model": "Chi-Squared", 
        "grid": {"alpha": [0.05, 0.10]}
    },
    "pre_processors": {"methods": [[]]},
    "classifiers": {
        "Decision Tree": {"min_samples_split": [2,0.01]},  
        "Logistic Regression": {"C": [1, 10]}
    }
}
```
- Running the following:
```
    from classificator.classify import Classificator

    clf = Classificator(config_loc="config.json")
    clf.choose_model()
```
will build out a text pre-processing pipeline using a `TF-IDF Vectorizer` (with the supplied parameters) for the column `Industry` and a OneHotEncoder for the column `Sector` in order to predict classes `CapBucket_Label`, with data located here: `doc/data/tickers.tsv`.  It will send logging, validation metrics, and a pickled model object here: `tmp`.  It will select features from the vectorized data using Chi-Squared feature selection, attemtping multiple alpha values in grid search.  It will run separate grid search processes for a `Decision Tree` and for a `Logistic Regression` using the same pre-processing pipeline.  The best model will be chosen by the best `Accuracy` score among parameter maps across classifier choices.  

