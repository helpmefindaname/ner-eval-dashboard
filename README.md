[![PyPI version](https://badge.fury.io/py/ner-eval-dashboard.svg)](https://badge.fury.io/py/ner-eval-dashboard)
[![GitHub Issues](https://img.shields.io/github/issues/helpmefindaname/ner-eval-dashboard.svg)](https://github.com/helpmefindaname/ner-eval-dashboard/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

# ner-eval-dashboard
Dashboard for Quality-driven NER.

## concept

The idea of this project is to provide a more elaborated evaluation for NER models.
That way, it should be easier to fix labeling mistakes,
better understand the positive and negative aspects of the trained NER model and see how it applies on unlabeled data.

## progress

version `0.1.0` provides standard F1 scores for Exact Match, Type Match, and Position Match. So far only [Flair](https://github.com/flairNLP/flair) models are implemented.
See [Issues](https://github.com/helpmefindaname/ner-eval-dashboard/issues) to view planned features

## installation

The ner eval dashboard can be installed via:
````
pip install ner-eval-dashboard==0.1
````

## usage

The ner eval dashboard can be used on various ways: cli, api or via docker.
It is important to always specify a model, a dataset and a tokenizer.

**Note:** To run the examples, you need to once manually download the CONLL03 dataset and put it into the `{FLAIR_CACHE_ROOT}/.flair/datasets/conll_03` folder.

### cli

The ner eval dashboard can be use via the command line interface:
````
ner_eval_dashboard [--dataset_path DATASET_PATH] [--extra_unlabeled_data EXTRA_UNLABELED_DATA] [--use USE [USE ...]] [--exclude EXCLUDE [EXCLUDE ...]] {FLAIR} predictor_name_or_path {SPACE} {RAW,COLUMN_DATASET,JSONL_DATASET,CONLL03,CONLL03_GERMAN,CONLL03_DUTCH,CONLL03_SPANISH,WNUT17,ARABIC_ANER,ARABIC_AQMAR,BASQUE,WEIBO,DANE,MOVIE_SIMPLE,MOVIE_COMPLEX,SEC_FILLINGS,RESTAURANT,STACKOVERFLOW,TWITTER
,PERSON,WEBPAGES,WNUT2020,WIKIGOLD,FINER,BIOFID,EUROPARL,LEGAL_NER,GERMEVAL,POLITICS,BUSINESS,ICELANDIC_NER,HIRONSAN,MASAKHANE,MULTI_CONER,WIKIANN,XTREME,WIKINER,SWEDISH_NER,TURKU}
````
For example the following can be used to evaluate the Bi-LSTM-CRF model based on Flair embeddings on CONLL03:

````
ner_eval_dashboard FLAIR flair/ner-english SPACE CONLL03
````

### api

````python
from ner_eval_dashboard.dataset.flair import FlairConll03
from ner_eval_dashboard.predictor import FlairPredictor
from ner_eval_dashboard.tokenizer import SpaceTokenizer
from ner_eval_dashboard.app import create_app

tokenizer = SpaceTokenizer()
dataset = FlairConll03(tokenizer)
predictor = FlairPredictor("flair/ner-english")

app = create_app("my-app", predictor, dataset)

app.run_server()
````

### docker

docker images are publicly available at [docker hub](https://hub.docker.com/repository/docker/helpmefindaname/ner-eval-dashboard/general)

````
docker run -it --rm -p 8050:8050 helpmefindaname/ner-eval-dashboard FLAIR flair/ner-english SPACE CONLL03
````
