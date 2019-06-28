import NER_gridsearch_vars as vars
from numpy.random import seed
seed(vars.SEED)
from tensorflow import set_random_seed
set_random_seed(vars.SEED)

# import sys
# sys.path.append('/home/bschulz/repos/datascience_app_tm_gutenberg/gutenberg/')
# from textpreparation.NER import NERTagger, NERIterator

from src.NER import NERTagger
from src.util.metrics import get_precision_recall
from src.util.logging import NerFormatter
import traceback
import logging

import mlflow
import json
import os
import shutil
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from gutenberg.classification.keras import mlp


def setup( GPU_ID=None, clear_session=False):
    """Classification setup

    :param config: configuration
    :param clear_session: flag whether Tensorflow session is cleaned
    """
    if clear_session:
        K.clear_session()

    if GPU_ID is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
        runtime_classifier_config = tf.ConfigProto(allow_soft_placement=True)
        runtime_classifier_config.gpu_options.allow_growth = True
        sess = tf.Session(config=runtime_classifier_config)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        runtime_classifier_config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=runtime_classifier_config)

    K.set_session(sess)


def load_sorted_taggers(path):
    tagger_paths = [x for x in os.listdir(path) if x not in ['iterated_stats.json', 'scores.png']]
    tagger_paths = sorted(tagger_paths, key=lambda x: int(x.split('_')[1]))
    setup(GPU_ID='2')
    ners = list()

    for tp in tagger_paths:
        ners.append(NERTagger.load(os.path.join(path, tp)))
    return ners


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = NerFormatter()
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    init_config = {
        'window': 3,
        'n_jobs': 20,
        'train_min_pos_rate': 0.5
    }

    generate_config = {
        'max_iterations': 20,
        'min_probability': 0.7,
        'min_update_rate': 0.02
    }


    mlflow.set_tracking_uri("http://deepsense-mlflow.openshift-prd.ov.otto.de")

    mlflow.set_experiment("NER_tagging_names_luk")

    setup(GPU_ID='0', clear_session=True)

    ner = NERTagger(vars.CORPUS,
                    entities=[{'name': vars.ENTITY_NAME, 'seed': vars.SEED_LIST}],
                    seed=vars.SEED,
                    **init_config
                    )

    model_dims = ner.get_required_dimensions()

    EMBEDDING_SIZE = 100

    mlp_model = Sequential()
    mlp_model.add(Embedding(model_dims['num_labels'], EMBEDDING_SIZE, input_length=model_dims['in_dim']))
    mlp_model.add(Flatten())
    mlp_model.add(Dense(500, activation='relu'))
    mlp_model.add(Dropout(0.5))
    mlp_model.add(Dense(model_dims['out_dim'], activation='softmax'))

    MODEL_PARAMS = {
        "epochs": 2,
        "batch_size": 512,
        "loss": "categorical_crossentropy",
        "metrics": ["accuracy"],
        "optimizer": keras.optimizers.Adam(amsgrad=False,
                                           beta_1=0.9,
                                           beta_2=0.999,
                                           decay=0.00,
                                           epsilon=1e-8,
                                           lr=0.001),
    }

    model = KerasClassifier(build_fn=mlp.compile_model, model=mlp_model, **MODEL_PARAMS)

    ner.set_model(model)

    with mlflow.start_run():
        shutil.rmtree('/home/bschulz/ner/gridsearch', ignore_errors=True)
        os.makedirs('/home/bschulz/ner/gridsearch', exist_ok=True)
        ner.generate_predictive_rules(iteration_save_path='/home/bschulz/ner/gridsearch',
                                      save_iterations=list(range(generate_config['max_iterations']+1)),
                                      **generate_config)

        ners = load_sorted_taggers('/home/bschulz/ner/gridsearch')

        stats = [get_precision_recall(ner, vars.LABELLED, 150, vars.SEED_DOCS) for ner in ners]

        mlflow.log_param('train_min_pos_rate', ner.train_min_pos_rate)
        mlflow.log_param('iterations', ner.iterations)
        mlflow.log_param('max_iterations', generate_config['max_iterations'])
        mlflow.log_param('min_probability', generate_config['min_probability'])
        mlflow.log_param('min_update_rate', generate_config['min_update_rate'])

        final_stat = stats[-1]
        mlflow.log_metric('precision', final_stat[0])
        mlflow.log_metric('recall', final_stat[1])
        mlflow.log_metric('f1_score', final_stat[2])
        mlflow.log_metric('support', final_stat[3])

        stats = [
            {'precision': s[0],
             'recall': s[1],
             'f1': s[2],
             'support': s[3]
             }
            for s in stats]

        with open('/home/bschulz/ner/gridsearch/iterated_stats.json', 'w') as f:
            f.write(json.dumps(stats))
        try:
            mlflow.log_artifact('/home/bschulz/ner/gridsearch/iterated_stats.json')
        except:
            pass

        try:
            stat_frame = pd.DataFrame(stats)
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.lineplot(ax=ax, data=stat_frame.drop(['support'], axis=1), size=20)
            fig.savefig('/home/bschulz/ner/gridsearch/scores.png')
            mlflow.log_artifact('/home/bschulz/ner/gridsearch/scores.png')
        except Exception as e:
            print('Exception in plot...')
            traceback.print_tb(e.__traceback__)