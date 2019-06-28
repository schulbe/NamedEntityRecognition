import NER_gridsearch_vars as vars
from numpy.random import seed
seed(vars.SEED)
from tensorflow import set_random_seed
set_random_seed(vars.SEED)

# import sys
# sys.path.append('/home/bschulz/repos/datascience_app_tm_gutenberg/gutenberg/')
# from textpreparation.NER import NERTagger, NERIterator

from NER import NERTagger
import traceback
import logging
from datetime import datetime
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
from keras.layers import Dense, Conv1D, Flatten, Embedding, Dropout
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


def get_precision_recall(ner, labels, max_num, seed_docs):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    false_pos_list = list()

    for ix in [i for i in range(max_num) if i not in seed_docs]:
        probas = ner.predict_token_probabilities_iterated(ner.corpus[ix].strip(), thres=0.499, mask_thres=0.5)
        for token, prob in probas:
            if ix in labels:
                if prob > 0 and token in labels[ix]:
                    true_pos += 1
                elif prob > 0 and token not in labels[ix]:
                    false_pos += 1
                    false_pos_list.append((ix, token))
                elif prob == 0 and token in labels[ix]:
                    false_neg += 1
                else:
                    true_neg += 1
            else:
                if prob > 0:
                    false_pos += 1
                    false_pos_list.append((ix, token))
                else:
                    true_neg += 1

    try:
        precision = true_pos/(true_pos+false_pos)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = true_pos/(true_pos+false_neg)
    except ZeroDivisionError:
        recall = 0

    try:
        f_score = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        f_score=0

    support = true_pos+false_neg
    return precision, recall, f_score, support
    # return (precision, recall, f_score, support), false_pos_list


def get_precision_recall_spacy(nlp, labels, max_num, corpus, tokenize_fn, seed_docs):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for ix in [i for i in range(max_num) if i not in seed_docs]:
        doc = nlp(corpus[ix])
        names = list()
        for ent in doc.ents:
            if ent.label_ == 'PER':
                names.extend(tokenize_fn(ent.text))

        for token in tokenize_fn(corpus[ix]):
            if ix in labels:
                if token in names and token.lower() in labels[ix]:
                    true_pos += 1
                elif token in names and token.lower() not in labels[ix]:
                    false_pos += 1
                elif token not in names and token.lower() in labels[ix]:
                    false_neg += 1
                else:
                    true_neg += 1
            else:
                if token in names:
                    false_pos += 1
                else:
                    true_neg += 1

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f_score = 2*precision*recall/(precision+recall)
    support = true_pos+false_neg
    return precision, recall, f_score, support


class NerFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_string = "{passed_time} -- {message}"
        self.start_time = datetime.now()

    def format(self, record):
        record = record.__dict__

        vars = {
            'message': record.get('message', None),
            'passed_time': datetime.fromtimestamp(record.get('created')-self.start_time).strftime('%H:%M:%S.%f'),
        }
        return self.base_string.format(**vars)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

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

    EMBEDDING_SIZE = 250

    mlp_model = Sequential()
    mlp_model.add(Embedding(model_dims['num_labels'], EMBEDDING_SIZE, input_length=model_dims['in_dim']))
    mlp_model.add(Flatten())
    mlp_model.add(Dense(1000, activation='relu'))
    mlp_model.add(Dropout(0.5))
    mlp_model.add(Dense(model_dims['out_dim'], activation='softmax'))

    MODEL_PARAMS = {
        "epochs": 2,
        "batch_size": 256,
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