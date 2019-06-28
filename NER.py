from multiprocessing import Pool
import math
import re
from gensim.models import Word2Vec
import numpy as np
from gutenberg.classification.keras import mlp
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from keras.preprocessing.text import text_to_word_sequence
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras import backend as K
import os
from collections import defaultdict
import keras
from datetime import datetime, timedelta
import sys
from gutenberg.util.io_operations import pickle_object
from gutenberg.util.io_operations import load_pickle
from pathlib import Path
import gc


class NERTagger:

    def __init__(self,
                 corpus,
                 entities, # list of dicts {name: "ENTITY_NAME", seed: [seed_1, seed_2,...]}
                 window,
                 n_jobs,
                 train_min_pos_rate,
                 seed,
                 **kwargs):
        t = datetime.now()
        print(f'{(datetime.now()-t)} -- Initializing...')
        self.corpus = corpus

        # TODO: change for multiple possible labels
        self.seedlist = entities[0]['seed']
        self.entity_name = entities[0]['name']
        self.window = window
        self.n_jobs = n_jobs

        self.train_min_pos_rate = train_min_pos_rate
        self.seed = seed

        print(f'{(datetime.now()-t)} -- Tokenizing Corpus...')
        self.tokenized_corpus = [text_to_word_sequence(doc) for doc in self.corpus]

        self.encoder = FastEncoder()
        print(f'{(datetime.now()-t)} -- Fitting Encoder...')
        self.encoder.fit([token.lower() for doc in self.tokenized_corpus for token in doc]
                         + self.seedlist +
                         ['\u0002PADDING\u0002'] +
                         [f'\u0002{ent["name"]}\u0002' for ent in entities])
        self.encoded_corpus = None
        self.encode_corpus()
        self.encoded_seedlist = self.encoder.transform(self.seedlist)

        # TODO: change for multiple possible labels
        self.encoded_padding = self.encoder.transform(['\u0002PADDING\u0002'])
        self.encoded_entity = self.encoder.transform([f'\u0002{self.entity_name}\u0002'])

        if not kwargs.get('load', False):
            print(f'{(datetime.now()-t)} -- Getting Token Counts...')
            self.token_rel = self.get_token_count(relative=True)
            self.token_abs = self.get_token_count(relative=False)

            self.model = None

        self.iterations = 0

    def set_model(self, model):
        self.model = model

    def get_required_dimensions(self):
        return {
            'in_dim': 2*self.window,
            #TODO: Change or multiple possible labels
            'out_dim': 2,
            'num_labels': len(self.encoder.classes_)
        }

    def encode_corpus(self):
        p = Progressor(len(self.corpus))

        self.encoded_corpus = list()
        for i, doc in enumerate(self.tokenized_corpus):
            try:
                self.encoded_corpus.append(self.encoder.transform(doc))
            except:
                import ipdb; ipdb.set_trace()
            p.print_progress(i+1)

    def predict_token_probabilities_iterated(self, text, thres=0, mask_thres=0.5):
        token_probas = self.predict_token_probabilities(text=text, thres=thres)
        masked_tokens = ' '.join([t[0] if t[1] <= mask_thres else f'\u0002{self.entity_name}\u0002' for t in token_probas])

        while True:
            masked_token_probas = self.predict_token_probabilities(text=masked_tokens, thres=thres)
            masked_tokens_new = ' '.join([t[0] if t[1] <= mask_thres else f'\u0002{self.entity_name}\u0002' for t in masked_token_probas])
            if masked_tokens_new == masked_tokens:
                break
            masked_tokens = masked_tokens_new
        return list(zip([t[0] for t in token_probas], [t[1] for t in masked_token_probas]))

    def predict_token_probabilities(self, text, thres=0):
        # import ipdb; ipdb.set_trace()
        encoded_text = self.encoder.transform(text_to_word_sequence(text))
        ngrams, _, tokens = self.create_training_data(np.array([encoded_text]))

        # mask
        ngrams[np.isin(ngrams, self.encoded_seedlist)] = self.encoded_entity

        decoded_tokens = self.encoder.inverse_transform(tokens)
        return list(zip(decoded_tokens, [i if i > thres else 0 for num, i in enumerate(self.model.predict_proba(ngrams)[:,1])]))

    def get_token_count(self, relative=True):
        token_count = defaultdict(lambda: 0)
        overall_tokens = 0
        for doc_num, doc in enumerate(self.encoded_corpus):
            for token in doc:
                overall_tokens += 1
                token_count[token] += 1
        if relative:
            return {k: v/overall_tokens for k, v in token_count.items()}
        else:
            return dict(token_count)

    def get_mean_name_probas(self, name_probas):
        token_probas = defaultdict(lambda: list())
        p = Progressor(len(name_probas))
        for i, (token, proba) in enumerate(name_probas):
            token_probas[token].append(proba)
            p.print_progress(i+1)
        token_mean_proba = {k: sum(v)/len(v) for k, v in token_probas.items()}
        return token_mean_proba

    def do_iteration(self, t=None):
        gc.collect()

        if t is None:
            t = datetime.now()
        print(f'{(datetime.now()-t)} -- Creating training data...')
        X, y, tokens = self.create_training_data(self.encoded_corpus, progressbar=True)
        print(f'{(datetime.now()-t)} -- Masking Training Data...')
        X_masked = X.copy()
        X_masked[np.isin(X_masked, self.encoded_seedlist)] = self.encoded_entity
        print(f'{(datetime.now()-t)} -- Stratifying for equal positives and negatives...')
        X_strat, y_strat = self.duplicate_positives(X_masked, y)
        print(f'{(datetime.now()-t)} -- Training Classifier...')
        _ = self.model.fit(X_strat, y_strat)
        print()
        print(f'{(datetime.now()-t)} -- Calculating Name Probabilities...')
        name_probas = list(zip(tokens, self.model.predict_proba(X)[:, 1]))
        print()
        del X_masked, X, y, X_strat, y_strat, tokens
        gc.collect()
        return name_probas

    def generate_predictive_rules(self, max_iterations=10, min_probability=0.9, min_update_rate=0.1,
                                  iteration_save_path=None, save_iterations=None):
        iteration = 0
        update_rate = None

        t = datetime.now()
        while iteration < max_iterations and (update_rate is None or update_rate >= min_update_rate):
            iteration += 1
            print(f'\n{(datetime.now()-t)} -- ITERATION {iteration}....\n')
            name_probas = self.do_iteration(t=t)
            print(f'{(datetime.now()-t)} -- Get New Seed Probabilities...')
            self.token_mean_probas = self.get_mean_name_probas(name_probas)
            print(f'{(datetime.now()-t)} -- Updating Seed List...')
            new_whitelist_items = list()
            p = Progressor(len(self.token_mean_probas))
            for i, (token, proba) in enumerate(self.token_mean_probas.items()):
                p.print_progress(i+1)
                if (min_probability is None or proba >= min_probability) and self.token_abs[token] > 1 and token not in self.encoded_seedlist:
                    new_whitelist_items.append(token)
            update_rate = len(new_whitelist_items)/len(self.encoded_seedlist)
            self.encoded_seedlist = np.append(self.encoded_seedlist, new_whitelist_items)

            print(f'\n{(datetime.now()-t)} -- Update Rate of Seed List: {update_rate}')
            print(f'{(datetime.now()-t)} -- Size of Seed List: {len(self.encoded_seedlist)}\n')

            if iteration_save_path and save_iterations is None or iteration in save_iterations:
                if not iteration_save_path.endswith('/'):
                    iteration_save_path = iteration_save_path + '/'
                p = iteration_save_path + f'iteration_{iteration}'
                os.makedirs(p, exist_ok=True)
                self.save(p)
                print(f'{(datetime.now()-t)} -- Saved Iteration {iteration}\n')

        print(f'{(datetime.now()-t)} -- Successfully iterated {iteration} times!')
        self.iterations = iteration

    def create_training_data(self, encoded_text, progressbar=False):
        y_train = list()
        X_train = list()
        tokens = list()
        p = Progressor(len(encoded_text))
        for i, doc in enumerate(encoded_text):
            padded_doc = [self.encoded_padding]*self.window + doc.tolist() + [self.encoded_padding]*self.window
            for j, encoded_token in enumerate(doc):
                tokens.append(encoded_token)
                X_train.append(padded_doc[j:j+self.window] + padded_doc[j+self.window+1:j+2*self.window+1])
                if encoded_token in self.encoded_seedlist:
                    y_train.append(1)
                else:
                    y_train.append(0)

            if progressbar:
                p.print_progress(i+1)

        X_train = np.array(X_train)
        if X_train.ndim > 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
        y_train = np.array(y_train)
        y_train = to_categorical(y_train, num_classes=2)
        tokens = np.array(tokens)

        return X_train, y_train, tokens


    #
    # def worker_get_name_probas(self, ngrams, wv, progressbar=False):
    #     def replace_seed(listlike):
    #         return [x if x not in self.whitelist else f'\u0002{self.entity_name}\u0002' for x in listlike]
    #     vecs = list()
    #     tokens = list()
    #     p = Progressor(len(ngrams))
    #     for i, (token, neighbours) in enumerate(ngrams):
    #         try:
    #             vector = wv[neighbours].reshape(1,-1)
    #         except KeyError:
    #             vector = wv[replace_seed(neighbours)].reshape(1,-1)
    #         vecs.append(vector)
    #         tokens.append(token)
    #         if progressbar:
    #             p.print_progress(i+1)
    #     vecs = np.array(vecs)
    #     vecs = vecs.reshape((vecs.shape[0], vecs.shape[2]))
    #     return tokens, vecs
    #
    # def get_name_probabilities(self, ngrams, wv, clf):
    #     # TODO
    #     if self.n_jobs <= -1:
    #         return self.worker_get_name_probas(ngrams, wv)
    #     else:
    #         chunks = self.get_chunks(ngrams, self.n_jobs)
    #
    #         pool = Pool(processes=self.n_jobs)
    #         results = [pool.apply_async(self.worker_get_name_probas, args=(chunk, wv)) for chunk in chunks[:-1]]
    #         results.append(pool.apply_async(self.worker_get_name_probas, args=(chunks[-1], wv), kwds={'progressbar': True}))
    #         tokens = list()
    #         vecs = None
    #
    #         for r in results:
    #             t, v = r.get()
    #             if vecs is None:
    #                 vecs = v
    #             else:
    #                 vecs = np.append(vecs, v, axis=0)
    #             tokens.extend(t)
    #         pool.close()
    #         pool.join()
    #
    #         return list(zip(tokens, clf.predict_proba(vecs)[:,1]))
    #
    def duplicate_positives(self, X, y):
        n_pos = int(np.sum(y[:, 1]))
        n_all = y.shape[0]
        if n_pos/n_all < self.train_min_pos_rate:
            missing_examples = int(n_all * self.train_min_pos_rate + 0.5) - n_pos
            new_positive_idx = np.random.choice(np.argwhere(y[:, 1] == 1).reshape(-1,),
                                                missing_examples)
            X = np.append(X, X[new_positive_idx, :], axis=0)

            y = np.append(y, np.array(len(new_positive_idx)*[[0, 1]]).reshape(-1, 2), axis=0)
        return X, y

    def save(self, path):
        #TODO
        if isinstance(path, str):
            path = Path(path)

        attributes = {
            'seedlist': self.seedlist,
            'entity_name': self.entity_name,
            'corpus': self.corpus,
            'window': self.window,
            'n_jobs': self.n_jobs,
            'token_abs': self.token_abs,
            'token_rel': self.token_rel,
            'token_mean_probas': self.token_mean_probas,
            'train_min_pos_rate': self.train_min_pos_rate,
            'seed': self.seed,
        }

        os.makedirs(str(path), exist_ok=True)
        self.model.model.save(str(path / 'model.h5'))
        pickle_object(str(path / 'attributes.pkl'), obj=attributes)

    @classmethod
    def load(cls, path):
        #TODO
        if isinstance(path, str):
            path = Path(path)

        attributes = load_pickle(path / 'attributes.pkl')
        ner_instance = cls(corpus=attributes['corpus'],
                           entities=[{'name': attributes['entity_name'], 'seed': attributes['seedlist']}],
                           window=attributes['window'],
                           n_jobs=attributes['n_jobs'],
                           train_min_pos_rate=attributes['train_min_pos_rate'],
                           seed=attributes['seed'],
                           load=True
                           )
        from contextlib import suppress
        with suppress(KeyError):
            ner_instance.token_rel = attributes['token_rel']
        with suppress(KeyError):
            ner_instance.token_abs = attributes['token_abs']
        with suppress(KeyError):
            ner_instance.token_mean_probas = attributes['token_mean_probas']

        def load_build_fn():
            model = keras.models.load_model(str(path / 'model.h5'))

            if isinstance(model.optimizer, keras.optimizers.Adam):
                optimizer = 'adam'
            else:
                raise NotImplementedError(f'optimizer not implemented in load_model: {model.optimizer}')

            model.compile(loss=model.loss, optimizer=optimizer)

            return model

        keras_clf = KerasClassifier(build_fn=load_build_fn)
        keras_clf.model = load_build_fn()

        ner_instance.model = keras_clf

        return ner_instance


class FastEncoder:

    def __init__(self):
        self.labels = None
        self.dic = None
        self.reverse_dic = None
        self.classes_ = None

    def fit(self, tokens):
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens)
        self.labels = np.unique(tokens, return_inverse=True)[1]
        self.dic = dict(zip(tokens.flatten(), self.labels))
        self.reverse_dic = dict(zip(self.labels, tokens.flatten()))
        self.classes_ = [x[0] for x in sorted(self.dic.items(), key=lambda x: x[1])]

    def transform(self, tokens):
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens)
        if tokens.ndim > 1:
            return np.reshape([self.transform(sublist) for sublist in tokens], tokens.shape)
        else:
            return np.reshape([self.dic[t] for t in tokens], tokens.shape)

    def fit_transform(self, tokens):
        self.fit(tokens)
        return self.transform(tokens)

    def inverse_transform(self, trf_tokens):
        if not isinstance(trf_tokens, np.ndarray):
            trf_tokens = np.array(trf_tokens)
        if trf_tokens.ndim > 1:
            return np.reshape([self.inverse_transform(sublist) for sublist in trf_tokens], trf_tokens.shape)
        else:
            return np.reshape([self.reverse_dic[t] for t in trf_tokens], trf_tokens.shape)

class Progressor:
    def __init__(self, n):
        self.total = n
        self.t_start = datetime.now()
        self.t_last_msg = datetime.now()

    def print_progress(self, iteration, prefix='', suffix='', decimals=2, bar_length=50):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(self.total)))
        filled_length = int(round(bar_length * iteration / float(self.total)))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        msg_time = datetime.now()
        if self.t_start and iteration > 0 and iteration < self.total:
            passed_seconds = (msg_time-self.t_start).total_seconds()
            remaining_seconds = (1/iteration*self.total-1)*passed_seconds
            suffix = f'[{str(timedelta(seconds=remaining_seconds)).split(".")[0]} remaining]'
        elif self.t_start and iteration == self.total:
            suffix = f'[{str(msg_time-self.t_start).split(".")[0]} process time]'

        if (msg_time- self.t_last_msg).total_seconds() > 1/500 or iteration == self.total:
            self.t_last_msg = msg_time
            sys.stdout.write('\r%s [%s] %s%s %s' % (prefix, bar, percents, '%', suffix)),

            if iteration == self.total:
                sys.stdout.write('\n')
            sys.stdout.flush()