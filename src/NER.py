import dill as dill
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import os
from collections import defaultdict
import tensorflow.keras as keras
from datetime import datetime, timedelta
import sys
from pathlib import Path
import gc
import logging


class NERTagger:
    def __init__(self,
                 corpus,
                 entities, # list of dicts {name: "ENTITY_NAME", seed: [seed_1, seed_2,...]}
                 window,
                 n_jobs,
                 train_min_pos_rate,
                 seed,
                 **kwargs):
        """
        Main Class for approaching the iterative tagging process. Since this is mainly a proof of concept,
        it is not really ready to be used for anything else but "quick and dirty" experimentation on different datat sets.
        It was written some years back using Tensorflow 1 and has been updated to match a TF2 version since then.
        Howerver it remains a proof of concept.

        The Tagger is supposed to use a small list of seeds on a large corpus to find entities that appear in similar
        contexts and might therefore also be of the same entity.

        :param corpus: List of documents to be processed
        :param entities: entity name and a list of seeds, should accept multiple in the future
        :param window: The window around each word to be concidered
        :param n_jobs: used for multiprocessing
        :param train_min_pos_rate: In the iterative training process, how many of all occurences of a word have to be positive
                in order for the seedlist to be updated
        :param seed: just for reproduceability
        :param kwargs: used for compatibility at some point
        """
        t = datetime.now()
        logging.info('Initializing...')
        self.corpus = corpus

        # TODO: change for multiple possible labels
        self.seedlist = entities[0]['seed']
        self.entity_name = entities[0]['name']
        self.window = window
        self.n_jobs = n_jobs

        self.train_min_pos_rate = train_min_pos_rate
        self.seed = seed

        logging.info('Tokenizing Corpus...')
        self.tokenized_corpus = [text_to_word_sequence(doc) for doc in self.corpus]

        logging.info('Fitting Encoder...')
        self.encoder = FastEncoder()
        self.encoder.fit([token.lower() for doc in self.tokenized_corpus for token in doc]
                         + self.seedlist +
                         ['\u0002PADDING\u0002'] +
                         [f'\u0002{ent["name"]}\u0002' for ent in entities] +
                         ['\u0002UNKNOWN\u0002'])
        self.encoded_corpus = None
        self.encode_corpus()
        self.encoded_seedlist = self.encoder.transform(self.seedlist)

        # TODO: change for multiple possible labels
        self.encoded_padding = self.encoder.transform(['\u0002PADDING\u0002'])[0]
        self.encoded_entity = self.encoder.transform([f'\u0002{self.entity_name}\u0002'])[0]

        if not kwargs.get('load', False):
            logging.info('Getting Token Counts...')
            self.token_rel = self.get_token_count(relative=True)
            self.token_abs = self.get_token_count(relative=False)

            self.model = None

        self.iterations = 0

    def set_model(self, model):
        """
        Set the keras model to be reinitialized each iteration
        :param model: KerasClassifier
        :return:
        """
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
            self.encoded_corpus.append(self.encoder.transform(doc))
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
        """
        Predicts the probability of  a token to belong to the given entity.
        :param text: Text to be evaluated
        :param thres: Can be set to some value to avoid very small values like 1E-7 (could also be rounded later)
        :return: List of tuples conatining tokens and their probabilities
        """
        word_sequence = text_to_word_sequence(text)
        encoded_text = self.encoder.transform(word_sequence)
        ngrams, _, tokens = self.create_training_data(np.array([encoded_text]))

        # mask
        ngrams[np.isin(ngrams, self.encoded_seedlist)] = self.encoded_entity

        decoded_tokens = [t if 'UNKNOWN' not in t else word_sequence[i]
                          for i, t in enumerate(self.encoder.inverse_transform(tokens))]

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

    def do_iteration(self):
        gc.collect()

        logging.info('Creating training data...')
        X, y, tokens = self.create_training_data(self.encoded_corpus, progressbar=True)

        logging.info('Masking Training Data...')
        X_masked = X.copy()
        X_masked[np.isin(X_masked, self.encoded_seedlist)] = self.encoded_entity

        # logging.info('Stratifying for equal positives and negatives...')
        # X_strat, y_strat = self.duplicate_positives(X_masked, y)
        #
        # logging.info('Training Classifier...')
        # _ = self.model.fit(X_strat, y_strat)

        logging.info('Training Classifier...')
        _ = self.model.fit(X_masked, y, class_weight={0: 1., 1: sum(y[:, 0])/sum(y[:, 1])})

        logging.info('Calculating Name Probabilities...')
        name_probas = list(zip(tokens, self.model.predict_proba(X)[:, 1]))

        del X_masked, X, y, tokens
        # del X_masked, X, y, X_strat, y_strat, tokens
        gc.collect()
        return name_probas

    def generate_predictive_rules(self, max_iterations=10, min_probability=0.9, min_update_rate=0.1,
                                  iteration_save_path=None, save_iterations=None):
        """
        Generates the contextual Rules for entities to be discovered.

        :param max_iterations: maximum number of times the seedlist gets updated and training is redone
        :param min_probability: minimum probability for a token to be concidered of the given entity
        :param min_update_rate: minimum rate of updates to be carried out before training is concidered complete
        :param iteration_save_path: where to save iterations (for later analysis)
        :param save_iterations: whether or not to save iterations
        :return:
        """
        iteration = 0
        update_rate = None

        while iteration < max_iterations and (update_rate is None or update_rate >= min_update_rate):
            iteration += 1
            logging.info(f'ITERATION {iteration}....')
            name_probas = self.do_iteration()

            logging.info('Get New Seed Probabilities...')
            self.token_mean_probas = self.get_mean_name_probas(name_probas)

            logging.info('Updating Seed List...')
            new_whitelist_items = list()
            p = Progressor(len(self.token_mean_probas))
            for i, (token, proba) in enumerate(self.token_mean_probas.items()):
                p.print_progress(i+1)
                if (min_probability is None or proba >= min_probability) and self.token_abs[token] > 1 and token not in self.encoded_seedlist:
                    new_whitelist_items.append(token)
            update_rate = len(new_whitelist_items)/len(self.encoded_seedlist)
            self.encoded_seedlist = np.append(self.encoded_seedlist, new_whitelist_items)

            logging.info(f'Update Rate of Seed List: {update_rate}')
            logging.info(f'Size of Seed List: {len(self.encoded_seedlist)}\n')

            if iteration_save_path and save_iterations is None or iteration in save_iterations:
                if not iteration_save_path.endswith('/'):
                    iteration_save_path = iteration_save_path + '/'
                p = iteration_save_path + f'iteration_{iteration}'
                os.makedirs(p, exist_ok=True)
                self.save(p)
                logging.info(f'Saved Iteration {iteration}')

        logging.info(f'Successfully iterated {iteration} times!')
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
        # self.model.save(str(path / 'model.h5'))

        with open(os.path.join(path, 'attributes.pkl'), 'wb') as f:
            dill.dump(attributes, f)

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
            return np.reshape([self.dic.get(t, self.dic['\u0002UNKNOWN\u0002']) for t in tokens], tokens.shape)

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