SEED = 1234
from numpy.random import seed
seed(SEED)
from tensorflow import set_random_seed
set_random_seed(SEED)

from NER import NERTagger
import mlflow


if __name__ == '__main__':
    ner = NERTagger(CORPUS, ENTITY_NAME, SEED_LIST, window=WINDOW, non_word_boundaries=NON_WORD_BOUNDARIES, n_jobs=25)
    ner.generate_predictive_rules(max_iterations=30, min_probability=0.90, min_update_rate=0.01,
                                  iteration_save_path='/home/bschulz/ner/names_luk_5',
                                  save_iterations=[1, 3, 5, 7, 10, 15, 20, 25, 30])
    if ner.iterations != 30:
        ner.save(f'/home/bschulz/ner/names_luk_5/iteration_{ner.iterations}')
