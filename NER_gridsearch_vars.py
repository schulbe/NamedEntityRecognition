from flashtext.keyword import KeywordProcessor
import xlrd

KEYWORD_PROCESSOR = KeywordProcessor()
NON_WORD_BOUNDARIES = list(KEYWORD_PROCESSOR.non_word_boundaries | {'Ä', 'Ö', 'Ü', 'ä', 'ö', 'ü', 'ß', '\u0002'})

SEED = 1234

ENTITY_NAME = 'NAME'
SEED_LIST = ['scholz',
             'tim',
             'kühnel',
             'martina',
             'rudolf',
             'vladimir',
             'rosemarie',
             'hiltrud',
             'schumacher',
             'wieczorek',
             'kisunas',
             'kim',
             'laktin',
             'angeli',
             'molzberger',
             'schütt',
             'wenke',
             'ursan',
             'stammberger',
             'stranz']


workbook = xlrd.open_workbook("/data/lob_und_kritik/data_raw/2018_10_UXM_LobKritik_fuerBI_v1.xlsx", "rb")
sheets = workbook.sheet_names()
sh = workbook.sheet_by_index(0)
CORPUS = [cell.value.lower() for cell in sh.col_slice(colx=7) if isinstance(cell.value, str)]

LABELLED = {
    1: ['ingrid', 'siems'],
    7: ['lemniddem'],
    8: ['martina', 'wieczorek'],
    9: ['karin', 'böhmer'],
    11: ['k', 'kempe'],
    13: ['klaus', 'hasselmann'],
    19: ['a', 'brandes'],
    23: ['suska', 'meyer', 'landrut'],
    25: ['holger', 'wiegandt'],
    26: ['scholz'],
    29: ['h', 'frast'],
    32: ['vladimir', 'kisunas'],
    33: ['doraci', 'marques', 'thurau'],
    37: ['hans', 'ulla', 'mueller'],
    39: ['stefan', 'wimmer'],
    41: ['h', 'kranefeld'],
    43: ['birgit', 'salzwedel'],
    46: ['daniel', 'elke', 'batschke'],
    49: ['sandra', 'witte'],
    50: ['christa', 'ralph', 'sauermann'],
    51: ['j', 'jakob'],
    53: ['susanne', 'drost'],
    54: ['tim', 'kühnel'],
    55: ['h', 'p', 'schmitz'],
    56: ['kreisel', 'cornelia'],
    61: ['johann', 'horwath'],
    62: ['katja', 'hagemann'],
    64: ['karin', 'kwasniak'],
    70: ['kim', 'schumacher'],
    71: ['m', 'kasling'],
    75: ['karin', 'hartmann'],
    78: ['g', 'riedel'],
    86: ['zimmermann', 'siebus', 'tietz', 'silvia', 'werner'],
    89: ['stephan', 'köppel'],
    91: ['h', 'ivers'],
    94: ['heidi', 'speier'],
    96: ['rosemarie', 'von', 'der', 'ehe'],
    97: ['hiltrud', 'rudolf'],
    100: ['heike', 'harwig'],
    102: ['kerstin', 'fürst'],
    104: ['daniela', 'haag'],
    105: ['steffen', 'fischer'],
    107: ['n', 'wambach'],
    108: ['monika', 'fornes', 'ulrike', 'ringler'],
    114: ['karin', 'bernadi'],
    115: ['lagutin'],
    117: ['katharina', 'huda'],
    120: ['regina', 'gogolin'],
    121: ['nancy', 'pankow'],
    125: ['n', 'polmann'],
    128: ['steffen', 'lewandowski'],
    130: ['elisa', 'buhse'],
    132: ['tanja', 'decker'],
    134: ['jacobs'],
    135: ['henning', 'heiss', 'norbert', 'kirsch', 'lissewski'],
    136: ['sabine', 'kinzler'],
    137: ['wolfram', 'bittdorf'],
    139: ['sabine', 'kreisz']
}

SEED_DOCS = [8, 26, 32, 54, 70, 96, 97]


init_configs = [
    {
        'window': 3,
        'n_jobs': 20,
        'train_min_pos_rate': 0.5
    }
]

generate_configs = [
    {
        'max_iterations': 20,
        'min_probability': 0.7,
        'min_update_rate': 0.02
    }
]