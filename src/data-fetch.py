import os

import kaggle

DATASET_NAME = 'raseluddin/bengali-empathetic-conversations-corpus'
DATA_DIRECTORY = '../data'

DATA_FILE_NEW_NAME = '../data/BengaliEmpatheticConversationsCorpus.csv'
DATA_FILE_ORIG_NAME = '../data/BengaliEmpatheticConversationsCorpus .csv'

if not os.path.exists('../data'):
    os.mkdir('../data')

kaggle.api.authenticate()
kaggle.api.dataset_download_files(DATASET_NAME, DATA_DIRECTORY, unzip=True,
                                  quiet=False)

os.rename(DATA_FILE_ORIG_NAME, DATA_FILE_NEW_NAME)
