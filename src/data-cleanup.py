import os.path

import pandas as pd
from bnlp import CleanText


def get_clean_data(input_data, attr):
    cleaner = CleanText(remove_url=True, remove_email=True, remove_punct=True, remove_digits=True,
                        replace_with_punct='', replace_with_number='')

    input_data[attr] = input_data[attr].str.strip()
    input_data[attr] = input_data[attr].str.replace('[a-zA-Z0-9]*', '', regex=True)

    for i in range(0, len(input_data)):
        input_data[attr][i] = cleaner(input_data[attr][i])

    return input_data[attr]


dataset = pd.read_csv('../data/BengaliEmpatheticConversationsCorpus.csv', encoding='utf-8')
dataset = dataset[['Questions', 'Answers']]

bot_dataset = pd.read_csv('../bot-data/OnuvutiBot Dataset.csv')
dataset = pd.merge(dataset, bot_dataset, on=['Questions', 'Answers'], how='outer')

print('\nDataset Statistics:')
print(dataset.describe())

print('\nChecking empty or duplicate values in dataset...')
print('Dataset Shape:', dataset.shape)
print('Duplicate Count:', dataset.duplicated().sum())
print('Empty Count:', dataset.isna().sum(), sep='\n')

print('\nDropping empty data')
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)
dataset.reset_index(drop=True, inplace=True)

print('Dataset Shape:', dataset.shape)
print('Duplicate Count:', dataset.duplicated().sum())
print('Empty Count:', dataset.isna().sum(), sep='\n')

print('\nCleaning up data [removing english characters, punctuations, numbers, etc.]')
dataset['Questions'] = get_clean_data(input_data=dataset, attr='Questions')
dataset['Answers'] = get_clean_data(input_data=dataset, attr='Answers')

print('\nFirst 10 Rows of Clean Data')
print(dataset.head(10))

print('\nDropping empty data (after cleanup)')
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)
dataset.reset_index(drop=True, inplace=True)

print('Write cleaned dataset to file...')
if not os.path.exists('../data'):
    os.mkdir('../data')

dataset.to_csv('../data/clean_data.csv', encoding='utf-8', index=False)
print('Cleaned data written to file')
