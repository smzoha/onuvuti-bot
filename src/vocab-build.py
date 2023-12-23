import pickle

import pandas as pd

import utils

dataset = pd.read_csv('../data/clean_data.csv')

vocab = utils.construct_vocab(dataset, dataset.columns.values.tolist())
print('Word Count:', len(vocab.keys()))

formatted_answers = utils.get_formatted_answers(dataset['Answers'])

inv_vocab = {word_id: word for word, word_id in vocab.items()}

encoder_inp = utils.get_tokenized_texts(dataset['Questions'], vocab)
decoder_inp = utils.get_tokenized_texts(dataset['Answers'], vocab)

print('First Encoder Input:', encoder_inp[0])
print('First Decoder Input:', decoder_inp[0])

vocab_data = {'vocab': vocab, 'inv_vocab': inv_vocab, 'encoder_inp': encoder_inp, 'decoder_inp': decoder_inp,
              'formatted_answers': formatted_answers}

with open('../bot-data/vocab.pkl', 'wb') as vocab_file:
    pickle.dump(vocab_data, vocab_file)
