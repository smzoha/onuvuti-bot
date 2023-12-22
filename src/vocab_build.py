import pickle

import pandas as pd

marker_tokens = {
    'pad_token': '<PAD>',
    'start_token': '<SOS>',
    'end_token': '<EOS>',
    'unk_token': '<UNK>'
}

vocab_freq_thres = 5


def add_to_vocab(data, vocab, counter, word_counter):
    for text in data:
        for word in str(text).split():
            if word not in vocab and word_counter[word] > 5:
                vocab[word] = counter
                counter += 1

    return counter


def run_word_counter(word_counter, data):
    for text in data:
        for word in str(text).split():
            if word not in word_counter:
                word_counter[word] = 1
            else:
                word_counter[word] = word_counter[word] + 1


def create_tokenized_seq(data, vocab):
    tokenized_seq = []

    for text in data:
        tokenized_seq.append([vocab[word] if word in vocab else vocab[marker_tokens['unk_token']]
                              for word in str(text).split()])

    return tokenized_seq


dataset = pd.read_csv('../data/clean_data.csv')

word_counter = {}
run_word_counter(word_counter, dataset['Questions'])
run_word_counter(word_counter, dataset['Answers'])

vocab = {marker_tokens['pad_token']: 0}
counter = 1
counter = add_to_vocab(dataset['Questions'], vocab, counter, word_counter)
counter = add_to_vocab(dataset['Answers'], vocab, counter, word_counter)

for token in marker_tokens.values():
    if token not in vocab:
        vocab[token] = counter
        counter += 1

print('Word Count:', len(vocab.keys()))

formatted_answers = [marker_tokens['start_token'] + ' ' + str(answer) + ' ' + marker_tokens['end_token']
                     for answer in dataset['Answers']]

inv_vocab = {word_id: word for word, word_id in vocab.items()}

encoder_inp = create_tokenized_seq(dataset['Questions'], vocab)
decoder_inp = create_tokenized_seq(dataset['Answers'], vocab)

print('First Encoder Input:', encoder_inp[0])
print('First Decoder Input:', decoder_inp[0])

vocab_data = {'vocab': vocab, 'inv_vocab': inv_vocab, 'encoder_inp': encoder_inp, 'decoder_inp': decoder_inp,
              'formatted_answers': formatted_answers}

with open('../data/vocab.pkl', 'wb') as vocab_file:
    pickle.dump(vocab_data, vocab_file)
