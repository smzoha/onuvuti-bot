import pandas as pd

marker_tokens = {
    'pad_token': '<PAD>',
    'start_token': '<SOS>',
    'end_token': '<EOS>',
    'unk_token': '<UNK>'
}


def add_to_vocab(data, vocab, counter):
    for text in data:
        for word in str(text).split():
            if word not in vocab:
                vocab[word] = counter
                counter += 1

    return counter


def create_tokenized_seq(data, vocab):
    tokenized_seq = []

    for text in data:
        tokenized_seq.append([vocab[word] if word in vocab else vocab[marker_tokens['unk_token']]
                              for word in str(text).split()])

    return tokenized_seq


dataset = pd.read_csv('../data/clean_data.csv')

vocab = {marker_tokens['pad_token']: 0}
counter = 1

counter = add_to_vocab(dataset['Questions'], vocab, counter)
counter = add_to_vocab(dataset['Answers'], vocab, counter)

for token in marker_tokens.values():
    if token not in vocab:
        vocab[token] = counter
        counter += 1

print('Word Count:', len(vocab.keys()))

formatted_answers = [marker_tokens['start_token'] + ' ' + str(answer) + ' ' + marker_tokens['end_token']
                     for answer in dataset['Answers']]

inv_vocab = {word_id: word for word, word_id in vocab.items()}
print(inv_vocab)

encoder_inp = create_tokenized_seq(dataset['Questions'], vocab)
decoder_inp = create_tokenized_seq(dataset['Answers'], vocab)

print('First Encoder Input:', encoder_inp[0])
print('First Decoder Input:', decoder_inp[0])
