import pickle

import numpy as np
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

with open('../data/vocab.pkl', 'rb') as vocab_file:
    vocab_data = pickle.load(vocab_file)

vocab = vocab_data['vocab']
inv_vocab = vocab_data['inv_vocab']
full_encoder_inp = vocab_data['encoder_inp']
full_decoder_inp = vocab_data['decoder_inp']

train_val_split = int(len(full_encoder_inp) * 0.8)
encoder_inp = full_encoder_inp[:train_val_split]
decoder_inp = full_decoder_inp[:train_val_split]

val_encoder_inp = full_encoder_inp[train_val_split + 1:]
val_decoder_inp = full_decoder_inp[train_val_split + 1:]

encoder_seq = pad_sequences(encoder_inp, 32, padding='post', truncating='post')
decoder_seq = pad_sequences(decoder_inp, 32, padding='post', truncating='post')

decoder_target_data = []
for token_seq in decoder_seq:
    decoder_target_data.append(token_seq[1:])

decoder_seq_shifted = pad_sequences(decoder_target_data, 32, padding='post')
one_hot_decoder_seq = to_categorical(decoder_seq_shifted, len(vocab))
decoder_target_data = np.array(one_hot_decoder_seq)

# Training model
enc_inp_layer = Input(shape=(32,))
dec_inp_layer = Input(shape=(32,))

embed = Embedding(input_dim=len(vocab) + 1, output_dim=64, input_length=32, trainable=True)

enc_embed = embed(enc_inp_layer)
enc_lstm = LSTM(512, return_sequences=True, return_state=True)
enc_seq_out, enc_mem_state, enc_carry_state = enc_lstm(enc_embed)
enc_states = [enc_mem_state, enc_carry_state]

dec_embed = embed(dec_inp_layer)
dec_lstm = LSTM(512, return_sequences=True, return_state=True)
dec_seq_out, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(len(vocab), activation='softmax')
dense_out = dense(dec_seq_out)

model = Model([enc_inp_layer, dec_inp_layer], dense_out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([encoder_seq, decoder_seq], decoder_target_data, epochs=40)

# Inference Engine
enc_model = Model([enc_inp_layer], enc_states)

decoder_mem_state_input = Input(shape=(512,))
decoder_carry_state_input = Input(shape=(512,))

decoder_state_inputs = [decoder_mem_state_input, decoder_carry_state_input]
dec_out, dec_mem_state, dec_carry_state = dec_lstm(dec_embed, initial_state=decoder_state_inputs)
dec_model = Model([dec_inp_layer] + decoder_state_inputs, [dec_out] + [dec_mem_state, dec_carry_state])

user_input = ''

while user_input != 'bye':
    user_input = input('You: ')
    user_input = user_input.lower()
    user_input = re.sub('[^a-zA-Z\\s]', '', user_input)
    user_inputs = [user_input]

    txt = []
    for user_text in user_inputs:
        tokens = []

        for token in user_text.split():
            if token in vocab:
                tokens.append(vocab[token])
            else:
                tokens.append(vocab['<UNK>'])

        txt.append(tokens)

    txt = pad_sequences(txt, 32, padding='post')
    enc_out = enc_model.predict(txt)

    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = vocab['<SOS>']

    stop_condition = False
    result = ''

    while not stop_condition:
        output, mem_state, cache_state = dec_model.predict([empty_target_seq] + enc_out)
        decoder_inp_concat = dense(output)

        sampled_word_index = np.argmax(decoder_inp_concat[0, -1, :])
        sampled_word = inv_vocab[sampled_word_index] + ' '

        if sampled_word != '<EOS>':
            result += sampled_word

        if sampled_word == '<EOS>' or len(result.split()) > 32:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        enc_out = [mem_state, cache_state]

    print('Bot: ', result)
