import os.path
import pickle

from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('../bot-data/vocab.pkl', 'rb') as vocab_file:
    vocab_data = pickle.load(vocab_file)

vocab = vocab_data['vocab']
inv_vocab = vocab_data['inv_vocab']

encoder_inp = vocab_data['encoder_inp']
decoder_inp = vocab_data['decoder_inp']

encoder_seq = pad_sequences(encoder_inp, 32, padding='post', truncating='post')
decoder_seq = pad_sequences(decoder_inp, 32, padding='post', truncating='post')

decoder_target_data = []
for token_seq in decoder_seq:
    decoder_target_data.append(token_seq[1:])

decoder_target_data = pad_sequences(decoder_target_data, 32, padding='post')

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

dense = Dense(len(vocab) + 1, activation='softmax')
dense_out = dense(dec_seq_out)

model = Model([enc_inp_layer, dec_inp_layer], dense_out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit([encoder_seq, decoder_seq], decoder_target_data, epochs=20, validation_split=0.2)

if not os.path.exists('../model'):
    os.mkdir('../model')

model.save('../model/seq2seq.keras')
