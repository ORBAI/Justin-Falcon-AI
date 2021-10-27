import glob

import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers , activations , models , preprocessing

from tensorflow.keras import preprocessing , utils
import os
import yaml
from tensorflow.keras.models import load_model
from preparsing import Preprocessing

prepobj = Preprocessing()
paired_events = prepobj._load_case_documents()
print(len(paired_events))

questions = [i[0] for i in paired_events]
answers = [i[1] for i in paired_events]

answers_with_tags = list()
for i in range( len( answers ) ):
    if type( answers[i] ) == str:
        answers_with_tags.append( answers[i] )
    else:
        print(answers[i], "pop")
        questions.pop( i )

answers = list()
for i in range( len( answers_with_tags ) ) :
    answers.append( '<start> ' + answers_with_tags[i] + ' <end>' )

tokenizer = preprocessing.text.Tokenizer()
lvocab = list()
for event in questions + answers:
    for word in event.split():
        if word not in lvocab:
            lvocab.append(word)

tokenizer.word_index = dict()
for index, w in enumerate(lvocab):
    tokenizer.word_index[w] = index
print("HIII",tokenizer.word_index)
VOCAB_SIZE = len( tokenizer.word_index )+1
'''tokenizer.fit_on_texts( questions + answers )
print("hi",tokenizer.word_index)
VOCAB_SIZE = len( tokenizer.word_index )+1'''
print(VOCAB_SIZE)
print( 'VOCAB SIZE : {}'.format( VOCAB_SIZE ))

from gensim.models import Word2Vec
import re

vocab = []
for word in tokenizer.word_index:
    vocab.append( word )

# encoder_input_data
#tokenized_questions = tokenizer.texts_to_sequences( questions )
tokenized_questions = []
for event in questions:
    sub = []
    for word in event.split():
        sub.append(tokenizer.word_index[word])
    tokenized_questions.append(sub)
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions , maxlen=maxlen_questions , padding='post' )
encoder_input_data = np.array( padded_questions )
print( encoder_input_data.shape , maxlen_questions )

# decoder_input_data
#tokenized_answers = tokenizer.texts_to_sequences( answers )
tokenized_answers = []
for event in answers:
    sub = []
    for word in event.split():
        sub.append(tokenizer.word_index[word])
    tokenized_answers.append(sub)
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
print( decoder_input_data.shape , maxlen_answers )

# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
tokenized_answers = []
for event in answers:
    sub = []
    for word in event.split():
        sub.append(tokenizer.word_index[word])
    tokenized_answers.append(sub)
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
print( decoder_output_data.shape )


encoder_inputs = tf.keras.layers.Input(shape=( maxlen_questions , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( maxlen_answers ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax )
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()

#print(encoder_input_data , decoder_input_data, decoder_output_data)
model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150 )
model.save( 'model.h5' )
model = load_model('model.h5')

def make_inference_models():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] )
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


enc_model, dec_model = make_inference_models()

for _ in range(10):
    states_values = enc_model.predict(str_to_tokens(input('Enter Event : ')))
    print(states_values)
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index['<start>']
    print(empty_target_seq)
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format(word)
                sampled_word = word

        if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)







