import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, models, preprocessing, utils
import pandas as pd
from tensorflow.python.keras.models import load_model


def read_data(filename):
    data = list()
    myfile = open(filename)
    for line in myfile:
        line = line.strip()
        tokens = line.split('- ')
        if len(tokens) == 2:
            data.append(tokens[1])
        elif len(tokens) == 3:
            data.append(tokens[2])
        else:
            data.append(line)
    return data


if __name__ == "__main__":
    questions = read_data('questions.txt')
    answers = read_data('responses.txt')

    print(questions)
    print(answers)

    question_lines = list()
    for line in questions:
        question_lines.append(line)

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(question_lines)
    tokenized_question_lines = tokenizer.texts_to_sequences(question_lines)

    length_list = list()
    for token_seq in tokenized_question_lines:
        length_list.append(len(token_seq))
    max_input_length = np.array(length_list).max()
    print('English max length is {}'.format(max_input_length))

    padded_question_lines = preprocessing.sequence.pad_sequences(
        tokenized_question_lines, maxlen=max_input_length, padding='post')
    encoder_input_data = np.array(padded_question_lines)
    print('Encoder input data shape -> {}'.format(encoder_input_data.shape))

    question_word_dict = tokenizer.word_index
    num_question_tokens = len(question_word_dict)+1
    print('Number of English tokens = {}'.format(num_question_tokens))

    answer_lines = list()
    for line in answers:
        answer_lines.append('<START> ' + line + ' <END>')

    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(answer_lines)
    tokenized_answer_lines = tokenizer.texts_to_sequences(answer_lines)

    length_list = list()
    for token_seq in tokenized_answer_lines:
        length_list.append(len(token_seq))
    max_output_length = np.array(length_list).max()
    print('answer max length is {}'.format(max_output_length))

    padded_answer_lines = preprocessing.sequence.pad_sequences(
        tokenized_answer_lines, maxlen=max_output_length, padding='post')
    decoder_input_data = np.array(padded_answer_lines)
    print('Decoder input data shape -> {}'.format(decoder_input_data.shape))

    answer_word_dict = tokenizer.word_index
    num_answer_tokens = len(answer_word_dict)+1
    print('Number of answer tokens = {}'.format(num_answer_tokens))

    decoder_target_data = list()
    for token_seq in tokenized_answer_lines:
        decoder_target_data.append(token_seq[1:])

    padded_answer_lines = preprocessing.sequence.pad_sequences(
        decoder_target_data, maxlen=max_output_length, padding='post')
    onehot_answer_lines = utils.to_categorical(
        padded_answer_lines, num_answer_tokens)
    decoder_target_data = np.array(onehot_answer_lines)
    print('Decoder target data shape -> {}'.format(decoder_target_data.shape))

    import tensorflow as tf

    encoder_inputs = tf.keras.layers.Input(shape=(None, ))
    encoder_embedding = tf.keras.layers.Embedding(
        num_question_tokens, 200, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
        200, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(
        num_answer_tokens, 200, mask_zero=True)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(
        200, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(
        num_answer_tokens, activation=tf.keras.activations.softmax)
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='categorical_crossentropy')

    model.summary()

    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data, batch_size=250, epochs=300)
    model.save('model.h5')


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


def str_to_tokens(sentence: str):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(question_word_dict[word])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=max_input_length, padding='post')


enc_model, dec_model = make_inference_models()

for epoch in range(encoder_input_data.shape[0]):
    states_values = enc_model.predict(
        str_to_tokens(input('Enter question : ')))
    
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = answer_word_dict['start']
    stop_condition = False
    decoded_translation = ''
    while not stop_condition:
        dec_outputs, h, c = dec_model.predict(
            [empty_target_seq] + states_values)
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        for word, index in answer_word_dict.items():
            if sampled_word_index == index:
                decoded_translation += ' {}'.format(word)
                sampled_word = word
z
        if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
            stop_condition = True

        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]

    print(decoded_translation)
