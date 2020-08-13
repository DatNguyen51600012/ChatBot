
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text)
    # delete stopwors from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


if __name__ == "__main__":

    ### TRAINING ###
    df = pd.read_csv('train.csv')
    df = df[pd.notnull(df['tags'])]

    df['post'] = df['post'].apply(clean_text)

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = 10000
    # Max number of words in each complaint.
    MAX_SEQUENCE_LENGTH = 250
    # This is fixed.
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df['post'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

    X = tokenizer.texts_to_sequences(df['post'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    
    num_tokens = len(set(df['tags']))
    Y = pd.get_dummies(df['tags']).values
    

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_tokens, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 50
    batch_size = 64

    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[
                        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

   
    post = ""
    tags = list(set(df['tags']))
    while post != "exit":
        print('\nInput a sentence to check (\'exit\' to exit): ')
        sentence = input()
        if (post != "exit"):
            tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                                  filters='!"#$%&()*+,./:;<=>?@[\\]^_`{|}~\t\n')
            tokenizer.fit_on_texts([sentence])
            sequences = tokenizer.texts_to_sequences([sentence])
            padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
            pred = model.predict(padded)
            print(tags[np.argmax(pred)])
        else:
            print("Good bye")
            break
    
