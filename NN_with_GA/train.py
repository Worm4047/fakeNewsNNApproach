"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

ngram_range = (1, 1)
penal = 'l1'
# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)


def get_dataset():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 2
    batch_size = 128
    input_shape = (20,)

    df_all = pd.read_csv('dataset/Complete_Dataset_Clean.csv', sep="{")
    X_body_text = df_all.body.values[:-10]
    y = df_all.fakeness.values[:-10]
    print(len(y))
    
    # print("For the parameters of\nngram_range=",ngram_range,"penalty as=",penal)
    tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,
                        ngram_range=ngram_range,max_features= 20)

    X_body_tfidf = tfidf.fit_transform(X_body_text)
    x_train, x_test, y_train, y_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)
    x_train = x_train.todense()
    x_test = x_test.todense()

    x_train = x_train.reshape(14346,20)
    x_test = x_test.reshape(3587,20)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_dataset()

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
