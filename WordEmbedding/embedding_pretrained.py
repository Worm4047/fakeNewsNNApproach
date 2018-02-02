'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
'''

from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
#For MPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
#for RNN
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, './data/glove.6B')
MAX_SEQUENCE_LENGTH = 500
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

df_all = pd.read_csv('./data/dataset/Complete_Dataset_Clean.csv', sep="{")
X_body_text = df_all.body.values[:-10]
y = df_all.fakeness.values[:-10]

texts = X_body_text
labels = y

# print(texts)
print('Found %s texts.' % len(texts))
# print(labels)
# print(labels_index)
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]


###
###Simple multi perceptron model
###

# print("At neural net")
# # using the basic neural net
# #Neural Networks
# clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1)

# # print(X_body_train_tfidf)
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_val)

# accu_score = accuracy_score(y_val, y_pred)
# preci_score = precision_score(y_val, y_pred, average="macro")
# rec_score = recall_score(y_val, y_pred, average="macro")  
# # print("Predicted value: ",y_pred)   
# print("Metrics Are Scores : \n")
# print("Accuracy ", accu_score)
# print("Precision ", preci_score)
# print("Recall ", rec_score)

###
###RNN with LSTM
###

# build the model: a single LSTM
# learning_rate = 0.001 #learning rate

# print('Build LSTM model.')
# model = Sequential()
# model.add(LSTM(128, input_shape=(MAX_SEQUENCE_LENGTH,)))
# model.add(Dense(MAX_SEQUENCE_LENGTH))
# model.add(Activation('softmax'))

# #adam optimizer
# optimizer = Adam(lr=learning_rate)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# #fit the model
# model.fit(x_train, y_train,
#           batch_size=128,
#           epochs=5,
#           validation_data=(x_val, y_val))


###
###CNN with embedding layer
###

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

model = Sequential()
model.add(embedding_layer)
# model.add(Dropout(0.2))
model.add(Conv1D(128,5,activation='relu'))
model.add(MaxPooling1D(5))
# model.add(Dropout(0.2))
model.add(Conv1D(128,5,activation='relu'))
model.add(MaxPooling1D(5))
# model.add(Dropout(0.2))
model.add(Conv1D(128,5,activation='relu'))
model.add(GlobalMaxPooling1D())
# model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

fashion_train = model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))

#Plots
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()