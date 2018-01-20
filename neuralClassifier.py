import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


df_all = pd.read_csv('dataset/Complete_Dataset_Clean.csv', sep="{")
X_body_url = df_all.site_url.values
X_body_text = df_all.body.values[:-10]
X_headline_text = df_all.headline.values
y = df_all.fakeness.values[:-10]
# print(len(y))
ngram_range = (1, 1)
penal = 'l1'
f1_sc_lst = []
acc_lst = []

# print("For the parameters of\nngram_range=",ngram_range,"penalty as=",penal)

tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,
                        ngram_range=ngram_range,max_features= 20)

X_body_tfidf = tfidf.fit_transform(X_body_text)

X_body_train_tfidf, X_body_test_tfidf, y_body_train, y_body_test = train_test_split(X_body_tfidf,y, test_size = 0.2, random_state=1234)
# clf = LogisticRegression()

print(X_body_train_tfidf.shape)
print(X_body_test_tfidf.shape)
print(y_body_train.shape)

#Neural Networks
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

X_body_train_tfidf= [list(line.nonzero()[1]) for line in X_body_train_tfidf]
print("shape")
print(len(X_body_train_tfidf))
# print(X_body_train_tfidf)
clf.fit(X_body_train_tfidf, y_body_train)
y_pred = clf.predict(X_body_test_tfidf)

# print(y_pred)


accu_score = accuracy_score(y_body_test, y_pred)
preci_score = precision_score(y_body_test, y_pred, average="macro")
rec_score = recall_score(y_body_test, y_pred, average="macro")  
print("Predicted value: ",y_pred)   
print("Metrics Are Scores : \n")
print("Accuracy ", accu_score)
print("Precision ", preci_score)
print("Recall ", rec_score)