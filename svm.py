from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import re
import pandas as pd
import warnings; warnings.simplefilter('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utility import preprocessing2, clean_text, load_train_test_imdb_data
import matplotlib.pyplot as plt

def gridSearch_SVM(X_train, y_train):

    trainX, trainY = preprocessing2("./aclImdb/train/labeledBow.feat")

    c_range = []
    d_range = [True, False]
    c = 0.1
    for i in range(5):
        c *= 0.6
        c_range.append(c)

    param_grid = dict(C=c_range, dual=d_range)
    svm = LinearSVC(C=c_range, dual=d_range)
    gs = GridSearchCV(svm, param_grid, scoring='accuracy', n_jobs=-1, verbose=50)
    gs.fit(X_train, y_train)
    best_score = gs.best_score_
    best_param = gs.best_params_
    print("The Accuracy = {} \nC = {}\nDual = {}".format(best_score, best_param.get('C'), best_param.get('dual')))

# gridSearch_SVM(trainX, trainY)

def SVM_ngrams(train, test, n):
    # Transform each text into a vector of word counts
    vectorizer = TfidfVectorizer(stop_words="english", preprocessor=clean_text, ngram_range=(1, n))

    training_features = vectorizer.fit_transform(train["text"])
    test_features = vectorizer.transform(test["text"])

    temp_c = 10
    best_acc = 0
    max_c = 1
    c_range = []
    acc_range = []
    for i in range(30):
        model = LinearSVC(C=temp_c)
        model.fit(training_features, train["sentiment"])
        y_pred = model.predict(test_features)
        acc = accuracy_score(test["sentiment"], y_pred)
        if (acc > best_acc):
            best_acc = acc
            max_c = temp_c
        c_range.append(temp_c)
        acc_range.append(acc)
        temp_c *= 0.7
        print("Accuracy on the IMDB dataset: {:.2f}".format(acc * 100))

    # Evaluation
    print("Best accuracy is {:.2f} \n Best C is {}".format(best_acc * 100, max_c))
    return c_range, acc_range

def funny_plot(x, y):
    plt.plot(x, y, 'ro')
    plt.title('Unigram C vs Accuracy')
    plt.xlabel('Hyper Parameter C')
    plt.ylabel('Accuracy')
    plt.show()

train_data, test_data = load_train_test_imdb_data(data_dir="aclImdb/")
c_array, acc_array = SVM_ngrams(train_data, test_data, 3)
funny_plot(c_array, acc_array)