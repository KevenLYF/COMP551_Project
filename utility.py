import numpy as np
import os
import re
import pandas as pd


def preprocessing(features):
    x = []
    y = []

    with open(features, 'r') as f:

        for row in f:
            y.append(row[0])
            feature = row[1:]
            r = [0] * 10000

            for ratio in feature.split(' ')[1:]:
                ratio = ratio.split(':')

                if int(ratio[0]) >= 10000:
                    break

            x.append(r)

    return x, y


def preprocessing2(features):
    x = []
    y = []

    with open(features, 'r') as f:

        for row in f:
            rate = -1 if int(row[0]) <= 4 else 1
            y.append(rate)
            feature = row[1:]
            r = [0] * 10000

            for ratio in feature.split(' ')[1:]:
                ratio = ratio.split(':')

                if int(ratio[0]) >= 10000:
                    break
                # ***********************************************
                r[int(ratio[0])] = 0 if int(ratio[1]) == 0 else 1

            x.append(r)
   
    return x, y

def readValue(file):
    with open(file, 'r') as f:
        values = f.read().splitlines()

    return values

def fvpair(file):
    lcount = 0
    with open(file, 'r') as f:
        data = []
        for row in f:
            newRow = []
            lcount += 1
            feature = row[1:]
            rating = int(row[0])
            if rating <= 4:
                rating = -1
            else:
                rating = 1
            newRow.append(rating)
            for ratio in feature.split(' ')[1:]:
                pair = []
                ratio = ratio.split(':')
                word = int(ratio[0])
                word += 1
                pair.append(word)
                pair.append(ratio[1])
                newRow.append(pair)
            data.append(newRow)

    with open('train.txt', 'w') as f:
        for i in range(len(data)):
            f.write(str(data[i][0]))
            for j in range(1, len(data[i])):
                f.write(" " + str(data[i][j][0]) + ":" + str(data[i][j][1]))

def fvpair_value(file, value):
    lcount = 0
    with open(file, 'r') as f:
        data = []
        for row in f:
            newRow = []
            lcount += 1
            feature = row[1:]
            rating = int(row[0])
            if rating <= 4:
                rating = -1
            else:
                rating = 1
            newRow.append(rating)
            for ratio in feature.split(' ')[1:]:
                pair = []
                ratio = ratio.split(':')
                word = int(ratio[0])
                word += 1
                pair.append(word)
                pair.append(value[int(ratio[0])])
                newRow.append(pair)
            data.append(newRow)

    with open('test.dat', 'w') as f:
        for i in range(len(data)):
            f.write(str(data[i][0]))
            for j in range(1, len(data[i])):
                f.write(" " + str(data[i][j][0]) + ":" + str(data[i][j][1]))
            f.write("\n")

def fvpair_binary(file):
    lcount = 0
    with open(file, 'r') as f:
        data = []
        for row in f:
            newRow = []
            lcount += 1
            feature = row[1:]
            rating = int(row[0])
            if rating <= 4:
                rating = -1
            else:
                rating = 1
            newRow.append(rating)
            for ratio in feature.split(' ')[1:]:
                pair = []
                ratio = ratio.split(':')
                word = int(ratio[0])
                word += 1
                pair.append(word)
                pair.append(1)
                newRow.append(pair)
            data.append(newRow)

    with open('test.dat', 'w') as f:
        for i in range(len(data)):
            f.write(str(data[i][0]))
            for j in range(1, len(data[i])):
                f.write(" " + str(data[i][j][0]) + ":" + str(data[i][j][1]))
            f.write("\n")

"""
Applies some pre-processing on the given text.

Steps :
- Removing HTML tags
- Removing punctuation
- Lowering text
"""
def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

"""Loads the IMDB train/test datasets from a folder path.
Input:
data_dir: path to the "aclImdb" folder.

Returns:
train/test datasets as pandas dataframes.
"""
def load_train_test_imdb_data(data_dir):

    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])

    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])

    return data["train"], data["test"]


# preprocessing2("./aclImdb/train/labeledBow.feat")[1]
# fvpair("./aclImdb/train/labeledBow.feat")
# for i in range(len(data)):
#     print(data[i])

# fvpair_binary("./aclImdb/test/labeledBow.feat")

