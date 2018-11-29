import numpy as np


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
                r[int(ratio[0])] = int(ratio[1])

            x.append(r)


    return x, y


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




fvpair("./aclImdb/train/labeledBow.feat")
# fvpair("./aclImdb/test/labeledBow.feat")
# for i in range(len(data)):
#     print(data[i])


