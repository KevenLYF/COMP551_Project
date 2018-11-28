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


file1 = "./aclImdb/train/labeledBow.feat"
x_train, y_train = preprocessing(file1)

x_train = np.array(x_train)
y_train = np.array(y_train)
