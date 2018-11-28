file1 =

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

            x.extend(r)


