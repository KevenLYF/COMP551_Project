from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import warnings; warnings.simplefilter('ignore')
from utility import preprocessing

trainX, trainY = preprocessing("./aclImdb/train/labeledBow.feat")

c_range = []
d_range = [True, False]
c = 1
for i in range(30):
    c *= 0.8
    c_range.append(c)

def gridSearch_SVM(X_train, y_train):

    param_grid = dict(C=c_range, dual=d_range)
    svm = LinearSVC(C=c_range, dual=d_range)
    gs = GridSearchCV(svm, param_grid, scoring='f1_micro', n_jobs=-1, verbose=50)
    gs.fit(X_train, y_train)
    best_score = gs.best_score_
    best_param = gs.best_params_
    print("The F1 measure = {} \nC = {}\nDual = {}".format(best_score, best_param.get('C'), best_param.get('dual')))

gridSearch_SVM(trainX, trainY)
