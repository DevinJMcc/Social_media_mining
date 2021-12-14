from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
import pickle

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
'''
Run a normal decision tree classifier. Out put the top 5 classifiers and the the features importance for all the
features being used
'''


def runDecisionTreeClassifier(x_train, y_train, x_test, y_test, p):

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    # Here we instantiate the decision tree classifier
    clf = tree.DecisionTreeClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    dt_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    dt_score = accuracy_score(y_test, dt_predictions)
    pt_score = precision_score(y_test, dt_predictions)
    rt_score = recall_score(y_test, dt_predictions)
    f1t_score = f1_score(y_test, dt_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(dt_predictions)
    for i in range(len(yt)):
        
        if  yt[i] == 0:
          
            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:
      
            l1 += [yt[i]]
            l2 += [pred[i]]
    
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)

    print("decision tree classification accuracy on test data is " + str(dt_score), file=sys.stderr)
    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score, pt_score, rt_score, f1t_score,severe_acc,mild_acc)


'''
Function to run the gradient boosting classifier on the data given. Out put the top 5 classifiers and the the features
importance for all the features being used
'''


def runGradientBoostingClassifier(x_train, y_train, x_test, y_test, p):
    #
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = GradientBoostingClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    gbc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, gbc_predictions)
    pt_score = precision_score(y_test, gbc_predictions)
    rt_score = recall_score(y_test, gbc_predictions)
    f1t_score = f1_score(y_test, gbc_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(gbc_predictions)
    for i in range(len(yt)):

        if  yt[i] == 0:

            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:

            l1 += [yt[i]]
            l2 += [pred[i]]
       
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)

    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score, pt_score, rt_score, f1t_score,severe_acc,mild_acc)


'''
Function to run the adaboost classifier on the data given. Out put the top 5 classifiers and the the features
importance for all the features being used
'''


def runAdaBoostClassifier(x_train, y_train, x_test, y_test, p):

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = AdaBoostClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    ada_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    dt_score = accuracy_score(y_test, ada_predictions)
    pt_score = precision_score(y_test, ada_predictions)
    rt_score = recall_score(y_test, ada_predictions)
    f1t_score = f1_score(y_test, ada_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(ada_predictions)
    for i in range(len(yt)):
    
        if  yt[i] == 0:
    
            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:

            l1 += [yt[i]]
            l2 += [pred[i]]
        
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)

    print("adaboost classification accuracy on test data is " + str(dt_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)
    return (train_score, dt_score, pt_score, rt_score, f1t_score,severe_acc,mild_acc)


'''
Function to run the extra trees classifier on the data given. Out put the top 5 classifiers and the the features
importance for all the features being used
'''


def runExtraTreesClassifier(x_train, y_train, x_test, y_test, p):

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = ExtraTreesClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    et_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    et_score = accuracy_score(y_test, et_predictions)
    pt_score = precision_score(y_test, et_predictions)
    rt_score = recall_score(y_test, et_predictions)
    f1t_score = f1_score(y_test, et_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(et_predictions)
    for i in range(len(yt)):
    
        if  yt[i] == 0:
    
            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:
        
            l1 += [yt[i]]
            l2 += [pred[i]]
         
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)
    print("Extra Tree classification accuracy on test data is " + str(et_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score,pt_score, rt_score, f1t_score,severe_acc,mild_acc)


def runSupportVec(x_train, y_train, x_test, y_test, p):

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = svm.SVC()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    svm_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    svm_score = accuracy_score(y_test, svm_predictions)
    pt_score = precision_score(y_test, svm_predictions)
    rt_score = recall_score(y_test, svm_predictions)
    f1t_score = f1_score(y_test, svm_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(svm_predictions)
    for i in range(len(yt)):

        if  yt[i] == 0:

            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:

            l1 += [yt[i]]
            l2 += [pred[i]]
      
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)

    print("SVM classification accuracy on test data is " + str(svm_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score,pt_score, rt_score, f1t_score,severe_acc,mild_acc)


def runXGBoost(x_train, y_train, x_test, y_test, p):

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = XGBClassifier()
    clf.set_params(**p)


    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    xgb_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    xgb_score = accuracy_score(y_test, xgb_predictions)
    pt_score = precision_score(y_test, xgb_predictions)
    rt_score = recall_score(y_test, xgb_predictions)
    f1t_score = f1_score(y_test, xgb_predictions)
    
    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(xgb_predictions)
    for i in range(len(yt)):

        if  yt[i] == 0:

            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:

            l1 += [yt[i]]
            l2 += [pred[i]]
       
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)
    print("XGB classification accuracy on test data is " + str(xgb_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score,pt_score, rt_score, f1t_score,severe_acc,mild_acc)


def random_trees(x_train, y_train, x_test, y_test, p):

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = RandomForestClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    rt_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    rt_score = accuracy_score(y_test, rt_predictions)
    print("Random Tree classification accuracy on test data is " + str(rt_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    pt_score = precision_score(y_test, etc_predictions)
    rt_score = recall_score(y_test, etc_predictions)
    f1t_score = f1_score(y_test, etc_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(etc_predictions)
    for i in range(len(yt)):
    
        if  yt[i] == 0:
    
            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:

            l1 += [yt[i]]
            l2 += [pred[i]]
       
    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score,pt_score, rt_score, f1t_score,severe_acc,mild_acc)

def nueral_net(x_train, y_train, x_test, y_test, p):
    clf = MLPClassifier()
    clf.set_params(**p)

    clf.fit(x_train, y_train)

    # now, make the predictions using our classifier
    rt_predictions = clf.predict(x_test)

    # now we have to computer the classification accuracy
    # think about what two variables we have to compare
    rt_score = accuracy_score(y_test, rt_predictions)
    print("MLP classification accuracy on test data is " + str(rt_score), file=sys.stderr)

    etc_predictions = clf.predict(x_test)
    dt_score = accuracy_score(y_test, etc_predictions)
    pt_score = precision_score(y_test, etc_predictions)
    rt_score = recall_score(y_test, etc_predictions)
    f1t_score = f1_score(y_test, etc_predictions)

    v1 = []
    v2 = []
    l1 = []
    l2 = []
    yt = list(y_test)
    pred = list(etc_predictions)
    for i in range(len(yt)):

        if  yt[i] == 0:

            v1 += [yt[i]]
            v2 += [pred[i]]
        elif yt[i] == 1:

            l1 += [yt[i]]
            l2 += [pred[i]]

    severe_acc = accuracy_score(v1, v2)
    mild_acc = accuracy_score(l1, l2)
    print("accuracy score on test data: " + str(dt_score), file=sys.stderr)
    train_score = accuracy_score(y_train, clf.predict(x_train))
    print("accuracy score on training data: " + str(train_score), file=sys.stderr)

    return (train_score, dt_score,pt_score, rt_score, f1t_score,severe_acc,mild_acc)


if __name__ == '__main__':

    # Command line arguments need to run the code
    # data = file
    # alg = classifier to run
    # fs = features to use
    data = sys.argv[1]
    alg = int(sys.argv[2])
    params = sys.argv[3]

    p_dict = {}
    with open(params, 'rb') as handle:
        p_dict = pickle.load(handle)

#    print(p_dict.keys())
#    exit()

    # load in the data
    tweet_data = pd.read_csv(data)

    cols = list(tweet_data.columns)
    if 'vax-sideeffects-FINAL-2.csv' in data:
        tweet_data = tweet_data.drop(columns=['text'])

    # one-hot encoding for categorical variables
    tweet_data_onehot = pd.get_dummies(tweet_data,
                    columns=['sentiment','V/L'],
                    dummy_na=True)

    # Start bootstrap loop
    n_iterations = 100
    n_size = int(len(tweet_data_onehot) * 0.50)

#    rows = defendent_strikes_onehot.values
    for i in range(n_iterations):

        train, test = train_test_split(tweet_data_onehot, test_size=0.5,
                                       stratify=tweet_data_onehot['V/L_L'])

        # drop outcome from features
#        feature_subset.remove('MotionResultCode_GR')


        # data to train on
        x_train = train.drop(['V/L_L', 'V/L_V', 'V/L_nan'],axis=1)
        # and the population as the outcome (what we want to predict)
        y_train = train['V/L_L']

        # this is the test data, we do not train using this data
        x_test = test.drop(['V/L_L', 'V/L_V', 'V/L_nan'],axis=1)
        y_test = test['V/L_L']

        import ast
        if alg == 0:
            print("decision tree classifier for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:

                p = ast.literal_eval(p_dict[dd]['decision tree'])
            except:
                jj = {}


                jj = p_dict[dd]['decision tree']

                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = runDecisionTreeClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['decision tree', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)
        elif alg == 1:
            print("gradient boosting classifier for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}

            try:
                p = ast.literal_eval(p_dict[dd]['gradient boosting'])

            except:
                jj = {}

                jj = p_dict[dd]['gradient boosting']

                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = runGradientBoostingClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['gradient boosting', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)
        elif alg == 2:
            print("extra trees classifier for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:
                p = ast.literal_eval(p_dict[dd]['extra trees'])

            except:
                jj = {}

                jj = p_dict[dd]['extra trees']

                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = runExtraTreesClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['extra trees', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)
        elif alg == 3:
            print("adaboost classifier for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:
                p = ast.literal_eval(p_dict[dd]['adaboost'])
            except:
                jj = {}

                jj = p_dict[dd]['adaboost']

                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random',max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = runAdaBoostClassifier(x_train, y_train, x_test, y_test,p)
            print("\t".join(['adaboost', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)
        elif alg == 4:
            print("SVM for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:
                p = ast.literal_eval(p_dict[dd]['svm'])
            except:
                jj = {}

                jj = p_dict[dd]['svm']
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = runSupportVec(x_train, y_train, x_test, y_test,p)
            print("\t".join(['svm', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)
        elif alg == 5:
            print("XGB for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:
                p = ast.literal_eval(p_dict[dd]['xgb'])
            except:
                jj = {}
                jj = p_dict[dd]['xgb']
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = runXGBoost(x_train, y_train, x_test, y_test,p)
            print("\t".join(['xgb', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)
        elif alg == 6:
            print("random forest for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:
                p = ast.literal_eval(p_dict[dd]['random forest'])
            except:
                jj = {}
                jj = p_dict[dd]['random forest']
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = random_trees(x_train, y_train, x_test, y_test,p)
            print("\t".join(['random forest', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)

        elif alg == 7:

            print("MLP for " + data, file=sys.stderr)
            dd = data.split('/')[-1]
            p = {}
            try:
                p = ast.literal_eval(p_dict[dd]['MLP'])
            except:
                jj = {}
                jj = p_dict[dd]['MLP']
                jj = jj.lstrip('{').rstrip('}')
                p = {}
                sx = jj.split(':')
                nx = []
                for item in sx:

                    item = item.lstrip().rstrip()

                    items = []
                    if '),' in item:
                        items = item.split('),')
                    else:
                        items = item.split(',', 1)

                    for i in items:
                        i = i.lstrip().rstrip()
                        i = i.lstrip('\'').rstrip('\'')
                        i = i.lstrip('\"').rstrip('\"')
                        nx += [i]

                for i in range(len(nx)):
                    if i % 2 == 0:
                        if nx[i] == 'base_estimator':
                            p['base_estimator'] = tree.DecisionTreeClassifier(splitter='random', max_depth=1)
                        elif nx[i] == 'learning_rate':
                            p[nx[i].lstrip().rstrip()] = float(nx[i + 1])
                        elif nx[i] == 'n_estimators':
                            p[nx[i].lstrip().rstrip()] = int(nx[i + 1])
                        else:
                            p[nx[i].lstrip().rstrip()] = nx[i + 1] if type(nx[i + 1]) != str else nx[
                                i + 1].lstrip().rstrip()

            (accuracy_train, accuracy_test,pt_score, rt_score, f1t_score,severe_acc,mild_acc) = nueral_net(x_train, y_train, x_test, y_test,p)
            print("\t".join(['MLP', data, str(accuracy_train), str(accuracy_test), str(pt_score), str(rt_score), str(f1t_score),str(severe_acc),str(mild_acc)]))
            print("\n", file=sys.stderr)

