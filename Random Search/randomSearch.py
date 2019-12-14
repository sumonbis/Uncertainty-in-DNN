#!/bin/python
# -*- coding: utf8 -*-

'''
Description: Get best hyperparameter with random search
Date: 2019-12-14
'''

import sys, random, time
sys.path.append(".")
# sys.path.append("../")
sys.path.append("../Training")

from dnn_model import input_data, create_model, train, evaluate
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV


def randomSearch(distributions, x_train, y_train, x_test, y_test, n_iter):
    # epoch = 2
    para = []
    acc = []
    for i in range(n_iter):
        hidden_unit = random.choice(distributions)
        model = create_model(hidden_unit);
        model = train(model, x_train, y_train, x_test, y_test, 2)
        loss, accuracy = evaluate(model, x_test, y_test)
        para.append(hidden_unit)
        acc.append(accuracy)
    accuracy = max(acc)
    parameter = para[acc.index(accuracy)]
    return parameter


def main():
    x_train, y_train, x_test, y_test, input_shape = input_data(30)

    # random search
    t = time.time()
    distributions = [i for i in range(1, 1001)]
    n_iter = 30
    best_hidden_unit = randomSearch(distributions, x_train, y_train, x_test, y_test, n_iter)

    print("time for {} iteration random search: {}".format(n_iter, time.time() - t))

    model = create_model(best_hidden_unit)
    model = train(model, x_train, y_train, x_test, y_test, 2)
    loss, accuracy = evaluate(model, x_test, y_test)
    result = "{},{},{}".format(best_hidden_unit, loss, accuracy)
    print("++ Result : best_hidden_unit, loss, accuracy \n \t\t" + result)
    f = open("RS.csv", "w+")
    f.write(result)
    f.close()


    # model = KerasClassifier(build_fn=create_model)
    # distributions = dict(hidden_unit=[i for i in range(1, 1001)])
    # clf = RandomizedSearchCV(model, distributions, random_state=0, n_iter=20) #, scoring="accuracy")
    # search = clf.fit(x_train, y_train)
    # print(search.best_params_)
    # epoch = 2
    # pass


if __name__ == "__main__":
    main()
