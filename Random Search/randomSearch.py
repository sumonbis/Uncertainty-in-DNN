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


def randomSearch(distributions, x_train, y_train, x_test, y_test, no_of_trials, index_exp):
    # epoch = 1
    file = open("result/rs_experiment_" + str(index_exp+1)+"_detail.csv", "w", encoding="utf-8")
    file.write("hidden unit size, accuracy\n")

    para = []
    acc = []
    for i in range(no_of_trials):
        hidden_unit = random.choice(distributions)
        print(" + random search {} iteration : hidden_unit = {}".format(i, hidden_unit))
        model = create_model(hidden_unit);
        model = train(model, x_train, y_train, x_test, y_test, 1)
        accuracy = evaluate(model, x_test, y_test)
        print("   accuracy = {}".format(accuracy))
        file.write(str(hidden_unit)+','+str(accuracy)+"\n")
        para.append(hidden_unit)
        acc.append(accuracy)
    file.close()
    accuracy = max(acc)
    parameter = para[acc.index(accuracy)]
    return parameter


def main():
    file_rs = open("result/randomSearch.csv", "w+", encoding="utf-8")
    file_rs.write("hidden unit size, accuracy\n")

    no_of_exp = 3
    no_of_trials = 10

    # x_train, y_train, x_test, y_test, input_shape = input_data(30)
    x_train, y_train, x_test, y_test, input_shape = input_data(30,100)
    distributions = [i for i in range(1, 1001)]

    # random search
    start_time = time.time()

    for i in range(no_of_exp):
        print("{} experiment start ... ".format(i+1))
        best_hidden_unit = randomSearch(distributions, x_train, y_train, x_test, y_test, no_of_trials, i)
        model = create_model(best_hidden_unit)
        model = train(model, x_train, y_train, x_test, y_test, 1)
        accuracy = evaluate(model, x_test, y_test)
        file_rs.write(str(best_hidden_unit) + "," + str(accuracy) + "\n")
        print("{} experiment, best_hidden_unit: {}, accuracy: {}\n".format(i+1, best_hidden_unit, accuracy))

    elapsed_time = time.time() - start_time

    print("time for {} experiment(s) random search: {} (second)".format(no_of_exp, elapsed_time))
    file_rs.close()


    # model = KerasClassifier(build_fn=create_model)
    # distributions = dict(hidden_unit=[i for i in range(1, 1001)])
    # clf = RandomizedSearchCV(model, distributions, random_state=0, n_iter=20) #, scoring="accuracy")
    # search = clf.fit(x_train, y_train)
    # print(search.best_params_)
    # epoch = 2
    # pass


if __name__ == "__main__":
    main()
