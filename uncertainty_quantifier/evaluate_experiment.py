import random
from uncertainty_quantifier.uncertain import Uncertain
from Training.dnn_model import train_and_evaluate,input_data, evaluate,create_model,train
import time
import csv

def getRandomHiddenSizeSample():
    return random.randint(1,1000)


def main():
    file_overall = open('temp/overall.csv', 'w', encoding='utf-8')
    file_overall.write('hidden unit size,accuracy\n')

    no_of_exp=10

    x_train, y_train, x_test, y_test, input_shape = input_data(30,100)

    start_time = time.time()

    for j in range(no_of_exp):

        result = []

        with open('temp/experiment_'+str(j+1)+'_detail.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader, None)  # skip the headers
            for row in csv_reader:
                 result.append((int(row[0]), float(row[1])-float(row[2])))


        result=sorted(result, key=lambda x: x[1])
        print(result)

        model = create_model(result[len(result)-1][0])
        model = train(model, x_train, y_train, x_test, y_test, 1)
        accuracy = evaluate(model, x_test, y_test)
        file_overall.write(str(result[len(result)-1][0])+','+str(accuracy)+'\n')

    elapsed_time = time.time() - start_time

    print('Elapsd time: ', elapsed_time)

    file_overall.close()


if __name__=="__main__":
    main()