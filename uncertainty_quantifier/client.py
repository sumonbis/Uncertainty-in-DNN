import random
from uncertainty_quantifier.uncertain import Uncertain
from Training.dnn_model import train_and_evaluate,input_data, evaluate
import time

def getRandomHiddenSizeSample():
    return random.randint(1,1000)


def main():

    file = open('result.csv', 'w', encoding='utf-8')
    file.write('hidden unit size,expected accuracy, margin of error\n')

    _hiddenUnitSampler=Uncertain(getRandomHiddenSizeSample)

    no_of_exp=2
    no_of_trials=2

    x_train, y_train, x_test, y_test, input_shape = input_data(30,100)

    start_time = time.time()

    for j in range(no_of_exp):

        result = []

        for i in range(no_of_trials):
            hiddenUnitSize=_hiddenUnitSampler.sample()
            _accuracySampler=Uncertain(train_and_evaluate, 10, 1, hiddenUnitSize)
            _accuracySampler.id=str(hiddenUnitSize)
            _accuracySampler.plotDensity=True

            e=_accuracySampler.E()
            result.append((hiddenUnitSize, e[0]-e[1]))
            file.write(str(hiddenUnitSize)+','+str(e[0])+','+str(e[1])+'\n')
            print("(Hidden unit size, expected accuracy, margin of error): ",(hiddenUnitSize, e[0], e[1]))

        result=sorted(result, key=lambda x: x[1])
        print(result)

    elapsed_time = time.time() - start_time

    print('Elapsd time: ', elapsed_time)
    file.close()


if __name__=="__main__":
    main()