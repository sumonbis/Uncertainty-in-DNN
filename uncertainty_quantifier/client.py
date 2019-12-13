import random
from uncertainty_quantifier.uncertain import Uncertain
from Training.dnn_model import train_and_evaluate

def getRandomHiddenSizeSample():
    return random.randint(1,1000)


def main():
    _hiddenUnitSampler=Uncertain(getRandomHiddenSizeSample)

    no_of_trials=30

    for i in range(no_of_trials):
        hiddenUnitSize=_hiddenUnitSampler.sample()



if __name__=="__main__":
    main()