import random

from uncertainty_quantifier.uncertain import Uncertain


def getSamplingFunction(low,high):
    return random.randint(low,high)

def testLt():
    sampler1=getSamplingFunction
    _uncertain1=Uncertain(sampler1, 40,40)

    sampler2 = getSamplingFunction #
    _uncertain2 = Uncertain(sampler2, 30,50)

    assert _uncertain1<_uncertain2, "First uncertain not less than second uncertain"

def testE():
    sampler1 = getSamplingFunction
    _uncertain1 = Uncertain(sampler1)

    print(_uncertain1.E())

def testFunctPass(f, *args):
    a=list(args)
    print(f(*a))
    print(a)

if __name__=="__main__":
    testLt()
    # testE()

    # testFunctPass(getSamplingFunction, 30,30)
