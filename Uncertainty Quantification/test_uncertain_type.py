import random

from uncertain import Uncertain


def getSamplingFunction():
    return random.randint(1,100)

def testLt():
    sampler1=getSamplingFunction
    _uncertain1=Uncertain(sampler1)

    sampler2 = getSamplingFunction
    _uncertain2 = Uncertain(sampler2)

    assert _uncertain1<_uncertain2, "First uncertain not less than second uncertain"



if __name__=="__main__":
    testLt()