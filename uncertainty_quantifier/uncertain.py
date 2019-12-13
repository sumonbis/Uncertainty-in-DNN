import operator as op
import statistics as stat
import math

class Uncertain:
    def __init__(self,sampler, *args):
        self.sampleSize=1000
        self.samplingFunction = sampler
        self.args = list(args)

    def __lt__(self, other):
        return self.hypothesis_test(other, op.lt)

    def __le__(self, other):
        return self.sample() <= other.sample()

    def __gt__(self, other):
        return self.sample() > other.sample()

    def __ge__(self, other):
        return self.sample() >= other.sample()

    def __eq__(self, other):
        return self.sample() == other.sample()

    def __ne__(self, other):
        return self.sample() != other.sample()

    def sample(self):
        return self.samplingFunction(*self.args)

    def hypothesis_test(self, other, H0):
        t1 = self.sample()
        t2 = other.sample()
        print(t1)
        print(t2)
        print(H0(t1,t2))
        return H0(t1,t2)

    #returns sample mean and margin of error in 95% CI
    def E(self):
        data=[]
        for i in range(self.sampleSize):
            data.append(self.sample())

        print(data)
        std=stat.stdev(data)
        moe= (2*std)/math.sqrt(self.sampleSize)

        return [stat.mean(data), moe]
