

class Uncertain:
    def __init__(self, sampler):
        self.samplingFunction=sampler

    def __lt__(self, other):
        t1=self.samplingFunction()
        t2=other.samplingFunction()
        print(t1)
        print(t2)
        return t1<t2

    def __le__(self, other):
        return self.samplingFunction() <= other.samplingFunction()

    def __gt__(self, other):
        return self.samplingFunction() > other.samplingFunction()

    def __ge__(self, other):
        return self.samplingFunction() >= other.samplingFunction()

    def __eq__(self, other):
        return self.samplingFunction() == other.samplingFunction()

    def __ne__(self, other):
        return self.samplingFunction() != other.samplingFunction()