from random import random
from features import featureMap, FEATURESET

class Rule():
    def __init__(self, focus = [], structuralChange = [], leftTriggers = [], rightTriggers = []):
        '''
        Each matrix is represented as a list of (Boolean,string)
        The first component is the polarity, the second component is feature.
        '''
        
        self.focus = focus
        self.structuralChange = structuralChange
        self.leftTriggers = leftTriggers
        self.rightTriggers = rightTriggers

        # Remove unimportant guard
        while len(leftTriggers) > 0 and len(leftTriggers[0]) == 0:
            leftTriggers = leftTriggers[1:]
        self.relevantLeftTriggers = leftTriggers
        while len(rightTriggers) > 0 and len(rightTriggers[-1]) == 0:
            rightTriggers = rightTriggers[:-1]
        self.relevantRightTriggers = rightTriggers
    
    def applyRule(self, input):
        output = []

        for i in range(len(input)):
            if self.matchesContext(input, i):
                output.append(self.change(input[i]))
            else:
                output.append(input[i])
        return output

    def matchesContext(self, input, i):
        if i < len(self.relevantLeftTriggers): return False
        if i >= len(input) - len(self.relevantRightTriggers): return False
        if not matchesTemplate(input[i], self.focus): return False

        numberLeft = len(self.relevantLeftTriggers)
        for i_,l_ in zip(input[(i-numberLeft):i], self.relevantLeftTriggers):
            if not matchesTemplate(i_,l_): return False

        numberRight = len(self.relevantRightTriggers)
        for i_,r_ in zip(input[(i+1):(i+1+numberRight)], self.relevantRightTriggers):
            if not matchesTemplate(i_,r_): return False
        
        return True
    
    def change(self, input):
        newPhoneme = [ f for f in input ]
        for polarity,f in self.structuralChange:
            if polarity: newPhoneme.append(f)
            elif f in newPhoneme: newPhoneme.remove(f)
        return newPhoneme

    def descriptionLength(self):
        return len(self.focus) + len(self.structuralChange) + len([f for l in self.leftTriggers for f in l ]) + len([f for l in self.rightTriggers for f in l ])

    def mutate(self):
        return Rule(mutateMatrix(self.focus),
                    mutateMatrix(self.structuralChange),
                    map(mutateMatrix,self.leftTriggers),
                    map(mutateMatrix,self.rightTriggers))

    def __str__(self):
        return "%s ---> %s / %s _ %s" % (stringOfMatrix(self.focus),
                                      stringOfMatrix(self.structuralChange),
                                      " ".join(map(stringOfMatrix,self.leftTriggers)),
                                      " ".join(map(stringOfMatrix,self.rightTriggers)))

def mutateMatrix(m):
    # On average we add 1/4
    expectedChanges = 0.25
    additionProbability = expectedChanges / len(FEATURESET)
    removalProbability = expectedChanges / (len(m) + 0.001)

    return set([f for f in m if random() > removalProbability ] +
               [(random() > 0.5, f) for f in FEATURESET if random() < additionProbability ])

def matchesTemplate(candidate, template):
    for polarity,f in template:
        if polarity and (not (f in candidate)): return False
        if not polarity and (f in candidate): return False
    return True


def stringOfMatrix(m):
    return "[ %s ]" % " ".join([ ('+' if polarity else '-')+f for polarity,f in m ])
