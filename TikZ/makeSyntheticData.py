import os
from language import *
from random import random

def renderSamples(outputDirectory, k, numberOfSamples = 1000):
    for s in range(numberOfSamples):
        source = str(k.sample())
        with open("/tmp/sampledSource.tex", "w") as f:
            f.write(source)
        os.system("python generateOutput/render.py /tmp/sampledSource.tex %s/%d.png" % (outputDirectory,s))

renderSamples("generateOutput/lines", Line)
renderSamples("generateOutput/rectangles", Rectangle)
renderSamples("generateOutput/circles", Circle)
