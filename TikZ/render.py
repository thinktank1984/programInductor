import tempfile
import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.animation as animation

def render(source, showImage = False, output = None, yieldsPixels = False, canvas = (10,10)):
    if canvas == None: canvas = ""
    else: canvas = '''
\draw[fill = white, white] (0,0) rectangle (%d,%d);
'''%(canvas[0],canvas[1])
    
    source = '''
\documentclass[convert={density=300,size=300x300,outext=.png}]{standalone}
\usepackage{tikz}

\\begin{document}
\\begin{tikzpicture}
%s
%s
\end{tikzpicture}
\end{document}
''' % (canvas, source)
    
    fd, temporaryName = tempfile.mkstemp(suffix = ".tex")
    with os.fdopen(fd, 'w') as new_file:
        new_file.write(source)
    os.system("cd /tmp; echo X|pdflatex -shell-escape %s > /dev/null 2> /dev/null" % temporaryName)

    temporaryPrefix = temporaryName[:-3]
    temporaryImage = temporaryPrefix + "png"

    if showImage:
        os.system("feh %s" % temporaryImage)

    returnValue = []
    if output != None:
        os.system("mv %s %s 2> /dev/null" % (temporaryImage, output))
        temporaryImage = output
        
    if yieldsPixels:
        im = Image.open(temporaryImage).convert('L')
        (width, height) = im.size
        greyscale_map = list(im.getdata())
        greyscale_map = np.array(greyscale_map)
        greyscale_map = greyscale_map.reshape((height, width))
        returnValue = greyscale_map/255.0


    os.system("rm %s*" % temporaryPrefix)
    if returnValue != []: return returnValue

def animateMatrices(matrices):
    fig = plot.figure() # make figure
    im = plot.imshow(matrices[0], cmap=plot.get_cmap('bone'), vmin=0.0, vmax=1.0)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(matrices[j])
        # return the artists set
        return im,
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(matrices)), 
                              interval=50, blit=True)
    plot.show()

if __name__ == "__main__":
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    i = sys.stdin if inputFile == '-' else open(inputFile, "r")
    source = i.read()
    render(source, outputFile)
