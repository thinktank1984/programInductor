import tempfile
import sys
import os


def render(source, output = None, canvas = (10,10)):
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

    if output != None:
        os.system("mv %s %s 2> /dev/null" % (temporaryImage, output))
    else:
        os.system("feh %s" % temporaryImage)

    os.system("rm %s*" % temporaryPrefix)

if __name__ == "__main__":
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    i = sys.stdin if inputFile == '-' else open(inputFile, "r")
    source = i.read()
    render(source, outputFile)
