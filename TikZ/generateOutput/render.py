import tempfile
import sys
import os

inputFile = sys.argv[1]
#outputFile = sys.argv[2]

i = sys.stdin if inputFile == '-' else open(inputFile, "r")
source = '''
\documentclass[convert={density=300,size=300x300,outext=.png}]{standalone}
\usepackage{tikz}

\\begin{document}
\\begin{tikzpicture}
\draw[fill = white, white] (0,0) rectangle (10,10);
%s
\end{tikzpicture}
\end{document}
''' % i.read()
print source

fd, temporaryName = tempfile.mkstemp(suffix = ".tex")
with os.fdopen(fd, 'w') as new_file:
    new_file.write(source)
#os.close(fd)
print temporaryName
os.system("cd /tmp; pdflatex -shell-escape %s" % temporaryName)

temporaryPrefix = temporaryName[:-3]
temporaryImage = temporaryPrefix + "png"

if len(sys.argv) > 2:
    os.system("mv %s %s" % (temporaryImage, sys.argv[2]))
else:
    os.system("feh %s" % temporaryImage)

os.system("rm %s*" % temporaryPrefix)
