from random import random,choice

MAXIMUMCOORDINATE = 10

def randomCoordinate():
    return int(random()*(MAXIMUMCOORDINATE - 1)) + 1

def randomPoint():
    return (randomCoordinate(), randomCoordinate())

def inbounds(p):
    if p is tuple: return inbounds(p[0]) and inbounds(p[1])
    return p >= 1 and p <= MAXIMUMCOORDINATE - 1

class Line():
    def __init__(self, points): self.points = points
    def __str__(self):
        return "\\draw [ultra thick] %s;" % " -- ".join([str(p) for p in self.points ])
    def mutate(self):
        if random() > 0.5:
            return Line([randomPoint(),self.points[1]])
        else:
            return Line([self.points[0],randomPoint()])
    @staticmethod
    def sample():
        p1 = randomPoint()
        if random() > 0.5: # horizontal line
            while True:
                x2 = randomCoordinate()
                if x2 != p1[0]:
                    p2 = (x2, p1[1])
                    return Line([p1, p2])
        else:
            while True:
                y2 = randomCoordinate()
                if y2 != p1[1]:
                    p2 = (p1[0],y2)
                    return Line([p1, p2])
    

class Rectangle():
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def __str__(self):
        return "\\draw [ultra thick] %s rectangle %s;" % (self.p1, self.p2)
    def mutate(self):
        if random() > 0.5:
            return Rectangle(randomPoint(),self.p2)
        else:
            return Rectangle(self.p1,randomPoint())
    @staticmethod
    def sample():
        while True:
            p1 = randomPoint()
            p2 = randomPoint()
            if p1[0] != p2[0] and p1[1] != p2[1]:
                return Rectangle(p1, p2)

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
    def __str__(self):
        return "\\draw [ultra thick] %s circle (%s);" % (self.center, self.radius)
    def mutate(self):
        return self
    @staticmethod
    def sample():
        while True:
            p = randomPoint()
            r = 2
            if inbounds(p[0] + r) and inbounds(p[0] - r) and inbounds(p[1] + r) and inbounds(p[1] - r):
                return Circle(p,r)

class Sequence():
    def __init__(self, lines): self.lines = lines
    def __str__(self):
        return "\n".join(map(str, self.lines))
    @staticmethod
    def sample():
        return Sequence([ Sequence.samplePart() for _ in range(choice([1,2,3])) ])
    @staticmethod
    def samplePart():
        x = random()
        if x < 0.5: return Line.sample()
#        if x < 0.66: return Rectangle.sample()
        return Circle.sample()

    def mutate(self):
        r = random()
        if r < 0.3 or self.lines == []:
            return Sequence(self.lines + [Sequence.samplePart()])
        elif r < 0.6:
            r = choice(self.lines)
            return Sequence([ l for l in self.lines if l != r ])
        else:
            r = choice(self.lines)
            return Sequence([ (l if l != r else l.mutate()) for l in self.lines ])
