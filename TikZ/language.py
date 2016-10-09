from random import random

MAXIMUMCOORDINATE = 10

def randomCoordinate():
    return int(random()*(MAXIMUMCOORDINATE+1))

def randomPoint():
    return (randomCoordinate(), randomCoordinate())

def inbounds(p):
    if p is tuple: return inbounds(p[0]) and inbounds(p[1])
    return p >= 0 and p <= MAXIMUMCOORDINATE

class Line():
    def __init__(self, points): self.points = points
    def __str__(self):
        return "\\draw %s;" % " -- ".join([str(p) for p in self.points ])
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
        return "\\draw %s rectangle %s;" % (self.p1, self.p2)
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
        return "\\draw %s circle (%s);" % (self.center, self.radius)
    @staticmethod
    def sample():
        while True:
            p = randomPoint()
            r = 2
            if inbounds(p[0] + r) and inbounds(p[0] - r) and inbounds(p[1] + r) and inbounds(p[1] - r):
                return Circle(p,r)

