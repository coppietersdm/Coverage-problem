import matplotlib.pyplot as plt
from numpy import *
import numpy as np

class Circle:
    def __init__(self, x, y, r, N):
        self.C = array([x,y])
        self.r = r
        self.points = []
        self.N = N
        self.edges = []
        self.force = array([0.0,0.0])
        
    
    def plot(self, color):
        theta = linspace(0,2*pi,1000)
        plt.plot(self.C[0], self.C[1], 'ko')
        plt.plot(self.C[0]+self.r*cos(theta), self.C[1] + self.r*sin(theta), color)
        plt.text(self.C[0], self.C[1], str(self.N))
    
    def __str__(self):
        return f"Cercle de centre ({self.x}, {self.y}) et de rayon {self.r}"
class Point:
    def __init__(self, P, N, C1, C2):
        self.P = P
        self.N = N
        self.C1 = C1
        self.C2 = C2
        self.order = 0
            
    def plot(self):
        plt.plot(self.P[0], self.P[1], 'ro')
        plt.text(self.P[0], self.P[1], str(self.order))
        
class Edge:
    def __init__(self, p, q, C, order):
        self.p = p
        self.q = q
        self.C = C
        self.angles = [arctan2(p.P[1] - C.C[1], p.P[0] - C.C[0]), arctan2(q.P[1] - C.C[1], q.P[0] - C.C[0])]
        self.order = order
    
    def plot(self):
        colordict = {0: 'b', 1: 'y', 2: 'r', 3: 'k'}
        plt.plot([self.p.P[0], self.q.P[0]], [self.p.P[1], self.q.P[1]], colordict[self.order])



class Graph:
    def __init__(self, color):
        self.points = []
        self.circles = []
        self.faces = []
        self.edges = []
        self.color = color

    def add_circle(self, x, y, r):
        NC = Circle(x, y, r, len(self.circles))
        for C in self.circles:
            V = NC.C - C.C
            A = arctan2(V[1], V[0])
            d = linalg.norm(V)
            if(d < C.r + NC.r):
                x = (d*d + C.r*C.r - NC.r*NC.r) / (2*d)
                y = sqrt(C.r*C.r - x*x)
                c, s = cos(A), sin(A)
                R = array(((c, -s), (s, c)))
                self.add_point(C.C + R@array((x, y)), NC, C)
                self.add_point(C.C + R@array((x, -y)), C, NC)
                C.points.append(self.points[-1])
                NC.points.append(self.points[-1])
                C.points.append(self.points[-2])
                NC.points.append(self.points[-2])
                
                
        self.circles.append(NC)
    
    def add_point(self, P, C1, C2):
        self.points.append(Point(P,len(self.points), C1, C2))
        
        
    def draw(self):
        for C in self.circles:
            plt.arrow(C.C[0], C.C[1], C.force[0], C.force[1], width = 0.01)
            C.plot(self.color)
            #for E in C.edges:
                #E.plot()
        for P in self.points:
            P.plot()
        plt.axis('equal')
        
    def compute_edges_and_faces(self):
        for C in self.circles:
            C.points.sort(key=lambda x: arctan2(x.P[1] - C.C[1],x.P[0] - C.C[0]))
            N = []
            for P in C.points:
                N.append((P.N, P.C1 == C))
            print(N)
            for i in range(len(C.points)):
                P = C.points[i]
                if(P.C1 == C):
                    P.order += 1
                    for j in range(i+1,i+len(C.points)):
                        Q = C.points[j%len(C.points)]
                        Q.order += 1
                        if(Q.C1 == P.C2):
                            C.edges.append(Edge(P, Q, C,0))
                            C.force -= array([-(P.P[1]-Q.P[1]), P.P[0]-Q.P[0]])
                            break
                        elif(Q.C2 == C):
                            C.edges.append(Edge(P, Q, C, 1))
                            C.force += array([-(P.P[1]-Q.P[1]), P.P[0]-Q.P[0]])

            
            
                        
            

epsilon = 0.001
graph = Graph('green')
graph.add_circle(-0.7,0,1)
graph.add_circle(1,0.2,1)
graph.add_circle(1,-0.6,1)
graph.add_circle(0.5,0.9,1)
graph.add_circle(0.2,0.2,0.5)
for i in range(1000):
    newgraph = Graph('black')
    for C in graph.circles:
        newgraph.add_circle(C.C[0]+epsilon*C.force[0], C.C[1] + epsilon*C.force[1], C.r)
    newgraph.compute_edges_and_faces()
    graph = newgraph
    if(i%1000 == -1):
        graph.draw()
    for C in graph.circles:
        plt.plot(C.C[0], C.C[1], 'ko', markersize = 1)
graph.draw()
plt.show()










