from numpy import *
import matplotlib.pyplot as plt

class Point:
    N = 0
    def __init__(self,P,C1,C2):
        self.P = P
        self.C1 = C1
        self.C2 = C2
        self.N = Point.N
        Point.N += 1
    
    def plot(self):
        plt.plot(self.P[0], self.P[1], 'go', markersize = 5)
        plt.text(self.P[0]+0.05, self.P[1]+0.05, str(self.N))

class Edge:
    def __init__(self,P1,P2,C):
        self.P1 = P1
        self.P2 = P2
        self.C = C
        self.exterior = True
    
    def plot(self):
        theta1 = arctan2(self.P1.P[1] - self.C.C[1], self.P1.P[0] - self.C.C[0])
        theta2 = arctan2(self.P2.P[1] - self.C.C[1], self.P2.P[0] - self.C.C[0])
        if(theta1 > theta2):
            theta2 += 2*pi
        theta = linspace(theta1, theta2, 1000)
        plt.plot(self.C.C[0] + self.C.r*cos(theta), self.C.C[1]+ self.C.r*sin(theta), 'b', linewidth=2)
        #plt.plot([self.P1.P[0], self.P2.P[0]], [self.P1.P[1], self.P2.P[1]], "b", linewidth=2)

class Circle:
    N = 0
    def __init__(self, C, r):
        self.C = C
        self.r = r
        self.N = Circle.N
        Circle.N += 1
        self.points = []
        self.edges = []
        self.force = array([0.0,0.0])
        
    def plot(self):
        plt.plot(self.C[0], self.C[1], 'ko', markersize = 1)
        plt.text(self.C[0], self.C[1], str(self.N))
        theta = linspace(0,2*pi,1000)
        plt.plot(self.C[0] + self.r*cos(theta), self.C[1] + self.r*sin(theta), 'k')
        for point in self.points:
            point.plot()
        for edge in self.edges:
            if(edge.exterior):
                edge.plot()
        plt.arrow(self.C[0], self.C[1], self.force[0], self.force[1], width = 0.01)

    def intersect(self, C):
        d = linalg.norm(C.C - self.C)
        D = (C.C - self.C)/d
        Q = array([-D[1], D[0]])
        if(d < self.r+C.r and d > abs(self.r-C.r)):
            x = (self.r*self.r + d*d - C.r*C.r)/(2*d)
            h = sqrt(self.r*self.r - x*x)
            P1 = Point(self.C + x*D + h*Q, self, C)
            P2 = Point(self.C + x*D - h*Q, C, self)
            self.points.append(P1); self.points.append(P2)
            C.points.append(P1); C.points.append(P2)
    
    def organize_edges(self):
        self.points.sort(key=lambda x: arctan2(x.P[1]-self.C[1], x.P[0]-self.C[0]))
        self.edges = []
        for i in range(len(self.points)):
            A = self.points[i]; B = self.points[(i+1)%len(self.points)]
            self.edges.append(Edge(A, B,self))
            
        for i in range(len(self.edges)):
            E1 = self.edges[i]
            if(E1.P1.C2 == self):
                for j in range(i, len(self.points)+i):
                    E2 = self.edges[j%len(self.points)]
                    E2.exterior = False
                    if(E2.P2.C2 == E1.P1.C1):
                        break
class Pedge:
    def __init__(self, P1, P2, exterior):
        self.P1 = P1
        self.P2 = P2
        self.exterior = exterior
        
    def plot(self):
        if(self.exterior):
            plt.plot([self.P1.P[0], self.P2.P[0]],[self.P1.P[1], self.P2.P[1]], 'g', markersize =5)
        else:
            plt.plot([self.P1.P[0], self.P2.P[0]],[self.P1.P[1], self.P2.P[1]], 'b', markersize =5)
    
class Polygon:
    def __init__(self, polygon):
        self.poly = array(polygon)
        self.N = -1
        self.pedges = []
        self.points = [[]]*len(polygon)
        

    def plot(self):
        plt.plot(self.poly[:,0], self.poly[:,1], 'k')
        
    def intersect(self, C):
        for i in range(len(self.poly)-1):
            A = self.poly[i]; B = self.poly[i+1]; D = C.C
            U = B-A; V = D-A
            u = U/linalg.norm(U)
            w = array((u[1],-u[0]))
            height = u[0]*V[1] - u[1]*V[0]
            if(abs(height) < C.r):
                b = sqrt(C.r*C.r - height*height)
                a1 = (A-D)@u; a2 = (B-D)@u
                if(a1 <= -b and a2 >= -b):
                    C.points.append(Point(D + height*w - b*u, C, self))
                    self.points[i].append(C.points[-1])
                if(a1 <= b and a2 >= b):
                    C.points.append(Point(D + height*w + b*u, self, C))
                    self.points[i].append(C.points[-1])
            print(i, ":" , [i for i in map(lambda x: x.N, self.points[i])])
        """for i in range(len(EItot)):
            P1 = EItot[i]; P2 = EItot[(i+1)%len(EItot)]
            self.pedges.append(Pedge(P1,P2,True))
        for i in range(len(self.pedges)):
            E1 = self.pedges[i]
            if(E1.P1.C2 != self):
                for j in range(i, len(self.pedges)+i):
                    E2 = self.pedges[j%len(self.pedges)]
                    E2.exterior = False
                    if(E2.P2.C1 == E1.P1.C2):
                        break
        """    
        
            
                    
                
class Graph:
    def __init__(self, polygon):
        self.circles = []
        self.polygon = Polygon(polygon)
        
    def add_circle(self, C, r):
        NC = Circle(C, r)
        for C in self.circles:
            NC.intersect(C)
        self.polygon.intersect(NC)
        self.circles.append(NC)
    
    def organize_edges(self):
        for C in self.circles:
            C.organize_edges()
            for edge in C.edges:
                if(edge.exterior):
                    C.force += edge.P2.P - edge.P1.P
            C.force = array([C.force[1], -C.force[0]])
        
        
    def plot(self):
        for C in self.circles:
            C.plot()
        self.polygon.plot()
        for pedge in self.polygon.pedges:
            pedge.plot()
        for points in self.polygon.points:
            for point in points:
                point.plot()
        
import numpy as np

print([[]] * 3)
graph = Graph([[0,0],[0.75,1],[1,0],[0,0]])
R = 0.4
a = 1
#graph.polygon.poly = array([[0,0],[0,a],[a,a],[a,0],[0,0]])
for i in range(3):
    graph.add_circle(np.random.rand(2)/100+graph.polygon.poly[i%len(graph.polygon.poly)],R)

graph.organize_edges()

trajectories = [[]] * len(graph.circles)
for i in range(0):
    newgraph = Graph(graph.polygon.poly)
    for i in range(len(graph.circles)):
        C = graph.circles[i]
        newgraph.add_circle(C.C+C.force*0.05+random.rand(2)/1000, C.r)
        trajectories[i].append(C.C)
    newgraph.organize_edges()
    graph = newgraph

graph.plot()

"""for i in trajectories:
    i = array(i)
    plt.plot(i.T[0], i.T[1], 'ko', markersize = 3)"""
plt.axis('equal')
plt.show()