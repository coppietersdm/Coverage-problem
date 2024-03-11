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
        #plt.text(self.P[0], self.P[1], str(self.N))

class Edge:
    def __init__(self,P1,P2):
        self.P1 = P1
        self.P2 = P2
        self.exterior = True
    
    def plot(self):
        plt.plot([self.P1.P[0], self.P2.P[0]], [self.P1.P[1], self.P2.P[1]], "b", linewidth=2)

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
        print(self.C, self.force)
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
            self.edges.append(Edge(A, B))
            
        for i in range(len(self.edges)):
            E1 = self.edges[i]
            if(E1.P1.C2 == self):
                for j in range(i, len(self.points)+i):
                    E2 = self.edges[j%len(self.points)]
                    E2.exterior = False
                    if(E2.P2.C2 == E1.P1.C1):
                        break

class Polygon:
    def __init__(self, polygon):
        self.poly = array(polygon)
        self.N = -1
    
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
                if(a1 <= b and a2 >= b):
                    C.points.append(Point(D + height*w + b*u, self, C))
                    
                
class Graph:
    def __init__(self):
        self.circles = []
        self.polygon = Polygon([])
        
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
        
            
            
graph = Graph()
graph.polygon.poly = array([[0,0],[1,1],[1,0],[0,0]])
graph.add_circle(array([0.5, 0.25]),0.45)
graph.add_circle(array([0.5, 0.75]),0.45)
graph.add_circle(array([1, 0.5]),0.45)
graph.organize_edges()


graph.plot()

plt.axis('equal')
plt.show()