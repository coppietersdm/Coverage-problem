import numpy as np
import matplotlib.pyplot as plt
   

class Circle:
    def __init__(self, C, r, N):
        self.r = r
        self.C = C
        self.N = N
        self.points = []
        
    def add_point(self, P, C1, C2):
        self.points.append(Point(P, C1, C2))
        
    def plot(self):
        theta = np.linspace(0,2*np.pi,1000)
        plt.plot(self.C[0], self.C[1], 'ko')
        plt.text(self.C[0], self.C[1], str(self.N))
        plt.plot(self.C[0] + self.r*np.cos(theta), self.C[1] + self.r*np.sin(theta), 'k')
    
    def __str__(self):
        return str(self.N)

class Point:
    N = 0
    def __init__(self, P, C1, C2):
        self.P = P
        self.C1 = C1
        self.C2 = C2
        self.N = Point.N
        Point.N += 1
        
    def plot(self):
        plt.plot(self.P[0], self.P[1], 'ro')
        plt.text(self.P[0], self.P[1], str(self.N))
        
    def __str__(self):
        return str(self.N)
class Edge:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        self.exterior = True
    
    def __str__(self):
        return str(self.P1) + " - " + str(self.P2)
    
    def plot(self):
        plt.plot([self.P1.P[0], self.P2.P[0]], [self.P1.P[1], self.P2.P[1]], 'b', linewidth=5)
    
class Graph:
    def __init__(self, color):
        self.circles = []
        self.polygon = np.array([])
        
    def plot(self):
        for circle in self.circles:
            circle.plot()
            for point in circle.points:
                point.plot()
        plt.plot(self.polygon[:,0], self.polygon[:,1], 'r')
        
            
    def add_circle(self, NC):
        self.circles.append(NC)
        
    def intersections(self):
        for i in range(len(self.circles)):
            C1 = self.circles[i]
            for j in range(i+1,len(self.circles)):
                C2 = self.circles[j]
                if(C1 != C2):
                    V = C1.C - C2.C
                    A = np.arctan2(V[1], V[0])
                    d = np.linalg.norm(V)
                    if(d < C1.r + C2.r):
                        x = (d*d + C1.r*C1.r - C2.r*C2.r) / (2*d)
                        y = np.sqrt(C1.r*C1.r - x*x)
                        c, s = np.cos(A), np.sin(A)
                        R = np.array(((c, -s), (s, c)))
                        P1 = Point(C1.C - R@np.array((x, y)), C2, C1)
                        P2 = Point(C1.C - R@np.array((x, -y)), C1, C2)
                        C1.points.append(P1)
                        C2.points.append(P1)
                        C1.points.append(P2)
                        C2.points.append(P2)
                        
            for i in range(len(self.polygon)-1):
                A = self.polygon[i]; B = self.polygon[i+1]; D = C1.C
                U = B-A; V = D-A
                
                u = U/np.linalg.norm(U)
                w = np.array((u[1],-u[0]))
                height = u[0]*V[1] - u[1]*V[0]
                if(abs(height) < C1.r):
                    b = np.sqrt(C1.r*C1.r - height*height)
                    a1 = (A-D)@u; a2 = (B-D)@u
                    if(a1 <= b and a2 >= b):
                        P1 = Point(D + height*w + b*u, None, C1)
                        plt.plot(P1.P[0], P1.P[1], 'bo', markersize= 10)
                        C1.points.append(P1)
                    if(a1 <= -b and a2 >= -b):
                        P2 = Point(D + height*w - b*u, C1, None)
                        plt.plot(P1.P[0], P1.P[1], 'go',markersize= 10)
                        C1.points.append(P2)
                        
        for C in self.circles:
            C.points.sort(key=lambda x: np.arctan2(x.P[1]-C.C[1], x.P[0]-C.C[0]))
            C.edges = [Edge(C.points[i], C.points[(i+1)%len(C.points)]) for i in range(len(C.points))]
            for i in range(len(C.edges)):
                E1 = C.edges[i]
                if(E1.P1.C1 == C):
                    for j in range(i, len(C.points)+i):
                        E2 = C.edges[j%len(C.points)]
                        E2.exterior = False
                        if(E2.P1.C1 == E1.P2.C2):
                            break
            for E in C.edges:
                if(E.exterior):
                    E.plot()
                
            
        
        
graph = Graph('green')
graph.polygon = np.array([[0,0],[1,1],[1,0],[0,0]])
R = 0.6
graph.add_circle(Circle(np.array([0,0.1]), R, 1))
graph.add_circle(Circle(np.array([1,1]), R, 2))
graph.add_circle(Circle(np.array([1,0]), R, 3))

graph.intersections()
graph.plot()

plt.axis('equal')
plt.show()
