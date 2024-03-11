from numpy import *
import matplotlib.pyplot as plt

class Point:
    Npoints = 0
    def __init__(self,P,C1,C2):
        self.P = P
        self.C1 = C1 # cercle rentrant
        self.C2 = C2 # cercle sortant
        self.N = Point.Npoints
        Point.Npoints += 1
    
    def plot(self):
        plt.plot(self.P[0], self.P[1], 'go', markersize = 5)
        plt.text(self.P[0]+0.05, self.P[1]+0.05, str(self.N))
        
    def __str__(self):
        return "P" + str(self.N)

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

    def integral(self):
        theta1 = arctan2(self.P1.P[1] - self.C.C[1], self.P1.P[0] - self.C.C[0])
        theta2 = arctan2(self.P2.P[1] - self.C.C[1], self.P2.P[0] - self.C.C[0])
        if(theta1 > theta2):
            theta2 += 2*pi
        theta = linspace(theta1, theta2, 10000)
        x = self.C.C[0] + self.C.r*cos(theta)
        y = self.C.C[1] + self.C.r*sin(theta)
        dx = diff(x)
        dy = diff(y)
        return (dx@y[:-1]-dy@x[:-1])/2
        
    
    
class Circle:
    Ncircles = 0
    def __init__(self, C, r):
        self.C = C
        self.r = r
        self.N = Circle.Ncircles
        Circle.Ncircles += 1
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
                    
    def surface(self):
        result = 0
        for edge in self.edges:
            result += edge.integral()
        return result
        
class Pedge:
    def __init__(self, P1, P2, exterior):
        self.P1 = P1
        self.P2 = P2
        self.exterior = exterior
        
        
    def plot(self):
        if(self.exterior):
            plt.plot([self.P1.P[0], self.P2.P[0]],[self.P1.P[1], self.P2.P[1]], 'b', markersize =5)
        else:
            plt.plot([self.P1.P[0], self.P2.P[0]],[self.P1.P[1], self.P2.P[1]], 'g', markersize =5)
    
    def integral(self):
        x = linspace(self.P1.P[0], self.P2.P[0], 1000)
        y = linspace(self.P1.P[1], self.P2.P[1], 1000)
        dx = diff(x)
        dy = diff(y)
        return (dx@y[:-1]-dy@x[:-1])/2
        
    
class Polygon:
    def __init__(self, polygon):
        self.poly = array(polygon)
        self.pedges = []
        self.points = []
        self.N = -1
        self.kakapoints = []
        for i in range(len(polygon)):
            self.kakapoints.append((Point(array(polygon[i]),self,self),i, 0))
        
    def plot(self):
        plt.plot(self.poly[:,0], self.poly[:,1], 'k')
        for pedge in self.pedges:
            pedge.plot()
        for point in self.points:
            point.plot()
        
    def intersect(self, C):
        self.pedges = []
        for i in range(len(self.poly)-1):
            A = array(self.poly[i]); B = array(self.poly[i+1]); D = C.C
            U = B-A; V = D-A
            u = U/linalg.norm(U)
            w = array((u[1],-u[0]))
            height = u[0]*V[1] - u[1]*V[0]
            if(abs(height) < C.r):
                b = sqrt(C.r*C.r - height*height)
                a1 = (A-D)@u; a2 = (B-D)@u
                if(a1 <= -b and a2 >= -b):
                    P = Point(D + height*w - b*u, C, self)
                    C.points.append(P)
                    self.kakapoints.append((P,i, (P.P-A)@u))
                if(a1 <= b and a2 >= b):
                    P = Point(D + height*w + b*u, self, C)
                    C.points.append(P)
                    self.kakapoints.append((P,i,(P.P-A)@u))
        self.kakapoints.sort(key = lambda x: (x[1],x[2]))
        self.points = [i for i in map(lambda x: x[0], self.kakapoints)]
        for i in range(len(self.points)-1):
            P1 = self.points[i]; P2 = self.points[i+1]
            self.pedges.append(Pedge(P1,P2,True))
        for i in range(len(self.pedges)):
            E1 = self.pedges[i]
            if(E1.P1.C1 != self):
                for j in range(i, len(self.pedges)+i):
                    E2 = self.pedges[j%len(self.pedges)]
                    E2.exterior = False
                    if(E2.P2.C2 == E1.P1.C1):
                        break
    
    def surface(self):
        result = 0
        for pedge in self.pedges:
            result += pedge.integral()
        return result
    
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
        
            
    def not_covered_surface(self):
        result = 0
        for pedge in self.polygon.pedges:
            if(pedge.exterior):
                result += pedge.integral()
                
        for C in self.circles:
            if(len(C.edges)==0):
                result -= C.r*C.r*pi
            for edge in C.edges:
                if(edge.exterior):
                    result += edge.integral()
        return result

    def surface(self):
        return 1 - graph.not_covered_surface()/graph.polygon.surface()
        
import numpy as np
a = 5
graph = Graph(array([[0,0],[0,1],[1,1],[1,0],[0,0]])*a)

R = 2

for i in range(3):
    graph.add_circle(np.random.rand(2)*a,R)#+graph.polygon.poly[i%(len(graph.polygon.poly)-1)],R)
graph.organize_edges()
print('coverage surface 1 =', 1 - graph.not_covered_surface()/graph.polygon.surface(), "%")
trajectories = [[]] * len(graph.circles)
surfaces = []
grad = []

epsilon = 0.1
for i in range(1):
    if(i==150):
        epsilon = 0.01
    print(i)
    newgraph = Graph(graph.polygon.poly)
    for j in range(len(graph.circles)):
        C = graph.circles[j]
        newgraph.add_circle(C.C+C.force*epsilon+(random.rand(2)-0.5)/5000, R)
        trajectories[j].append(C.C)
    newgraph.organize_edges()
    graph = newgraph
    surfaces.append(graph.surface())
    if i in []: #[0,5,10,20,40,80]:
        graph.plot()
        plt.title('Iteration ' + str(i))
        plt.axis('equal')
        plt.show()
    grad.append(sum([k**2 for k in map(lambda x:linalg.norm(x.force), graph.circles)]))
    
    Gradient = array([k for k in map(lambda x:x.force,graph.circles)])
    print(Gradient.flatten())
    H = zeros((len(Gradient), len(Gradient)))
    for circle in graph.circles:
        for edge in circle.edges:
            if(edge.exterior):
                print(edge.P1.N, "   --  ", edge.P1.C1.N, edge.P1.C2.N)
                R2 = edge.P1.P - edge.P1.C1.C
                plt.arrow(edge.P1.P[0], edge.P1.P[1], R2[0], R2[1])
                print(edge.P2.N, "   --  ", edge.P2.C1.N, edge.P2.C2.N)
    

graph.plot()
print('coverage surface 1 =', graph.surface(), "%")

plt.title("Final convergence position after " + str(i) + " iterations")

for i in trajectories:
    i = array(i)
    plt.plot(i.T[0], i.T[1], 'ko', markersize = 2)
plt.axis('equal')
plt.show()

#plt.ylim(0,1)
plt.grid(True)
plt.plot(surfaces, label = 'covered ratio of the polygon')
plt.plot(np.diff(array(surfaces))*graph.polygon.surface(), label = r'$f(i+1) - f(i)$')
plt.plot(array(grad)*epsilon, label = r'$\epsilon |\nabla f|^2$')
plt.legend()
plt.title(r"Evolution of the covered area ($\epsilon$ = {})".format(epsilon))
plt.show()