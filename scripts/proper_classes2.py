from numpy import *
import matplotlib.pyplot as plt
import numpy as np

class Circle:
    N = 0
    def __init__(self, C, r):
        """
        Initialize a Circle object.

        Parameters:
        - C: The center point of the circle.
        - r: The radius of the circle.
        """
        self.C = C
        self.r = r
        self.points = []
        self.gradient = zeros(2)
        self.N = Circle.N
        Circle.N += 1
        self.edges = []
        
    def CC_intersection(self, circle):
        """
        Find the intersection points between two circles.

        Parameters:
        - circle: The other circle to find the intersection with.

        Returns:
        - None

        The method calculates the intersection points between the current circle and the given circle.
        If the circles intersect, it adds the intersection points to the `points` list of both circles.
        """
        
        D = circle.C - self.C        # distance vector
        d = linalg.norm(D)           # distance
        
        # if the circles intersect
        if(d < self.r + circle.r):
            # Unitary vectors
            Ud = D/d; UQd = array([-Ud[1], Ud[0]])
            
            # distances along unitary vectors
            x = (self.r*self.r + d*d - circle.r*circle.r)/(2*d)
            y = sqrt(self.r*self.r - x*x)
            
            # creation of the points
            P1 = Point(self.C + Ud*x + UQd*y, circle, self)
            P2 = Point(self.C + Ud*x - UQd*y, self, circle)
            
            # creation of the subhessians h
            P1.subhessian(P1.P - P1.Cin.C, P1.P - P1.Cout.C)
            P2.subhessian(P2.P - P2.Cin.C, P2.P - P2.Cout.C)
            
            # add points to the lists of points of the circles
            self.points.append(P1); self.points.append(P2)
            circle.points.append(P1); circle.points.append(P2)
    
    def order_filter_points(self):
        # sort points in CCW order
        self.points.sort(key=lambda x: arctan2(x.P[1]-self.C[1], x.P[0]-self.C[0]))
        
        # assign an order to the points corresponding to twice the number of agents containing it
        for i in range(len(self.points)):
            if(self.points[i].Cin == self):
                for j in range(i+1, i + len(self.points)):
                    if(self.points[j%len(self.points)].Cin == self.points[i].Cout):
                        break
                    else:
                        self.points[j%len(self.points)].order += 1
        
        #filter points with an order bigger than 0
        self.points = array(list(filter(lambda x: x.order == 0, self.points)))
    
    def calculate_gradient(self):
        if(len(self.points) != 0):
            for i in range(int(self.points[0].Cout != self), len(self.points), 2):
                self.gradient += array([[0,-1],[1,0]])@(self.points[i].P - self.points[i-1].P)
                self.edges.append(Edge(self.points[i-1].P,self.points[i].P,self))
        for edge in self.edges:
            edge.plot()
        
    def plot(self):
        plt.plot(self.C[0], self.C[1], 'ro')
        theta = linspace(0,2*pi, 1000)
        plt.plot(self.C[0] + self.r*cos(theta), self.C[1] + self.r*sin(theta), 'k')

class Edge():
    def __init__(self, P1, P2, C):
        self.P1 = P1
        self.P2 = P2
        self.C = C
    
    def integrate(self):
        theta1 = arctan2(self.P1[1]   - self.C.C[1], self.P1[0]   - self.C.C[0])
        theta0 = arctan2(self.P2[1]   - self.C.C[1], self.P2[0]   - self.C.C[0])
        if(theta1 < theta0):
            theta1 += 2*pi
        theta = linspace(theta1, theta0, 100)
        x = self.C.C[0] + self.C.r*cos(theta)
        y = self.C.C[1] + self.C.r*sin(theta)
        dx = diff(x)
        dy = diff(y)
        x = x[:-1]
        y = y[:-1]
        return (dx@y - dy@x)/2
        
        
        
    def plot(self):
        theta0 = arctan2(self.P1[1]   - self.C.C[1], self.P1[0]   - self.C.C[0])
        theta1 = arctan2(self.P2[1]   - self.C.C[1], self.P2[0]   - self.C.C[0])
        if(theta1 < theta0):
            theta1 += 2*pi
        theta = linspace(theta0, theta1, 100)
        plt.plot(self.C.C[0]+ self.C.r*cos(theta), self.C.C[1]+ self.C.r*sin(theta), 'b', linewidth=3)

class Point:
    def __init__(self, P, Cin, Cout):
        self.P = P
        self.Cin = Cin
        self.Cout = Cout
        self.h = zeros((2,2))
        self.order = 0
    
    def subhessian(self, R1, R2):
        self.h = outer(R1, R2)/abs(cross(R1,R2))
        R1 = R1/linalg.norm(R1)
        R2 = R2/linalg.norm(R2)
    
    def plot(self):
        plt.plot(self.P[0],self.P[1], 'bo')
     
class Graph():
    def __init__(self):
        self.circles = []
        self.pedges = []
    
    def polygon(self, lst):
        for i in range(len(lst)):
            self.pedges.append(Pedge(array(lst[i]),array(lst[(i+1)%len(lst)])))

    def calculations(self):
        self.clean_calculations()
        for i in range(len(self.circles)):
            for j in range(i+1, len(self.circles)):
                self.circles[i].CC_intersection(self.circles[j])
        for pedge in self.pedges:
            for circle in self.circles:
                pedge.SC_intersection(circle)
            
        for circle in self.circles:   
            circle.order_filter_points()
            circle.calculate_gradient()
        
        for pedge in self.pedges:
            pedge.filter_points()
    
    def clean_calculations(self):
        for circle in self.circles:
            circle.points = []
            circle.gradient *= 0
            circle.edges = []
        for pedge in self.pedges:
            pedge.points = []
        
    def update_with_gradient(self, epsilon):
        self.clean_calculations()
        self.calculations()
        for circle in self.circles:
            plt.plot(circle.C[0], circle.C[1], 'r+')
            circle.C += epsilon*circle.gradient
            
        self.clean_calculations()
        self.calculations()

    def gradient(self):
        G = zeros(2*len(self.circles)+2)
        for i in range(len(self.circles)):
            G[2*i:2*i+2] = self.circles[i].gradient
        return G
    
    def update_with_hessian(self):
        self.calculations()
        G = self.gradient()
        H = self.hessian()
        DX = -linalg.solve(H,G)
        for i in range(len(self.circles)):
            self.circles[i].C += DX[2*i:2*i+2] - DX[-2:]
            plt.plot(self.circles[i].C[0], self.circles[i].C[1], 'b+')     
        
    def hessian(self):
        H = zeros((2*len(self.circles)+2, 2*len(self.circles)+2))
        for circle in self.circles:
            for point in circle.points:
                i = len(self.circles) if(point.Cin == None) else point.Cin.N ; j = len(self.circles) if(point.Cout == None) else point.Cout.N
                SH = zeros((2*len(self.circles)+2, 2*len(self.circles)+2))
                SH[2*i:2*i+2, 2*i:2*i+2] = point.h 
                SH[2*j:2*j+2, 2*j:2*j+2] = point.h.T 
                SH[2*i:2*i+2, 2*j:2*j+2] = -point.h
                SH[2*j:2*j+2, 2*i:2*i+2] = -point.h.T
                H += SH
        for pedge in self.pedges:
            for point in pedge.points:
                i = len(self.circles) if(point.Cin == None) else point.Cin.N ; j = len(self.circles) if(point.Cout == None) else point.Cout.N
                SH = zeros((2*len(self.circles)+2, 2*len(self.circles)+2))
                SH[2*i:2*i+2, 2*i:2*i+2] = point.h 
                SH[2*j:2*j+2, 2*j:2*j+2] = point.h.T 
                SH[2*i:2*i+2, 2*j:2*j+2] = -point.h
                SH[2*j:2*j+2, 2*i:2*i+2] = -point.h.T
                H += SH
        SH = zeros((2*len(self.circles)+2, 2*len(self.circles)+2))
        SH[-2] = array([i%2 for i in range(2*len(self.circles)+2)])
        SH[-1] = array([(i+1)%2 for i in range(2*len(self.circles)+2)])
        H += SH
        return H
    
    def plot(self):
        
        for circle in self.circles:
            circle.plot()
            plt.arrow(circle.C[0], circle.C[1], circle.gradient[0], circle.gradient[1])
            for point in circle.points:
                point.plot()
            # for edge in circle.edges:
            #     edge.plot()
                
        
        for pedge in self.pedges:
            pedge.plot()
        
    
class Pedge():
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        self.points = []
        self.gradient = zeros(2)
    
    def SC_intersection(self, circle):
        A = self.P1; B = self.P2; C = circle.C
        U = B-A; V = C-A
        u = linalg.norm(U)
        Eu = U/u
        y = cross(Eu,V)
        if(abs(y) < circle.r):
            x = sqrt(circle.r*circle.r - y*y)
            s = V@Eu
            if(s-x > 0 and s-x < u):
                P1 = Point(A + (s-x)*Eu, None, circle)
                P1.subhessian(P1.P-circle.C, array([[0,1],[-1,0]])@(self.P2-self.P1))
                circle.points.append(P1)
                self.points.append(P1)
            if(s+x > 0 and s+x < u):
                P2 = Point(A + (s+x)*Eu, circle, None)
                P2.subhessian(array([[0,1],[-1,0]])@(self.P2-self.P1),P2.P-circle.C)
                circle.points.append(P2)
                self.points.append(P2)
    
    def filter_points(self):
        self.points = array(list(filter(lambda x: x.order == 0, self.points)))
    
    def plot(self):
        plt.plot([self.P1[0], self.P2[0]],[self.P1[1], self.P2[1]], 'r')
        for point in self.points:
            point.plot()

graph = Graph()
for i in range(1):
    graph.circles.append(Circle(random.rand(2)*10, 12))

graph.polygon([[0,0],[0,20],[20,20],[20,0]])

print(graph.gradient())

    
graph.calculations()
    
graph.plot()

plt.axis('equal')
plt.show()



