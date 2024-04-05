from numpy import *
import matplotlib.pyplot as plt
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

Rpi2 = np.array([[0,-1],[1,0]])

class Circle():
    N = 0
    def __init__(self, C, r):
        self.C = array(C, dtype=float)
        self.r = r
        self.points = []
        self.N = Circle.N
        Circle.N += 1
        self.gradient = zeros(2)
        self.edges = []
        self.isolated = False
        self.Rgradient = 0
        
    def clean(self):
        self.points = []
        self.gradient *= 0
        self.edges = []
        self.isolated = False
        self.Rgradient = 0
        
        
    def CC_intersection(self, circle):
        D = circle.C - self.C
        d = linalg.norm(D)
        if(d < self.r + circle.r):
            Ud = D/d; UQd = array([-Ud[1], Ud[0]])
            x = (self.r*self.r + d*d - circle.r*circle.r)/(2*d)
            y = sqrt(self.r*self.r - x*x)
            P1 = Point(self.C + Ud*x + UQd*y, circle, self)
            P2 = Point(self.C + Ud*x - UQd*y, self, circle)
            
            P1.subhessian(P1.P - P1.Cin.C, P1.P - P1.Cout.C)
            P2.subhessian(P2.P - P2.Cin.C, P2.P - P2.Cout.C)
            
            self.points.append(P1); self.points.append(P2)
            circle.points.append(P1); circle.points.append(P2)
            
    def CS_intersection(self, segment):
        A = segment.P1; B = segment.P2; C = self.C
        U = B-A; V = C-A
        u = linalg.norm(U)
        Eu = U/u
        y = cross(Eu,V)
        if(abs(y) < self.r):
            x = sqrt(self.r*self.r - y*y)
            s = V@Eu
            
            P1 = Point(A + (s-x)*Eu, segment, self)
            self.points.append(P1); segment.points.append(P1)
            P2 = Point(A + (s+x)*Eu, self, segment)
            self.points.append(P2); segment.points.append(P2)
            
            P1.subhessian(P1.Cin.normal, P1.P - P1.Cout.C)
            P2.subhessian(P2.P - P2.Cin.C, P2.Cout.normal)
    
    def filter_points(self):
        self.isolated = len(self.points) == 0
        self.points.sort(key = lambda x: arctan2(x.P[1] - self.C[1],x.P[0] - self.C[0]))
        for i in range(len(self.points)):
            if(self.points[i].Cin == self):
                for j in range(i+1, i+len(self.points)):
                    if(self.points[j%len(self.points)].Cin == self.points[i].Cout):
                        break
                    else:
                        self.points[j%len(self.points)].order += 1
        self.points = list(filter(lambda x: x.order == 0, self.points))

    def compute_gradient(self):
        if(len(self.points)==0):
            return zeros(2)
        gradient = zeros(2)
        if(self.points[0].Cin == self):
            self.points = self.points[1:] + self.points[0:1]
        for i in range(0, len(self.points),2):
            gradient -= self.points[i+1].P - self.points[i].P
            self.Rgradient += (arctan2(self.points[i+1][1],self.points[i+1][0])- arctan2(self.points[i][1]-self.points[i][0]))*self.R
        gradient = array([-gradient[1], gradient[0]])
        self.gradient = gradient
        return gradient
    
    def integrate(self, plot=False):
        self.edges = []
        for i in range(0, len(self.points),2):
            self.edges.append([self.points[i].P,self.points[i+1].P])
        integral = 0
        for edge in self.edges:
            theta1 = arctan2(edge[0][1] - self.C[1],edge[0][0] - self.C[0])
            theta2 = arctan2(edge[1][1] - self.C[1],edge[1][0] - self.C[0])
            if(theta1 > theta2):
                theta2 += 2*pi
            theta = linspace(theta1, theta2, 10000)
            x = self.C[0] + self.r*cos(theta)
            y = self.C[1] + self.r*sin(theta)
            if(plot):
                plt.plot(x,y, 'b')
            dx = diff(x)
            dy = diff(y)
            integral += (dx@y[1:] - dy@x[1:])/2
        if(self.isolated):
            integral -= pi*self.r**2
        return integral

    def plot(self):
        theta = linspace(0,2*pi, 1000)
        plt.plot(self.C[0] + self.r*cos(theta), self.C[1] + self.r*sin(theta), 'k')
        plt.plot(self.C[0], self.C[1], 'ro')
        plt.text(self.C[0], self.C[1], str(self.N))
        plt.arrow(self.C[0], self.C[1], self.gradient[0], self.gradient[1])

class Segment():
    N = 0
    def __init__(self, P1, P2):
        self.P1 = array(P1)
        self.P2 = array(P2)
        self.points = []
        self.N = Segment.N
        Segment.N += 1
        self.normal = array([[0,-1],[1,0]])@(self.P2-self.P1)

    def clean(self):
        self.points = []
    
    def filter_points(self):
        self.points.append(Point(self.P1, None, self))
        self.points.append(Point(self.P2, self, None))
        self.points.sort(key = lambda x: (x.P-self.P1)@(self.P2-self.P1))
        for i in range(len(self.points)):
            s = (self.points[i].P - self.P1)@(self.P2-self.P1)/linalg.norm(self.P2-self.P1)**2
            if(self.points[i].Cin == self):
                for j in range(i+1, len(self.points)):
                    if(self.points[j].Cin == self.points[i].Cout):
                        break
                    else:
                        self.points[j].order += 1
        self.points = [x for x in filter(lambda x: x.order == 0, self.points)]

    def integrate(self, plot = False):
        integral = 0
        
        for i in range(1,len(self.points),2):
            if(plot):
                plt.plot([self.points[i-1].P[0], self.points[i].P[0]],[self.points[i-1].P[1], self.points[i].P[1]], 'b')
            x = linspace(self.points[i-1].P[0], self.points[i].P[0], 10000)
            y = linspace(self.points[i-1].P[1], self.points[i].P[1], 10000)
            dx = diff(x)
            dy = diff(y)
            integral += (dx@y[1:] - dy@x[1:])/2
        
        return integral
    
    def plot(self):
        plt.plot([self.P1[0], self.P2[0]],[self.P1[1], self.P2[1]], 'r')
        plt.text((self.P1[0] + self.P2[0])/2,(self.P1[1] + self.P2[1])/2,str(self.N))

class Point():
    N = 0
    def __init__(self, P, Cin, Cout):
        self.P = P
        self.Cin = Cin
        self.Cout = Cout
        self.h = zeros((2,2))
        self.N = Point.N
        Point.N += 1
        self.order = 0
    
    def subhessian(self, R1, R2):
        #self.h = outer(R1, R2)/abs(cross(R1,R2))
        self.h = outer(R1, R2)/(R1@(Rpi2@R2))
    
    def plot(self):
        plt.plot(self.P[0], self.P[1], 'bo')
        plt.text(self.P[0], self.P[1], str(self.order))   
        
class Graph():
    def __init__(self):
        self.segments = []
        self.circles = []
        Circle.N = 0
        Segment.N = 0
    
    def clean_graph(self):
        for circle in self.circles:
            circle.clean()
        for segment in self.segments:
            segment.clean()
        Point.N  = 0
    
    def compute_gradients(self):
        self.clean_graph()
        for i in range(len(self.circles)):
            for j in range(i+1,len(self.circles)):
                self.circles[i].CC_intersection(self.circles[j])
            for j in range(len(self.segments)):
                self.circles[i].CS_intersection(self.segments[j])
        
        for circle in self.circles:
            circle.filter_points()
            circle.compute_gradient()
        for segment in self.segments:
            segment.filter_points()
    
    def polygon(self, lst):
        for i in range(len(lst)):
            self.segments.append(Segment(lst[i], lst[(i+1)%len(lst)]))
    
    def plot_positions(self):
        for circle in self.circles:
            plt.plot(circle.C[0], circle.C[1], 'r+')
    
    def plot(self):
        for circle in self.circles:
            circle.plot()
            circle.integrate(plot=True)
            for point in circle.points:
                point.plot()
        for segment in self.segments:
            segment.plot()
            segment.integrate(plot=True)
            for point in segment.points:
                point.plot()
    
    def integral(self):
        integral = 0
        for circle in self.circles:
            integral += circle.integrate()
        for segment in self.segments:
            integral += segment.integrate()
        return integral
    
    def graph_gradient(self):
        gradient = zeros(2*len(self.circles))
        for i in range(len(self.circles)):
            gradient[2*i:2*i+2] = self.circles[i].gradient
        return gradient
    
    def graph_hessian(self):
        H = zeros((2*len(self.circles), 2*len(self.circles)))
        for circle in self.circles:
            for point in circle.points:
                SH = zeros((2*len(self.circles), 2*len(self.circles)))
                if(type(point.Cin) == type(point.Cout)):
                    i = point.Cin.N; j = point.Cout.N
                    SH[2*i:2*i+2,2*i:2*i+2] = -point.h
                    SH[2*i:2*i+2,2*j:2*j+2] = point.h
                    SH[2*j:2*j+2,2*i:2*i+2] = point.h.T
                    SH[2*j:2*j+2,2*j:2*j+2] = -point.h.T
                elif(point.Cin == circle):
                    i = circle.N
                    SH[2*i:2*i+2,2*i:2*i+2] = -2*point.h
                else:
                    j = circle.N
                    SH[2*j:2*j+2,2*j:2*j+2] = -2*point.h.T
                H += SH
                    
        return H/2


R = 0.6
 
def plot(x):
    graph = Graph()
    graph.polygon(array([[-1,-1],[-1,1],[1,1],[1,-1]])+1)
    positions = np.reshape(x, (-1,2))

    for position in positions:
        graph.circles.append(Circle(position, R))
    graph.compute_gradients()
    graph.plot()
            
def function(x):
    graph = Graph()
    graph.polygon(array([[-1,-1],[-1,1],[1,1],[1,-1]])+1)
    positions = np.reshape(x, (-1,2))

    for position in positions:
        graph.circles.append(Circle(position, R))
        
    graph.compute_gradients()
    graph.plot_positions()
    return graph.integral(), graph.graph_gradient(), graph.graph_hessian()
    
from scipy.optimize import minimize


def fun(x):
    return function(x)[0]

def jac(x):
    return -function(x)[1]

def hessi(x):
    return -function(x)[2]

def hessi_diag(x):
    hessi = -function(x)[2]
    H = hessi*0
    for i in range(len(x)//2):
        H[2*i:2*i+2,2*i:2*i+2] = hessi[2*i:2*i+2,2*i:2*i+2]
    return H
        

poly_surf = fun([])
N = 4
x = random.rand(2*N)*2

print(fun(x))
R += 0.01
print(fun(x))


# surface = []
# prediction = []
# epsilon = 0.1
# hessian_prediction = []
# hessian_diagonal_prediction = []

# max_step = 0.1

# for i in range(100):
#     F, G, H = function(x)
#     dx = G*epsilon
#     x += dx
#     surface.append((poly_surf-F)/poly_surf)
#     prediction.append(G@G * epsilon)
#     hessian_prediction.append(G@dx - 0.5*dx@(H@dx))
    
# plot(x)
# plt.axis('equal')
# plt.show()

# # plt.plot(surface, label = 'surface')
# GP = prediction[:-1]/poly_surf
# HP = hessian_prediction[:-1]/poly_surf
# # HDP = hessian_diagonal_prediction[:-1]/poly_surf
# P = np.diff(surface)
# plt.plot(P, label = 'dS')
# plt.plot(GP, label = 'gradient')
# plt.plot(HP, label = 'hessian')
# # plt.plot(abs(HDP-P), label = 'hessian_diagonal')
# print(np.exp(np.mean(np.log((abs(HP - P)/abs(GP - P))))))
# # print(np.exp(np.mean(np.log((abs(HDP - P)/abs(GP - P))))))
# plt.legend()
# plt.show()



# F0 = fun(x)
# G0 = jac(x)
# H0 = hessi(x)
# print(np.round(H0,3))

# dx = random.rand(2*N)/1000
# # dx = np.array([0,0,0.01,0.01,0,0])

# F1 = fun(x + dx)
# print(F1-F0)
# print(G0@dx)

# G1 = jac(x + dx)
# print((G1 - G0))
# print((-H0@dx))

# plot(x)
# plt.axis('equal')
# plt.show()
