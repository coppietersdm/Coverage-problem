from numpy import *
import matplotlib.pyplot as plt
import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) <= 0)

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
        
    def clean(self):
        self.points = []
        self.gradient *= 0
        self.edges = []
        self.isolated = False
        
        
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
            gradient -= self.points[i+1].P- self.points[i].P
        gradient = array([-gradient[1], gradient[0]])
        self.gradient = gradient
        return gradient
    
    def integrate(self, plot=False):
        self.edges = []
        for i in range(0, len(self.points),2):
            self.edges.append([self.points[i].P,self.points[i+1].P])
        if(plot):
            print(self.edges)
        integral = 0
        for edge in self.edges:
            theta1 = arctan2(edge[0][1] - self.C[1],edge[0][0] - self.C[0])
            theta2 = arctan2(edge[1][1] - self.C[1],edge[1][0] - self.C[0])
            if(theta1 > theta2):
                theta2 += 2*pi
            theta = linspace(theta1, theta2, 1000)
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
        if(plot):
            print('hello', [x for x in map(lambda x:x.P,self.points)])
        for i in range(1,len(self.points),2):
            if(plot):
                plt.plot([self.points[i-1].P[0], self.points[i].P[0]],[self.points[i-1].P[1], self.points[i].P[1]], 'b')
            x = linspace(self.points[i-1].P[0], self.points[i].P[0], 1000)
            y = linspace(self.points[i-1].P[1], self.points[i].P[1], 1000)
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
        self.h = outer(R1, R2)/abs(cross(R1,R2))
    
    def plot(self):
        plt.plot(self.P[0], self.P[1], 'bo')
        plt.text(self.P[0], self.P[1], str(self.order))   
        
class Graph():
    def __init__(self):
        self.segments = []
        self.circles = []
    
    def clean_graph(self):
        for circle in self.circles:
            circle.clean()
        for segment in self.segments:
            segment.clean()
        Point.N  = 0
    
    def compute_gradients(self):
        graph.clean_graph()
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
    
    def plot(self):
        self.compute_gradients()
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
        self.compute_gradients()
        gradient = zeros(2*len(self.circles))
        for i in range(len(self.circles)):
            gradient[2*i:2*i+2] = self.circles[i].gradient
        return gradient
    
    def update_with_gradient(self, epsilon):
        self.clean_graph()
        self.compute_gradients()
        for circle in self.circles:
            plt.plot(circle.C[0], circle.C[1], 'rx')
            circle.C += circle.gradient*epsilon
    
    def update_with_saturated_gradient(self, epsilon):
        self.clean_graph()
        self.compute_gradients()
        for circle in self.circles:
            plt.plot(circle.C[0], circle.C[1], 'rx')
            circle.gradient *= 1
            if(linalg.norm(circle.gradient) != 0):
                circle.C += circle.gradient/linalg.norm(circle.gradient)*min(epsilon, linalg.norm(circle.gradient))

    
    def graph_hessian(self):
        self.clean_graph()
        self.compute_gradients()
        H = zeros((2*len(self.circles), 2*len(self.circles)))
        for circle in self.circles:
            for point in circle.points:
                SH = zeros((2*len(self.circles), 2*len(self.circles)))
                if(type(point.Cin) == type(point.Cout)):
                    i = point.Cin.N; j = point.Cout.N
                    SH[2*i:2*i+2,2*i:2*i+2] = point.h
                    SH[2*i:2*i+2,2*j:2*j+2] = -point.h.T
                    SH[2*j:2*j+2,2*i:2*i+2] = -point.h
                    SH[2*j:2*j+2,2*j:2*j+2] = point.h.T
                elif(point.Cin == circle):
                    i = circle.N
                    SH[2*i:2*i+2,2*i:2*i+2] = -2*point.h.T
                else:
                    i = circle.N
                    SH[2*i:2*i+2,2*i:2*i+2] = -2*point.h
                H += SH
                    
        return H/2
        
    def update_with_saturated_local_hessian(self, epsilon):
        self.clean_graph()
        self.compute_gradients()
        H = self.graph_hessian()
        for circle in self.circles:
            h = H[2*circle.N:2*circle.N+2,2*circle.N:2*circle.N+2]
            if(is_pos_def(h)):
                circle.gradient = -linalg.pinv(h)@circle.gradient
                plt.plot(circle.C[0], circle.C[1], 'kx')
            else:
                plt.plot(circle.C[0], circle.C[1], 'rx')
            if(linalg.norm(circle.gradient) != 0):
                circle.C += circle.gradient/linalg.norm(circle.gradient)*min(epsilon, linalg.norm(circle.gradient))
            
    def update_with_local_hessian(self, epsilon):
        self.clean_graph()
        self.compute_gradients()
        H = self.graph_hessian()
        for circle in self.circles:
            h = H[2*circle.N:2*circle.N+2,2*circle.N:2*circle.N+2]
            if(is_pos_def(h)):
                circle.gradient = -linalg.pinv(h)@circle.gradient
            plt.plot(circle.C[0], circle.C[1], 'rx')
            circle.C += circle.gradient*epsilon




def gradient_descent(graph, epsilon,steps):
    for i in range(steps):
        graph.update_with_gradient(epsilon)
    graph.plot()
    plt.axis('equal')
    
def saturated_gradient_descent(graph, epsilon,steps):
    surface = []
    for i in range(steps):
        graph.update_with_saturated_gradient(epsilon)
        surface.append(graph.integral())
    graph.plot()
    plt.axis('equal')
    
    return surface

def local_hessian_descent(graph,epsilon,steps):
    for i in range(steps):
        graph.update_with_local_hessian(epsilon)
    graph.plot()
    plt.axis('equal')
    
def local_saturated_hessian_descent(graph,epsilon,steps):
    surface = []
    for i in range(steps):
        graph.update_with_saturated_local_hessian(epsilon)
        surface.append(graph.integral())
    graph.plot()
    plt.axis('equal')
    return surface
    
    
steps = 20


graph = Graph()
graph.polygon(array([[-1,-1],[-1,1],[1,1],[1,-1]])+1)
graph.compute_gradients()
polygon_surface = graph.integral()

Ncircles = 1
R = 1/sqrt(pi)
positions = [random.rand(2) for i in range(Ncircles)]
positions[0] = array([0,0])

for i in range(Ncircles):
    graph.circles.append(Circle(positions[i], R))
    
epsilon1 = 0.1
surface1 = saturated_gradient_descent(graph, epsilon1, steps)
plt.title("Saturated Gradient Descent: stepsize = " + str(epsilon1))
plt.savefig("graph1.pdf")
plt.clf()

graph.clean_graph()
    

print(polygon_surface)
plt.plot((polygon_surface - array(surface1))/polygon_surface, label = str(epsilon1))
plt.legend()
plt.grid()
plt.savefig("surface.pdf")