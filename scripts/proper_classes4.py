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

            
    def remove_inner_points(self):
        self.points = list(filter(lambda x: x.order == 0, self.points))

    def compute_gradient(self):
        if(len(self.points)==0):
            return zeros(2)
        gradient = zeros(2)
        if(self.points[0].Cin == self):
            self.points = self.points[1:] + self.points[0:1]
        for i in range(0, len(self.points),2):
            gradient -= self.points[i+1].P - self.points[i].P
        gradient = array([-gradient[1], gradient[0]])
        self.gradient = gradient
        return gradient

    
    def integrate(self, segments, plot=False):
        self.edges = []
        for i in range(0, len(self.points),2):
            self.edges.append([self.points[i].P,self.points[i+1].P])
        integral = 0
        for edge in self.edges:
            theta1 = arctan2(edge[0][1] - self.C[1],edge[0][0] - self.C[0])
            theta2 = arctan2(edge[1][1] - self.C[1],edge[1][0] - self.C[0])
            if(theta1 > theta2):
                theta2 += 2*pi
            
            if(plot):
                theta = linspace(theta1, theta2, 100)
                x = self.C[0] + self.r*cos(theta)
                y = self.C[1] + self.r*sin(theta)
                plt.plot(x,y, 'b')
            integral -= self.r**2*(theta2-theta1)/2 + np.cross(self.C, edge[1] - edge[0])/2
        if(self.isolated):
            vector = np.array([self.C, np.array([1.0,0])])
            intersections = []
            for segment in segments:
                A = segment.P1
                B = segment.P2
                AB = B - A
                if(np.cross(vector[1], AB)):
                    t = np.cross(A - vector[0], vector[1]) / np.cross(vector[1], AB)
                    if 0 <= t <= 1:
                        intersection = A + t * AB
                        intersections.append(intersection)    
            intersections = [x for x in list(map(lambda x: (x - self.C)@(np.array([1.0,0]))/self.r, intersections))]
            intersections = [x for x in list(filter(lambda x: x < -1, intersections))]
            if(len(intersections) %2 == 1):
                integral -= pi*self.r**2
        return integral

    def plot(self):
        theta = linspace(0,2*pi, 1000)
        plt.plot(self.C[0] + self.r*cos(theta), self.C[1] + self.r*sin(theta), 'k')
        plt.plot(self.C[0], self.C[1], 'ro')
        # plt.text(self.C[0], self.C[1], str(self.N))
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
            if(self.points[i].Cin == None):
                for j in range(0,i):
                    self.points[j].order += 1
                    
    def remove_inner_points(self):
        self.points = [x for x in filter(lambda x: x.order == 0, self.points)]

    def integrate(self, plot = False):
        integral = 0
        for i in range(1,len(self.points),2):
            AA = self.points[i-1].P
            BB = self.points[i].P
            if(plot):
                plt.plot([self.points[i-1].P[0], self.points[i].P[0]],[self.points[i-1].P[1], self.points[i].P[1]], 'b')
            integral += (AA[1]*BB[0] - AA[0]*BB[1])/2
            
        return integral
    
    def plot(self):
        plt.plot([self.P1[0], self.P2[0]],[self.P1[1], self.P2[1]], 'r')
        # plt.text((self.P1[0] + self.P2[0])/2,(self.P1[1] + self.P2[1])/2,str(self.N))

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
        # plt.text(self.P[0], self.P[1], str(self.order))   
        
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
        
        for segment in self.segments:
            segment.filter_points()
        for circle in self.circles:
            circle.filter_points()
            
        for segment in self.segments:
            segment.remove_inner_points()
        for circle in self.circles:
            circle.remove_inner_points()
            
        for circle in self.circles:
            circle.compute_gradient()
    
    def polygon(self, lst):
        for i in range(len(lst)):
            self.segments.append(Segment(lst[i], lst[(i+1)%len(lst)]))
    
    def plot_positions(self):
        for circle in self.circles:
            plt.plot(circle.C[0], circle.C[1], 'r+')
    
    def plot(self):
        for segment in self.segments:
            segment.plot()
            segment.integrate(plot=True)
            for point in segment.points:
                point.plot()
        for circle in self.circles:
            circle.plot()
            circle.integrate(self.segments, plot=True)
            for point in circle.points:
                point.plot()
        
    
    def integral(self):
        integral = 0
        for circle in self.circles:
            integral += circle.integrate(self.segments)
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

default_polygon = array([[0,0],[0,100],[100,100],[100,0]])
default_radius = 6

def plot(x, polygon=default_polygon, R=default_radius):
    global iii  # Add this line to access the global variable iii
    graph = Graph()
    graph.polygon(polygon)
    positions = np.reshape(x, (-1,2))

    for position in positions:
        graph.circles.append(Circle(position, R))
    graph.compute_gradients()
    graph.plot()

    plt.axis('equal')
    # plt.show()
            
def function(x, polygon=default_polygon, R=default_radius):
    graph = Graph()
    graph.polygon(polygon)
    positions = np.reshape(x, (-1,2))

    for position in positions:
        graph.circles.append(Circle(position, R))
        
    graph.compute_gradients()
    #graph.plot_positions()
    return graph.integral(), graph.graph_gradient(), graph.graph_hessian()


def fun(x, polygon=default_polygon, R=default_radius):
    return function(x, polygon=polygon, R=R)[0]

def jac(x, polygon=default_polygon, R=default_radius):
    return -function(x, polygon=polygon, R=R)[1]

def hessi(x, polygon=default_polygon, R=default_radius):
    return -function(x, polygon=polygon, R=R)[2]

def pinv_hessi_diag(x, polygon=default_polygon, R=default_radius):
    hessi = function(x)[2]
    H = hessi*0
    for i in range(len(x)//2):
        if(is_pos_def(hessi[2*i:2*i+2,2*i:2*i+2])):
            H[2*i:2*i+2,2*i:2*i+2] = linalg.pinv(hessi[2*i:2*i+2,2*i:2*i+2])
        else:
            H[2*i:2*i+2,2*i:2*i+2] = eye(2)
    return H

def pinv_hessi(x, polygon=default_polygon, R=default_radius):
    H = hessi(x, polygon=polygon, R=R)
    if(is_pos_def(H)):
        return linalg.pinv(H)
    else:
        return eye(len(x))


def algorithm1(x, polygon = default_polygon, R=default_radius):
    return -jac(x, polygon=polygon, R=R)

def algorithm2(x, polygon = default_polygon, R=default_radius):
    H = pinv_hessi_diag(x, polygon=polygon, R=R)
    return -H@jac(x, polygon=polygon, R=R)

def algorithm3(x, polygon = default_polygon, R=default_radius):
    H = pinv_hessi(x, polygon=polygon, R=R)
    return -H@jac(x, polygon=polygon, R=R)

def algorithm4(x, polygon = default_polygon, R=default_radius):
    H = pinv_hessi_diag(x, polygon=polygon, R=R)
    return (1-H/2)@jac(x, polygon=polygon, R=R)

def max_step(dx, d):
    dx = dx.reshape((-1,2))
    for i, dxi in enumerate(dx):
        if linalg.norm(dxi) > d:
            dx[i] = dxi/linalg.norm(dxi)*d
    return dx.reshape(-1)


R = 1.0

N = 30

Z = np.array([[fun([x,y], polygon = array([[0,0],[0,1],[1,1],[1,0]])*1.8,R = 1.0) for x in linspace(-1.2,3, N)] for y in linspace(-1.2,3,N)])
U = np.array([[jac([x,y], polygon = array([[0,0],[0,1],[1,1],[1,0]])*1.8,R = 1.0)[0] for x in linspace(-1.2,3, N)] for y in linspace(-1.2,3,N)])
V = np.array([[jac([x,y], polygon = array([[0,0],[0,1],[1,1],[1,0]])*1.8,R = 1.0)[1] for x in linspace(-1.2,3, N)] for y in linspace(-1.2,3,N)])
X = np.array([[x for x in linspace(-1.2,3, N)] for y in linspace(-1.2,3,N)])
Y = np.array([[y for x in linspace(-1.2,3, N)] for y in linspace(-1.2,3,N)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracé de la surface
surf = ax.plot_surface(X, Y, Z)
plt.show()

plt.streamplot(X, Y, -U, -V, density=2, arrowsize=2, color=np.sqrt(U*U+V*V), cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Streamplot of fun(x, y)')
polygon = (array([[-1,-1],[-1,1],[1,1],[1,-1],[-1,-1]])+1)*0.9
plt.plot(polygon.T[0], polygon.T[1], 'k')
plt.axis('equal')
plt.colorbar(label='Intensity')
plt.show()


