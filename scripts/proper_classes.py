from numpy import *
import matplotlib.pyplot as plt
import numpy as np

class Circle:
    Ncircle = 0
    def __init__(self, C, r):
        self.C = C
        self.r = r
        self.N = Circle.Ncircle
        Circle.Ncircle += 1
        self.points = []
        self.edges = []
        self.gradient = zeros(2)
        
    def C_C_intersection(self,C):
        D = C.C - self.C
        d = linalg.norm(D)
        Ed = D/d
        EQd = array([-Ed[1], Ed[0]])
        if(d < self.r + C.r):
            x = (self.r*self.r + d*d - C.r*C.r)/(2*d)
            y = sqrt(self.r*self.r - x*x)
            self.points.append(Point(self.C + Ed*x + EQd*y, C,IN = False, Cbis = self))
            C.points.append(Point(self.C + Ed*x + EQd*y,self,Cbis = C))
            self.points.append(Point(self.C + Ed*x - EQd*y, C, Cbis = self))
            C.points.append(Point(self.C + Ed*x - EQd*y, self, IN = False, Cbis = C))
            
    def filter(self):
        self.points.sort(key=lambda x: arctan2(x.P[1]-self.C[1], x.P[0]-self.C[0]))
        for i in range(len(self.points)):
            if(self.points[i].IN == True):
                for j in range(i+1, i + len(self.points)):
                    if(self.points[j%len(self.points)].C != self.points[i].C):
                        self.points[j%len(self.points)].EXT = False
                    else:
                        break
        self.points = list(filter(lambda x: x.EXT,self.points))

    def generate_edges(self):
        for i in range(len(self.points)):
            if(self.points[i].IN == False):
                self.edges.append(Edge(self.points[i].P, self.points[(i+1)%(len(self.points))].P,self))
                self.gradient += array([[0,1],[-1,0]])@(self.edges[-1].P2-self.edges[-1].P1)
        
         
    def plot(self):
        plt.plot(self.C[0],self.C[1], 'ro')
        theta = linspace(0,2*pi, 1000)
        plt.plot(self.C[0] + self.r*cos(theta), self.C[1] + self.r*sin(theta), 'k')
        for edge in self.edges:
            edge.plot()
        #plt.arrow(self.C[0], self.C[1], self.gradient[0], self.gradient[1])
            
class Point:
    N = 0
    def __init__(self, P, C, IN = True, Cbis = None, normal = None):
        self.P = P
        self.C = C
        self.IN = IN
        self.N = Point.N 
        Point.N += 1
        self.EXT = True
        self.Cbis = Cbis
        self.normal = normal
    
    def plot(self,color):
        plt.plot(self.P[0],self.P[1], color)
        plt.text(self.P[0]+0.05, self.P[1]-0.6+0.3*self.IN, str(self.N))

class Edge:
    def __init__(self,P1,P2,C):
        self.P1 = P1
        self.P2 = P2
        self.C = C
    
    def plot(self):
        R1 = self.P1 - self.C.C; R2 = self.P2 - self.C.C
        t1 = arctan2(R1[1], R1[0]); t2 = arctan2(R2[1],R2[0])
        if(t1 > t2):
            t2 += 2*pi
        theta = linspace(t1, t2, 1000)
        plt.plot(self.C.C[0] + self.C.r*cos(theta), self.C.C[1]+ self.C.r*sin(theta), 'b')
        

class Graph():
    def __init__(self):
        self.circles = []
        self.pedges  = []
        self.ppedges = []
    
    def compute_intersections(self):
        self.ppedges = []
        for i in range(len(self.circles)):
            self.circles[i].points = []
            self.circles[i].edges = []
            self.circles[i].gradient *= 0
        for i in range(len(self.circles)):
            for j in range(i+1,len(self.circles)):
                self.circles[i].C_C_intersection(self.circles[j])
        for pedge in self.pedges:
            for circle in self.circles:
                pedge.P_C_intersection(circle)
                
        for circle in self.circles:
            circle.filter()
            circle.generate_edges()
        
        points = []
        I = 0
        for pedge in self.pedges:
            pedge.filter()
            points += pedge.points
        if(len(points) == 0 or points[0].IN):
            I=1
        points = []
        for pedge in self.pedges:
            points += [pedge.P1] + list(map(lambda x:x.P, pedge.points)) + [pedge.P2]
        for i in range(I, len(points)-1,2):
            self.ppedges.append([points[i], points[i+1]])
            
    def gradient(self):
        gradient = zeros(len(self.circles)*2+2)
        for i in range(len(self.circles)):
            gradient[2*i] += self.circles[i].gradient[0]
            gradient[2*i+1] += self.circles[i].gradient[1]
        return gradient
    
    def hessian(self):
        H = zeros((len(self.circles)*2+2,len(self.circles)*2))
        for circle in self.circles:
            for point in circle.points:
                Ri = point.P-circle.C
                Ri /= linalg.norm(Ri)
                Rj = point.normal
                if(point.Cbis != None):
                    Rj = point.P-point.C.C
                    Rj /= linalg.norm(Rj)
                plt.arrow(point.P[0], point.P[1], Rj[0], Rj[1])
                plt.arrow(point.P[0], point.P[1], Ri[0]/2, Ri[1]/2, width = 0.03)
                SM = outer(Rj,Ri)/abs(cross(Ri,Rj))
                H[2*circle.N,   2*circle.N  ] += SM[0,0]
                H[2*circle.N,   2*circle.N+1] += SM[0,1]
                H[2*circle.N+1, 2*circle.N  ] += SM[1,0]
                H[2*circle.N+1, 2*circle.N+1] += SM[1,1]
                if(point.Cbis != None):
                    H[2*point.C.N,   2*circle.N  ] -= SM[0,0]
                    H[2*point.C.N,   2*circle.N+1] -= SM[0,1]
                    H[2*point.C.N+1, 2*circle.N  ] -= SM[1,0]
                    H[2*point.C.N+1, 2*circle.N+1] -= SM[1,1]
        H[-2] = array([i%2 for i in range(len(H)-2)])
        H[-1] = array([(i+1)%2 for i in range(len(H)-2)])
        return H
            
    def update(self):
        self.compute_intersections()
        for circle in self.circles:
            plt.plot(circle.C[0], circle.C[1],'r+')
            circle.C += circle.gradient*0.1
            circle.points = []
            circle.edges = []
            circle.gradient = zeros(2)
            Point.N = 0
            
        self.ppedges = []
        for pedge in self.pedges:
            pedge.points = []
            
    def update_with_hessian(self):
        self.compute_intersections()
        G = self.gradient()
        H = self.hessian()
        G = -linalg.inv(H)@G
        for i in range(len(self.circles)):
            circle = self.circles[i]
            plt.plot(circle.C[0], circle.C[1],'r+')
            circle.C += G[2*i:2*i+2]
            circle.points = []
            circle.edges = []
            circle.gradient = zeros(2)
            Point.N = 0
            
        self.ppedges = []
        for pedge in self.pedges:
            pedge.points = []
        return linalg.norm(G)
    
    def plot(self):
        for circle in self.circles:
            circle.plot()
            plt.arrow(circle.C[0], circle.C[1], circle.gradient[0], circle.gradient[1])
            for point in circle.points:
                point.plot('go')
        for pedge in self.pedges:
            pedge.plot()
        for ppedge in self.ppedges:
            plt.plot([ppedge[0][0], ppedge[1][0]],[ppedge[0][1], ppedge[1][1]], 'b')

class Pedge:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        self.points = []
        self.edges = []
        
    def P_C_intersection(self, C):
        P = self.P1; Q = self.P2; R = C.C
        PQ = Q - P; PR = R - P
        h = (PQ[0]*PR[1] - PQ[1]*PR[0])/(linalg.norm(PQ))
        if(abs(h) < C.r):
            Epq = PQ/linalg.norm(PQ)
            EQpq = array([-Epq[1], Epq[0]])
            S = C.C - EQpq*h
            b = sqrt(C.r*C.r - h*h)
            p = (P-S)@Epq
            q = (Q-S)@Epq
            if(p<b and q>b):
                POINT = Point(S + b*Epq, C, normal = -EQpq)
                self.points.append(POINT)
                C.points.append(POINT)
            if(p<-b and q>-b):
                POINT = Point(S - b*Epq, C, IN = False, normal = -EQpq)
                self.points.append(POINT)
                C.points.append(POINT)

    def filter(self):
        self.points = list(filter(lambda x: x.EXT, self.points))
        self.points.sort(key = lambda x: (x.P-self.P1)@(self.P2-self.P1))
            
    def plot(self):
        plt.plot([self.P1[0],self.P2[0]],[self.P1[1],self.P2[1]], 'r')
        for point in self.points:
            if(point.IN):
                point.plot('bo')
            else:
                point.plot('go')

        

graph = Graph()
#graph.circles.append(Circle(array([0.0,0]), 10))
#graph.circles.append(Circle(array([0.0,10*sqrt(2)]), 10))
#graph.circles.append(Circle(array([0.0,15]), 10))
graph.circles.append(Circle(array([0.0,10/sqrt(2)]), 10))
graph.circles.append(Circle(array([0.0,-10/sqrt(2)]), 10))
graph.circles.append(Circle(array([15,0]), 10))


'''graph.pedges.append(Pedge(array([0,-15]),array([0,15])))
graph.pedges.append(Pedge(array([0,15]),array([50,0])))
graph.pedges.append(Pedge(array([50,0]),array([0,-15])))
'''
graph.pedges.append(Pedge(array([0,-30]),array([-30,0])))
graph.pedges.append(Pedge(array([-30,0]),array([0,30])))
graph.pedges.append(Pedge(array([0,30]),array([30,0])))
graph.pedges.append(Pedge(array([30,0]),array([0,-30])))


graph.compute_intersections()
G1 = graph.gradient()
print("G1 = ", G1)
H = graph.hessian()
print("H = \n", np.round(H,2))

Hinv = np.linalg.pinv(H)
print("Hinv = ", Hinv@G1)

#for i in range(10):
#    print(graph.update_with_hessian())
    
graph.plot()
plt.axis('equal')
plt.show()


