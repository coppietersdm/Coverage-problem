import numpy as np
import matplotlib.pyplot as plt
   

class Circle: 
    def __init__(self, C, r, N):
        self.C = C
        self.r = r
        self.N = N
        self.force = np.array([0.0,0.0])
        
    def plot(self):
        theta = np.linspace(0,2*np.pi,1000)
        plt.plot(self.C[0], self.C[1], 'ko')
        plt.text(self.C[0], self.C[1], str(self.N))
        plt.plot(self.C[0]+self.r*np.cos(theta), self.C[1] + self.r*np.sin(theta), 'r')
        plt.arrow(circle.C[0], circle.C[1], circle.force[0], circle.force[1], width = 0.03)


class Polygon:
    def __init__(self, polygon):
        self.poly = np.array(polygon)
        
    def plot(self):
        plt.plot(self.poly[:,0], self.poly[:,1], 'k')
        
class Graph:
    def __init__(self, polygon, color):
        self.circles = []
        self.polygon = polygon
        
    def add_circle(self, NC):
        self.circles.append(NC)
        
        #compute the intersection with the polygon
        PI = []
        for i in range(len(self.polygon.poly)-1):
            A = self.polygon.poly[i]; B = self.polygon.poly[i+1]; D = NC.C
            U = B-A; V = D-A
            u = U/np.linalg.norm(U); w = np.array((u[1],-u[0]))
            height = u[0]*V[1] - u[1]*V[0]
            if(abs(height) < NC.r):
                b = np.sqrt(NC.r*NC.r - height*height)
                P1 = NC.C + height*w + b*u
                P2 = NC.C + height*w - b*u
                if((P1-A)@u/np.linalg.norm(U) < 1 and (P1-A)@u > 0):
                    #plt.plot(P1[0], P1[1], 'go')
                    PI.append([P1,True])
                if((P2-A)@u < np.linalg.norm(U) and (P2-A)@u > 0):
                    #plt.plot(P2[0], P2[1], 'yo')
                    PI.append([P2, False])
                    
        PI.sort(key=lambda x: np.arctan2(x[0][1] - NC.C[1], x[0][0] - NC.C[0]))
        if(len(PI) != 0):
            if(PI[0][1] == False):
                PI = PI[1:] + PI[0:1]
        for i in range(0,len(PI),2):
            A = PI[i][0]
            B = PI[i+1][0]
            C = (A+B)/2
            dC = np.array([-B[1] + A[1], B[0] - A[0]])
            #plt.plot([A[0], B[0]],[A[1],B[1]], 'g--')
            #plt.arrow(C[0], C[1],dC[0]/5,dC[1]/5, width = 0.03)
            NC.force += dC
            
    def plot_graph(self):
        for circle in self.circles:
            circle.plot()
            
        self.polygon.plot()


graph = Graph(Polygon([[0,0],[0,6],[6,6],[1,5],[6,4],[1,3],[6,2],[1,1],[6,0],[0,0]]),'green')
#graph = Graph(Polygon([[0,0],[0,6],[6,0],[0,0]]),'green')

radius = 0.5
R = 1.5
"""for i in np.linspace(0,2*np.pi,20):
    graph.add_circle(Circle([3 + R*np.cos(i),3 + R*np.sin(i)], radius, 0))
"""

for i in np.linspace(-0.1,6.1,10):
    for j in np.linspace(-0.1,6.1,10):
        graph.add_circle(Circle([i,j], radius, 0))
trajectories = [[]]*len(graph.circles)
for i in range(1000):
    print(i)
    newgraph = Graph(graph.polygon,'black')
    for circle in graph.circles:
        circle.force += np.random.randn(2)/1000
        newgraph.add_circle(Circle(circle.C+circle.force*0.05, circle.r, circle.N))
        
    graph = newgraph
    for j in range(len(graph.circles)):
        trajectories[j].append(graph.circles[j].C)

for traj in trajectories:
    plt.plot(np.array(traj)[:,0], np.array(traj)[:,1], 'ko', markersize = 0.01)
graph.plot_graph()

plt.axis('equal')
plt.show()
