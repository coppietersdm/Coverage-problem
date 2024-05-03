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
        try:
            gradient = zeros(2)
            if(self.points[0].Cin == self):
                self.points = self.points[1:] + self.points[0:1]
            for i in range(0, len(self.points),2):
                gradient -= self.points[i+1].P - self.points[i].P
            gradient = array([-gradient[1], gradient[0]])
            self.gradient = gradient
            return gradient
        except:
            return zeros(2)
    
    def integrate(self, plot=False):
        self.edges = []
        try:
            for i in range(0, len(self.points),2):
                self.edges.append([self.points[i].P,self.points[i+1].P])
        except:
            return -1
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
            circle.integrate(plot=True)
            for point in circle.points:
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


R = 1.0
iii = 0
def plot(x):
    global iii  # Add this line to access the global variable iii
    graph = Graph()
    graph.polygon(array([[-1,-1],[-1,1],[1,1],[1,-1]])*0.9)
    positions = np.reshape(x, (-1,2))

    for position in positions:
        graph.circles.append(Circle(position, R))
    graph.compute_gradients()
    graph.plot()

    plt.axis('equal')
    plt.show()
            
def function(x):
    graph = Graph()
    graph.polygon(array([[-1,-1],[-1,1],[1,1],[1,-1]]))
    positions = np.reshape(x, (-1,2))

    for position in positions:
        graph.circles.append(Circle(position, R))
        
    graph.compute_gradients()
    #graph.plot_positions()
    return graph.integral(), graph.graph_gradient(), graph.graph_hessian()
    
from scipy.optimize import minimize


def fun(x):
    return function(x)[0]

def jac(x):
    return -function(x)[1]

def hessi(x):
    return -function(x)[2]

def pinv_hessi_diag(x, epsilon):
    hessi = function(x)[2]
    H = hessi*0
    for i in range(len(x)//2):
        if(is_pos_def(hessi[2*i:2*i+2,2*i:2*i+2])):
            H[2*i:2*i+2,2*i:2*i+2] = linalg.pinv(hessi[2*i:2*i+2,2*i:2*i+2])
        else:
            H[2*i:2*i+2,2*i:2*i+2] = eye(2)*epsilon
    return H






xb = [-0.3740442, -1.99896368, -0.28323218, -1.84930609,  1.83896229,  1.47239203]
xb = np.array(xb)
plot(xb)
try:
    for i in range(100):
        x = np.random.rand(6)*4 - 2
        plot(x)
except:
    print(x)


# N = 100

# Z = np.array([[fun([x,y]) for x in linspace(-2,2, N)] for y in linspace(-2,2,N)])
# U = np.array([[jac([x,y])[0] for x in linspace(-2,2, N)] for y in linspace(-2,2,N)])
# V = np.array([[jac([x,y])[1] for x in linspace(-2,2, N)] for y in linspace(-2,2,N)])
# X = np.array([[x for x in linspace(-2,2, N)] for y in linspace(-2,2,N)])
# Y = np.array([[y for x in linspace(-2,2, N)] for y in linspace(-2,2,N)])
# print(Z)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')

# # # Tracé de la surface
# # surf = ax.plot_surface(X, Y, Z)
# # plt.show()

# plt.streamplot(X, Y, -U, -V, density=2, arrowsize=2)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Streamplot of fun(x, y)')
# plt.show()


# results99 = np.array([0,0])
# results999 = np.array([0,0])

# Ntrials = 20

# for k in range(Ntrials):
#     print(k)
#     poly_surf = 4.0 #fun([])
#     N = 5
#     x0 = random.rand(2*N)*2
#     epsilon = 0.1
#     max_step = 0.1


#     x= x0.copy()
#     surface1 = []
#     gradient1 = []
#     for i in range(100):
#         F, G, H = function(x)
#         dx = G*epsilon
#         if(is_pos_def(H)):
#             dx = linalg.pinv(H)@G
#             # plt.plot(x.reshape(-1,2)[:,0], x.reshape(-1,2)[:,1], 'kx')
#         if(linalg.norm(dx) != 0):
#             dx = dx/linalg.norm(dx)*min(max_step, linalg.norm(dx))        
#         x += dx
#         surface1.append((poly_surf-F)/poly_surf)
#         gradient1.append(linalg.norm(G)/poly_surf)
        
#     x= x0.copy()
#     surface2 = []
#     gradient2 = []
#     for i in range(100):
#         F, G, H = function(x)
#         dx = G*epsilon
#         if(linalg.norm(dx) != 0):
#             dx = dx/linalg.norm(dx)*min(max_step, linalg.norm(dx))
#         x += dx
#         surface2.append((poly_surf-F)/poly_surf)
#         gradient2.append(linalg.norm(G)/poly_surf)



    # x= x0.copy()
    # surface3 = []
    # for i in range(100):
    #     F, G, H = function(x)
    #     dx = G*epsilon
    #     dx = pinv_hessi_diag(x,epsilon)@G
    #     dx = dx/linalg.norm(dx)*min(max_step, linalg.norm(dx))
    #     x += dx
    #     surface3.append((poly_surf-F)/poly_surf)
        
    # plot(x)
    # plt.axis('equal')
    # plt.show()

#     plt.plot(surface1, color = 'b', label = 'S_hessian')
#     plt.plot(surface2, color = 'g', label = 'S_gradient')
#     plt.plot(gradient1, color = 'b', linestyle = '--')
#     plt.plot(gradient2, color = 'g', linestyle = '--')
#     # plt.plot(surface3, label = 'S_diag')
#     results99[0] += sum(np.array(surface1) < 0.95092*0.99)
#     results99[1] += sum(np.array(surface2) < 0.95092*0.99)
    
#     results999[0] += sum(np.array(surface1) < 0.95092*0.999)
#     results999[1] += sum(np.array(surface2) < 0.95092*0.999)


# print(results99/Ntrials)
# print(results999/Ntrials)
# plt.plot(0.95092*0.99*np.ones(100), color = 'k',linewidth = 2, label = '99%')
# #plt.legend()
# plt.show()



# N = 4
# error_hessian = []
# for i in range(1000):
#     x = np.random.rand(2*N)

#     F0 = fun(x)
#     G0 = jac(x)
#     H0 = hessi(x)
#     #print(np.round(H0,3))

#     dx = 2*(random.rand(2*N)+0.5)/10
#     # dx = np.array([0,0,0.01,0.01,0,0])
#     F1 = fun(x + dx)
#     G1 = jac(x + dx)
#     error_hessian.append(linalg.norm(G1 - G0 + H0@dx))

# print("mean = ", np.mean(error_hessian))
# print("rms  = ", np.mean(np.array(error_hessian)**2)**0.5)
# plt.plot(error_hessian)
# plt.show()


# plot(x)




# ---------------------------------------------------------------------
# Test of the gradient modeling
# ---------------------------------------------------------------------

# R = 0.4
# Ndrones = 10
# epsilon = 0.1
# x = np.random.rand(Ndrones*2)*2
# plot(x)
# G_pred = []
# surface = []
# for i in range(40):
#     F, G, H = function(x)
#     dx = G*epsilon
#     G_pred.append(-G@G*epsilon)
#     x += dx
#     surface.append(F)
# plot(x)

# plt.title("Evolution of the uncovered surface,\n its increment per iteration and the prediction based on the gradient")
# plt.plot(surface, label=r'$C(\vec{x})$')
# plt.plot(np.diff(np.array(surface)), label=r'$\Delta C(\vec{x})$')
# plt.plot(G_pred, label=r'$\epsilon |\nabla C(\vec{x})|^2$')
# plt.legend()
# plt.savefig("evolution_F_DF_G.pdf")


################################################################
################################################################
################################################################

# R = 1
# Ndrones = 4
# epsilon = 0.1
# rate_ = []
# stdev_grad = []
# mean_grad = []
# stdev_hess = []
# mean_hess = []
# for rate in np.arange(0.001, 0.1, 0.005):
#     rate_.append(rate)
#     print(rate)
#     gradient_error = []
#     hessian_error = []   
#     for i in range(300):
#         x = np.random.rand(Ndrones*2)*2
#         F0, G0, H0 = function(x)
#         dx = (np.random.rand(Ndrones*2)-0.5)*2*rate
#         dx = rate*G0
#         F1, G1, H1 = function(x+dx)
#         gradient_error.append(((F1-F0) + dx@G0))
#         hessian_error.append(((F1-F0) + dx@G0 - 0.5*dx@H0@dx))

#     stdev_grad.append(np.std(np.array(gradient_error)))
#     mean_grad.append(np.mean(np.array(gradient_error)))
#     stdev_hess.append(np.std(np.array(hessian_error)))
#     mean_hess.append(np.mean(np.array(hessian_error)))
#     # plt.plot(hessian_error, label = 'hessian_error')
#     # plt.plot(np.array(gradient_error)+1, label = 'gradient_error')
#     # plt.legend()
#     # plt.show()
# plt.plot(rate_, stdev_grad, label = 'stdev_grad')
# plt.plot(rate_, mean_grad, label = 'mean_grad')


# plt.plot(rate_, stdev_hess, label = 'stdev_hess')
# plt.plot(rate_, mean_hess, label = 'mean_hess')
# plt.legend()
# plt.show()



################################################################
# evaluation du gradient
################################################################

# def funny(x, a):
#     return a*x**2
# R = 0.5
# scale = arange(0.001,0.1,0.01)
# error = []
# error2 = []
# for rate in scale:
#     print(rate)
#     error_bis = []
#     error_tris = []
#     for k in range(100):
#         x = np.random.rand(2)*2
#         F0, G0, H0 = function(x)
#         dx = (np.random.rand(2)-0.5)*2*rate
#         F1, G1, H1 = function(x+dx)
#         error_bis.append(F1 - F0 + dx@G0)
#         error_tris.append(F1 - F0 + dx@(G0 - H0@dx/2))
#     error.append(np.std(error_bis))
#     error2.append(np.std(error_tris))
# plt.plot(scale, error, label = 'error')
# plt.plot(scale, error2, label = 'error2')
# import scipy
# a = (scipy.optimize.curve_fit(funny, scale, error))[0][0]
# plt.plot(scale,a*scale**2, label = 'fit')
# print(a)
# plt.legend()
# plt.show()