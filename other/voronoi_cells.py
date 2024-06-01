from foronoi import Voronoi, Polygon
import numpy as np
import matplotlib.pyplot as plt

# Define some points (a.k.a sites or cell points)
x = np.array([
    [2.5, 2.5],
    [4, 7.5],
    [7.5, 2.5],
    [6, 7.5],
    [4, 4],
    [3, 3],
    [6, 3]
]) 

# Define a bounding box / polygon
p = np.array([
    (2.5, 10),
    (5, 10),
    (10, 5),
    (10, 2.5),
    (5, 0),
    (2.5, 0),
    (0, 2.5),
    (0, 5),
])

def center_of_mass(vertices):
    vx = vertices.T[0]
    vy = vertices.T[1]
    A = 0.5*sum(vx[1:]*vy[:-1] - vx[:-1]*vy[1:])
    xG = 1/(6*A)*sum( (vx[:-1] + vx[1:])*(vx[1:]*vy[:-1] - vx[:-1]*vy[1:]))
    yG = 1/(6*A)*sum( (vy[:-1] + vy[1:])*(vx[1:]*vy[:-1] - vx[:-1]*vy[1:]))
    return np.array([xG,yG])

def centroids(points, polygon, plot = False):
    v = Voronoi(Polygon(polygon))
    v.create_diagram(points)
    gradient = np.zeros((len(points), 2))
    for i,site in enumerate(v.sites):
        edge = site.first_edge
        vertices = []
        while(edge!= site.first_edge.next or len(vertices) == 1):
            vertices.append([edge.origin.x, edge.origin.y])
            edge = edge.next
        G = center_of_mass(np.array(vertices))
        P = np.array([site.x, site.y])
        gradient[i] = G - P
        if(plot):
            plt.plot(site.x, site.y, 'bo')
            plt.text(site.x,site.y, str(site.name))
            plt.plot(np.array(vertices).T[0], np.array(vertices).T[1])
            plt.plot(G[0], G[1],'bo')
    if(plot):
        plt.axis('equal')
        plt.show()
    return gradient




import math

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)
def circle_from_two_points(p1, p2):
    center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    radius = dist(p1, p2) / 2
    return center, radius
def circle_from_three_points(p1, p2, p3):
    A, B, C = p1, p2, p3
    D = 2 * (A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
    Ux = ((A[0]**2 + A[1]**2)*(B[1] - C[1]) + (B[0]**2 + B[1]**2)*(C[1] - A[1]) + (C[0]**2 + C[1]**2)*(A[1] - B[1])) / D
    Uy = ((A[0]**2 + A[1]**2)*(C[0] - B[0]) + (B[0]**2 + B[1]**2)*(A[0] - C[0]) + (C[0]**2 + C[1]**2)*(B[0] - A[0])) / D
    center = (Ux, Uy)
    radius = dist(center, A)
    return center, radius
def is_in_circle(p, center, radius):
    return dist(p, center) <= radius + 1e-9
def welzl(P, R=[]):
    if not P or len(R) == 3:
        if len(R) == 0: return ((0, 0), 0)
        if len(R) == 1: return (R[0], 0)
        if len(R) == 2: return circle_from_two_points(R[0], R[1])
        return circle_from_three_points(R[0], R[1], R[2])
    
    p = P.pop()
    center, radius = welzl(P, R)
    
    if is_in_circle(p, center, radius):
        P.append(p)
        return center, radius
    
    R.append(p)
    center, radius = welzl(P, R)
    P.append(p)
    R.pop()
    return center, radius
def smallest_enclosing_circle(points):
    P = list(points)
    return welzl(P)[0]

center = smallest_enclosing_circle(p)

def minimax(points, polygon, plot = False):
    v = Voronoi(Polygon(polygon))
    v.create_diagram(points)
    gradient = np.zeros((len(points), 2))
    for i,site in enumerate(v.sites):
        edge = site.first_edge
        vertices = []
        while(edge!= site.first_edge.next or len(vertices) == 1):
            vertices.append([edge.origin.x, edge.origin.y])
            edge = edge.next
        G = smallest_enclosing_circle(np.array(vertices))
        P = np.array([site.x, site.y])
        gradient[i] = G - P
        # plt.plot(site.x, site.y, 'rx')
        if(plot):
            plt.plot(site.x, site.y, 'bo')
            plt.text(site.x,site.y, str(site.name))
            plt.plot(np.array(vertices).T[0], np.array(vertices).T[1])
            plt.plot(G[0], G[1],'bo')
    if(plot):
        plt.axis('equal')
        plt.show()
    return gradient


def alternative_algorithm(x, name):
    poly = p
    if name == 'minimax':
        return minimax(x, poly)
    if name == 'centroid':
        return centroids(x, poly)
    else:
        return None
    
    

    

