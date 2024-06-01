import numpy as np
import matplotlib.pyplot as plt

d_crit = 0.2
R = 1

v_crit = 2*(1-np.sqrt(1-d_crit/(2*R)))
A = v_crit/d_crit


def f(d):
    return A*(d - 2*d_crit)**2 * (d<=2*d_crit)

def fd(d):
    return 2*A*(d - 2*d_crit) * (d<=2*d_crit)

def fdd(d):
    return 2*A * (d<=2*d_crit)


def avoidance(x):
    x = x.reshape(-1,2)
    print(x)
    toReturn = 0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if(i<j):
                dij = np.linalg.norm(xi-xj)
                toReturn += f(dij)
    return toReturn

def avoidance_d(x):
    x = x.reshape(-1,2)
    gradient = x*0
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if(i<j):
                dij = np.linalg.norm(xi-xj)
                gradient[i] += fd(dij)*(xi-xj)/dij
                gradient[j] -= fd(dij)*(xi-xj)/dij
            
    return gradient.reshape(-1)

def avoidance_dd(x):
    hessian = np.zeros((len(x), len(x)))
    x = x.reshape(-1,2)
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            if(i<j):
                subhessian = hessian.copy()*0
                dij = np.linalg.norm(xi-xj)
                grad_i_dij = (xi-xj)/dij
                h = (dij*np.eye(2) - np.outer(grad_i_dij,(xi-xj)))/(dij*dij)
                subhessian[2*i:2*i+2,2*i:2*i+2] = h
                subhessian[2*j:2*j+2,2*i:2*i+2] = -h
                subhessian[2*i:2*i+2,2*j:2*j+2] = -h
                subhessian[2*j:2*j+2,2*j:2*j+2] = h
                
                grad_dij = np.zeros(2*len(x))
                grad_dij[2*i:2*i+2] = grad_i_dij
                grad_dij[2*j:2*j+2] = -grad_i_dij
                hessian += fdd(dij)*np.outer(grad_dij,grad_dij) + fd(dij)*subhessian

    return hessian


def avoidance_function(x):
    return avoidance(x), avoidance_d(x), avoidance_dd(x)

x0, y0 = 0.2,0
x = np.array([x0,y0,0,0])
F = avoidance(x)
G = avoidance_d(x)[0:2]
H = avoidance_dd(x)[0:2,0:2]
print('F = ', F)
print('G = ', G)
print('h = ',H)



# Generate 3D grid of points
x = np.linspace(-3*d_crit, 3*d_crit, 100)
y = np.linspace(-3*d_crit, 3*d_crit, 100)
X, Y = np.meshgrid(x, y)

# Compute distance from origin for each point
d = np.sqrt(X**2 + Y**2)

# Compute f(d) for each point
Z = f(d)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

# Compute the Taylor series approximation
Z_approx = F + G[0]*(X-x0) + G[1]*(Y-y0) + 0.5*(H[0,0]*(X-x0)**2 + H[1,1]*(Y-y0)**2 + 2*H[0,1]*(X-x0)*(Y-y0))

# Plot the Taylor series approximation on a small region around the point
ax.plot_surface(X, Y, Z_approx, alpha=0.5)

# Plot a point at the evaluation of the Taylor series
ax.scatter(x0, y0, F, color='red', s=100)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel(r'$D(\vec{x})$')
ax.set_title(r'3D Plot of $D(\vec{x})$ with Taylor Series Approximation')

# Set z-axis limit
ax.set_zlim(0, 0.1)

# Show the plot
plt.savefig('avoidance 3D.pdf')
plt.show()

# def Dd(x):
#     return avoidance_d(np.array(list(x) + [0,0]))[0:2]


# # Generate 2D grid of points
# x = np.linspace(-3*d_crit, 3*d_crit, 100)
# y = np.linspace(-3*d_crit, 3*d_crit, 100)
# X, Y = np.meshgrid(x, y)

# # Compute Dd(x) for each point
# U, V = np.zeros_like(X), np.zeros_like(Y)
# for i in range(len(x)):
#     for j in range(len(y)):
#         position = np.array([X[i, j], Y[i, j]])
#         vector = Dd(position)
#         U[i, j] = vector[0]
#         V[i, j] = vector[1]

# # Compute amplitude of the vector field
# amplitude = np.sqrt(U**2 + V**2)

# # Create streamplot with color showing amplitude
# plt.streamplot(X, Y, -U, -V, color=amplitude, cmap='viridis', density=3)

# # Set labels and title
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title(r'Streamplot of $\nabla D(\vec{x})$ with Amplitude')
# plt.axis('equal')
# plt.colorbar(label='Amplitude')
# plt.savefig('avoidance_streamplot.pdf')
# # Show the plot
# plt.show()