# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 08:16:13 2023

@author: priti
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
r = 0.25 # Interaction rate
d = 0.01   # Death rate for infected
L = 100      # Length of the domain
T = 100   # Total time
Nx = 100   # Number of spatial grid points
Nt = 1000  # Number of time steps
D = 0.25  #Diffusion rate

# Discretization of space and time
dx = L/Nx
dt =T/Nt
print(dx, dt)

# Initialize arrays
S = np.zeros((Nx, Nt)) # Susceptible population
I = np.zeros((Nx, Nt)) # Infected population

# Initial condition

S[:,0] = np.random.rand(Nx)
S[int(Nx/2):int(Nx*0.6),0] = 0.5 # heterogeneous distribution of susceptible population
I[:,0] = np.zeros(Nx)
I[int(Nx/2),0] = 0.02


#Looping over time and space with explicit Forward time and centered symmetrical space FD approach with periodic boundary conditions
for n in range(0, Nt-1):
    for i in range(0, Nx-1):
        I[i,n+1] = I[i,n] + dt*r*S[i,n]*I[i,n] - dt*d*I[i,n] + dt*D*(I[(i+1) % Nx,n] + I[(i-1) % Nx,n] - 2*I[i,n])/dx**2
    # no diffusion for S population
        S[:,n+1] = S[:,n] - dt*r*S[:,n]*I[:,n]
    # Periodic boundary conditions
    S[0,n+1] = S[Nx-1,n+1] # derivative of S at the left boundary is equal to the value at the right boundary
    S[Nx-1,n+1] = S[0,n+1] # derivative of S at the right boundary is equal to the value at the left boundary
    I[0,n+1] = I[Nx-1,n+1] # derivative of I at the left boundary is equal to the value at the right boundary
    I[Nx-1,n+1] = I[0,n+1] # derivative of I at the right boundary is equal to the value at the left boundary




# Plot the Space vs population density
for i in range(Nt):
    plt.plot( np.linspace(0, L, Nx),S[:,i], label='t={}'.format(i*dt), linestyle='--', color='blue')
    plt.plot( np.linspace(0, L, Nx),I[:,i],  label='t={}'.format(i*dt), color='red')
plt.ylabel('Population density')
plt.xlabel('Space')
plt.title('2D plot of Population density vs Spcae with periodic boundary conditions & Explicit FTCS FD scheme')
plt.text(10, .98, 'Susceptible', fontsize=10, color='blue')
plt.text(60, .98, 'Infected',fontsize=10, color='red')
plt.show()
plt.text

# Plot the Time vs population density
for i in range(Nx):
    plt.plot(np.linspace(0, T, Nt), S[i,:], label='x={}'.format(i*dx), linestyle='--', color='blue')
    plt.plot(np.linspace(0, T, Nt), I[i,:], label='x={}'.format(i*dx), color='red')
plt.ylabel('Population density')
plt.xlabel('Time')
plt.title('2D plot of Population density vs Time with periodic boundary conditions & Explicit FTCS FD scheme')
plt.text(60, .98, 'Susceptible', fontsize=10, color='blue')
plt.text(90, .98, 'Infected',fontsize=10, color='red')
plt.show()
plt.text


# 3D plot for susceptible population
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
X, Z = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, T, Nt))
ax1.plot_surface(X, Z, S.T, cmap='YlGn')
ax1.set_xlabel('Position')
ax1.set_ylabel('Time')
ax1.set_zlabel('Population density')
#ax1.invert_zaxis()
ax1.view_init(30, 45)
cbar = fig1.colorbar(ax1.collections[0], ax=ax1)
cbar.set_label('Susceptible population range')
ax1.set_title('3D plot of Susceptible population in time and space with Explicit FTCS FD scheme')

# 3D plot for infected population
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
X, Z = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, T, Nt))
ax2.plot_surface(X, Z, I.T, cmap='YlOrRd')
ax2.set_xlabel('Position')
ax2.set_ylabel('Time')
ax2.set_zlabel('Population density')
#ax2.invert_zaxis()
ax2.view_init(30, 45)
cbar = fig2.colorbar(ax2.collections[0], ax=ax2)
cbar.set_label('Infected Population range')
ax2.set_title('3D plot of Infected population in time and space with Explicit FTCS FD scheme')
plt.show()




