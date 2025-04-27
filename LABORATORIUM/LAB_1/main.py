import numpy as np

def generate_flat_surface_xy(width, length, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.random.uniform(0, length, num_points)
    z = np.zeros(num_points)
    return np.column_stack((x, y, z))

def generate_flat_surface_xz(width, height, num_points):
    x = np.random.uniform(0, width, num_points)
    y = np.zeros(num_points)
    z = np.random.uniform(0, height, num_points)
    return np.column_stack((x, y, z))

def generate_cylindrical_surface(radius, height, num_points):
    theta = np.random.uniform(0, 2*np.pi, num_points)
    z = np.random.uniform(0, height, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))

def save_to_xyz(points, filename):
    np.savetxt(filename, points, fmt='%.5f')

# Generowanie chmur punktów
flat_xy = generate_flat_surface_xy(width=10, length=10, num_points=1000)

flat_xz = generate_flat_surface_xz(width=10, height=5, num_points=1000)
flat_xz[:, 0] += 20  # przesunięcie w osi X

cylinder = generate_cylindrical_surface(radius=5, height=10, num_points=1000)
cylinder[:, 1] += 20  # przesunięcie w osi Y

# Łączenie wszystkich punktów razem
all_points = np.vstack((flat_xy, flat_xz, cylinder))

# Zapis do jednego pliku
save_to_xyz(all_points, "combined.xyz")
