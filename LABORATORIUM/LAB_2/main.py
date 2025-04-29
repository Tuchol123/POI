import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from pyransac3d import Plane
import os
import matplotlib.pyplot as plt

# 1. Wczytaj chmurę punktów
def load_xyz(file_path):
    points = np.loadtxt(file_path)
    return points

# 2. Podział na 3 grupy k-means
def cluster_kmeans(points, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    labels = kmeans.labels_
    clusters = [points[labels == i] for i in range(n_clusters)]
    return clusters

# 3. Podział DBSCAN
def cluster_dbscan(points, eps=0.5, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_
    unique_labels = set(labels)
    clusters = [points[labels == label] for label in unique_labels if label != -1]
    return clusters

# 4. Dopasowanie płaszczyzny RANSAC
def fit_plane(points):
    plane = Plane()
    coefficients, inliers = plane.fit(points, thresh=0.01)
    a, b, c, d = coefficients
    normal_vector = np.array([a, b, c])
    return normal_vector, d

# 5. Klasyfikacja płaszczyzny
def classify_plane(normal_vector):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    z_axis = np.array([0, 0, 1])
    dot = np.abs(np.dot(normal_vector, z_axis))

    if dot > 0.9:
        return "Płaszczyzna pozioma"
    elif dot < 0.1:
        return "Płaszczyzna pionowa"
    else:
        return "Inna orientacja"

# 6. Zapis do .ply z kolorem
def save_cluster_as_ply(points, filename, color):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))
    o3d.io.write_point_cloud(filename, pc)

# 7. Główna funkcja
def main():
    file_path = "C:/Users/Adam/Desktop/STUDIA_POI/POI/LABORATORIUM/LAB_1/combined.xyz"
    output_dir = "clusters_output"
    os.makedirs(output_dir, exist_ok=True)

    points = load_xyz(file_path)

    # ================== KMeans ===================
    print("Klasteryzacja KMeans...")
    clusters = cluster_kmeans(points)

    kmeans_colors = [
        [1, 0, 0],  # czerwony
        [0, 1, 0],  # zielony
        [0, 0, 1],  # niebieski
    ]

    for idx, cluster in enumerate(clusters):
        normal_vector, d = fit_plane(cluster)
        classification = classify_plane(normal_vector)
        print(f"KMeans - Klaster {idx}: {classification}, wektor normalny: {normal_vector}")

        color = kmeans_colors[idx % len(kmeans_colors)]
        filename = os.path.join(output_dir, f"kmeans_cluster_{idx}.ply")
        save_cluster_as_ply(cluster, filename, color)

    # ================== DBSCAN ===================
    print("\nKlasteryzacja DBSCAN...")
    clusters_dbscan = cluster_dbscan(points)

    colormap = plt.get_cmap("tab20")  # 20 różnych kolorów
    for idx, cluster in enumerate(clusters_dbscan):
        normal_vector, d = fit_plane(cluster)
        classification = classify_plane(normal_vector)
        print(f"DBSCAN - Klaster {idx}: {classification}, wektor normalny: {normal_vector}")

        color = colormap(idx % 20)[:3]  # RGB (0-1)
        filename = os.path.join(output_dir, f"dbscan_cluster_{idx}.ply")
        save_cluster_as_ply(cluster, filename, color)

if __name__ == "__main__":
    main()
