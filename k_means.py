# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 17:15:19 2025

@author: Nagesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from sklearn.cluster import KMeans, AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Load dataset
data = pd.read_csv(r"C:\Users\Nagesh\Desktop\Naresh IT\Machine_Learning\(18)2.K-MEANS CLUSTERING\Mall_Customers.csv")
x = data.iloc[:, [3, 4]].values

# Plot functions
def kmeans_clustering():
    kmeans = KMeans(n_clusters=5, random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    plt.figure()
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('KMeans Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1–100)')
    plt.legend()
    plt.show()

def hierarchical_clustering():
    plt.figure()
    dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
    plt.title('Dendrogram (Hierarchical Clustering)')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    plt.show()

    hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(x)

    plt.figure()
    plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.title('Hierarchical Clustering')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1–100)')
    plt.legend()
    plt.show()

# Tkinter GUI
def run_clustering():
    method = algo_choice.get().lower()
    if method == "kmeans":
        kmeans_clustering()
    elif method == "hierarchical":
        hierarchical_clustering()
    else:
        messagebox.showerror("Error", "Please enter either 'kmeans' or 'hierarchical'.")

root = Tk()
root.title("Clustering Algorithm Selector")
root.geometry("400x200")

Label(root, text="Enter Clustering Method (kmeans/hierarchical):").pack(pady=10)
algo_choice = Entry(root, width=30)
algo_choice.pack()

Button(root, text="Run Clustering", command=run_clustering).pack(pady=20)

root.mainloop()
