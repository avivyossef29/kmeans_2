import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

def calculate_distance(x1, x2):
    # Function to calculate the Euclidean distance between two points
    dis = 0
    for j in range(len(x1)):
        dis += pow(x1[j] - x2[j], 2)
    return dis

def plot_elbow_method(x_vals, y_vals):
    # Function to plot the elbow method for selecting the optimal "K" clusters
    plt.plot(x_vals, y_vals, color='#0080FF')  # Plot the x and y values
    plt.scatter(x_vals[2], y_vals[2], facecolors='none', edgecolors='black', marker='o', s=200, linestyle='--')  # Scatter plot of the elbow point
    plt.annotate('Elbow Point', xy=(x_vals[2], y_vals[2]), xytext=(len(x_vals) / 3, np.max(y_vals) * 0.3),
                 arrowprops=dict(facecolor='black', arrowstyle='-|>', linewidth=1, linestyle='--', shrinkB=10), fontweight="bold")  # Annotation for the elbow point with an arrow
    plt.title("Elbow Method for selection of optimal \"K\" clusters", color='#994C00', style='italic', fontweight="bold", fontname="Times New Roman")  # Title of the plot
    plt.text(8, min(y_vals) - 0.18 * (max(y_vals) - min(y_vals)), 'K', fontsize=12, ha='center', fontweight="bold", fontname="Arial")  # Text annotation for 'K'
    plt.ylabel("Average Dispersion", fontsize=12, ha='center', fontweight="bold", fontname="Arial")  # Y-axis label for average dispersion

    plt.xticks(range(1, len(x_vals) + 1))  # Set the x-axis ticks
    plt.savefig("elbow.png", transparent=True)  # Save the plot as a PNG file
    plt.show()  # Display the plot

if __name__ == '__main__':
    data = load_iris()  # Load the Iris dataset
    df = pd.DataFrame(data.data, columns=data.feature_names)  # Create a DataFrame from the dataset
    dist = [None for _ in range(12)]  # Initialize a list for storing distances
    X = np.array(df)  # Convert the DataFrame to a numpy array

    for k in range(1, 11):
        # Iterate over the range of 'K' values
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)  # Perform K-means clustering
        centers = kmeans.cluster_centers_  # Get the cluster centers
        total = 0  # Initialize the total distance
        for i in range(df.shape[0]):
            # Iterate over each data point
            total += calculate_distance(list(df.iloc[i]), list(centers[kmeans.labels_[i]]))  # Calculate the distance and add it to the total
        dist[k] = total  # Store the total distance for the current 'K' value

    dist.remove(None)  # Remove the 'None' values from the distances list

    x_vals = list(range(1, 11))  # Create a list of 'K' values
    y_vals = dist[:10]  # Get the corresponding distances for the 'K' values

    plot_elbow_method(x_vals, y_vals)  # Call the function to plot the elbow method
