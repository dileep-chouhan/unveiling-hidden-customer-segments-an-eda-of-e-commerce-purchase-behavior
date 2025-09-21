import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_customers = 100
data = {
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, num_customers),
    'AnnualIncome': np.random.randint(20000, 150000, num_customers),
    'SpendingScore': np.random.randint(1, 101, num_customers),
    'ProductA': np.random.randint(0, 10, num_customers), # Number of Product A purchased
    'ProductB': np.random.randint(0, 10, num_customers), # Number of Product B purchased
    'ProductC': np.random.randint(0, 10, num_customers)  # Number of Product C purchased
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this example, the data is already clean.
# --- 3. Analysis: Customer Segmentation using Hierarchical Clustering ---
# Select features for clustering (exclude CustomerID)
X = df[['Age', 'AnnualIncome', 'SpendingScore', 'ProductA', 'ProductB', 'ProductC']]
# Perform hierarchical clustering
linked = linkage(X, 'ward')
# --- 4. Visualization: Dendrogram ---
plt.figure(figsize=(12, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.tight_layout()
# Save the dendrogram plot
dendrogram_filename = 'dendrogram.png'
plt.savefig(dendrogram_filename)
print(f"Plot saved to {dendrogram_filename}")
# --- 5. Visualization: Customer Segmentation based on two features ---
plt.figure(figsize=(8,6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue=df['CustomerID'], data=df, palette='viridis', legend=False)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.tight_layout()
# Save the scatter plot
scatter_filename = 'customer_segmentation.png'
plt.savefig(scatter_filename)
print(f"Plot saved to {scatter_filename}")
print("Analysis Complete.")